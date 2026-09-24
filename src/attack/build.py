"""Composicao do M3: mescla a campanha as semanas 5 a 8 do trafego legitimo.

O M3 **le o `requests.csv` do M2 e escreve outro**, na pasta de sigma. Nao
acrescenta linhas ao arquivo do M2, porque cada arquivo tem um unico modulo
que o escreve (D-049). E o que impede o M3 de mutar a entrada que o M4 do ramo
da semente vai ler.

Tres coisas saem daqui, e a ordem entre elas importa:

1. O **administrador comprometido**, primeiro sorteio do fluxo de ataque. Vem
   antes de qualquer coisa que dependa de sigma, e por isso e o mesmo nas onze
   condicoes da mesma semente (D-011).
2. O **`requests.csv` mesclado**, renumerado do zero em ordem cronologica.
3. O **`compromised_sessions.csv`**, com os identificadores das sessoes da
   campanha — o rotulo, que viaja fora do log (D-063).
"""

from __future__ import annotations

from typing import NamedTuple

import pandas as pd
from numpy.random import Generator

from src.attack.campaign import campaign_starts
from src.attack.parameters import AttackSpecification
from src.attack.sessions import CompromisedSession, compromised_rows
from src.attack.stealth import Stealth, stealth_of
from src.globals.phases import EVALUATED, belongs_to
from src.globals.rng import ATTACK, stream
from src.traffic.build import COLUMNS, chronological, with_event_ids
from src.traffic.operators import Operator, read_operators
from src.traffic.regimes import REGIMES
from src.traffic.repository import build_repository
from src.traffic.parameters import TrafficSpecification

ADMINISTRATOR = "administrator"

COMPROMISED_COLUMNS = ("session_id",)
"""As colunas de `compromised_sessions.csv` (D-084).

Uma so. O arquivo responde a unica pergunta que o M7 lhe faz — esta sessao e
comprometida? —, e quem e o administrador ja esta em `run.csv`. Acrescentar
`operator_id` aqui repetiria em 58 linhas o que ja esta registrado em uma.
"""

RUN_COLUMNS = ("seed", "sigma", "compromised_admin")
"""As colunas de `run.csv`, uma linha por execucao (D-012, D-085)."""


class AttackOutput(NamedTuple):
    """Os tres arquivos que o M3 produz."""

    requests: pd.DataFrame
    compromised: pd.DataFrame
    run: pd.DataFrame


def administrators_of(operators: list[Operator]) -> list[Operator]:
    """Os oito administradores, em ordem estavel de identificador.

    Ordem fixada antes do sorteio: sem isso, mudar a ordem das linhas de
    `operators.csv` trocaria o administrador comprometido da mesma semente, e
    o pareamento da D-011 dependeria da ordenacao de um arquivo.
    """
    holders = [
        operator for operator in operators if operator.profile == ADMINISTRATOR
    ]

    return sorted(holders, key=lambda operator: operator.operator_id)


def draw_compromised_admin(rng: Generator, operators: list[Operator]) -> Operator:
    """Qual administrador tem a credencial comprometida (D-010).

    **Primeiro sorteio do fluxo de ataque**, antes de qualquer consumo que
    dependa de sigma. E o que garante que as onze condicoes da mesma semente
    compartilhem o alvo, isolando o efeito de sigma (D-011).
    """
    holders = administrators_of(operators)
    chosen = int(rng.integers(len(holders)))

    return holders[chosen]


def plan_campaign(
    rng: Generator, admin: Operator, sigma: float,
    traffic: TrafficSpecification, attack: AttackSpecification,
) -> tuple[list[CompromisedSession], Stealth]:
    """As sessoes da campanha, situadas no calendario e ainda sem conteudo.

    O identificador aqui e provisorio: ele vale so para casar linha com
    sessao ate a renumeracao final, que e quem decide o identificador que vai
    para o disco.
    """
    regime = REGIMES[admin.regime]
    stealth = stealth_of(sigma, regime, traffic, attack)

    starts = campaign_starts(rng, attack.campaign_sessions, stealth)
    planned = [
        CompromisedSession(f"attack_{number:05d}", admin, start)
        for number, start in enumerate(starts, start=1)
    ]

    return planned, stealth


def known_addresses_of(operators: list[Operator]) -> frozenset[str]:
    """Todo endereco habitual da populacao, para o inedito ser mesmo inedito."""
    return frozenset(
        address for operator in operators for address in operator.usual_ips
    )


def legitimate_of_evaluated(requests: pd.DataFrame) -> pd.DataFrame:
    """As semanas 5 a 8 do trafego legitimo, que e onde a campanha entra.

    O arquivo do M2 tem as oito semanas; as quatro primeiras ficam no ramo da
    semente e nao podem aparecer aqui, porque o aquecimento e anterior ao
    ataque (D-048).
    """
    of_phase = requests[belongs_to(EVALUATED, requests["timestamp"])]

    return of_phase.drop(columns=["event_id"]).reset_index(drop=True)


def renumbered(merged: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Numera sessoes e eventos do zero, em ordem cronologica (D-083).

    Devolve tambem de-para do identificador provisorio para o definitivo, que
    e como o `compromised_sessions.csv` descobre os numeros que as sessoes da
    campanha receberam depois de embaralhadas com as legitimas.

    Renumerar tudo e obrigatorio, nao arrumacao. Mantidos os identificadores
    do M2 e dados numeros novos so as sessoes do atacante, elas ficariam todas
    no fim da faixa, e o identificador de sessao — que a D-015 mantem fora dos
    atributos justamente para nao carregar sinal — passaria a **anunciar o
    rotulo** para quem abrisse o arquivo.
    """
    ordered = chronological(merged)

    identifiers = {
        session: f"session_{number:05d}"
        for number, session in enumerate(ordered["session_id"].unique(), start=1)
    }
    ordered = ordered.assign(session_id=ordered["session_id"].map(identifiers))

    return with_event_ids(ordered), identifiers


def build_attack(
    seed: int,
    sigma: float,
    operators_table: pd.DataFrame,
    keys_table: pd.DataFrame,
    requests: pd.DataFrame,
    traffic: TrafficSpecification,
    attack: AttackSpecification,
) -> AttackOutput:
    """Do trafego legitimo das semanas 5 a 8 ao arquivo com a campanha dentro.

    Emite tentativas, nunca desfechos (D-013), e nao marca as linhas do
    atacante de forma nenhuma: quem sabe quais sao e o
    `compromised_sessions.csv`, que o M7 consulta so na fase avaliada.
    """
    rng = stream(seed, ATTACK)

    operators = read_operators(operators_table)
    admin = draw_compromised_admin(rng, operators)

    planned, stealth = plan_campaign(rng, admin, sigma, traffic, attack)

    repository = build_repository(keys_table, operators)
    keys = repository[admin.operator_id]
    known = known_addresses_of(operators)

    attacker_rows = [
        row
        for session in planned
        for row in compromised_rows(rng, session, keys, stealth, traffic, known)
    ]

    merged = pd.concat(
        [legitimate_of_evaluated(requests), pd.DataFrame(attacker_rows)],
        ignore_index=True,
    )
    numbered, identifiers = renumbered(merged)

    compromised = pd.DataFrame(
        {"session_id": [identifiers[session.session_id] for session in planned]}
    ).sort_values("session_id", ignore_index=True)

    run = pd.DataFrame(
        [{"seed": seed, "sigma": sigma, "compromised_admin": admin.operator_id}]
    )

    return AttackOutput(numbered[list(COLUMNS)], compromised, run)
