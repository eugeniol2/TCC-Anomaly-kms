"""O que acontece dentro de uma sessao comprometida.

Este arquivo chama **as mesmas funcoes que o M2 chama**, com os valores que
sigma produziu. Nao ha um gerador de sessao legitima e outro de sessao
maliciosa: ha um so, parametrizado.

Isso nao e economia de codigo, e a garantia central do desenho. Em sigma 1
todos os parametros interpolados valem o que o M2 usaria, entao a sessao
comprometida sai literalmente da mesma distribuicao que uma legitima daquele
administrador. Com dois geradores separados, a indistinguibilidade dependeria
de duas implementacoes concordarem, e bastaria calibrar uma delas para o piso
da varredura deixar de ser piso sem que nada acusasse.

A mistura de operacoes vem do perfil do administrador personificado (D-055) e
**nao** e dimensao de sigma: perfil de operacoes proprio carregaria sinal fora
do controle de sigma e sobreviveria em 1,0.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from typing import NamedTuple

from numpy.random import Generator

from src.attack.stealth import Stealth
from src.traffic.operations import draw_operations
from src.traffic.operators import Operator
from src.traffic.parameters import TrafficSpecification
from src.traffic.repository import OperatorKeys
from src.traffic.sessions import (
    choose_source_address,
    draw_request_count,
    request_instants,
)
from src.traffic.targets import session_targets

ADDRESS_PREFIX = "10."


class CompromisedSession(NamedTuple):
    """Uma sessao da campanha ja situada no calendario, antes do conteudo."""

    session_id: str
    operator: Operator
    start: datetime


def draw_novel_address(rng: Generator, known: frozenset[str]) -> str:
    """Endereco na mesma faixa privada que nenhum operador usa.

    Mesma faixa de proposito, pela razao da D-009: endereco com formato
    proprio separaria as classes sozinho, e o sinal tem de vir de o endereco
    ser inedito **para aquele operador**, nao de ele parecer estrangeiro.
    """
    while True:
        octets = rng.integers([0, 0, 1], [256, 256, 255])
        candidate = f"{ADDRESS_PREFIX}{octets[0]}.{octets[1]}.{octets[2]}"

        is_known = candidate in known

        if not is_known:
            return candidate


def choose_address(
    rng: Generator, operator: Operator, stealth: Stealth, known: frozenset[str]
) -> str:
    """De onde a sessao comprometida parte.

    Com chance `novel_address_chance` e um endereco nunca visto; caso
    contrario e um dos habituais do operador, sorteado com a **mesma
    geometrica** que o M2 usa. Sortear uniformemente entre os habituais
    pareceria furtivo e nao seria: o operador real usa o principal em 80 % das
    sessoes, e distribuicao diferente deixaria rastro em sigma 1.
    """
    is_novel = rng.random() < stealth.novel_address_chance

    if is_novel:
        return draw_novel_address(rng, known)

    return choose_source_address(rng, operator.usual_ips)


def targeting_specification(
    traffic: TrafficSpecification, stealth: Stealth
) -> TrafficSpecification:
    """A especificacao do M2 com as tres grandezas que sigma move.

    Substituir campos em vez de escrever outro seletor de alvo mantem uma
    implementacao so. Em sigma 1 os tres valores substituidos sao iguais aos
    originais, e o objeto devolvido e igual ao recebido.
    """
    return replace(
        traffic,
        distinct_keys_range=stealth.distinct_keys_range,
        stale_scope_rate=stealth.stale_scope_rate,
        absent_identifier_rate=stealth.absent_identifier_rate,
    )


def compromised_rows(
    rng: Generator,
    session: CompromisedSession,
    keys: OperatorKeys,
    stealth: Stealth,
    traffic: TrafficSpecification,
    known_addresses: frozenset[str],
) -> list[dict[str, str]]:
    """As requisicoes de uma sessao comprometida, sem desfecho e sem rotulo.

    Sem rotulo aqui de proposito (D-063): a marca de comprometimento viaja em
    `compromised_sessions.csv` e so encontra estas linhas no M7. Sem desfecho
    pela D-013: quem decide se a requisicao passa e o M4.

    A ordem dos sorteios e a mesma de `traffic.sessions.session_rows`, e
    precisa continuar sendo: e ela que faz sigma 1 consumir o fluxo do mesmo
    jeito que o trafego legitimo consumiria.
    """
    operator = session.operator

    source_address = choose_address(rng, operator, stealth, known_addresses)
    quantity = draw_request_count(rng, stealth.requests_range, traffic)

    instants = request_instants(
        rng, session.start, quantity, stealth.seconds_between_requests
    )
    operations = draw_operations(rng, operator.profile, quantity)
    targets = session_targets(
        rng, keys, quantity, targeting_specification(traffic, stealth)
    )

    return [
        {
            "session_id": session.session_id,
            "operator_id": operator.operator_id,
            "timestamp": instant.isoformat(timespec="seconds"),
            "source_ip": source_address,
            "operation": operation,
            "key_id": target,
        }
        for instant, operation, target in zip(instants, operations, targets)
    ]
