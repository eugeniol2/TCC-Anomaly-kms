from __future__ import annotations

from typing import NamedTuple

import pandas as pd
from numpy.random import Generator

from src.entities.scenario_engine.attack.campaign import campaign_starts
from src.entities.scenario_engine.attack.parameters import AttackSpecification
from src.entities.scenario_engine.attack.sessions import CompromisedSession, compromised_rows
from src.entities.scenario_engine.attack.stealth import Stealth, stealth_of
from src.shared.phases import EVALUATED, belongs_to
from src.shared.rng import ATTACK, stream
from src.entities.scenario_engine.traffic.build import COLUMNS, chronological, with_event_ids
from src.entities.scenario_engine.traffic.operators import Operator, read_operators
from src.entities.scenario_engine.traffic.regimes import REGIMES
from src.entities.scenario_engine.traffic.repository import build_repository
from src.entities.scenario_engine.traffic.parameters import TrafficSpecification

ADMINISTRATOR = "administrator"

COMPROMISED_COLUMNS = ("session_id",)

RUN_COLUMNS = ("seed", "sigma", "compromised_admin")


class AttackOutput(NamedTuple):
    requests: pd.DataFrame
    compromised: pd.DataFrame
    run: pd.DataFrame


def identifier_of(operator: Operator) -> str:
    """Como os administradores se ordenam: pelo identificador."""
    return operator.operator_id


def administrators_of(operators: list[Operator]) -> list[Operator]:
    """Os oito administradores, ordenados pelo identificador: o sorteio nao depende da
    ordem das linhas do `operators.csv`.
    """
    holders = []

    for operator in operators:
        is_administrator = operator.profile == ADMINISTRATOR

        if is_administrator:
            holders.append(operator)

    return sorted(holders, key=identifier_of)


def draw_compromised_admin(rng: Generator, operators: list[Operator]) -> Operator:
    """Qual administrador tem a credencial comprometida.

    E o primeiro sorteio do fluxo de ataque, antes de qualquer um que dependa de sigma:
    as onze condicoes da mesma semente comprometem o mesmo administrador.
    """
    holders = administrators_of(operators)
    chosen = int(rng.integers(len(holders)))

    return holders[chosen]


def plan_campaign(
    rng: Generator, admin: Operator, sigma: float,
    traffic: TrafficSpecification, attack: AttackSpecification,
) -> tuple[list[CompromisedSession], Stealth]:
    """As sessoes da campanha, situadas no calendario e ainda sem conteudo.

    O identificador e provisorio: a renumeracao final decide o que vai para o disco.
    """
    regime = REGIMES[admin.regime]
    stealth = stealth_of(sigma, regime, traffic, attack)

    starts = campaign_starts(rng, attack.campaign_sessions, stealth)
    planned = []

    for number, start in enumerate(starts, start=1):
        session = CompromisedSession(f"attack_{number:05d}", admin, start)
        planned.append(session)

    return planned, stealth


def known_addresses_of(operators: list[Operator]) -> frozenset[str]:
    """Todo endereco habitual da populacao, para o inedito ser mesmo inedito."""
    addresses = set()

    for operator in operators:
        for address in operator.usual_ips:
            addresses.add(address)

    return frozenset(addresses)


def legitimate_of_evaluated(requests: pd.DataFrame) -> pd.DataFrame:
    """As semanas 5 a 8 do trafego legitimo, que e onde a campanha entra."""
    of_phase = requests[belongs_to(EVALUATED, requests["timestamp"])]

    return of_phase.drop(columns=["event_id"]).reset_index(drop=True)


def renumbered(merged: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Numera sessoes e eventos do zero, em ordem cronologica.

    Devolve tambem o de-para do identificador provisorio para o definitivo, de onde o
    `compromised_sessions.csv` tira os numeros das sessoes da campanha.
    """
    ordered = chronological(merged)

    identifiers = {}

    for number, session in enumerate(ordered["session_id"].unique(), start=1):
        identifiers[session] = f"session_{number:05d}"

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

    Emite tentativas, sem desfecho, e as linhas do atacante nao levam marca nenhuma.
    """
    rng = stream(seed, ATTACK)

    operators = read_operators(operators_table)
    admin = draw_compromised_admin(rng, operators)

    planned, stealth = plan_campaign(rng, admin, sigma, traffic, attack)

    repository = build_repository(keys_table, operators)
    keys = repository[admin.operator_id]
    known = known_addresses_of(operators)

    attacker_rows = []

    for session in planned:
        rows = compromised_rows(rng, session, keys, stealth, traffic, known)
        attacker_rows.extend(rows)

    merged = pd.concat(
        [legitimate_of_evaluated(requests), pd.DataFrame(attacker_rows)],
        ignore_index=True,
    )
    numbered, identifiers = renumbered(merged)

    compromised_ids = []

    for session in planned:
        compromised_ids.append(identifiers[session.session_id])

    compromised = pd.DataFrame({"session_id": compromised_ids}).sort_values(
        "session_id", ignore_index=True
    )

    run = pd.DataFrame(
        [{"seed": seed, "sigma": sigma, "compromised_admin": admin.operator_id}]
    )

    return AttackOutput(numbered[list(COLUMNS)], compromised, run)
