from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from typing import NamedTuple

from numpy.random import Generator

from src.entities.scenario_engine.attack.stealth import Stealth
from src.entities.scenario_engine.traffic.operations import draw_operations
from src.entities.scenario_engine.traffic.operators import Operator
from src.entities.scenario_engine.traffic.parameters import TrafficSpecification
from src.entities.scenario_engine.traffic.repository import OperatorKeys
from src.entities.scenario_engine.traffic.sessions import (
    choose_source_address,
    draw_request_count,
    request_instants,
)
from src.entities.scenario_engine.traffic.targets import session_targets

ADDRESS_PREFIX = "10."


class CompromisedSession(NamedTuple):
    session_id: str
    operator: Operator
    start: datetime


def draw_novel_address(rng: Generator, known: frozenset[str]) -> str:
    """Endereco na mesma faixa privada que nenhum operador usa."""
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

    Com chance `novel_address_chance` e um endereco nunca visto; senao, um dos habituais
    do operador, sorteado com a mesma geometrica do trafego legitimo.
    """
    is_novel = rng.random() < stealth.novel_address_chance

    if is_novel:
        return draw_novel_address(rng, known)

    return choose_source_address(rng, operator.usual_ips)


def targeting_specification(
    traffic: TrafficSpecification, stealth: Stealth
) -> TrafficSpecification:
    """A especificacao do trafego legitimo com as tres grandezas que sigma move.

    Em sigma 1 os tres valores sao os originais, e o objeto devolvido e igual ao
    recebido.
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

    Sorteia na mesma ordem que o `session_rows` do trafego legitimo, e precisa continuar
    assim: e essa ordem que iguala os dois em sigma 1.
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

    rows = []

    for instant, operation, target in zip(instants, operations, targets):
        rows.append({
            "session_id": session.session_id,
            "operator_id": operator.operator_id,
            "timestamp": instant.isoformat(timespec="seconds"),
            "source_ip": source_address,
            "operation": operation,
            "key_id": target,
        })

    return rows
