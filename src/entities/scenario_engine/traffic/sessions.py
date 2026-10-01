from __future__ import annotations

from datetime import datetime, timedelta
from typing import NamedTuple

import numpy as np
from numpy.random import Generator

from src.formulas.distributions import geometric_weights
from src.entities.scenario_engine.traffic.operations import draw_operations
from src.entities.scenario_engine.traffic.operators import Operator
from src.entities.scenario_engine.traffic.repository import OperatorKeys
from src.entities.scenario_engine.traffic.parameters import PRIMARY_ADDRESS_SHARE, TrafficSpecification
from src.entities.scenario_engine.traffic.regimes import REGIMES
from src.entities.scenario_engine.traffic.targets import session_targets


class PlannedSession(NamedTuple):
    session_id: str
    operator: Operator
    start: datetime


def address_weights(quantity: int) -> np.ndarray:
    """Peso de cada origem habitual, decaindo geometricamente da principal, que vem
    primeiro na lista.
    """
    return geometric_weights(quantity, PRIMARY_ADDRESS_SHARE)


def choose_source_address(rng: Generator, addresses: tuple[str, ...]) -> str:
    """De onde a sessao inteira parte: uma origem por sessao, nao por requisicao."""
    chosen = rng.choice(len(addresses), p=address_weights(len(addresses)))

    return addresses[int(chosen)]


def draw_request_count(
    rng: Generator,
    requests_range: tuple[int, int],
    specification: TrafficSpecification,
) -> int:
    """Quantas requisicoes a sessao emite: um valor da faixa e, com chance
    `long_session_chance`, um excesso geometrico de media `long_session_mean_excess`.
    """
    lowest, highest = requests_range
    typical = int(rng.integers(lowest, highest + 1))

    runs_long = rng.random() < specification.long_session_chance

    if not runs_long:
        return typical

    excess = int(rng.geometric(1.0 / specification.long_session_mean_excess))

    return typical + excess


def request_instants(
    rng: Generator, start: datetime, quantity: int, seconds_between: float
) -> list[datetime]:
    """Os instantes das requisicoes, a partir da abertura da sessao, com intervalo
    exponencial da media recebida.
    """
    gaps = rng.exponential(seconds_between, size=quantity - 1)
    offsets = np.concatenate(([0.0], np.cumsum(gaps)))
    instants = []

    for offset in offsets:
        instants.append(start + timedelta(seconds=float(offset)))

    return instants


def session_rows(
    rng: Generator,
    planned: PlannedSession,
    keys: OperatorKeys,
    specification: TrafficSpecification,
) -> list[dict[str, str]]:
    """As requisicoes de uma sessao, sem desfecho e sem rotulo.

    A ordem dos sorteios faz parte do resultado: todos consomem do mesmo fluxo,
    entao trocar dois de lugar muda o trafego inteiro com a mesma semente.
    """
    operator = planned.operator
    regime = REGIMES[operator.regime]

    source_address = choose_source_address(rng, operator.usual_ips)
    quantity = draw_request_count(rng, regime.requests_range, specification)

    instants = request_instants(
        rng, planned.start, quantity, regime.seconds_between_requests
    )
    operations = draw_operations(rng, operator.profile, quantity)
    targets = session_targets(rng, keys, quantity, specification)

    rows = []

    for instant, operation, target in zip(instants, operations, targets):
        rows.append({
            "session_id": planned.session_id,
            "operator_id": operator.operator_id,
            "timestamp": instant.isoformat(timespec="seconds"),
            "source_ip": source_address,
            "operation": operation,
            "key_id": target,
        })

    return rows
