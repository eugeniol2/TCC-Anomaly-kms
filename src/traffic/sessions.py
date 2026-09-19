"""O que acontece dentro de uma sessao: de onde vem, quanto dura, o que pede.

A sessao e a unidade de analise do trabalho (D-061), entao o que este arquivo
produz e exatamente uma linha do conjunto que os modelos vao classificar, ainda
em forma de requisicoes.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import NamedTuple

import numpy as np
from numpy.random import Generator

from src.traffic.operations import draw_operations
from src.traffic.operators import Operator
from src.traffic.repository import OperatorKeys
from src.traffic.specification import (
    PRINCIPAL_ADDRESS_SHARE,
    REGIMES,
    Regime,
    TrafficSpecification,
)
from src.traffic.targets import session_targets


class PlannedSession(NamedTuple):
    """Uma sessao ja situada no calendario, antes de ter conteudo."""

    session_id: str
    operator: Operator
    start: datetime


def address_weights(quantity: int) -> np.ndarray:
    """Peso de cada origem habitual, decaindo geometricamente da principal.

    A lista vem ordenada do M1, com a principal primeiro. O decaimento e o que
    faz um endereco pouco usado ter chance real de nao aparecer no aquecimento
    e produzir origem inedita legitima no periodo avaliado (D-040).
    """
    positions = np.arange(quantity)
    weights = PRINCIPAL_ADDRESS_SHARE * (1 - PRINCIPAL_ADDRESS_SHARE) ** positions

    return weights / weights.sum()


def choose_source_address(rng: Generator, addresses: tuple[str, ...]) -> str:
    """De onde a sessao inteira parte.

    Uma origem por sessao, nao por requisicao: quem muda de rede troca de
    sessao, e o atributo de origem inedita e lido no nivel da sessao.
    """
    chosen = rng.choice(len(addresses), p=address_weights(len(addresses)))

    return addresses[int(chosen)]


def draw_request_count(rng: Generator, regime: Regime) -> int:
    """Quantas requisicoes a sessao emite, pela faixa do regime."""
    lowest, highest = regime.requests_range

    return int(rng.integers(lowest, highest + 1))


def request_instants(
    rng: Generator, start: datetime, quantity: int, regime: Regime
) -> list[datetime]:
    """Os instantes das requisicoes, a partir da abertura da sessao.

    Intervalo exponencial, com media do regime. E dele que sai a duracao da
    sessao, e portanto a taxa de requisicoes — uma das cinco dimensoes que
    sigma interpola no M3.
    """
    gaps = rng.exponential(regime.seconds_between_requests, size=quantity - 1)
    offsets = np.concatenate(([0.0], np.cumsum(gaps)))

    return [start + timedelta(seconds=float(offset)) for offset in offsets]


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
    quantity = draw_request_count(rng, regime)

    instants = request_instants(rng, planned.start, quantity, regime)
    operations = draw_operations(rng, operator.profile, quantity)
    targets = session_targets(rng, keys, quantity, specification)

    return [
        {
            "session_id": planned.session_id,
            "operator_id": operator.operator_id,
            "timestamp": instant.isoformat(timespec="seconds"),
            "source_ip": source_address,
            "operation": operation,
            "key_id": target,
        }
        for instant, operation, target in zip(instants, operations, targets)
    ]
