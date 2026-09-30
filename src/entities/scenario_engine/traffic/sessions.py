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

from src.formulas.distributions import geometric_weights
from src.entities.scenario_engine.traffic.operations import draw_operations
from src.entities.scenario_engine.traffic.operators import Operator
from src.entities.scenario_engine.traffic.repository import OperatorKeys
from src.entities.scenario_engine.traffic.parameters import PRIMARY_ADDRESS_SHARE, TrafficSpecification
from src.entities.scenario_engine.traffic.regimes import REGIMES
from src.entities.scenario_engine.traffic.targets import session_targets


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
    return geometric_weights(quantity, PRIMARY_ADDRESS_SHARE)


def choose_source_address(rng: Generator, addresses: tuple[str, ...]) -> str:
    """De onde a sessao inteira parte.

    Uma origem por sessao, nao por requisicao: quem muda de rede troca de
    sessao, e o atributo de origem inedita e lido no nivel da sessao.
    """
    chosen = rng.choice(len(addresses), p=address_weights(len(addresses)))

    return addresses[int(chosen)]


def draw_request_count(
    rng: Generator,
    requests_range: tuple[int, int],
    specification: TrafficSpecification,
) -> int:
    """Quantas requisicoes a sessao emite: o tipico, e as vezes muito mais.

    Recebe a faixa, e nao o regime inteiro, porque o M3 chama esta mesma
    funcao com uma faixa interpolada por sigma. Em sigma 1 a faixa recebida e
    a do regime, e as duas chamadas passam a ser indistinguiveis (D-082), a
    cauda inclusive, porque ela vem da especificacao, que e a mesma nos dois.

    **A faixa e o comprimento tipico, nao um teto** (D-097). Uma sessao em
    vinte se estende por um excesso geometrico: e a migracao em lote, a
    reprocessagem, a tentativa que repete. Sem isso a faixa era teto rigido e
    nenhuma sessao legitima passava de 40 eventos, o que fazia a regra `events`
    separar as classes sozinha em sigma baixo, por aritmetica de faixa e nao
    por comportamento.
    """
    lowest, highest = requests_range
    typical = int(rng.integers(lowest, highest + 1))

    runs_long = rng.random() < specification.long_session_chance

    if not runs_long:
        return typical

    excess = int(rng.geometric(1.0 / specification.long_session_excess))

    return typical + excess


def request_instants(
    rng: Generator, start: datetime, quantity: int, seconds_between: float
) -> list[datetime]:
    """Os instantes das requisicoes, a partir da abertura da sessao.

    Intervalo exponencial, com a media recebida. E dele que sai a duracao da
    sessao, e portanto a taxa de requisicoes, que e uma das cinco dimensoes que
    sigma interpola no M3, que reaproveita esta funcao com outra media.
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
