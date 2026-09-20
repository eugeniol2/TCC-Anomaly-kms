"""Composicao do M2: das duas tabelas estaticas ao trafego das sete semanas.

Mora separado da linha de comando para que o teste exercite exatamente o que a
execucao real exercita, e nao uma copia da sequencia de chamadas.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
from numpy.random import Generator

from src.globals.rng import TRAFFIC, stream
from src.traffic.calendar import session_starts
from src.traffic.operators import Operator, read_operators
from src.traffic.repository import OperatorKeys, build_repository
from src.traffic.sessions import PlannedSession, session_rows
from src.traffic.parameters import TrafficSpecification
from src.traffic.regimes import REGIMES

COLUMNS = (
    "event_id",
    "session_id",
    "operator_id",
    "timestamp",
    "source_ip",
    "operation",
    "key_id",
)
"""As colunas de `requests.csv`.

Sao as oito do `log.csv` (D-064) menos `outcome`, que so existe depois do M4.
O M5 junta as duas coisas e o log fecha, sem renomear nada — por isso o
identificador ja nasce `event_id` aqui, e nao `request_id`: e a mesma linha em
estagios diferentes, e dois nomes para um conceito so seria ruido.
"""


def plan_sessions(
    rng: Generator, operators: list[Operator], specification: TrafficSpecification
) -> list[PlannedSession]:
    """Todas as sessoes de todos os operadores, numeradas em ordem cronologica.

    O identificador sai da posicao no tempo, e nao do operador, para nao
    carregar perfil nem agrupamento. Ele fica preservado no conjunto mas nao e
    exposto como atributo do modelo (D-015).
    """
    scheduled: list[tuple[datetime, Operator]] = []

    for operator in operators:
        regime = REGIMES[operator.regime]
        scheduled.extend(
            (start, operator) for start in session_starts(rng, regime, specification)
        )

    scheduled.sort(key=lambda item: (item[0], item[1].operator_id))

    return [
        PlannedSession(f"session_{number:05d}", operator, start)
        for number, (start, operator) in enumerate(scheduled, start=1)
    ]


def request_rows(
    rng: Generator,
    planned: list[PlannedSession],
    repository: dict[str, OperatorKeys],
    specification: TrafficSpecification,
) -> list[dict[str, str]]:
    """As requisicoes de todas as sessoes, na ordem em que foram planejadas."""
    rows: list[dict[str, str]] = []

    for session in planned:
        keys = repository[session.operator.operator_id]
        rows.extend(session_rows(rng, session, keys, specification))

    return rows


def chronological(frame: pd.DataFrame) -> pd.DataFrame:
    """Ordena o arquivo no tempo, que e como um log de auditoria se le.

    O instante e ISO com segundos, entao a ordem alfabetica coincide com a
    cronologica. A ordenacao e estavel: requisicoes que caem no mesmo segundo
    mantem a ordem em que a sessao as emitiu.
    """
    ordered = frame.sort_values(["timestamp", "session_id"], kind="stable")

    return ordered.reset_index(drop=True)


def with_event_ids(frame: pd.DataFrame) -> pd.DataFrame:
    """Numera os eventos na ordem cronologica em que o log os registra."""
    identifiers = [f"event_{number:06d}" for number in range(1, len(frame) + 1)]

    numbered = frame.copy()
    numbered.insert(0, "event_id", identifiers)

    return numbered


def build_traffic(
    seed: int,
    operators_table: pd.DataFrame,
    keys_table: pd.DataFrame,
    specification: TrafficSpecification,
) -> pd.DataFrame:
    """Das duas tabelas do M1 as requisicoes legitimas das sete semanas.

    Emite tentativas, nunca desfechos (D-013). Nenhuma coluna diz se a
    requisicao vai ser autorizada: isso e do M4, e e o que faz o rotulo ser
    derivado da politica em vez de inventado aqui.
    """
    rng = stream(seed, TRAFFIC)

    operators = read_operators(operators_table) # Leitura de `operators.csv` e conversao para `Operator`.
    repository = build_repository(keys_table, operators) # Leitura de `keys.csv` e organizacao de chaves por operador.
    planned = plan_sessions(rng, operators, specification)
    rows = request_rows(rng, planned, repository, specification)

    return with_event_ids(chronological(pd.DataFrame(rows)))
