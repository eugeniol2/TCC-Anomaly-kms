from __future__ import annotations

from datetime import datetime

import pandas as pd
from numpy.random import Generator

from src.shared.rng import TRAFFIC, stream
from src.entities.scenario_engine.traffic.calendar import session_starts
from src.entities.scenario_engine.traffic.operators import Operator, read_operators
from src.entities.scenario_engine.traffic.repository import OperatorKeys, build_repository
from src.entities.scenario_engine.traffic.sessions import PlannedSession, session_rows
from src.entities.scenario_engine.traffic.parameters import TrafficSpecification
from src.entities.scenario_engine.traffic.regimes import REGIMES

COLUMNS = (
    "event_id",
    "session_id",
    "operator_id",
    "timestamp",
    "source_ip",
    "operation",
    "key_id",
)


def start_then_operator(scheduled: tuple[datetime, Operator]) -> tuple[datetime, str]:
    """Como as sessoes se ordenam: pelo instante de abertura, e no empate pelo operador."""
    start, operator = scheduled

    return start, operator.operator_id


def plan_sessions(
    rng: Generator, operators: list[Operator], specification: TrafficSpecification
) -> list[PlannedSession]:
    """Todas as sessoes de todos os operadores, numeradas em ordem cronologica."""
    scheduled: list[tuple[datetime, Operator]] = []

    for operator in operators:
        regime = REGIMES[operator.regime]
        starts = session_starts(rng, regime, specification)

        for start in starts:
            scheduled.append((start, operator))

    scheduled.sort(key=start_then_operator)

    planned = []

    for number, (start, operator) in enumerate(scheduled, start=1):
        planned.append(PlannedSession(f"session_{number:05d}", operator, start))

    return planned


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
    """Ordena o arquivo no tempo.

    O instante e ISO com segundos, entao a ordem alfabetica e a cronologica. A ordenacao
    e estavel: requisicoes no mesmo segundo mantem a ordem em que a sessao as emitiu.
    """
    ordered = frame.sort_values(["timestamp", "session_id"], kind="stable")

    return ordered.reset_index(drop=True)


def with_event_ids(frame: pd.DataFrame) -> pd.DataFrame:
    """Numera os eventos na ordem cronologica em que o log os registra."""
    identifiers = []

    for number in range(1, len(frame) + 1):
        identifiers.append(f"event_{number:06d}")

    numbered = frame.copy()
    numbered.insert(0, "event_id", identifiers)

    return numbered


def build_traffic(
    seed: int,
    operators_table: pd.DataFrame,
    keys_table: pd.DataFrame,
    specification: TrafficSpecification,
) -> pd.DataFrame:
    """Da tabela de operadores e da de chaves as requisicoes legitimas das oito semanas,
    sem desfecho.
    """
    rng = stream(seed, TRAFFIC)

    operators = read_operators(operators_table) # Leitura de `operators.csv` e conversao para `Operator`.
    repository = build_repository(keys_table, operators) # Leitura de `keys.csv` e organizacao de chaves por operador.
    planned = plan_sessions(rng, operators, specification) # Planejamento de todas as sessoes de todos os operadores, numeradas em ordem cronologica.
    rows = request_rows(rng, planned, repository, specification) # As requisicoes de todas as sessoes, na ordem em que foram planejadas.

    return with_event_ids(chronological(pd.DataFrame(rows))) # Ordena o arquivo no tempo, que e como um log de auditoria se le.
