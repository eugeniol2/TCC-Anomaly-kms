from __future__ import annotations

from statistics import median
from time import perf_counter_ns
from typing import Callable

import pandas as pd

DISCARDED_ROUNDS = 1

TIMED_ROUNDS = 10

NANOSECONDS_PER_MICROSECOND = 1_000
NANOSECONDS_PER_SECOND = 1_000_000_000

COLUMNS = (
    "mechanism",
    "sessions",
    "timed_rounds",
    "median_nanoseconds",
    "microseconds_per_session",
    "sessions_per_second",
)


def median_nanoseconds(decide: Callable[[], object]) -> int:
    """A mediana das rodadas cronometradas, depois de descartar o aquecimento.
    Cronometra so a chamada.
    """
    for _ in range(DISCARDED_ROUNDS):
        decide()

    rounds = []

    for _ in range(TIMED_ROUNDS):
        started = perf_counter_ns()
        decide()
        rounds.append(perf_counter_ns() - started)

    return int(median(rounds))


def timing_row(mechanism: str, sessions: int, nanoseconds: int) -> dict[str, object]:
    """Uma linha do arquivo de tempo, com as duas unidades derivadas."""
    per_session = nanoseconds / sessions / NANOSECONDS_PER_MICROSECOND
    throughput = sessions * NANOSECONDS_PER_SECOND / nanoseconds

    return {
        "mechanism": mechanism,
        "sessions": sessions,
        "timed_rounds": TIMED_ROUNDS,
        "median_nanoseconds": nanoseconds,
        "microseconds_per_session": round(per_session, 4),
        "sessions_per_second": round(throughput, 1),
    }


def time_decision(
    mechanism: str, sessions: int, decide: Callable[[], object]
) -> pd.DataFrame:
    """O tempo de um mecanismo decidindo sobre o holdout inteiro."""
    nanoseconds = median_nanoseconds(decide)
    row = timing_row(mechanism, sessions, nanoseconds)

    return pd.DataFrame([row])[list(COLUMNS)]
