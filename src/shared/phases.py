from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

FIRST_DAY = date(2026, 1, 5)

RULER_WEEKS = 4

EVALUATED_WEEKS = 4

WEEK_COUNT = RULER_WEEKS + EVALUATED_WEEKS

DAYS_PER_WEEK = 7

WARMUP = "warmup"

EVALUATED = "evaluated"

PHASES = (WARMUP, EVALUATED)

WEEKS_OF = {
    WARMUP: (1, RULER_WEEKS),
    EVALUATED: (RULER_WEEKS + 1, WEEK_COUNT),
}


def day_count(week_count: int = WEEK_COUNT) -> int:
    """Dias corridos de um numero de semanas."""
    return week_count * DAYS_PER_WEEK


def first_day_of(week: int) -> date:
    """O primeiro dia de uma semana, contada a partir de 1."""
    return FIRST_DAY + timedelta(days=(week - 1) * DAYS_PER_WEEK)


def day_after_last_of(week: int) -> date:
    """O dia seguinte ao ultimo de uma semana, para comparacao aberta."""
    return first_day_of(week + 1)


def belongs_to(phase: str, moments: pd.Series) -> pd.Series:
    """Quais instantes caem dentro da fase indicada.

    Recebe a coluna de `timestamp` como texto ISO e devolve mascara booleana.
    O limite superior e aberto para que a meia-noite do dia seguinte fique de
    fora sem depender de arredondamento.
    """
    is_unknown = phase not in WEEKS_OF

    if is_unknown:
        raise ValueError(f"fase desconhecida: {phase!r}. Use uma de {PHASES}")

    first_week, last_week = WEEKS_OF[phase]
    opens = pd.Timestamp(first_day_of(first_week))
    closes = pd.Timestamp(day_after_last_of(last_week))

    instants = pd.to_datetime(moments)

    return (instants >= opens) & (instants < closes)
