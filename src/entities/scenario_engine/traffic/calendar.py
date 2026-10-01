from __future__ import annotations

from datetime import date, datetime, time, timedelta

from numpy.random import Generator

from src.formulas.distributions import negative_binomial_success
from src.entities.scenario_engine.traffic.parameters import BUSINESS_WEEKDAYS, TrafficSpecification
from src.entities.scenario_engine.traffic.regimes import ArrivalRhythm, Regime, ScheduledRhythm


def simulated_days(specification: TrafficSpecification) -> list[date]:
    """Os dias corridos do periodo, do primeiro ao ultimo."""
    days = []

    for offset in range(specification.day_count):
        days.append(specification.first_day + timedelta(days=offset))

    return days


def business_days_among(days: list[date]) -> list[date]:
    """So os dias uteis. Fim de semana nao recebe sessao de pessoa."""
    business = []

    for day in days:
        is_business_day = day.weekday() in BUSINESS_WEEKDAYS

        if is_business_day:
            business.append(day)

    return business


def scheduled_starts(
    rng: Generator, rhythm: ScheduledRhythm, days: list[date]
) -> list[datetime]:
    """Lotes nas horas fixas de cada dia, com desvio de alguns minutos."""
    starts: list[datetime] = []

    for day in days:
        for hour in rhythm.hours:
            drift = float(rng.uniform(-rhythm.jitter_minutes, rhythm.jitter_minutes))
            starts.append(datetime.combine(day, time(hour)) + timedelta(minutes=drift))

    return starts


def daily_session_count(rng: Generator, rhythm: ArrivalRhythm) -> int:
    """Quantas sessoes aquele dia util recebe.

    Sem dispersao declarada e Poisson, com variancia igual a media; com dispersao e
    binomial negativa de mesma media e variancia maior. O `success` e o unico valor que
    faz a media sair igual a `sessions_per_business_day`.
    """
    is_overdispersed = rhythm.dispersion is not None

    if is_overdispersed:
        expected = rhythm.sessions_per_business_day
        success = negative_binomial_success(expected, rhythm.dispersion)

        return int(rng.negative_binomial(rhythm.dispersion, success))

    return int(rng.poisson(rhythm.sessions_per_business_day))


def arrival_starts(
    rng: Generator, rhythm: ArrivalRhythm, days: list[date]
) -> list[datetime]:
    """Chegadas aleatorias dentro da janela de horario de cada dia."""
    window_seconds = (rhythm.closes_at_hour - rhythm.opens_at_hour) * 3600
    starts: list[datetime] = []

    for day in days:
        opening = datetime.combine(day, time(rhythm.opens_at_hour))

        for _ in range(daily_session_count(rng, rhythm)):
            offset = timedelta(seconds=float(rng.uniform(0, window_seconds)))
            starts.append(opening + offset)

    return starts


def session_starts(
    rng: Generator, regime: Regime, specification: TrafficSpecification
) -> list[datetime]:
    """Os instantes em que um operador daquele regime abre sessao. Nao vem ordenado."""
    days = simulated_days(specification)
    is_scheduled = isinstance(regime.rhythm, ScheduledRhythm)

    if is_scheduled:
        return scheduled_starts(rng, regime.rhythm, days)

    return arrival_starts(rng, regime.rhythm, business_days_among(days))
