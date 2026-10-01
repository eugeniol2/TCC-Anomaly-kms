from __future__ import annotations

from datetime import date, datetime, time, timedelta

from numpy.random import Generator

from src.entities.scenario_engine.attack.stealth import HourWindow, Stealth
from src.shared.phases import EVALUATED, WEEKS_OF, day_after_last_of, first_day_of
from src.entities.scenario_engine.traffic.calendar import business_days_among


def evaluated_days() -> list[date]:
    """Os dias corridos das semanas 5 a 8, que e onde a campanha cabe."""
    first_week, last_week = WEEKS_OF[EVALUATED]

    opens = first_day_of(first_week)
    closes = day_after_last_of(last_week)
    days = []

    for offset in range((closes - opens).days):
        days.append(opens + timedelta(days=offset))

    return days


def moment_within(rng: Generator, day: date, window: HourWindow) -> datetime:
    """Um instante uniforme dentro da janela horaria daquele dia."""
    span_seconds = (window.closes_at_hour - window.opens_at_hour) * 3600

    opening = datetime.combine(day, time(window.opens_at_hour))
    offset = timedelta(seconds=float(rng.uniform(0, span_seconds)))

    return opening + offset


def session_start(
    rng: Generator, days: list[date], business: list[date], stealth: Stealth
) -> datetime:
    """Quando uma sessao comprometida abre.

    Com chance `atypical_hour_chance` cai na madrugada de qualquer dia, fim de semana
    inclusive; senao, em dia util dentro da janela do operador.
    """
    is_atypical = rng.random() < stealth.atypical_hour_chance

    if is_atypical:
        chosen = days[int(rng.integers(len(days)))]

        return moment_within(rng, chosen, stealth.atypical_window)

    chosen = business[int(rng.integers(len(business)))]

    return moment_within(rng, chosen, stealth.usual_window)


def campaign_starts(rng: Generator, quantity: int, stealth: Stealth) -> list[datetime]:
    """Os instantes de abertura das sessoes da campanha, em ordem cronologica."""
    days = evaluated_days()
    business = business_days_among(days)

    starts = []

    for _ in range(quantity):
        starts.append(session_start(rng, days, business, stealth))

    return sorted(starts)
