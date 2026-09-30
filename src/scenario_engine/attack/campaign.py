"""Quando as sessoes comprometidas acontecem, dentro das semanas 5 a 8.

O numero de sessoes e **fixo nas onze condicoes** (D-081): o que sigma move e
o comportamento dentro delas, nunca quantas sao. Se sigma mexesse na contagem,
a proporcao de anomalias mudaria junto, e a comparacao entre condicoes
confundiria furtividade com desbalanceamento, que e exatamente o que a D-002 existe
para impedir.

O momento de abertura e a primeira das cinco dimensoes. Ele carrega **dia e
hora juntos**, porque as duas coisas respondem a mesma pergunta: o operador
personificado trabalha em dia util, dentro de uma janela de expediente. Sessao
de madrugada e sessao de domingo sao a mesma marca de estranheza, e separa-las
em duas dimensoes daria peso dobrado ao horario contra as outras quatro.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta

from numpy.random import Generator

from src.scenario_engine.attack.stealth import HourWindow, Stealth
from src.shared.phases import EVALUATED, WEEKS_OF, day_after_last_of, first_day_of
from src.scenario_engine.traffic.calendar import business_days_among


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
    """Um instante uniforme dentro da janela horaria daquele dia.

    Uniforme, e nao concentrado no meio, porque e assim que o M2 sorteia a
    abertura das sessoes legitimas. Qualquer outra forma faria a distribuicao
    da hora divergir em sigma 1.
    """
    span_seconds = (window.closes_at - window.opens_at) * 3600

    opening = datetime.combine(day, time(window.opens_at))
    offset = timedelta(seconds=float(rng.uniform(0, span_seconds)))

    return opening + offset


def session_start(
    rng: Generator, days: list[date], business: list[date], stealth: Stealth
) -> datetime:
    """Quando uma sessao comprometida abre.

    Com chance `atypical_hour_chance` ela cai na madrugada de qualquer dia,
    inclusive fim de semana; caso contrario, em dia util dentro da janela do
    operador. Em sigma 1 a chance e zero e sobra so o segundo caso, que e o
    que o M2 faria.
    """
    is_atypical = rng.random() < stealth.atypical_hour_chance

    if is_atypical:
        chosen = days[int(rng.integers(len(days)))]

        return moment_within(rng, chosen, stealth.atypical_window)

    chosen = business[int(rng.integers(len(business)))]

    return moment_within(rng, chosen, stealth.usual_window)


def campaign_starts(rng: Generator, quantity: int, stealth: Stealth) -> list[datetime]:
    """Os instantes de abertura das sessoes da campanha, em ordem cronologica.

    Ordenados aqui porque a numeracao das sessoes comprometidas sai desta
    lista, e identificador fora de ordem cronologica seria ruido para quem
    inspeciona o arquivo.
    """
    days = evaluated_days()
    business = business_days_among(days)

    starts = []

    for _ in range(quantity):
        starts.append(session_start(rng, days, business, stealth))

    return sorted(starts)
