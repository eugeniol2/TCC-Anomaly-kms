"""Quando cada operador abre sessao, ao longo das oito semanas.

Dois tipos de ritmo, ambos lidos do regime, e a diferenca entre eles e o que
faz maquina e humano se distinguirem no log: o lote chega em
hora fixa todos os dias, inclusive fim de semana; a pessoa chega quando chega,
em dia util, dentro do horario de expediente.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta

from numpy.random import Generator

from src.scenario_engine.traffic.parameters import BUSINESS_WEEKDAYS, TrafficSpecification
from src.scenario_engine.traffic.regimes import ArrivalRhythm, Regime, ScheduledRhythm


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
    """Lotes nas horas fixas de cada dia, com desvio de alguns minutos.

    O desvio existe para que o horario nao seja identico ao segundo, o que
    seria assinatura boa demais. Ele e pequeno o bastante para o pico continuar
    visivel na contagem por hora, que e o que precisa aparecer.
    """
    starts: list[datetime] = []

    for day in days:
        for hour in rhythm.hours:
            drift = float(rng.uniform(-rhythm.jitter_minutes, rhythm.jitter_minutes))
            starts.append(datetime.combine(day, time(hour)) + timedelta(minutes=drift))

    return starts


def daily_session_count(rng: Generator, rhythm: ArrivalRhythm) -> int:
    """Quantas sessoes aquele dia util recebe.

    Sem dispersao declarada e Poisson, onde a variancia iguala a media. Com
    dispersao e binomial negativa de mesma media e variancia maior, que e o
    ritmo irregular exigido do administrador.

    **Quem cai em cada ramo, hoje:** o `routine` nao declara dispersao e vai
    para a Poisson; o `occasional_custody` declara e vai para a Pascal. O
    `periodic_batch` nao chega aqui, porque maquina nao sorteia quantas vezes
    roda. Essa correspondencia mora no `regimes.py`, e **esta funcao nao a
    conhece**: ela decide pelo que o ritmo declara, e nao por quem o operador
    e. Assim um regime novo, superdisperso ou nao, entra acrescentando uma
    entrada la, sem tocar aqui.

    O `success` nao e escolhido: e o unico valor que faz a media sair igual a
    `sessions_per_business_day`. O numpy parametriza a binomial negativa por
    `(n, p)`, com media `n(1-p)/p`; isolando `p` da igualdade com a media
    desejada `m` chega-se a `p = n / (m + n)`, que e a linha abaixo.
    """
    is_overdispersed = rhythm.dispersion is not None

    if is_overdispersed:
        expected = rhythm.sessions_per_business_day
        success = rhythm.dispersion / (rhythm.dispersion + expected)

        return int(rng.negative_binomial(rhythm.dispersion, success))

    return int(rng.poisson(rhythm.sessions_per_business_day))


def arrival_starts(
    rng: Generator, rhythm: ArrivalRhythm, days: list[date]
) -> list[datetime]:
    """Chegadas aleatorias dentro da janela de horario de cada dia.

    A janela estreita e o que da ao M6 uma faixa horaria habitual para extrair.
    Chegada uniforme nas 24 horas deixaria o atributo de hora atipica
    degenerado, porque nenhuma hora seria atipica.
    """
    window_seconds = (rhythm.closes_at - rhythm.opens_at) * 3600
    starts: list[datetime] = []

    for day in days:
        opening = datetime.combine(day, time(rhythm.opens_at))

        for _ in range(daily_session_count(rng, rhythm)):
            offset = timedelta(seconds=float(rng.uniform(0, window_seconds)))
            starts.append(opening + offset)

    return starts


def session_starts(
    rng: Generator, regime: Regime, specification: TrafficSpecification
) -> list[datetime]:
    """Os instantes em que um operador daquele regime abre sessao.

    Nao vem ordenado: a ordenacao acontece uma vez so, sobre todas as sessoes
    de todos os operadores, em `build`.
    """
    days = simulated_days(specification)
    is_scheduled = isinstance(regime.rhythm, ScheduledRhythm)

    if is_scheduled:
        return scheduled_starts(rng, regime.rhythm, days)

    return arrival_starts(rng, regime.rhythm, business_days_among(days))
