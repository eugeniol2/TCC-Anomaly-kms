"""Os tres regimes de uso, e a forma que cada um tem.

O que muda aqui e a **forma** do comportamento: quantos tipos de ritmo existem,
que campos um regime tem. Os numeros que preenchem essa forma moram em
`parameters.py`, e mudam por outro motivo — decisao de registro, nao de codigo.

A separacao entre os dois arquivos e essa: valores de um lado, vocabulario do
outro. Este importa daquele, nunca o contrario.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.traffic.parameters import (
    BATCH_HOURS,
    BATCH_JITTER_MINUTES,
    BATCH_REQUESTS_RANGE,
    BATCH_REQUEST_INTERVAL,
    CUSTODY_CLOSES_AT,
    CUSTODY_DISPERSION,
    CUSTODY_OPENS_AT,
    CUSTODY_REQUESTS_RANGE,
    CUSTODY_REQUEST_INTERVAL,
    CUSTODY_SESSIONS_PER_BUSINESS_DAY,
    SPORADIC_CLOSES_AT,
    SPORADIC_OPENS_AT,
    SPORADIC_REQUESTS_RANGE,
    SPORADIC_REQUEST_INTERVAL,
    SPORADIC_SESSIONS_PER_BUSINESS_DAY,
)


@dataclass(frozen=True)
class ScheduledRhythm:
    """Lotes em horas fixas, todos os dias do calendario."""

    hours: tuple[int, ...]
    jitter_minutes: int


@dataclass(frozen=True)
class ArrivalRhythm:
    """Chegadas aleatorias em dias uteis, dentro de uma janela de horario.

    `dispersion` ausente significa Poisson, onde a variancia iguala a media.
    Presente, e o parametro da Pascal, que produz variancia maior.
    """

    sessions_per_business_day: float
    opens_at: int
    closes_at: int
    dispersion: int | None = None


@dataclass(frozen=True)
class Regime:
    """Como um regime abre sessoes e o que acontece dentro delas."""

    rhythm: ScheduledRhythm | ArrivalRhythm
    requests_range: tuple[int, int]
    seconds_between_requests: float


REGIMES: dict[str, Regime] = {
    "periodic_batch": Regime(
        rhythm=ScheduledRhythm(BATCH_HOURS, BATCH_JITTER_MINUTES),
        requests_range=BATCH_REQUESTS_RANGE,
        seconds_between_requests=BATCH_REQUEST_INTERVAL,
    ),
    "sporadic": Regime(
        rhythm=ArrivalRhythm(
            SPORADIC_SESSIONS_PER_BUSINESS_DAY, SPORADIC_OPENS_AT, SPORADIC_CLOSES_AT
        ),
        requests_range=SPORADIC_REQUESTS_RANGE,
        seconds_between_requests=SPORADIC_REQUEST_INTERVAL,
    ),
    "occasional_custody": Regime(
        rhythm=ArrivalRhythm(
            CUSTODY_SESSIONS_PER_BUSINESS_DAY,
            CUSTODY_OPENS_AT,
            CUSTODY_CLOSES_AT,
            CUSTODY_DISPERSION,
        ),
        requests_range=CUSTODY_REQUESTS_RANGE,
        seconds_between_requests=CUSTODY_REQUEST_INTERVAL,
    ),
}
"""O comportamento de cada regime, e nao de cada perfil.

A coluna `regime` existe para que o M2 dependa do comportamento e nao do rotulo
do perfil. Este dicionario e o unico lugar do modulo que menciona os tres nomes:
todo o resto recebe um `Regime` ja resolvido e nao sabe qual e.
"""
