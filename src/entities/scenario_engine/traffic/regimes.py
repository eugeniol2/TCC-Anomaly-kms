from __future__ import annotations

from dataclasses import dataclass

from src.entities.scenario_engine.traffic.parameters import (
    BATCH_HOURS,
    BATCH_JITTER_MINUTES,
    BATCH_REQUESTS_RANGE,
    BATCH_SECONDS_BETWEEN_REQUESTS,
    CUSTODY_CLOSES_AT_HOUR,
    CUSTODY_DISPERSION,
    CUSTODY_OPENS_AT_HOUR,
    CUSTODY_REQUESTS_RANGE,
    CUSTODY_SECONDS_BETWEEN_REQUESTS,
    CUSTODY_SESSIONS_PER_BUSINESS_DAY,
    ROUTINE_CLOSES_AT_HOUR,
    ROUTINE_OPENS_AT_HOUR,
    ROUTINE_REQUESTS_RANGE,
    ROUTINE_SECONDS_BETWEEN_REQUESTS,
    ROUTINE_SESSIONS_PER_BUSINESS_DAY,
)


@dataclass(frozen=True)
class ScheduledRhythm:
    hours: tuple[int, ...]
    jitter_minutes: int


@dataclass(frozen=True)
class ArrivalRhythm:
    sessions_per_business_day: float
    opens_at_hour: int
    closes_at_hour: int
    dispersion: int | None = None


@dataclass(frozen=True)
class Regime:
    rhythm: ScheduledRhythm | ArrivalRhythm
    requests_range: tuple[int, int]
    seconds_between_requests: float


REGIMES: dict[str, Regime] = {
    "periodic_batch": Regime(
        rhythm=ScheduledRhythm(BATCH_HOURS, BATCH_JITTER_MINUTES),
        requests_range=BATCH_REQUESTS_RANGE,
        seconds_between_requests=BATCH_SECONDS_BETWEEN_REQUESTS,
    ),
    "routine": Regime(
        rhythm=ArrivalRhythm(
            ROUTINE_SESSIONS_PER_BUSINESS_DAY, ROUTINE_OPENS_AT_HOUR, ROUTINE_CLOSES_AT_HOUR
        ),
        requests_range=ROUTINE_REQUESTS_RANGE,
        seconds_between_requests=ROUTINE_SECONDS_BETWEEN_REQUESTS,
    ),
    "occasional_custody": Regime(
        rhythm=ArrivalRhythm(
            CUSTODY_SESSIONS_PER_BUSINESS_DAY,
            CUSTODY_OPENS_AT_HOUR,
            CUSTODY_CLOSES_AT_HOUR,
            CUSTODY_DISPERSION,
        ),
        requests_range=CUSTODY_REQUESTS_RANGE,
        seconds_between_requests=CUSTODY_SECONDS_BETWEEN_REQUESTS,
    ),
}
