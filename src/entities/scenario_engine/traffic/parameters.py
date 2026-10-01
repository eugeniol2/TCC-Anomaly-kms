from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from src.shared.phases import DAYS_PER_WEEK, FIRST_DAY, WEEK_COUNT

BUSINESS_WEEKDAYS = frozenset({0, 1, 2, 3, 4})

LONG_SESSION_CHANCE = 0.05
LONG_SESSION_MEAN_EXCESS = 15.0

PRIMARY_ADDRESS_SHARE = 0.80

IDENTIFIER_SPACE = 2**48
IDENTIFIER_PREFIX = "k_"
IDENTIFIER_HEX_DIGITS = 12

BATCH_HOURS = (2, 8, 14, 20)
BATCH_JITTER_MINUTES = 10
BATCH_REQUESTS_RANGE = (20, 40)
BATCH_SECONDS_BETWEEN_REQUESTS = 2.0

ROUTINE_SESSIONS_PER_BUSINESS_DAY = 1.5
ROUTINE_OPENS_AT_HOUR = 8
ROUTINE_CLOSES_AT_HOUR = 18
ROUTINE_REQUESTS_RANGE = (6, 20)
ROUTINE_SECONDS_BETWEEN_REQUESTS = 45.0

CUSTODY_SESSIONS_PER_BUSINESS_DAY = 2.0
CUSTODY_OPENS_AT_HOUR = 9
CUSTODY_CLOSES_AT_HOUR = 19
CUSTODY_DISPERSION = 2
CUSTODY_REQUESTS_RANGE = (8, 25)
CUSTODY_SECONDS_BETWEEN_REQUESTS = 90.0

USER_OPERATION_MIX = {
    "Decrypt": 0.55,
    "Encrypt": 0.25,
    "DescribeKey": 0.12,
    "ExportKeyMaterial": 0.08,
}
SERVICE_OPERATION_MIX = {
    "Decrypt": 0.45,
    "Encrypt": 0.35,
    "DescribeKey": 0.10,
    "ExportKeyMaterial": 0.10,
}
ADMIN_OPERATION_MIX = {
    "Decrypt": 0.35,
    "Encrypt": 0.15,
    "DescribeKey": 0.30,
    "ExportKeyMaterial": 0.20,
}

DEFAULT_STALE_SCOPE_RATE = 0.005
DEFAULT_ABSENT_IDENTIFIER_RATE = 0.003

DEFAULT_DISTINCT_KEYS_RANGE = (3, 12)


@dataclass(frozen=True)
class TrafficSpecification:
    first_day: date = FIRST_DAY
    week_count: int = WEEK_COUNT
    stale_scope_rate: float = DEFAULT_STALE_SCOPE_RATE
    absent_identifier_rate: float = DEFAULT_ABSENT_IDENTIFIER_RATE
    distinct_keys_range: tuple[int, int] = DEFAULT_DISTINCT_KEYS_RANGE
    long_session_chance: float = LONG_SESSION_CHANCE
    long_session_mean_excess: float = LONG_SESSION_MEAN_EXCESS

    @property
    def day_count(self) -> int:
        """Dias corridos do periodo simulado."""
        return self.week_count * DAYS_PER_WEEK
