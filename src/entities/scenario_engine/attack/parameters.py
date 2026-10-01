from __future__ import annotations

from dataclasses import dataclass

CAMPAIGN_SESSIONS = 58

OSTENSIVE_SECONDS_BETWEEN_REQUESTS = 6.0

OSTENSIVE_REQUESTS_RANGE = (40, 90)

OSTENSIVE_DISTINCT_KEYS_RANGE = (30, 70)

OSTENSIVE_OPENS_AT_HOUR = 0
OSTENSIVE_CLOSES_AT_HOUR = 6

OSTENSIVE_STALE_SCOPE_RATE = 0.30

OSTENSIVE_ABSENT_IDENTIFIER_RATE = 0.10


@dataclass(frozen=True)
class AttackSpecification:
    campaign_sessions: int = CAMPAIGN_SESSIONS
    ostensive_seconds_between_requests: float = OSTENSIVE_SECONDS_BETWEEN_REQUESTS
    ostensive_requests_range: tuple[int, int] = OSTENSIVE_REQUESTS_RANGE
    ostensive_distinct_keys_range: tuple[int, int] = OSTENSIVE_DISTINCT_KEYS_RANGE
    ostensive_opens_at_hour: int = OSTENSIVE_OPENS_AT_HOUR
    ostensive_closes_at_hour: int = OSTENSIVE_CLOSES_AT_HOUR
    ostensive_stale_scope_rate: float = OSTENSIVE_STALE_SCOPE_RATE
    ostensive_absent_identifier_rate: float = OSTENSIVE_ABSENT_IDENTIFIER_RATE
 
