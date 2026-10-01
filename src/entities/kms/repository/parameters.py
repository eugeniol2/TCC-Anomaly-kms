from __future__ import annotations

from dataclasses import dataclass

DEFAULT_TOTAL_KEYS = 300

DEFAULT_SCOPE_COUNT = 12

DEFAULT_DISABLED_RATE = 0.05

DEFAULT_MINIMUM_KEYS_PER_SCOPE = 3

DEFAULT_DIRICHLET_CONCENTRATION = 2.0


@dataclass(frozen=True)
class KeyRepositorySpecification:
    total_keys: int = DEFAULT_TOTAL_KEYS
    scope_count: int = DEFAULT_SCOPE_COUNT
    disabled_rate: float = DEFAULT_DISABLED_RATE
    minimum_keys_per_scope: int = DEFAULT_MINIMUM_KEYS_PER_SCOPE
    dirichlet_concentration: float = DEFAULT_DIRICHLET_CONCENTRATION
