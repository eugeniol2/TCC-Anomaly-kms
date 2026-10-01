from __future__ import annotations

SEEDS = tuple(range(1, 31))

def sigma_steps() -> tuple[float, ...]:
    """Os onze valores de sigma, de 0,0 a 1,0, em passo de 0,1."""
    sigmas = []

    for step in range(11):
        sigmas.append(step / 10)

    return tuple(sigmas)


SIGMAS = sigma_steps()

TOTAL_RUNS = len(SEEDS) * len(SIGMAS)

HYPERPARAMETER_SEARCH_SEED = 902

REHEARSAL_SEED = 903

RESERVED_SEEDS = (HYPERPARAMETER_SEARCH_SEED, REHEARSAL_SEED)

PREPARATION_SIGMA = 0.5
