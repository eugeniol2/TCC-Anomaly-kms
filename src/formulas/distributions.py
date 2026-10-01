from __future__ import annotations

import numpy as np


def geometric_weights(quantity: int, first_share: float) -> np.ndarray:
    """Pesos de `quantity` posicoes decaindo geometricamente, somando 1.

    A primeira leva `first_share` e cada seguinte leva `1 - first_share` da anterior; o
    conjunto e renormalizado, porque a geometrica e truncada em `quantity` posicoes.
    """
    positions = np.arange(quantity)
    weights = first_share * (1 - first_share) ** positions

    return weights / weights.sum()


def chance_never_drawn(probability: float, draws: int) -> float:
    """A chance de um valor de probabilidade `probability` faltar em `draws` sorteios."""
    return (1 - probability) ** draws


def negative_binomial_success(mean: float, dispersion: float) -> float:
    """O `p` da binomial negativa que tem aquela media, dado o `n`.

    O numpy parametriza a binomial negativa por `(n, p)`, com media `n(1-p)/p`.
    Isolando `p` da igualdade com a media desejada `m` chega-se a `p = n / (m + n)`.
    A variancia sai `m / p`, maior que a media: e o que a distingue da Poisson.
    """
    return dispersion / (dispersion + mean)
