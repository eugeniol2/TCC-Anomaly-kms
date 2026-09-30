"""As contas das distribuicoes que o gerador sorteia.

O sorteio mora na entidade; aqui mora so a conta que ele usa, e a que o viewer
usa para desenhar a mesma distribuicao sem reescrever a formula.
"""

from __future__ import annotations

import numpy as np


def geometric_weights(quantity: int, first_share: float) -> np.ndarray:
    """Pesos de `quantity` posicoes decaindo geometricamente, somando 1.

    A primeira leva `first_share` e cada seguinte leva `1 - first_share` da
    anterior, e o conjunto e renormalizado, porque a geometrica e truncada em
    `quantity` posicoes. E como a sessao escolhe a origem de rede (D-040).
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


def share_above(n: int, typical: np.ndarray) -> float:
    """A fracao de uma faixa uniforme de valores que passa de `n`."""
    return (typical > n).sum() / len(typical)


def share_above_with_tail(
    n: int, typical: np.ndarray, long_chance: float, mean_excess: float
) -> float:
    """A chance de passar de `n` quando a faixa uniforme ganha uma cauda (D-097).

    Com chance `long_chance`, o valor tipico recebe um excesso geometrico de media
    `mean_excess`. Tipico acima de `n` ja passa, com ou sem excesso. Tipico `t`
    abaixo so passa se o excesso for maior que `n - t`, e a geometrica sobrevive a
    `n - t` com `(1 - 1 / mean_excess)` elevado a essa diferenca.

    Calculada analiticamente, e nao somando uma densidade truncada: a soma daria
    zero no ultimo ponto somado, que e o proprio artefato que a cauda veio corrigir.
    """
    survives = 1.0 - 1.0 / mean_excess
    reached = 0.0

    for value in typical:
        is_below = value <= n

        if is_below:
            reached += long_chance * survives ** (n - value)

    return share_above(n, typical) + reached / len(typical)
