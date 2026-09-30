"""Reparticao de um total inteiro em partes proporcionais.

E como as chaves se dividem entre os escopos (D-037): a proporcao sai de uma
Dirichlet, e o total precisa fechar exato.
"""

from __future__ import annotations

import numpy as np


def largest_remainder(weights: np.ndarray, total: int) -> np.ndarray:
    """Reparte `total` em inteiros proporcionais a `weights`, somando exatamente `total`.

    O metodo dos maiores restos: cada parte recebe o piso da sua cota exata, e as
    unidades que sobram vao, uma a uma, para as partes de maior resto.
    """
    exact = weights * total
    allocated = np.floor(exact).astype(int)
    leftover = total - int(allocated.sum())

    has_leftover = leftover > 0

    if has_leftover:
        priority = np.argsort(-(exact - allocated))
        allocated[priority[:leftover]] += 1

    return allocated
