from __future__ import annotations

import math


def round_half_up(value: float) -> int:
    """O inteiro mais proximo, com o meio exato indo para cima, e nao para o par como no
    `round` do Python.
    """
    return math.floor(value + 0.5)
