"""Arredondamento com regra escrita."""

from __future__ import annotations

import math


def round_half_up(value: float) -> int:
    """O inteiro mais proximo, com o meio exato indo para cima.

    E nao o meio para o par do `round` do Python, que mudaria de lado conforme a
    paridade: 2,5 vira 2 e 3,5 vira 4.
    """
    return math.floor(value + 0.5)
