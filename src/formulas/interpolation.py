from __future__ import annotations


def interpolate(fraction: float, start: float, end: float) -> float:
    """O ponto a `fraction` do caminho de `start` a `end`.

    Na forma com dois pesos, que devolve o extremo exato em ponto flutuante quando
    `fraction` e 0 ou 1.
    """
    return (1.0 - fraction) * start + fraction * end


def interpolate_range(
    fraction: float, start: tuple[int, int], end: tuple[int, int]
) -> tuple[int, int]:
    """Interpola as duas pontas de uma faixa inteira, arredondando cada uma.

    O arredondamento e o `round` do Python, que leva o meio exato ao par.
    """
    lowest = round(interpolate(fraction, start[0], end[0]))
    highest = round(interpolate(fraction, start[1], end[1]))

    return lowest, highest
