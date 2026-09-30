"""Interpolacao linear entre dois extremos.

E como sigma move o atacante do ostensivo ao furtivo (D-082): em 0 o primeiro
extremo, em 1 o segundo, e em linha reta entre os dois.
"""

from __future__ import annotations


def interpolate(fraction: float, start: float, end: float) -> float:
    """O ponto a `fraction` do caminho de `start` a `end`.

    A forma com os dois pesos, e nao `start + fraction * (end - start)`, e
    deliberada. A segunda e algebricamente identica mas nao devolve o extremo
    **exato** em ponto flutuante: com 0,30 e 0,005 ela da 0,005000000000000004 em
    `fraction` 1. A diferenca nao muda sorteio nenhum, mas desfaz a igualdade exata
    contra os parametros do M2, que e a propriedade que o teste de convergencia do
    atacante verifica.
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
