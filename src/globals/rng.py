"""Derivacao dos fluxos de aleatoriedade do experimento (D-003).

Tres fluxos independentes saem da mesma semente. Fluxo unico compartilhado faria
o numero de sorteios de um subsistema deslocar os sorteios dos outros, desfazendo
o pareamento entre condicoes de sigma sem emitir erro nem aviso.
"""

from __future__ import annotations

from numpy.random import PCG64, Generator, SeedSequence

POPULATION = 0
"""Populacao de operadores e repositorio de chaves (M1)."""

TRAFFIC = 1
"""Trafego legitimo (M2)."""

ATTACK = 2
"""Campanha de ataque (M3)."""

STREAM_COUNT = 3


def stream(seed: int, subsystem: int) -> Generator:
    """Gerador do subsistema indicado, derivado de `seed`.

    A mesma semente devolve sempre os mesmos tres fluxos, e cada fluxo avanca
    de forma independente dos demais.
    """
    is_out_of_range = not 0 <= subsystem < STREAM_COUNT

    if is_out_of_range:
        raise ValueError(f"subsistema fora da faixa 0..{STREAM_COUNT - 1}: {subsystem}")

    seed_sequence = SeedSequence(seed)
    stream_sequences = seed_sequence.spawn(STREAM_COUNT)
    selected_sequence = stream_sequences[subsystem]

    bit_generator = PCG64(selected_sequence)

    return Generator(bit_generator)
