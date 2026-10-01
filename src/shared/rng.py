from __future__ import annotations

from numpy.random import PCG64, Generator, SeedSequence

POPULATION = 0

TRAFFIC = 1

ATTACK = 2

PARTITION = 3

MODELS = 4

IMPORTANCE = 5

STREAM_COUNT = 6


def stream(seed: int, subsystem: int) -> Generator:
    """Gerador do subsistema indicado, derivado de `seed`.

    A mesma semente devolve sempre os mesmos fluxos, e cada fluxo avanca de
    forma independente dos demais.
    """
    is_out_of_range = not 0 <= subsystem < STREAM_COUNT

    if is_out_of_range:
        raise ValueError(f"subsistema fora da faixa 0..{STREAM_COUNT - 1}: {subsystem}")

    seed_sequence = SeedSequence(seed) # recebe uma seed e retorna uma seed mais complexa, que pode ser usada para gerar outros geradores de numeros aleatorios
    stream_sequences = seed_sequence.spawn(STREAM_COUNT) # gera uma lista de SeedSequence, cada um derivado da seed original, para criar fluxos independentes
    selected_sequence = stream_sequences[subsystem] # seleciona a SeedSequence correspondente ao subsistema desejado

    bit_generator = PCG64(selected_sequence) # cria um gerador de bits (PCG64) a partir da SeedSequence selecionada, que é usado para gerar numeros aleatorios
    return Generator(bit_generator)
