"""Derivacao dos fluxos de aleatoriedade do experimento (D-003, D-102).

Fluxos independentes saem da mesma semente, um por subsistema. Fluxo unico
compartilhado faria o numero de sorteios de um subsistema deslocar os sorteios
dos outros, desfazendo o pareamento entre condicoes de sigma sem emitir erro nem
aviso.

**Acrescentar um fluxo nao altera os existentes.** O `SeedSequence.spawn` deriva
cada filho do indice dele, e nao de quantos filhos sao pedidos: os tres
primeiros de `spawn(4)` sao identicos aos de `spawn(3)`. Foi assim que a
particao ganhou o quarto fluxo sem mudar dado nenhum ja gerado (D-102).
"""

from __future__ import annotations

from numpy.random import PCG64, Generator, SeedSequence

POPULATION = 0
"""Populacao de operadores e repositorio de chaves (M1)."""

TRAFFIC = 1
"""Trafego legitimo (M2)."""

ATTACK = 2
"""Campanha de ataque (M3)."""

PARTITION = 3
"""Particao em treino e holdout (M9, D-102).

Derivado so da semente, e nao de sigma: a particao e a mesma nas onze condicoes
de uma semente, pela mesma logica da D-011.
"""

STREAM_COUNT = 4


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
