"""Que chave cada requisicao endereca, e os dois desvios que produzem falha.

O alvo normal e uma chave dos escopos do operador, que o M4 autoriza. Os dois
desvios da D-056 existem porque, sem eles, nenhuma requisicao legitima poderia
falhar por politica ou por identificador inexistente — e cada um desses
desfechos passaria a significar atacante, virando separador trivial.

Os desvios sao emitidos como tentativa comum, **sem marca nenhuma**. Quem os
distingue e o M4, ao avaliar a politica. Aqui eles sao apenas requisicoes que
apontam para o lugar errado, como uma referencia velha aponta.
"""

from __future__ import annotations

from numpy.random import Generator

from src.traffic.repository import OperatorKeys
from src.traffic.parameters import (
    IDENTIFIER_DIGITS,
    IDENTIFIER_PREFIX,
    IDENTIFIER_SPACE,
    TrafficSpecification,
)


def draw_absent_identifier(rng: Generator, existing: frozenset[str]) -> str:
    """Identificador no formato do repositorio que nao existe nele.

    Precisa ser indistinguivel de um real ate o M4 procura-lo e nao achar. Se
    tivesse formato proprio, o atributo de formato separaria as classes sozinho.
    """
    while True:
        drawn = int(rng.integers(0, IDENTIFIER_SPACE))
        candidate = f"{IDENTIFIER_PREFIX}{drawn:0{IDENTIFIER_DIGITS}x}"

        is_present = candidate in existing

        if not is_present:
            return candidate


def distinct_key_count(
    rng: Generator, ceiling: int, specification: TrafficSpecification
) -> int:
    """Quantas chaves distintas a sessao toca, limitado pelo que cabe nela.

    O teto e o menor entre o alcance do operador e o tamanho da sessao: nao da
    para tocar oito chaves distintas em seis requisicoes, nem para alcancar
    doze quando o escopo tem quatro.
    """
    lowest, highest = specification.distinct_keys_range
    drawn = int(rng.integers(lowest, highest + 1))

    return max(1, min(drawn, ceiling))


def spread_over_requests(
    rng: Generator, session_keys: tuple[str, ...], request_count: int
) -> list[str]:
    """Distribui as chaves escolhidas pelas requisicoes da sessao.

    Cada chave aparece ao menos uma vez — e por isso que a contagem de
    distintas e exatamente a sorteada — e as requisicoes restantes repetem
    alguma delas. A permutacao final evita que as primeiras requisicoes sejam
    sempre as de chave inedita.
    """
    surplus = request_count - len(session_keys)
    repeated = rng.integers(0, len(session_keys), size=surplus)

    sequence = list(session_keys) + [session_keys[index] for index in repeated]
    order = rng.permutation(request_count)

    return [sequence[index] for index in order]


def apply_deviations(
    rng: Generator,
    targets: list[str],
    keys: OperatorKeys,
    specification: TrafficSpecification,
) -> list[str]:
    """Substitui algumas requisicoes pelos dois desvios da D-056."""
    deviation_ceiling = specification.stale_scope_rate + specification.mistyped_rate
    has_unreachable = len(keys.out_of_reach) > 0
    draws = rng.random(len(targets))

    deviated: list[str] = []

    for target, draw in zip(targets, draws):
        is_stale_scope = has_unreachable and draw < specification.stale_scope_rate
        is_mistyped = not is_stale_scope and draw < deviation_ceiling

        if is_stale_scope:
            deviated.append(keys.out_of_reach[int(rng.integers(len(keys.out_of_reach)))])
        elif is_mistyped:
            deviated.append(draw_absent_identifier(rng, keys.existing))
        else:
            deviated.append(target)

    return deviated


def session_targets(
    rng: Generator,
    keys: OperatorKeys,
    request_count: int,
    specification: TrafficSpecification,
) -> list[str]:
    """A chave de cada requisicao da sessao, desvios ja aplicados."""
    ceiling = min(len(keys.in_reach), request_count)
    quantity = distinct_key_count(rng, ceiling, specification)

    chosen = rng.choice(len(keys.in_reach), size=quantity, replace=False)
    session_keys = tuple(keys.in_reach[index] for index in chosen)

    spread = spread_over_requests(rng, session_keys, request_count)

    return apply_deviations(rng, spread, keys, specification)
