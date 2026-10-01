from __future__ import annotations

import pandas as pd
from numpy.random import Generator

from src.entities.scenario_engine.population.profiles import PROFILES, total_scope_assignments
from src.shared.tables import MULTIVALUE_SEPARATOR

DEFAULT_COVERAGE_ATTEMPTS = 20


def holders_by_scope(operators: pd.DataFrame) -> dict[str, list[str]]:
    """Indice inverso da tabela: de cada escopo para os operadores que o detem."""
    holders: dict[str, list[str]] = {}

    for row in operators.itertuples():
        for scope in row.scopes.split(MULTIVALUE_SEPARATOR):
            holders.setdefault(scope, []).append(row.operator_id)

    return holders


def draw_usual_ips(rng: Generator, quantity: int) -> list[str]:
    """Enderecos habituais distintos, todos na mesma faixa privada."""
    seen: set[str] = set()
    addresses: list[str] = []

    while len(addresses) < quantity:
        octets = rng.integers([0, 0, 1], [256, 256, 255])
        candidate = f"10.{octets[0]}.{octets[1]}.{octets[2]}"

        is_repeated = candidate in seen

        if not is_repeated:
            seen.add(candidate)
            addresses.append(candidate)

    return addresses


def take_addresses(addresses: list[str], start: int, quantity: int) -> str:
    """Os `quantity` enderecos a partir da posicao `start`, ja unidos. O primeiro e o
    principal.
    """
    taken = addresses[start:start + quantity]

    return MULTIVALUE_SEPARATOR.join(taken)


def draw_address_counts(rng: Generator) -> list[int]:
    """Quantos enderecos habituais cada operador tem, sorteados na faixa do perfil, na
    ordem da tabela.

    O `high` do numpy e exclusivo, por isso o `+ 1`.
    """
    counts: list[int] = []

    for profile in PROFILES:
        lowest, highest = profile.addresses_range
        drawn = rng.integers(lowest, highest + 1, profile.operators)

        for count in drawn:
            counts.append(int(count))

    return counts


def draw_address_groups(rng: Generator) -> list[str]:
    """Os enderecos habituais de cada operador, ja unidos, na ordem da tabela.

    Os enderecos sao sorteados todos de uma vez, e cada operador leva os seus em
    sequencia: `start` anda o tanto que o operador anterior levou.
    """
    counts = draw_address_counts(rng)
    addresses = draw_usual_ips(rng, sum(counts))
    groups = []
    start = 0

    for count in counts:
        groups.append(take_addresses(addresses, start, count))
        start += count

    return groups


def draw_scopes(rng: Generator, pool: list[str], quantity: int) -> str:
    """Subconjunto de escopos de um operador, nunca a totalidade do repositorio."""
    chosen_indexes = rng.choice(len(pool), size=quantity, replace=False)
    chosen_scopes = []

    for index in chosen_indexes:
        chosen_scopes.append(pool[index])

    return MULTIVALUE_SEPARATOR.join(sorted(chosen_scopes))


def build_operators(rng: Generator, pool: list[str]) -> pd.DataFrame:
    """Tabela de operadores, um bloco por perfil da Tabela 1."""
    address_groups = draw_address_groups(rng)
    rows = []
    position = 0

    for profile in PROFILES:
        for number in range(1, profile.operators + 1):
            rows.append(
                {
                    "operator_id": f"{profile.id_prefix}_{number:02d}",
                    "profile": profile.name,
                    "scopes": draw_scopes(rng, pool, profile.scopes_each),
                    "regime": profile.regime,
                    "usual_ips": address_groups[position],
                }
            )
            position += 1

    return pd.DataFrame(rows)


def build_operators_covering_pool(
    rng: Generator, pool: list[str], attempts: int = DEFAULT_COVERAGE_ATTEMPTS
) -> pd.DataFrame:
    """Sorteia ate que todo escopo de chaves tenha ao menos um operador detentor."""
    for _ in range(attempts):
        operators = build_operators(rng, pool)

        every_scope_has_holder = len(holders_by_scope(operators)) == len(pool)

        if every_scope_has_holder:
            return operators

    raise RuntimeError(
        f"a populacao distribui {total_scope_assignments()} atribuicoes de escopo e nao "
        f"cobriu {len(pool)} escopos em {attempts} tentativas. "
        f"Reduza a quantidade de escopos ou aumente a populacao."
    )
