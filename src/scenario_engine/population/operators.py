"""Construcao de `operators.csv`: a populacao de operadores legitimos."""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd
from numpy.random import Generator

from src.scenario_engine.population.profiles import PROFILES, total_scope_assignments
from src.shared.tables import MULTIVALUE_SEPARATOR

DEFAULT_COVERAGE_ATTEMPTS = 20


def holders_by_scope(operators: pd.DataFrame) -> dict[str, list[str]]:
    """Indice inverso da tabela: de cada escopo para os operadores que o detem.

    A atribuicao de escopos acontece em `draw_scopes`. Aqui a tabela ja vem
    pronta e so e lida.
    """
    holders: dict[str, list[str]] = {}

    for row in operators.itertuples():
        for scope in row.scopes.split(MULTIVALUE_SEPARATOR):
            holders.setdefault(scope, []).append(row.operator_id)

    return holders


def draw_usual_ips(rng: Generator, quantity: int) -> list[str]:
    """Enderecos habituais distintos, todos na mesma faixa privada.

    Faixa unica de proposito: se o endereco codificasse o perfil, viraria
    separador por construcao, que e a armadilha da D-009.
    """
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


def take_addresses(addresses: Iterator[str], quantity: int) -> str:
    """Retira `quantity` enderecos do sorteio. O primeiro e o principal.

    A ordem importa para o M2: o principal responde pela maior parte das
    sessoes e os demais aparecem cada vez mais raramente, de modo que um
    endereco pouco usado possa nao ocorrer no aquecimento e produzir origem de
    rede nova em sessao legitima do periodo avaliado (D-040).
    """
    return MULTIVALUE_SEPARATOR.join(next(addresses) for _ in range(quantity))


def draw_address_counts(rng: Generator) -> list[int]:
    """Quantos enderecos habituais cada operador tem, na ordem da tabela.

    Sorteado por operador, dentro da faixa do perfil, e nao fixo por perfil:
    quantidade fixa daria a todos os operadores de um mesmo perfil a mesma taxa
    base de origem de rede nova, e o baseline poderia usar um limiar global em
    vez de comparar contra o historico daquele operador (D-041).

    O `high` do numpy e exclusivo, por isso o `+ 1`.
    """
    counts: list[int] = []

    for profile in PROFILES:
        lowest, highest = profile.addresses_range
        drawn = rng.integers(lowest, highest + 1, profile.operators)
        counts.extend(int(count) for count in drawn)

    return counts


def draw_address_groups(rng: Generator) -> list[str]:
    """Os enderecos habituais de cada operador, ja unidos, na ordem da tabela."""
    counts = draw_address_counts(rng)
    addresses = iter(draw_usual_ips(rng, sum(counts)))

    return [take_addresses(addresses, count) for count in counts]


def draw_scopes(rng: Generator, pool: list[str], quantity: int) -> str:
    """Subconjunto de escopos de um operador, nunca a totalidade do repositorio."""
    chosen_indexes = rng.choice(len(pool), size=quantity, replace=False)
    chosen_scopes = sorted(pool[index] for index in chosen_indexes)

    return MULTIVALUE_SEPARATOR.join(chosen_scopes)


def build_operators(rng: Generator, pool: list[str]) -> pd.DataFrame:
    """Tabela de operadores, um bloco por perfil da Tabela 1."""
    address_groups = iter(draw_address_groups(rng))
    rows = []

    for profile in PROFILES:
        for number in range(1, profile.operators + 1):
            rows.append(
                {
                    "operator_id": f"{profile.id_prefix}_{number:02d}",
                    "profile": profile.name,
                    "scopes": draw_scopes(rng, pool, profile.scopes_each),
                    "regime": profile.regime,
                    "usual_ips": next(address_groups),
                }
            )

    return pd.DataFrame(rows)


def build_operators_covering_pool(
    rng: Generator, pool: list[str], attempts: int = DEFAULT_COVERAGE_ATTEMPTS
) -> pd.DataFrame:
    """Sorteia ate que todo escopo de chaves tenha ao menos um usuário detentor.

    Escopo sem usuário detentor deixaria suas chaves fora do alcance de qualquer
    credencial legitima, o que não corresponde a nenhum cenario modelado.
    """
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
