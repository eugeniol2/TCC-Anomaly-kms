"""Linha de comando do M1: le a semente, escreve `operators.csv` e `keys.csv`.

    python -m src.population --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.population.build import build_population
from src.population.parameters import (
    DEFAULT_CONCENTRATION,
    DEFAULT_DISABLED_RATE,
    DEFAULT_KEYS,
    DEFAULT_SCOPE_FLOOR,
    DEFAULT_SCOPES,
    KeyRepositorySpecification,
)
from src.globals.layout import DEFAULT_ROOT, seed_directory
from src.globals.tables import write_csv


class Arguments(argparse.Namespace):
    """Atributos que a linha de comando produz.

    O argparse monta o Namespace em tempo de execucao, entao sem estas anotacoes
    a IDE nao sabe que `seed` e inteiro nem que `out` e caminho. Declarar aqui da
    autocompletar e deteccao de erro de digitacao.

    Cada atributo precisa ter um `add_argument` correspondente em `parse_args`:
    a sincronia entre as duas listas e manual.
    """

    seed: int
    out: Path
    keys: int
    scopes: int
    disabled_rate: float
    scope_floor: int
    concentration: float


def parse_args() -> Arguments:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, required=True, help="semente do experimento")
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_ROOT, help="raiz da pasta de dados"
    )
    parser.add_argument("--keys", type=int, default=DEFAULT_KEYS)
    parser.add_argument("--scopes", type=int, default=DEFAULT_SCOPES)
    parser.add_argument("--disabled-rate", type=float, default=DEFAULT_DISABLED_RATE)
    parser.add_argument("--scope-floor", type=int, default=DEFAULT_SCOPE_FLOOR)
    parser.add_argument("--concentration", type=float, default=DEFAULT_CONCENTRATION)
    return parser.parse_args(namespace=Arguments())


def key_repository_specification_from(args: Arguments) -> KeyRepositorySpecification:
    """Reune os parametros do repositorio numa unica estrutura."""
    return KeyRepositorySpecification(
        total_keys=args.keys,
        scope_count=args.scopes,
        disabled_rate=args.disabled_rate,
        scope_floor=args.scope_floor,
        concentration=args.concentration,
    )


def report(destination: Path, operators: pd.DataFrame, keys: pd.DataFrame) -> None:
    """Resumo da geracao, para conferencia imediata na linha de comando."""
    keys_per_scope = keys["scope"].value_counts()
    disabled_count = int((keys["status"] == "disabled").sum())

    print(f"{destination}")
    print(f"  operators.csv  {len(operators)} operadores")
    print(f"  keys.csv       {len(keys)} chaves em {len(keys_per_scope)} escopos")
    print(f"                 {disabled_count} desabilitadas")
    print(f"                 {keys_per_scope.min()} a {keys_per_scope.max()} por escopo")


def main() -> None:
    args = parse_args()
    specification = key_repository_specification_from(args)

    population = build_population(args.seed, specification)

    destination = seed_directory(args.out, args.seed)
    write_csv(population.operators, destination / "operators.csv")
    write_csv(population.keys, destination / "keys.csv")

    report(destination, population.operators, population.keys)


if __name__ == "__main__":
    main()
