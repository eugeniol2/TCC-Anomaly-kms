"""Linha de comando do M2: le as tabelas do M1, escreve `requests.csv`.

    python -m src.traffic --seed 1

Precisa que o M1 daquela semente ja tenha rodado: o trafego e dirigido pelos
escopos de `operators.csv` e pelas chaves de `keys.csv`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.globals.layout import DEFAULT_ROOT, seed_directory
from src.globals.tables import write_csv
from src.traffic.build import build_traffic
from src.traffic.parameters import (
    DEFAULT_DISTINCT_KEYS_RANGE,
    DEFAULT_ABSENT_IDENTIFIER_RATE,
    DEFAULT_STALE_SCOPE_RATE,
    FIRST_DAY,
    WEEK_COUNT,
    TrafficSpecification,
)


class Arguments(argparse.Namespace):
    """Atributos que a linha de comando produz.

    O argparse monta o Namespace em tempo de execucao, entao sem estas
    anotacoes a IDE nao sabe que `seed` e inteiro nem que `out` e caminho.

    Cada atributo precisa ter um `add_argument` correspondente em `parse_args`:
    a sincronia entre as duas listas e manual.
    """

    seed: int
    out: Path
    weeks: int
    stale_scope_rate: float
    absent_identifier_rate: float


def parse_args() -> Arguments:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, required=True, help="semente do experimento")
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_ROOT, help="raiz da pasta de dados"
    )
    parser.add_argument("--weeks", type=int, default=WEEK_COUNT)
    parser.add_argument("--stale-scope-rate", type=float, default=DEFAULT_STALE_SCOPE_RATE)
    parser.add_argument("--absent-identifier-rate", type=float, default=DEFAULT_ABSENT_IDENTIFIER_RATE)
    return parser.parse_args(namespace=Arguments())


def traffic_specification_from(args: Arguments) -> TrafficSpecification:
    """Reune os parametros do trafego numa unica estrutura."""
    return TrafficSpecification(
        first_day=FIRST_DAY,
        week_count=args.weeks,
        stale_scope_rate=args.stale_scope_rate,
        absent_identifier_rate=args.absent_identifier_rate,
        distinct_keys_range=DEFAULT_DISTINCT_KEYS_RANGE,
    )


def population_command(seed: int, root: Path) -> str:
    """ Comando que o M2 sugere ao usuario para gerar a populacao do M1, caso falte."""
    is_default_root = root == DEFAULT_ROOT
    elsewhere = "" if is_default_root else f" --out {root}"

    return f"python -m src.population --seed {seed}{elsewhere}"


def read_population(
    directory: Path, suggested_command: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Le as tabelas da respectiva seed: `operators.csv` e `keys.csv`."""
    operators_path = directory / "operators.csv"
    keys_path = directory / "keys.csv"

    is_missing = not operators_path.exists() or not keys_path.exists()

    if is_missing:
        raise FileNotFoundError(
            f"faltam as tabelas do M1 em {directory}"
            f"\n       rode antes:  {suggested_command}"
        )

    return pd.read_csv(operators_path), pd.read_csv(keys_path)


def report(destination: Path, requests: pd.DataFrame) -> None:
    """Resumo da geracao, para conferencia imediata na linha de comando."""
    sessions = requests["session_id"].nunique()
    operators = requests["operator_id"].nunique()
    first = requests["timestamp"].iloc[0][:10]
    last = requests["timestamp"].iloc[-1][:10]

    print(f"{destination}")
    print(f"  requests.csv   {len(requests)} requisicoes em {sessions} sessoes")
    print(f"                 {operators} operadores ativos")
    print(f"                 de {first} a {last}")


def main() -> None:
    args = parse_args()
    specification = traffic_specification_from(args)

    destination = seed_directory(args.out, args.seed)
    suggested_command = population_command(args.seed, args.out)


    operators, keys = read_population(destination, suggested_command)
    
    requests = build_traffic(args.seed, operators, keys, specification)

    print("requests")
    print(requests.describe().T)
    print()

    write_csv(requests, destination / "requests.csv")
    report(destination, requests)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as missing:
        raise SystemExit(f"erro: {missing}")
