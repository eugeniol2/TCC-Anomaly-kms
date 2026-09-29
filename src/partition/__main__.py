"""Linha de comando do M9: o conjunto avaliado em treino e holdout.

    python -m src.partition --seed 1 --sigma 0.5

Nao recebe `--fase`: so o periodo avaliado tem rotulo, e so ele se particiona. O
`--sigma` e obrigatorio porque o conjunto mora numa pasta de sigma, mas a divisao
nao depende dele (D-102): as onze condicoes de uma semente saem com as mesmas
sessoes legitimas de cada lado.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.dataset.build import LABEL
from src.globals.layout import DEFAULT_ROOT, run_directory
from src.globals.tables import write_csv
from src.partition.build import build_partition


class Arguments(argparse.Namespace):
    """Atributos que a linha de comando produz."""

    seed: int
    sigma: float
    out: Path


def parse_args() -> Arguments:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, required=True, help="semente do experimento")
    parser.add_argument("--sigma", type=float, required=True, help="a condicao de sigma")
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_ROOT, help="raiz da pasta de dados"
    )

    return parser.parse_args(namespace=Arguments())


def read_sessions(destination: Path) -> pd.DataFrame:
    """O conjunto avaliado da condicao, com erro claro quando falta."""
    path = destination / "sessions.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"falta `sessions.csv` em {destination}"
            f"\n       rode antes:  python -m src.dataset --seed N --fase evaluated --sigma S"
        )

    return pd.read_csv(path)


def describe(name: str, side: pd.DataFrame) -> str:
    """Uma linha do resumo: tamanho e proporcao de anomalias, contada (D-023)."""
    positives = int(side[LABEL].sum())

    return (f"  {name:<12} {len(side):>5} sessoes   {positives:>3} positivas   "
            f"{positives / len(side):.2%}")


def main() -> None:
    args = parse_args()

    destination = run_directory(args.out, args.seed, args.sigma)
    partition = build_partition(args.seed, read_sessions(destination))

    write_csv(partition.train, destination / "train.csv")
    write_csv(partition.holdout, destination / "holdout.csv")

    print(f"{destination}")
    print(describe("train.csv", partition.train))
    print(describe("holdout.csv", partition.holdout))


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as missing:
        raise SystemExit(f"erro: {missing}")
