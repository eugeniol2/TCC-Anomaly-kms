"""Linha de comando do M8: os limiares do baseline, do aquecimento.

    python -m src.calibration --seed 1

Nao recebe `--fase` nem `--sigma`. O aquecimento e anterior ao ataque e mora no
ramo da semente: nao ha segundo ramo onde este modulo pudesse rodar (D-043).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.calibration.build import build_thresholds
from src.globals.layout import DEFAULT_ROOT, seed_directory
from src.globals.tables import write_csv


class Arguments(argparse.Namespace):
    """Atributos que a linha de comando produz."""

    seed: int
    out: Path


def parse_args() -> Arguments:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, required=True, help="semente do experimento")
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_ROOT, help="raiz da pasta de dados"
    )

    return parser.parse_args(namespace=Arguments())


def read_sessions(tables: Path) -> pd.DataFrame:
    """O conjunto de sessoes do aquecimento, com erro claro quando falta."""
    path = tables / "sessions.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"falta `sessions.csv` em {tables}"
            f"\n       rode antes:  python -m src.dataset --seed N --fase warmup"
        )

    return pd.read_csv(path)


def report(destination: Path, thresholds: pd.DataFrame) -> None:
    """Resumo da execucao, com os limiares a vista para conferencia."""
    print(f"{destination}")
    print(f"  thresholds.csv   {len(thresholds)} regras de grandeza, "
          f"percentil {thresholds['percentile'].iloc[0]} de "
          f"{thresholds['sessions'].iloc[0]} sessoes do aquecimento")

    for row in thresholds.itertuples():
        print(f"     {row.attribute:<22} > {row.threshold:>10.4f}")


def main() -> None:
    args = parse_args()

    destination = seed_directory(args.out, args.seed)
    thresholds = build_thresholds(read_sessions(destination))

    write_csv(thresholds, destination / "thresholds.csv")
    report(destination, thresholds)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as missing:
        raise SystemExit(f"erro: {missing}")
