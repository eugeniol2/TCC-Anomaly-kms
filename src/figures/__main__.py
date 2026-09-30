"""Linha de comando das figuras: desenha a partir da grade ja rodada.

    python -m src.figures

Le o `metrics.csv` e o `comparison.csv` da raiz de dados, e as predicoes das 30
sementes em sigma 0,7, 0,8 e 0,9. Grava `f1_sigma` e `roc` em `figures/`, cada uma
em PNG e em PDF.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.figures.build import draw_f1_by_sigma, draw_roc
from src.figures.parameters import RESOLUTION
from src.globals.layout import COMPARISON, DEFAULT_ROOT, METRICS


class Arguments(argparse.Namespace):
    out: Path


def parse_args() -> Arguments:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_ROOT, help="raiz da pasta de dados"
    )

    return parser.parse_args(namespace=Arguments())


def save(figure, directory: Path, name: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)

    for suffix in ("png", "pdf"):
        path = directory / f"{name}.{suffix}"
        figure.savefig(path, dpi=RESOLUTION, facecolor=figure.get_facecolor())
        print(f"  {path}")


def main() -> None:
    args = parse_args()
    directory = args.out / "figures"

    metrics = pd.read_csv(args.out / METRICS)
    comparison = pd.read_csv(args.out / COMPARISON)

    save(draw_f1_by_sigma(metrics, comparison), directory, "f1_sigma")
    save(draw_roc(args.out), directory, "roc")


if __name__ == "__main__":
    main()
