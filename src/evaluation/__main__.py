"""Linha de comando do M12: a avaliacao da grade inteira.

    python -m src.evaluation

Le as 330 execucoes e escreve quatro tabelas na raiz de dados: `metrics.csv`,
`triviality.csv`, `comparison.csv` e `timing.csv`. Falha se faltar alguma
execucao: a comparacao pareada precisa das 30 sementes em todo sigma.

O resumo mostra so onde as tabelas ficaram. Os resultados se leem nelas.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.evaluation.build import build_evaluation, read_run
from src.globals.experiment import SEEDS, SIGMAS
from src.globals.layout import COMPARISON, DEFAULT_ROOT, METRICS, TIMING, TRIVIALITY
from src.globals.tables import write_csv


class Arguments(argparse.Namespace):
    out: Path


def parse_args() -> Arguments:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_ROOT, help="raiz da pasta de dados"
    )

    return parser.parse_args(namespace=Arguments())


def main() -> None:
    args = parse_args()

    runs = [read_run(args.out, seed, sigma) for seed in SEEDS for sigma in SIGMAS]
    evaluation = build_evaluation(runs)

    tables = {
        METRICS: evaluation.metrics,
        TRIVIALITY: evaluation.triviality,
        COMPARISON: evaluation.comparison,
        TIMING: evaluation.timing,
    }

    for name, table in tables.items():
        write_csv(table, args.out / name)
        print(f"  {args.out / name}   {len(table)} linhas")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as missing:
        raise SystemExit(f"erro: {missing}\n       rode antes a grade:  python -m src.pipeline --grade")
