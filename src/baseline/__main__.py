"""Linha de comando do M10: o baseline de regras decidindo sobre o holdout.

    python -m src.baseline --seed 1 --sigma 0.5

Le o `holdout.csv` da condicao e o `thresholds.csv` da semente, que e o mesmo
nas onze condicoes (D-043). Nao recebe `--fase`: so o periodo avaliado tem
holdout.

**O resumo nao mostra alertas nem acertos.** Olhar o resultado no holdout e um
momento que o registro marca (o campo Momento de cada decisao), e ele nao deve
acontecer por acidente no terminal. O arquivo tem tudo, para quando for a hora.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.baseline.build import build_baseline
from src.globals.layout import DEFAULT_ROOT, run_directory, seed_directory
from src.globals.tables import write_csv


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


def read_table(directory: Path, name: str, producer: str) -> pd.DataFrame:
    """Um arquivo de entrada, com o comando que o produz quando falta."""
    path = directory / name

    if not path.exists():
        raise FileNotFoundError(
            f"falta `{name}` em {directory}\n       rode antes:  {producer}"
        )

    return pd.read_csv(path)


def main() -> None:
    args = parse_args()

    destination = run_directory(args.out, args.seed, args.sigma)
    holdout = read_table(
        destination, "holdout.csv", "python -m src.partition --seed N --sigma S"
    )
    thresholds = read_table(
        seed_directory(args.out, args.seed), "thresholds.csv",
        "python -m src.calibration --seed N",
    )

    baseline = build_baseline(holdout, thresholds)

    write_csv(baseline.predictions, destination / "predictions_rules.csv")
    write_csv(baseline.timing, destination / "timing_rules.csv")

    timing = baseline.timing.iloc[0]
    print(f"{destination}")
    print(f"  predictions_rules.csv   {len(baseline.predictions)} sessoes do holdout")
    print(f"  timing_rules.csv        {timing['microseconds_per_session']:.2f} us por "
          f"sessao, mediana de {timing['timed_rounds']} rodadas")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as missing:
        raise SystemExit(f"erro: {missing}")
