"""Linha de comando do M11: os dois modelos treinados e decidindo sobre o holdout.

    python -m src.models --seed 1 --sigma 0.5

Le o `train.csv` e o `holdout.csv` da condicao, e a configuracao que a busca da 902
escolheu. A busca em si roda pelo orquestrador: `python -m src.pipeline --search`.

Como no baseline, o resumo mostra so tamanho e tempo, nunca acerto (D-113).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.globals.experiment import HYPERPARAMETER_SEARCH_SEED
from src.globals.layout import (
    CONFIGURATION,
    DEFAULT_ROOT,
    preparation_directory,
    run_directory,
)
from src.globals.tables import write_csv
from src.models.build import build_models, read_configuration


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


def read_side(directory: Path, name: str) -> pd.DataFrame:
    path = directory / name

    if not path.exists():
        raise FileNotFoundError(
            f"falta `{name}` em {directory}"
            "\n       rode antes:  python -m src.partition --seed N --sigma S"
        )

    return pd.read_csv(path)


def main() -> None:
    args = parse_args()

    destination = run_directory(args.out, args.seed, args.sigma)
    search = preparation_directory(args.out, HYPERPARAMETER_SEARCH_SEED)

    models = build_models(
        args.seed,
        read_side(destination, "train.csv"),
        read_side(destination, "holdout.csv"),
        read_configuration(search / CONFIGURATION),
    )

    write_csv(models.predictions, destination / "predictions_ml.csv")
    write_csv(models.timing, destination / "timing_ml.csv")

    print(f"{destination}")
    print(f"  predictions_ml.csv   {len(models.predictions)} sessoes do holdout")

    for row in models.timing.itertuples():
        print(f"  timing_ml.csv        {row.mechanism:<14} "
              f"{row.microseconds_per_session:.2f} us por sessao")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as missing:
        raise SystemExit(f"erro: {missing}")
