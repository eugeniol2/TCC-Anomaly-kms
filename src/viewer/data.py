"""De onde o viewer tira os dados: os arquivos que o comando unico gravou.

O viewer nao roda o pipeline. Ele le o que o `python -m src.main` gravou em
`data/`, e por isso mostra exatamente o resultado real, sem uma segunda copia da
ordem de execucao (D-121). Nada aqui depende do Streamlit: o app guarda em cache,
e os testes chamam direto.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.shared import layout


@dataclass(frozen=True)
class SeedFiles:
    """O ramo da semente: semanas 1 a 4, sem atacante."""

    operators: pd.DataFrame
    keys: pd.DataFrame
    requests: pd.DataFrame
    outcomes: pd.DataFrame
    log: pd.DataFrame
    profiles: pd.DataFrame
    sessions: pd.DataFrame
    thresholds: pd.DataFrame


@dataclass(frozen=True)
class RunFiles:
    """Uma execucao (semente, sigma): semanas 5 a 8, com a campanha e as decisoes."""

    requests: pd.DataFrame
    compromised: pd.DataFrame
    run: pd.DataFrame
    outcomes: pd.DataFrame
    log: pd.DataFrame
    sessions: pd.DataFrame
    train: pd.DataFrame
    holdout: pd.DataFrame
    predictions_rules: pd.DataFrame
    predictions_ml: pd.DataFrame
    timing: pd.DataFrame


@dataclass(frozen=True)
class GridFiles:
    """As quatro tabelas da avaliacao, na raiz."""

    metrics: pd.DataFrame
    triviality: pd.DataFrame
    comparison: pd.DataFrame
    timing: pd.DataFrame


SEED_NAMES = (
    layout.OPERATORS, layout.KEYS, layout.REQUESTS, layout.OUTCOMES,
    layout.LOG, layout.HISTORICAL_PROFILES, layout.SESSIONS, layout.THRESHOLDS,
)
RUN_NAMES = (
    layout.REQUESTS, layout.COMPROMISED_SESSIONS, layout.RUN, layout.OUTCOMES,
    layout.LOG, layout.SESSIONS, layout.TRAIN, layout.HOLDOUT,
    layout.PREDICTIONS_RULES, layout.PREDICTIONS_ML,
)
GRID_NAMES = (layout.METRICS, layout.TRIVIALITY, layout.COMPARISON, layout.TIMING)


def missing_files(directory: Path, names: tuple[str, ...]) -> list[str]:
    """Os arquivos que faltam numa pasta. Vazio quando o comando ja rodou."""
    return [name for name in names if not (directory / name).exists()]


def read_all(directory: Path, names: tuple[str, ...]) -> list[pd.DataFrame]:
    """Le os arquivos, na ordem dada, com erro claro quando falta algum."""
    absent = missing_files(directory, names)

    if absent:
        raise FileNotFoundError(
            f"faltam {', '.join(absent)} em {directory}; rode antes: python -m src.main"
        )

    return [pd.read_csv(directory / name) for name in names]


def read_seed(root: Path, seed: int) -> SeedFiles:
    return SeedFiles(*read_all(layout.seed_directory(root, seed), SEED_NAMES))


def read_run(root: Path, seed: int, sigma: float) -> RunFiles:
    directory = layout.run_directory(root, seed, sigma)
    files = read_all(directory, RUN_NAMES)
    timing = pd.concat(
        read_all(directory, (layout.TIMING_RULES, layout.TIMING_ML)), ignore_index=True
    )

    return RunFiles(*files, timing)


def read_grid(root: Path) -> GridFiles:
    return GridFiles(*read_all(root, GRID_NAMES))


def figure_paths(root: Path) -> dict[str, Path]:
    """As figuras que o comando desenhou, as que existirem."""
    directory = root / layout.FIGURES
    names = {"f1_sigma": "f1_sigma.png", "roc": "roc.png"}

    return {
        name: directory / file
        for name, file in names.items()
        if (directory / file).exists()
    }
