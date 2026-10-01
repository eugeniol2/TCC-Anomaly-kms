from __future__ import annotations

from pathlib import Path

DEFAULT_ROOT = Path("data")

OPERATORS = "operators.csv"
KEYS = "keys.csv"
REQUESTS = "requests.csv"
OUTCOMES = "outcomes.csv"
LOG = "log.csv"
HISTORICAL_PROFILES = "historical_profiles.csv"
SESSIONS = "sessions.csv"
THRESHOLDS = "thresholds.csv"
COMPROMISED_SESSIONS = "compromised_sessions.csv"
RUN = "run.csv"
TRAIN = "train.csv"
HOLDOUT = "holdout.csv"
PREDICTIONS_RULES = "predictions_rules.csv"
TIMING_RULES = "timing_rules.csv"
PREDICTIONS_ML = "predictions_ml.csv"
TIMING_ML = "timing_ml.csv"
FIGURES = "figures"

RUNS_INDEX = "runs.csv"

METRICS = "metrics.csv"

TRIVIALITY = "triviality.csv"

COMPARISON = "comparison.csv"

TIMING = "timing.csv"

IMPORTANCE = "importance.csv"

PREPARATION = "preparation"

CONFIGURATION = "config.csv"

SEARCH_RESULTS = "search_results.csv"


def seed_directory(root: Path, seed: int) -> Path:
    """A pasta do ramo da semente: o aquecimento e tudo que dele deriva."""
    return root / f"seed-{seed:02d}"


def run_directory(root: Path, seed: int, sigma: float) -> Path:
    """A pasta do ramo de sigma: o periodo avaliado de uma execucao.

    Um decimal em sigma mantem a ordem alfabetica igual a numerica: sigma-0.0 ate
    sigma-1.0.
    """
    return seed_directory(root, seed) / f"sigma-{sigma:.1f}"


def preparation_directory(root: Path, seed: int) -> Path:
    """A pasta de uma execucao preparatoria, a 902 ou a 903, fora das 330 replicas.

    Dentro dela o layout e o de uma replica.
    """
    return seed_directory(root / PREPARATION, seed)
