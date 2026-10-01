from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from src.entities.audit_logger.build import build_log
from src.entities.dataset_generator.dataset.build import build_dataset
from src.entities.dataset_generator.historical_profiles.build import build_profiles
from src.entities.dataset_generator.partition.build import build_partition
from src.entities.kms.build import build_outcomes
from src.entities.kms.repository.parameters import KeyRepositorySpecification
from src.entities.models.build import build_models
from src.pipeline.population import build_population
from src.pipeline.stages import LAST_STAGE, reaches, run_steps
from src.entities.policy_engine.baseline.build import build_baseline
from src.entities.policy_engine.calibration.build import build_thresholds
from src.entities.scenario_engine.attack.build import build_attack
from src.entities.scenario_engine.attack.parameters import AttackSpecification
from src.entities.scenario_engine.traffic.build import build_traffic
from src.entities.scenario_engine.traffic.parameters import TrafficSpecification
from src.shared import layout
from src.shared.phases import EVALUATED, WARMUP
from src.shared.tables import write_csv


class Specifications(NamedTuple):
    repository: KeyRepositorySpecification = KeyRepositorySpecification()
    traffic: TrafficSpecification = TrafficSpecification()
    attack: AttackSpecification = AttackSpecification()
    models: dict[str, dict] | None = None


@dataclass
class SeedBranch:
    seed: int
    root: Path
    specifications: Specifications
    operators: pd.DataFrame | None = None
    keys: pd.DataFrame | None = None
    requests: pd.DataFrame | None = None
    outcomes: pd.DataFrame | None = None
    log: pd.DataFrame | None = None
    profiles: pd.DataFrame | None = None
    sessions: pd.DataFrame | None = None
    thresholds: pd.DataFrame | None = None

    @property
    def directory(self) -> Path:
        return layout.seed_directory(self.root, self.seed)


@dataclass
class SigmaBranch:
    sigma: float
    seed_branch: SeedBranch
    requests: pd.DataFrame | None = None
    compromised: pd.DataFrame | None = None
    run: pd.DataFrame | None = None
    outcomes: pd.DataFrame | None = None
    log: pd.DataFrame | None = None
    sessions: pd.DataFrame | None = None
    train: pd.DataFrame | None = None
    holdout: pd.DataFrame | None = None
    predictions_rules: pd.DataFrame | None = None
    predictions_ml: pd.DataFrame | None = None

    @property
    def directory(self) -> Path:
        return layout.run_directory(self.seed_branch.root, self.seed_branch.seed, self.sigma)


def population(branch: SeedBranch) -> None:
    """Operadores (Scenario Engine) e repositorio de chaves (KMS), do mesmo sorteio."""
    built = build_population(branch.seed, branch.specifications.repository)
    write_csv(built.operators, branch.directory / layout.OPERATORS)
    write_csv(built.keys, branch.directory / layout.KEYS)
    branch.operators = built.operators
    branch.keys = built.keys


def legitimate_traffic(branch: SeedBranch) -> None:
    requests = build_traffic(
        branch.seed, branch.operators, branch.keys, branch.specifications.traffic
    )
    write_csv(requests, branch.directory / layout.REQUESTS)
    branch.requests = requests


def warmup_outcomes(branch: SeedBranch) -> None:
    outcomes = build_outcomes(branch.requests, branch.keys, branch.operators, WARMUP)
    write_csv(outcomes, branch.directory / layout.OUTCOMES)
    branch.outcomes = outcomes


def warmup_log(branch: SeedBranch) -> None:
    log = build_log(branch.requests, branch.outcomes, WARMUP)
    write_csv(log, branch.directory / layout.LOG)
    branch.log = log


def historical_profiles(branch: SeedBranch) -> None:
    profiles = build_profiles(branch.log)
    write_csv(profiles, branch.directory / layout.HISTORICAL_PROFILES)
    branch.profiles = profiles


def warmup_sessions(branch: SeedBranch) -> None:
    sessions = build_dataset(branch.log, branch.profiles, WARMUP)
    write_csv(sessions, branch.directory / layout.SESSIONS)
    branch.sessions = sessions


def thresholds(branch: SeedBranch) -> None:
    calibrated = build_thresholds(branch.sessions)
    write_csv(calibrated, branch.directory / layout.THRESHOLDS)
    branch.thresholds = calibrated


SEED_STEPS = (
    ("scenario_engine", population),
    ("scenario_engine", legitimate_traffic),
    ("kms", warmup_outcomes),
    ("audit_logger", warmup_log),
    ("dataset_generator", historical_profiles),
    ("dataset_generator", warmup_sessions),
    ("policy_engine", thresholds),
)


def attack_campaign(branch: SigmaBranch) -> None:
    seed = branch.seed_branch
    campaign = build_attack(
        seed.seed, branch.sigma, seed.operators, seed.keys, seed.requests,
        seed.specifications.traffic, seed.specifications.attack,
    )
    write_csv(campaign.requests, branch.directory / layout.REQUESTS)
    write_csv(campaign.compromised, branch.directory / layout.COMPROMISED_SESSIONS)
    write_csv(campaign.run, branch.directory / layout.RUN)
    branch.requests = campaign.requests
    branch.compromised = campaign.compromised
    branch.run = campaign.run


def evaluated_outcomes(branch: SigmaBranch) -> None:
    seed = branch.seed_branch
    outcomes = build_outcomes(branch.requests, seed.keys, seed.operators, EVALUATED)
    write_csv(outcomes, branch.directory / layout.OUTCOMES)
    branch.outcomes = outcomes


def evaluated_log(branch: SigmaBranch) -> None:
    log = build_log(branch.requests, branch.outcomes, EVALUATED)
    write_csv(log, branch.directory / layout.LOG)
    branch.log = log


def evaluated_sessions(branch: SigmaBranch) -> None:
    sessions = build_dataset(
        branch.log, branch.seed_branch.profiles, EVALUATED, branch.compromised
    )
    write_csv(sessions, branch.directory / layout.SESSIONS)
    branch.sessions = sessions


def partition(branch: SigmaBranch) -> None:
    sides = build_partition(branch.seed_branch.seed, branch.sessions)
    write_csv(sides.train, branch.directory / layout.TRAIN)
    write_csv(sides.holdout, branch.directory / layout.HOLDOUT)
    branch.train = sides.train
    branch.holdout = sides.holdout


def rule_decisions(branch: SigmaBranch) -> None:
    baseline = build_baseline(branch.holdout, branch.seed_branch.thresholds)
    write_csv(baseline.predictions, branch.directory / layout.PREDICTIONS_RULES)
    write_csv(baseline.timing, branch.directory / layout.TIMING_RULES)
    branch.predictions_rules = baseline.predictions


def model_decisions(branch: SigmaBranch) -> None:
    seed = branch.seed_branch
    models = build_models(
        seed.seed, branch.train, branch.holdout, seed.specifications.models
    )
    write_csv(models.predictions, branch.directory / layout.PREDICTIONS_ML)
    write_csv(models.timing, branch.directory / layout.TIMING_ML)
    branch.predictions_ml = models.predictions


SIGMA_STEPS = (
    ("scenario_engine", attack_campaign),
    ("kms", evaluated_outcomes),
    ("audit_logger", evaluated_log),
    ("dataset_generator", evaluated_sessions),
    ("dataset_generator", partition),
    ("policy_engine", rule_decisions),
    ("models", model_decisions),
)


def run_seed_branch(
    seed: int, root: Path, specifications: Specifications, until: str = LAST_STAGE
) -> SeedBranch:
    """As semanas 1 a 4 de uma semente, ate a entidade `until`."""
    branch = SeedBranch(seed, root, specifications)

    return run_steps(SEED_STEPS, branch, until)


def run_sigma_branch(branch: SeedBranch, sigma: float, until: str = LAST_STAGE) -> SigmaBranch:
    """As semanas 5 a 8 de uma condicao, ate a entidade `until`.

    Recebe o ramo da semente pronto. Recusa rodar os modelos sem a configuracao da
    busca, antes de gravar qualquer arquivo.
    """
    is_unconfigured = reaches(until, "models") and branch.specifications.models is None

    if is_unconfigured:
        raise ValueError("falta a configuracao dos modelos; rode antes a busca da 902")

    return run_steps(SIGMA_STEPS, SigmaBranch(sigma, branch), until)


def run_sweep(
    seed: int,
    sigmas: tuple[float, ...],
    root: Path,
    specifications: Specifications,
    until: str = LAST_STAGE,
) -> list[pd.DataFrame]:
    """Uma semente inteira: o aquecimento uma vez, e cada condicao de sigma.

    Devolve as linhas de `run.csv`, que o indice agregado junta.
    """
    branch = run_seed_branch(seed, root, specifications, until)
    rows = []

    for sigma in sigmas:
        sigma_branch = run_sigma_branch(branch, sigma, until)
        rows.append(sigma_branch.run)

    return rows


def write_runs_index(rows: list[pd.DataFrame], root: Path) -> pd.DataFrame:
    """Junta os `run.csv` das execucoes no indice, grava na raiz e o devolve."""
    index = pd.concat(rows, ignore_index=True).sort_values(
        ["seed", "sigma"], ignore_index=True
    )
    write_csv(index, root / layout.RUNS_INDEX)

    return index
