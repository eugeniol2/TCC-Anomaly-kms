"""Os dois ramos de uma execucao, como listas de passos (D-091, D-121).

A ordem de execucao e codigo, e mora aqui: `SEED_STEPS` e `SIGMA_STEPS` sao as
duas listas. Cada passo e um par, a entidade a que pertence e a funcao que roda.

**Dois ramos, porque o atacante so age nas semanas 5 a 8** (D-049). O ramo da
semente produz tudo que e anterior ao ataque e roda uma vez por semente; o ramo
de sigma produz o periodo avaliado e roda uma vez por condicao, recebendo o ramo
da semente pronto. Recomputar o aquecimento em cada condicao arriscaria perfis
diferentes entre condicoes da mesma semente, e o pareamento da D-002 quebraria.

**Cada passo grava o seu arquivo**, e passa o quadro adiante em memoria. O
arquivo e o que deixa cada entidade inspecionavel.
"""

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
    """Os parametros das entidades, reunidos para viajar por parametro.

    O teste troca um valor sem editar arquivo nenhum, e a funcao recebe o que
    precisa em vez de ler modulo global.
    """

    repository: KeyRepositorySpecification = KeyRepositorySpecification()
    traffic: TrafficSpecification = TrafficSpecification()
    attack: AttackSpecification = AttackSpecification()
    models: dict[str, dict] | None = None
    """A configuracao que a busca da 902 escolheu. Sem ela os modelos nao rodam."""


@dataclass
class SeedBranch:
    """O ramo da semente, preenchido passo a passo."""

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
    """Uma condicao (semente, sigma), preenchida passo a passo."""

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


def emit(frame: pd.DataFrame, directory: Path, name: str) -> pd.DataFrame:
    """Grava o arquivo e devolve o quadro, para o passo seguinte receber em memoria."""
    write_csv(frame, directory / name)

    return frame


# Os passos do ramo da semente: semanas 1 a 4, sem atacante.


def population(branch: SeedBranch) -> None:
    """Operadores (Scenario Engine) e repositorio de chaves (KMS), do mesmo sorteio."""
    built = build_population(branch.seed, branch.specifications.repository)
    branch.operators = emit(built.operators, branch.directory, layout.OPERATORS)
    branch.keys = emit(built.keys, branch.directory, layout.KEYS)


def legitimate_traffic(branch: SeedBranch) -> None:
    requests = build_traffic(
        branch.seed, branch.operators, branch.keys, branch.specifications.traffic
    )
    branch.requests = emit(requests, branch.directory, layout.REQUESTS)


def warmup_outcomes(branch: SeedBranch) -> None:
    outcomes = build_outcomes(branch.requests, branch.keys, branch.operators, WARMUP)
    branch.outcomes = emit(outcomes, branch.directory, layout.OUTCOMES)


def warmup_log(branch: SeedBranch) -> None:
    log = build_log(branch.requests, branch.outcomes, WARMUP)
    branch.log = emit(log, branch.directory, layout.LOG)


def historical_profiles(branch: SeedBranch) -> None:
    profiles = build_profiles(branch.log)
    branch.profiles = emit(profiles, branch.directory, layout.HISTORICAL_PROFILES)


def warmup_sessions(branch: SeedBranch) -> None:
    sessions = build_dataset(branch.log, branch.profiles, WARMUP)
    branch.sessions = emit(sessions, branch.directory, layout.SESSIONS)


def thresholds(branch: SeedBranch) -> None:
    branch.thresholds = emit(
        build_thresholds(branch.sessions), branch.directory, layout.THRESHOLDS
    )


SEED_STEPS = (
    ("scenario_engine", population),
    ("scenario_engine", legitimate_traffic),
    ("kms", warmup_outcomes),
    ("audit_logger", warmup_log),
    ("dataset_generator", historical_profiles),
    ("dataset_generator", warmup_sessions),
    ("policy_engine", thresholds),
)


# Os passos do ramo de sigma: semanas 5 a 8, com a campanha.


def attack_campaign(branch: SigmaBranch) -> None:
    seed = branch.seed_branch
    campaign = build_attack(
        seed.seed, branch.sigma, seed.operators, seed.keys, seed.requests,
        seed.specifications.traffic, seed.specifications.attack,
    )
    branch.requests = emit(campaign.requests, branch.directory, layout.REQUESTS)
    branch.compromised = emit(
        campaign.compromised, branch.directory, layout.COMPROMISED_SESSIONS
    )
    branch.run = emit(campaign.run, branch.directory, layout.RUN)


def evaluated_outcomes(branch: SigmaBranch) -> None:
    seed = branch.seed_branch
    outcomes = build_outcomes(branch.requests, seed.keys, seed.operators, EVALUATED)
    branch.outcomes = emit(outcomes, branch.directory, layout.OUTCOMES)


def evaluated_log(branch: SigmaBranch) -> None:
    log = build_log(branch.requests, branch.outcomes, EVALUATED)
    branch.log = emit(log, branch.directory, layout.LOG)


def evaluated_sessions(branch: SigmaBranch) -> None:
    sessions = build_dataset(
        branch.log, branch.seed_branch.profiles, EVALUATED, branch.compromised
    )
    branch.sessions = emit(sessions, branch.directory, layout.SESSIONS)


def partition(branch: SigmaBranch) -> None:
    sides = build_partition(branch.seed_branch.seed, branch.sessions)
    branch.train = emit(sides.train, branch.directory, layout.TRAIN)
    branch.holdout = emit(sides.holdout, branch.directory, layout.HOLDOUT)


def rule_decisions(branch: SigmaBranch) -> None:
    baseline = build_baseline(branch.holdout, branch.seed_branch.thresholds)
    branch.predictions_rules = emit(
        baseline.predictions, branch.directory, layout.PREDICTIONS_RULES
    )
    emit(baseline.timing, branch.directory, layout.TIMING_RULES)


def model_decisions(branch: SigmaBranch) -> None:
    seed = branch.seed_branch
    models = build_models(
        seed.seed, branch.train, branch.holdout, seed.specifications.models
    )
    branch.predictions_ml = emit(models.predictions, branch.directory, layout.PREDICTIONS_ML)
    emit(models.timing, branch.directory, layout.TIMING_ML)


SIGMA_STEPS = (
    ("scenario_engine", attack_campaign),
    ("kms", evaluated_outcomes),
    ("audit_logger", evaluated_log),
    ("dataset_generator", evaluated_sessions),
    ("dataset_generator", partition),
    ("policy_engine", rule_decisions),
    ("models", model_decisions),
)


# Os dois ramos, e a varredura de uma semente.


def run_seed_branch(
    seed: int, root: Path, specifications: Specifications, until: str = LAST_STAGE
) -> SeedBranch:
    """As semanas 1 a 4 de uma semente, ate a entidade `until`."""
    branch = SeedBranch(seed, root, specifications)

    return run_steps(SEED_STEPS, branch, until)


def run_sigma_branch(branch: SeedBranch, sigma: float, until: str = LAST_STAGE) -> SigmaBranch:
    """As semanas 5 a 8 de uma condicao, ate a entidade `until`.

    Recebe o ramo da semente pronto: e o que garante que as onze condicoes
    compartilhem exatamente o mesmo aquecimento. Recusa rodar os modelos sem a
    configuracao da busca, antes de gravar qualquer arquivo.
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
    """Junta os `run.csv` das execucoes no indice da D-085.

    Quem o escreve e quem percorre a grade, e por isso ele nasce completo ou nao
    nasce: reexecutar uma condicao isolada nao duplica linha.
    """
    index = pd.concat(rows, ignore_index=True).sort_values(
        ["seed", "sigma"], ignore_index=True
    )

    return emit(index, root, layout.RUNS_INDEX)
