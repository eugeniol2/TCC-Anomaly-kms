"""Testes do orquestrador: a ordem de execução, agora como código.

O risco que este arquivo guarda não é o de o orquestrador quebrar: ele é uma
lista de chamadas e quebraria ruidosamente. É o de ele **divergir dos módulos**
que orquestra: alguém muda a composição de um módulo, o `__main__` dele
acompanha, e a linha correspondente aqui fica para trás produzindo um arquivo
diferente com o mesmo nome.

Por isso o teste central compara, arquivo por arquivo, o que o orquestrador
escreve contra o que os módulos escreveriam chamados um a um. Se os dois
caminhos divergirem, o pipeline passa a ter duas ordens e nenhuma é a
verdadeira.

Tudo roda em `tmp_path`: nenhum teste escreve em `data/`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.entities.scenario_engine.attack.build import build_attack
from src.entities.audit_logger.build import build_log
from src.entities.policy_engine.baseline.build import build_baseline
from src.entities.policy_engine.calibration.build import build_thresholds
from src.entities.dataset_generator.dataset.build import build_dataset
from src.metrics.evaluation.build import MECHANISMS, SCOPES
from src.shared.experiment import PREPARATION_SIGMA, REHEARSAL_SEED, SIGMAS
from src.shared.layout import (
    METRICS,
    PREPARATION,
    RUNS_INDEX,
    TRIVIALITY,
    run_directory,
    seed_directory,
)
from src.shared.phases import EVALUATED, WARMUP
from src.shared.timing import COLUMNS as TIMING_COLUMNS
from src.entities.dataset_generator.historical_profiles.build import build_profiles
from src.entities.kms.build import build_outcomes
from src.entities.models.build import build_models
from src.entities.dataset_generator.partition.build import build_partition
from src.pipeline.build import (
    Specifications,
    run_seed_branch,
    run_sigma_branch,
    run_sweep,
    write_runs_index,
)
from src.pipeline.population import build_population
from src.pipeline.experiment import run_rehearsal
from src.entities.scenario_engine.traffic.build import build_traffic

TEST_CONFIGURATION = {
    "random_forest": {
        "n_estimators": 10, "max_depth": None, "min_samples_leaf": 1,
        "max_features": "sqrt", "class_weight": None,
    },
    "xgboost": {
        "n_estimators": 10, "max_depth": 3, "learning_rate": 0.1,
        "subsample": 1.0, "colsample_bytree": 1.0, "class_weight": "balanced",
    },
}
"""Uma configuracao pequena, so para os testes rodarem rapido. A de verdade sai da 902."""

SPECIFICATIONS = Specifications(models=TEST_CONFIGURATION)

SEED = 3
SAMPLE_SIGMAS = (0.0, 0.5, 1.0)

SEED_FILES = (
    "operators.csv",
    "keys.csv",
    "requests.csv",
    "outcomes.csv",
    "log.csv",
    "historical_profiles.csv",
    "sessions.csv",
    "thresholds.csv",
)

SIGMA_FILES = (
    "requests.csv",
    "compromised_sessions.csv",
    "run.csv",
    "outcomes.csv",
    "log.csv",
    "sessions.csv",
    "train.csv",
    "holdout.csv",
    "predictions_rules.csv",
    "predictions_ml.csv",
)

TIMING_FILES = {"timing_rules.csv": 1, "timing_ml.csv": 2}
"""Os arquivos de tempo, e quantos mecanismos cada um mede."""
"""Os arquivos de tempo, fora de toda comparacao de conteudo.

Tempo muda a cada execucao, entao eles nunca sao iguais aos de outra rodada, e
nem deveriam ser. O que se confere neles e que existem e tem a forma certa.
"""


def modules_one_by_one(seed: int, sigma: float) -> dict[str, pd.DataFrame]:
    """O que sai de chamar cada módulo na mão, sem passar pelo orquestrador.

    É deliberadamente uma segunda implementação da ordem. Duas implementações
    que precisam concordar são caras de manter, e é exatamente isso que as
    torna um teste: manter as duas em dia custa menos que descobrir tarde que
    o orquestrador produz outro arquivo.
    """
    population = build_population(seed, SPECIFICATIONS.repository)
    requests = build_traffic(
        seed, population.operators, population.keys, SPECIFICATIONS.traffic
    )

    warmup_outcomes = build_outcomes(
        requests, population.keys, population.operators, WARMUP
    )
    warmup_log = build_log(requests, warmup_outcomes, WARMUP)
    profiles = build_profiles(warmup_log)
    warmup_sessions = build_dataset(warmup_log, profiles, WARMUP)

    campaign = build_attack(
        seed, sigma, population.operators, population.keys, requests,
        SPECIFICATIONS.traffic, SPECIFICATIONS.attack,
    )
    evaluated_outcomes = build_outcomes(
        campaign.requests, population.keys, population.operators, EVALUATED
    )
    evaluated_log = build_log(campaign.requests, evaluated_outcomes, EVALUATED)
    evaluated_sessions = build_dataset(
        evaluated_log, profiles, EVALUATED, campaign.compromised
    )
    partition = build_partition(seed, evaluated_sessions)
    thresholds = build_thresholds(warmup_sessions)

    return {
        "operators.csv": population.operators,
        "keys.csv": population.keys,
        "requests.csv": requests,
        "outcomes.csv": warmup_outcomes,
        "log.csv": warmup_log,
        "historical_profiles.csv": profiles,
        "sessions.csv": warmup_sessions,
        "thresholds.csv": thresholds,
        "sigma/requests.csv": campaign.requests,
        "sigma/compromised_sessions.csv": campaign.compromised,
        "sigma/run.csv": campaign.run,
        "sigma/outcomes.csv": evaluated_outcomes,
        "sigma/log.csv": evaluated_log,
        "sigma/sessions.csv": evaluated_sessions,
        "sigma/train.csv": partition.train,
        "sigma/holdout.csv": partition.holdout,
        "sigma/predictions_rules.csv": build_baseline(
            partition.holdout, thresholds
        ).predictions,
        "sigma/predictions_ml.csv": build_models(
            seed, partition.train, partition.holdout, TEST_CONFIGURATION
        ).predictions,
    }


def test_the_orchestrator_writes_what_the_modules_would_write(tmp_path: Path) -> None:
    """A ordem codificada aqui é a mesma que os módulos executam sozinhos."""
    sigma = 0.5

    branch = run_seed_branch(SEED, tmp_path, SPECIFICATIONS)
    run_sigma_branch(branch, sigma)

    expected = modules_one_by_one(SEED, sigma)

    for name in SEED_FILES:
        written = pd.read_csv(seed_directory(tmp_path, SEED) / name)
        assert written.equals(
            pd.read_csv(_as_csv(expected[name], tmp_path, name))
        ), name

    for name in SIGMA_FILES:
        written = pd.read_csv(run_directory(tmp_path, SEED, sigma) / name)
        assert written.equals(
            pd.read_csv(_as_csv(expected[f"sigma/{name}"], tmp_path, f"s_{name}"))
        ), name


def _as_csv(frame: pd.DataFrame, directory: Path, name: str) -> Path:
    """Escreve e relê, para comparar depois da mesma travessia de CSV.

    Comparar o quadro em memória contra o arquivo lido acusaria diferença de
    tipo (um inteiro que volta como float, uma data que volta como texto) em
    vez de diferença de conteúdo, que é o que interessa.
    """
    path = directory / f"esperado_{name}"
    frame.to_csv(path, index=False, lineterminator="\n")

    return path


def test_every_expected_file_is_written(tmp_path: Path) -> None:
    """Nenhuma etapa deixa de escrever a sua saída."""
    branch = run_seed_branch(SEED, tmp_path, SPECIFICATIONS)
    run_sigma_branch(branch, 0.5)

    for name in SEED_FILES:
        assert (seed_directory(tmp_path, SEED) / name).exists(), name

    for name in SIGMA_FILES:
        assert (run_directory(tmp_path, SEED, 0.5) / name).exists(), name


def test_the_timing_files_are_written_with_their_columns(tmp_path: Path) -> None:
    """O tempo nao se compara byte a byte, mas o arquivo tem de existir e ter forma."""
    branch = run_seed_branch(SEED, tmp_path, SPECIFICATIONS)
    run_sigma_branch(branch, 0.5)

    for name, mechanisms in TIMING_FILES.items():
        timing = pd.read_csv(run_directory(tmp_path, SEED, 0.5) / name)

        assert tuple(timing.columns) == TIMING_COLUMNS, name
        assert len(timing) == mechanisms, name
        assert (timing["median_nanoseconds"] > 0).all(), name


def test_the_rehearsal_runs_the_whole_pipeline_on_the_reserved_seed(tmp_path: Path) -> None:
    """A 903 passa por todos os modulos e grava metricas e arvore rasa (D-107)."""
    evaluation = run_rehearsal(tmp_path, SPECIFICATIONS)
    directory = run_directory(tmp_path / PREPARATION, REHEARSAL_SEED, PREPARATION_SIGMA)

    assert len(evaluation.metrics) == len(MECHANISMS) * len(SCOPES)
    assert len(evaluation.triviality) == 1
    assert (directory / METRICS).exists()
    assert (directory / TRIVIALITY).exists()


def test_the_sigma_branch_refuses_to_run_without_a_configuration(tmp_path: Path) -> None:
    """Sem a busca da 902, nao ha modelo para treinar, e isso falha antes de escrever."""
    unconfigured = Specifications()
    branch = run_seed_branch(SEED, tmp_path, unconfigured)

    with pytest.raises(ValueError, match="configuracao"):
        run_sigma_branch(branch, 0.5)

    assert not run_directory(tmp_path, SEED, 0.5).exists()


@pytest.mark.parametrize("sigma", SAMPLE_SIGMAS)
def test_the_warmup_is_identical_across_conditions(
    tmp_path: Path, sigma: float
) -> None:
    """O aquecimento é computado uma vez e compartilhado pelas onze condições.

    Se cada condição o recomputasse, bastaria um sorteio consumido em ordem
    diferente para os perfis divergirem entre condições da mesma semente, e o
    pareamento que a D-002 assume quebraria sem erro e sem aviso.
    """
    branch = run_seed_branch(SEED, tmp_path, SPECIFICATIONS)
    before = (seed_directory(tmp_path, SEED) / "thresholds.csv").read_bytes()

    run_sigma_branch(branch, sigma)
    after = (seed_directory(tmp_path, SEED) / "thresholds.csv").read_bytes()

    assert before == after


def test_a_sweep_pairs_the_conditions(tmp_path: Path) -> None:
    """As onze condições de uma semente compartilham o administrador alvo.

    É a D-011 conferida do lado do orquestrador: não basta o M3 sortear igual,
    a varredura precisa entregar o mesmo ramo da semente às onze.
    """
    rows = run_sweep(SEED, SIGMAS, tmp_path, SPECIFICATIONS)
    index = pd.concat(rows, ignore_index=True)

    assert len(index) == len(SIGMAS)
    assert index["compromised_admin"].nunique() == 1
    assert sorted(index["sigma"]) == sorted(SIGMAS)


def test_the_runs_index_is_written_sorted_and_complete(tmp_path: Path) -> None:
    """D-085: o índice nasce completo, escrito por quem percorre a grade."""
    rows = run_sweep(SEED, SAMPLE_SIGMAS, tmp_path, SPECIFICATIONS)
    rows.extend(run_sweep(SEED + 1, SAMPLE_SIGMAS, tmp_path, SPECIFICATIONS))

    index = write_runs_index(rows, tmp_path)

    assert (tmp_path / RUNS_INDEX).exists()
    assert len(index) == 2 * len(SAMPLE_SIGMAS)
    assert list(index.columns) == ["seed", "sigma", "compromised_admin"]
    assert index.equals(index.sort_values(["seed", "sigma"], ignore_index=True))


def test_running_twice_produces_the_same_bytes(tmp_path: Path) -> None:
    """Determinismo de ponta a ponta, do orquestrador e não de um módulo só.

    Apagar `data/` inteiro e reexecutar é a forma de conferir determinismo, e
    este teste é essa conferência feita em duas pastas.
    """
    first = tmp_path / "primeira"
    second = tmp_path / "segunda"

    for root in (first, second):
        branch = run_seed_branch(SEED, root, SPECIFICATIONS)
        run_sigma_branch(branch, 0.5)

    for name in SEED_FILES:
        assert (seed_directory(first, SEED) / name).read_bytes() == (
            seed_directory(second, SEED) / name
        ).read_bytes(), name

    for name in SIGMA_FILES:
        assert (run_directory(first, SEED, 0.5) / name).read_bytes() == (
            run_directory(second, SEED, 0.5) / name
        ).read_bytes(), name
