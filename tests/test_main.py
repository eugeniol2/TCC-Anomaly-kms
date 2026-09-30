"""Testes do comando unico: os parametros e o corte pelo `--ate` (D-121).

O comando roda o experimento inteiro por padrao, e isso leva uns 25 minutos.
Aqui ele roda recortado: uma semente, um sigma, e parado cedo. O que se confere
e que cada entidade grava o que deve, e nenhuma depois do corte grava nada.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.main import parse_options, seeds_from, sigmas_from
from src.pipeline.experiment import Options, run_experiment
from src.pipeline.stages import LAST_STAGE, STAGES, reaches
from src.shared import layout
from src.shared.experiment import SEEDS, SIGMAS

SEED = 3
SIGMA = 0.5


# Os parametros.


def test_without_parameters_it_runs_everything() -> None:
    options = parse_options([])

    assert options.seeds == SEEDS
    assert options.sigmas == SIGMAS
    assert options.root == layout.DEFAULT_ROOT
    assert options.until == LAST_STAGE
    assert options.reuse_search is False


def test_seeds_as_range_or_list() -> None:
    assert seeds_from("1-3") == (1, 2, 3)
    assert seeds_from("1,5,9") == (1, 5, 9)


def test_reserved_seeds_are_refused() -> None:
    """A 902 e a 903 sao das preparatorias, e nunca viram replica (D-047)."""
    with pytest.raises(SystemExit):
        parse_options(["--sementes", "900-905"])


def test_sigmas_outside_the_range_are_refused() -> None:
    assert sigmas_from("0.0,0.5,1.0") == (0.0, 0.5, 1.0)

    with pytest.raises(SystemExit):
        parse_options(["--sigmas", "1.5"])


def test_an_unknown_stage_is_refused() -> None:
    with pytest.raises(SystemExit):
        parse_options(["--ate", "m4"])


# As etapas.


def test_the_stages_follow_the_architecture() -> None:
    assert STAGES[0] == "scenario_engine"
    assert STAGES[-1] == "figures"
    assert reaches("kms", "scenario_engine")
    assert reaches("kms", "kms")
    assert not reaches("kms", "audit_logger")


def run_until(root: Path, until: str) -> None:
    """Roda o comando recortado. O que ele escreve no terminal o pytest guarda e descarta."""
    run_experiment(Options((SEED,), (SIGMA,), root, until, reuse_search=False))


def test_until_kms_stops_after_the_kms(tmp_path: Path) -> None:
    """Grava a populacao, o trafego e os desfechos, e nada do Audit Logger em diante."""
    run_until(tmp_path, "kms")

    seed_directory = layout.seed_directory(tmp_path, SEED)
    run_directory = layout.run_directory(tmp_path, SEED, SIGMA)

    for name in (layout.OPERATORS, layout.KEYS, layout.REQUESTS, layout.OUTCOMES):
        assert (seed_directory / name).exists(), name

    for name in (layout.REQUESTS, layout.COMPROMISED_SESSIONS, layout.OUTCOMES):
        assert (run_directory / name).exists(), name

    assert not (seed_directory / layout.LOG).exists()
    assert not (run_directory / layout.LOG).exists()
    assert not (tmp_path / layout.PREPARATION).exists()


def test_until_policy_engine_decides_without_searching(tmp_path: Path) -> None:
    """Antes dos modelos nao ha busca: o baseline decide, e nenhum modelo treina."""
    run_until(tmp_path, "policy_engine")

    run_directory = layout.run_directory(tmp_path, SEED, SIGMA)

    assert (run_directory / layout.PREDICTIONS_RULES).exists()
    assert not (run_directory / layout.PREDICTIONS_ML).exists()
    assert not (tmp_path / layout.PREPARATION).exists()
    assert (tmp_path / layout.RUNS_INDEX).exists()
