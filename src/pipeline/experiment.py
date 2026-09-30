"""O experimento de ponta a ponta: busca, ensaio, grade, avaliacao e figuras (D-121).

A ordem e a que o protocolo exige, e agora o codigo a impoe: a busca de
hiperparametros na 902 escreve a configuracao que as execucoes leem (D-032); o
ensaio na 903 e a primeira olhada em acerto, numa semente reservada, **antes** da
grade (D-107); a grade roda as execucoes pedidas; a avaliacao e as figuras leem o
que a grade gravou.

Cada passo pertence a uma entidade, e so roda se o `--ate` a alcanca.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Callable, NamedTuple

import pandas as pd

from src.evaluation.build import (
    Evaluation,
    build_evaluation,
    read_run,
    run_metrics,
    run_triviality,
)
from src.evaluation.figures.build import draw_f1_by_sigma, draw_roc, save_figure
from src.models.build import (
    chosen_configuration,
    configuration_of,
    configuration_scores,
    read_configuration,
)
from src.pipeline.build import (
    Specifications,
    emit,
    run_seed_branch,
    run_sigma_branch,
    run_sweep,
    write_runs_index,
)
from src.pipeline.stages import Step, reaches
from src.shared import layout
from src.shared.experiment import (
    HYPERPARAMETER_SEARCH_SEED,
    PREPARATION_SIGMA,
    REHEARSAL_SEED,
)

Report = Callable[[str], None]


class Options(NamedTuple):
    """O que rodar e onde gravar. Os numeros do experimento nao passam por aqui."""

    seeds: tuple[int, ...]
    sigmas: tuple[float, ...]
    root: Path
    until: str
    reuse_search: bool


@dataclass
class Experiment:
    """O estado do experimento enquanto os passos rodam."""

    options: Options
    report: Report
    specifications: Specifications = field(default_factory=Specifications)


# A busca de hiperparametros, na 902.


def search_training_set(root: Path, specifications: Specifications) -> pd.DataFrame:
    """A 902 ate a particao, em sigma 0,5: o treino sobre o qual a busca roda.

    O holdout da 902 nao e consultado em momento nenhum (D-045).
    """
    preparation = root / layout.PREPARATION
    branch = run_seed_branch(HYPERPARAMETER_SEARCH_SEED, preparation, specifications)

    return run_sigma_branch(branch, PREPARATION_SIGMA, until="dataset_generator").train


def run_search(root: Path, specifications: Specifications, report: Report) -> dict[str, dict]:
    """Avalia cada configuracao da grade e grava a nota de todas e a escolhida."""
    started = perf_counter()
    train = search_training_set(root, specifications)
    report(f"busca na 902: {len(train)} sessoes de treino, "
           f"{int(train['compromised'].sum())} positivas")

    scores = []

    for score in configuration_scores(HYPERPARAMETER_SEARCH_SEED, train):
        scores.append(score)
        report(f"    {score['model']:<14} {score['position'] + 1:>3}   "
               f"F1 {score['mean_f1']:.4f}   {perf_counter() - started:.0f}s")

    directory = layout.preparation_directory(root, HYPERPARAMETER_SEARCH_SEED)
    results = emit(pd.DataFrame(scores), directory, layout.SEARCH_RESULTS)
    chosen = emit(chosen_configuration(results), directory, layout.CONFIGURATION)

    return configuration_of(chosen)


def configure_models(experiment: Experiment) -> None:
    """Roda a busca, ou reaproveita o `config.csv` de uma busca anterior."""
    options = experiment.options
    directory = layout.preparation_directory(options.root, HYPERPARAMETER_SEARCH_SEED)

    if options.reuse_search:
        configuration = read_configuration(directory / layout.CONFIGURATION)
        experiment.report(f"busca reaproveitada: {directory / layout.CONFIGURATION}")
    else:
        configuration = run_search(options.root, experiment.specifications, experiment.report)

    experiment.specifications = experiment.specifications._replace(models=configuration)


# O ensaio, na 903.


def run_rehearsal(root: Path, specifications: Specifications) -> Evaluation:
    """O pipeline inteiro na 903, em sigma 0,5, e as metricas e a arvore rasa dela (D-107)."""
    preparation = root / layout.PREPARATION
    branch = run_seed_branch(REHEARSAL_SEED, preparation, specifications)
    run_sigma_branch(branch, PREPARATION_SIGMA)

    run = read_run(preparation, REHEARSAL_SEED, PREPARATION_SIGMA)
    directory = layout.run_directory(preparation, REHEARSAL_SEED, PREPARATION_SIGMA)

    return Evaluation(
        emit(run_metrics(run), directory, layout.METRICS),
        emit(run_triviality(run), directory, layout.TRIVIALITY),
        comparison=pd.DataFrame(),
        timing=run.timing,
    )


def rehearse(experiment: Experiment) -> None:
    evaluation = run_rehearsal(experiment.options.root, experiment.specifications)
    summary = evaluation.metrics[["mechanism", "scope", "f1", "recall", "specificity"]]

    experiment.report("ensaio na 903, sigma 0,5:")
    experiment.report(summary.to_string(index=False))


# A grade, a avaliacao e as figuras.


def run_grid(experiment: Experiment) -> None:
    """As execucoes pedidas, semente a semente, e o indice agregado."""
    options = experiment.options
    rows = []
    experiment.report(f"grade: ate {options.until}")

    for seed in options.seeds:
        started = perf_counter()
        rows.extend(run_sweep(seed, options.sigmas, options.root,
                              experiment.specifications, options.until))
        experiment.report(f"  semente {seed:>2}: {len(options.sigmas)} condicoes, "
                          f"{perf_counter() - started:.1f}s")

    index = write_runs_index(rows, options.root)
    experiment.report(f"grade: {len(index)} execucoes, indice em {options.root / layout.RUNS_INDEX}")


def write_evaluation(root: Path, seeds: tuple[int, ...], sigmas: tuple[float, ...]) -> Evaluation:
    """O M12 sobre as execucoes pedidas, lidas do disco: as quatro tabelas na raiz."""
    runs = [read_run(root, seed, sigma) for seed in seeds for sigma in sigmas]
    evaluation = build_evaluation(runs)

    return Evaluation(
        emit(evaluation.metrics, root, layout.METRICS),
        emit(evaluation.triviality, root, layout.TRIVIALITY),
        emit(evaluation.comparison, root, layout.COMPARISON),
        emit(evaluation.timing, root, layout.TIMING),
    )


def evaluate(experiment: Experiment) -> None:
    options = experiment.options
    write_evaluation(options.root, options.seeds, options.sigmas)
    experiment.report("avaliacao:")

    for name in (layout.METRICS, layout.TRIVIALITY, layout.COMPARISON, layout.TIMING):
        experiment.report(f"  {options.root / name}")


def draw_figures(experiment: Experiment) -> None:
    options = experiment.options
    directory = options.root / layout.FIGURES
    metrics = pd.read_csv(options.root / layout.METRICS)
    comparison = pd.read_csv(options.root / layout.COMPARISON)

    figures = {
        "f1_sigma": draw_f1_by_sigma(metrics, comparison),
        "roc": draw_roc(options.root, options.seeds, options.sigmas),
    }
    experiment.report("figuras:")

    for name, figure in figures.items():
        if figure is not None:
            for path in save_figure(figure, directory, name):
                experiment.report(f"  {path}")


EXPERIMENT_STEPS = (
    Step("models", configure_models),
    Step("evaluation", rehearse),
    Step("scenario_engine", run_grid),
    Step("evaluation", evaluate),
    Step("figures", draw_figures),
)
"""A ordem do protocolo. Diferente dos ramos, aqui a entidade nao cresce passo a
passo: cada passo roda se o `--ate` o alcanca, e os outros sao pulados."""


def run_experiment(options: Options, report: Report) -> None:
    """O experimento inteiro, na ordem do protocolo, ate a entidade pedida."""
    experiment = Experiment(options, report)

    for step in EXPERIMENT_STEPS:
        if reaches(options.until, step.stage):
            step.run(experiment)
