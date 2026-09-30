"""A ordem de execução do pipeline, como código em vez de documentação.

Até aqui a ordem existia em seis lugares em prosa (`README.md`, `CLAUDE.md`,
três relatórios e um diagrama) e em nenhum deles executava. Seis cópias de um
fato divergem, e divergiram: o diagrama ficou quatro decisões atrás do código
sem que nada acusasse.

Este arquivo é a **única cópia executável**. Quem quiser saber em que ordem as
coisas acontecem lê as duas funções do fim, que são listas lineares de chamadas.

Duas propriedades vêm de decisão e não de conveniência:

**Os dois ramos existem porque o atacante só age nas semanas 5 a 8** (D-049).
O ramo da semente roda 30 vezes e produz tudo que é anterior ao ataque; o ramo
de sigma roda 330 e produz o período avaliado. Rodar o primeiro dentro do
segundo recomputaria onze vezes o mesmo aquecimento, e bastaria um sorteio
consumido em ordem diferente para os perfis divergirem entre condições da mesma
semente, quebrando o pareamento que a D-002 assume, sem erro e sem aviso.

**Tudo roda no mesmo processo.** Com os doze modulos, a grade completa sao 2521
invocacoes de modulo: sete no ramo da semente, 30 vezes cada; sete no ramo de
sigma, 330 vezes cada; e o M12 uma vez, sobre a grade inteira. Como subprocesso,
cada uma paga a partida do interpretador e o import do pandas, cerca de 0,57 s,
o que da uns 24 minutos so de inicializacao.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import pandas as pd

from src.attack.build import build_attack
from src.attack.parameters import AttackSpecification
from src.audit_logger.build import build_log
from src.baseline.build import build_baseline
from src.calibration.build import build_thresholds
from src.dataset.build import build_dataset
from src.evaluation.build import (
    Evaluation,
    build_evaluation,
    read_run,
    run_metrics,
    run_triviality,
)
from src.globals.experiment import (
    HYPERPARAMETER_SEARCH_SEED,
    PREPARATION_SIGMA,
    REHEARSAL_SEED,
    SEEDS,
    SIGMAS,
)
from src.globals.layout import (
    COMPARISON,
    CONFIGURATION,
    METRICS,
    PREPARATION,
    RUNS_INDEX,
    SEARCH_RESULTS,
    TIMING,
    TRIVIALITY,
    preparation_directory,
    run_directory,
    seed_directory,
)
from src.globals.phases import EVALUATED, WARMUP
from src.globals.tables import write_csv
from src.historical_profiles.build import build_profiles
from src.kms.build import build_outcomes
from src.models.build import build_models, chosen_configuration
from src.partition.build import build_partition
from src.population.build import build_population
from src.population.parameters import KeyRepositorySpecification
from src.traffic.build import build_traffic
from src.traffic.parameters import TrafficSpecification


class Specifications(NamedTuple):
    """Os parâmetros dos geradores, reunidos para viajar por parâmetro.

    Existem aqui pela mesma razão que existem nos módulos: o teste troca um
    valor sem editar arquivo nenhum, e a função recebe o que precisa em vez de
    ler módulo global.
    """

    repository: KeyRepositorySpecification = KeyRepositorySpecification()
    traffic: TrafficSpecification = TrafficSpecification()
    attack: AttackSpecification = AttackSpecification()
    models: dict[str, dict] | None = None
    """A configuracao que a busca da 902 escolheu. Sem ela o M11 nao roda."""


class SeedBranch(NamedTuple):
    """O que o ramo da semente produz, e o que o ramo de sigma consome."""

    operators: pd.DataFrame
    keys: pd.DataFrame
    requests: pd.DataFrame
    profiles: pd.DataFrame
    thresholds: pd.DataFrame


class EvaluatedSets(NamedTuple):
    """O ramo de sigma ate a particao: o que o baseline e os modelos recebem."""

    requests: pd.DataFrame
    compromised: pd.DataFrame
    run: pd.DataFrame
    sessions: pd.DataFrame
    train: pd.DataFrame
    holdout: pd.DataFrame


class SigmaBranch(NamedTuple):
    """O que uma execução (semente, sigma) produz."""

    requests: pd.DataFrame
    compromised: pd.DataFrame
    run: pd.DataFrame
    sessions: pd.DataFrame
    train: pd.DataFrame
    holdout: pd.DataFrame
    predictions_rules: pd.DataFrame
    predictions_ml: pd.DataFrame


def emit(frame: pd.DataFrame, directory: Path, name: str) -> pd.DataFrame:
    """Escreve o arquivo e devolve o quadro, para encadear as etapas.

    Devolver o que escreveu é o que permite a etapa seguinte receber o dado em
    memória em vez de reler o disco. O arquivo continua sendo escrito: ele é a
    fronteira entre módulos e o que torna cada etapa inspecionável.
    """
    write_csv(frame, directory / name)

    return frame


def run_seed_branch(seed: int, root: Path, specifications: Specifications) -> SeedBranch:
    """As semanas 1 a 4, e tudo que delas deriva. Roda 30 vezes, não 330.

    A ordem abaixo é a ordem. Cada linha depende do que as anteriores
    produziram, e nenhuma depende de sigma.
    """
    directory = seed_directory(root, seed)

    population = build_population(seed, specifications.repository)
    operators = emit(population.operators, directory, "operators.csv")
    keys = emit(population.keys, directory, "keys.csv")

    requests = emit(
        build_traffic(seed, operators, keys, specifications.traffic),
        directory, "requests.csv",
    )
    outcomes = emit(
        build_outcomes(requests, keys, operators, WARMUP),
        directory, "outcomes.csv",
    )
    log = emit(build_log(requests, outcomes, WARMUP), directory, "log.csv")

    profiles = emit(build_profiles(log), directory, "historical_profiles.csv")
    sessions = emit(
        build_dataset(log, profiles, WARMUP), directory, "sessions.csv"
    )
    thresholds = emit(build_thresholds(sessions), directory, "thresholds.csv")

    return SeedBranch(operators, keys, requests, profiles, thresholds)


def run_evaluated_sets(
    seed: int,
    sigma: float,
    root: Path,
    branch: SeedBranch,
    specifications: Specifications,
) -> EvaluatedSets:
    """As semanas 5 a 8 de uma condição, da campanha à partição.

    Recebe o ramo da semente pronto em vez de recomputá-lo: é o que garante
    que as onze condições compartilhem exatamente o mesmo aquecimento.
    """
    directory = run_directory(root, seed, sigma)

    campaign = build_attack(
        seed, sigma, branch.operators, branch.keys, branch.requests,
        specifications.traffic, specifications.attack,
    )
    requests = emit(campaign.requests, directory, "requests.csv")
    compromised = emit(
        campaign.compromised, directory, "compromised_sessions.csv"
    )
    run = emit(campaign.run, directory, "run.csv")

    outcomes = emit(
        build_outcomes(requests, branch.keys, branch.operators, EVALUATED),
        directory, "outcomes.csv",
    )
    log = emit(build_log(requests, outcomes, EVALUATED), directory, "log.csv")

    sessions = emit(
        build_dataset(log, branch.profiles, EVALUATED, compromised),
        directory, "sessions.csv",
    )

    partition = build_partition(seed, sessions)
    train = emit(partition.train, directory, "train.csv")
    holdout = emit(partition.holdout, directory, "holdout.csv")

    return EvaluatedSets(requests, compromised, run, sessions, train, holdout)


def run_sigma_branch(
    seed: int,
    sigma: float,
    root: Path,
    branch: SeedBranch,
    specifications: Specifications,
) -> SigmaBranch:
    """Uma condição inteira: os conjuntos, e as decisões do baseline e dos modelos.

    Roda 330 vezes. Exige a configuração dos modelos, que sai da busca da 902.
    """
    is_unconfigured = specifications.models is None

    if is_unconfigured:
        raise ValueError("falta a configuracao dos modelos; rode antes a busca da 902")

    directory = run_directory(root, seed, sigma)
    sets = run_evaluated_sets(seed, sigma, root, branch, specifications)

    baseline = build_baseline(sets.holdout, branch.thresholds)
    predictions_rules = emit(baseline.predictions, directory, "predictions_rules.csv")
    emit(baseline.timing, directory, "timing_rules.csv")

    models = build_models(seed, sets.train, sets.holdout, specifications.models)
    predictions_ml = emit(models.predictions, directory, "predictions_ml.csv")
    emit(models.timing, directory, "timing_ml.csv")

    return SigmaBranch(*sets, predictions_rules, predictions_ml)


def run_sweep(
    seed: int,
    sigmas: tuple[float, ...],
    root: Path,
    specifications: Specifications,
) -> list[pd.DataFrame]:
    """Uma varredura inteira: o aquecimento uma vez, e cada condição de sigma.

    Devolve as linhas de `run.csv` produzidas, que o índice agregado consome.
    """
    branch = run_seed_branch(seed, root, specifications)

    return [
        run_sigma_branch(seed, sigma, root, branch, specifications).run
        for sigma in sigmas
    ]


def write_runs_index(rows: list[pd.DataFrame], root: Path) -> pd.DataFrame:
    """Concatena os `run.csv` de cada execução no índice da D-085.

    O M3 não escreve este arquivo porque 330 invocações do mesmo módulo
    gravando no mesmo lugar dependeriam da ordem, e reexecutar uma condição
    isolada duplicaria a linha em vez de substituí-la. Quem o escreve é quem
    percorre a grade inteira, e por isso ele nasce completo ou não nasce.
    """
    index = pd.concat(rows, ignore_index=True).sort_values(
        ["seed", "sigma"], ignore_index=True
    )

    return emit(index, root, RUNS_INDEX)


def run_search_preparation(root: Path, specifications: Specifications) -> pd.DataFrame:
    """A preparatória 902 até a partição, em sigma 0,5: o treino da busca.

    Grava tudo em `preparation/seed-902/`, com o layout de uma réplica. Para na
    partição: o holdout da 902 não é consultado em momento nenhum (D-045).
    """
    preparation = root / PREPARATION
    seed = HYPERPARAMETER_SEARCH_SEED

    branch = run_seed_branch(seed, preparation, specifications)
    sets = run_evaluated_sets(seed, PREPARATION_SIGMA, preparation, branch, specifications)

    return sets.train


def write_search(scores: list[dict], root: Path) -> pd.DataFrame:
    """Grava a nota de cada configuração e a escolhida, e devolve a escolhida."""
    directory = preparation_directory(root, HYPERPARAMETER_SEARCH_SEED)
    results = emit(pd.DataFrame(scores), directory, SEARCH_RESULTS)

    return emit(chosen_configuration(results), directory, CONFIGURATION)


def run_rehearsal(root: Path, specifications: Specifications) -> Evaluation:
    """A preparatória 903: o pipeline inteiro numa semente reservada, em sigma 0,5 (D-107).

    Grava em `preparation/seed-903/`, e na pasta da execução as métricas e a árvore
    rasa dela. É a primeira vez que se vê acerto, e numa semente fora das 30.
    """
    preparation = root / PREPARATION
    seed = REHEARSAL_SEED

    branch = run_seed_branch(seed, preparation, specifications)
    run_sigma_branch(seed, PREPARATION_SIGMA, preparation, branch, specifications)

    run = read_run(preparation, seed, PREPARATION_SIGMA)
    directory = run_directory(preparation, seed, PREPARATION_SIGMA)

    return Evaluation(
        emit(run_metrics(run), directory, METRICS),
        emit(run_triviality(run), directory, TRIVIALITY),
        comparison=pd.DataFrame(),
        timing=run.timing,
    )


def write_evaluation(root: Path) -> Evaluation:
    """O M12 sobre a grade inteira, lida do disco: as quatro tabelas na raiz."""
    runs = [read_run(root, seed, sigma) for seed in SEEDS for sigma in SIGMAS]
    evaluation = build_evaluation(runs)

    return Evaluation(
        emit(evaluation.metrics, root, METRICS),
        emit(evaluation.triviality, root, TRIVIALITY),
        emit(evaluation.comparison, root, COMPARISON),
        emit(evaluation.timing, root, TIMING),
    )
