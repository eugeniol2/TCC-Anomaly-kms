"""Linha de comando do orquestrador: roda o pipeline na ordem certa.

    python -m src.pipeline --seed 1                # a varredura: aquecimento + 11 sigmas
    python -m src.pipeline --seed 1 --sigma 0.5    # so uma condicao
    python -m src.pipeline --warmup --seed 1       # so o ramo da semente
    python -m src.pipeline --grade                 # as 330, o runs.csv e a avaliacao
    python -m src.pipeline --search                # a preparatoria 902: a busca
    python -m src.pipeline --rehearsal             # a preparatoria 903: o ensaio

A ordem e: a busca, que escreve o `config.csv` que as execucoes leem (D-032,
D-115); o ensaio, que e a primeira vez que se ve acerto, numa semente reservada
(D-107); e so entao a grade.

Os modulos individuais continuam rodando sozinhos, e devem continuar: a
fronteira entre eles e o arquivo, e inspecionar a saida de um antes do proximo
e como se confere o gerador. O que este comando acrescenta e **a ordem**, que
antes so existia em prosa.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

from src.globals.experiment import HYPERPARAMETER_SEARCH_SEED, SEEDS, SIGMAS
from src.globals.layout import (
    CONFIGURATION,
    DEFAULT_ROOT,
    RUNS_INDEX,
    preparation_directory,
)
from src.models.build import configuration_scores, read_configuration
from src.pipeline.build import (
    Specifications,
    run_rehearsal,
    run_search_preparation,
    run_seed_branch,
    run_sigma_branch,
    run_sweep,
    write_evaluation,
    write_runs_index,
    write_search,
)


class Arguments(argparse.Namespace):
    """Atributos que a linha de comando produz."""

    seed: int | None
    sigma: float | None
    warmup: bool
    grade: bool
    search: bool
    rehearsal: bool
    out: Path


def parse_args() -> Arguments:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, help="uma semente; ausente com --grade")
    parser.add_argument("--sigma", type=float, help="uma condicao; ausente roda as 11")
    parser.add_argument(
        "--warmup", action="store_true", help="so o ramo da semente, sem sigma"
    )
    parser.add_argument(
        "--grade", action="store_true", help="as 330 execucoes e o indice agregado"
    )
    parser.add_argument(
        "--search", action="store_true", help="a busca de hiperparametros, na 902"
    )
    parser.add_argument(
        "--rehearsal", action="store_true", help="o ensaio do pipeline inteiro, na 903"
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_ROOT, help="raiz da pasta de dados"
    )

    return parser.parse_args(namespace=Arguments())


def check_request(args: Arguments) -> None:
    """Combinacoes que nao descrevem execucao nenhuma.

    A grade e a semente sao alternativas, e nao complementos: pedir as duas
    significaria coisas diferentes para quem escreve e para quem le.
    """
    is_preparation = args.search or args.rehearsal
    is_preparation_with_more = is_preparation and (
        (args.search and args.rehearsal) or args.grade or args.warmup
        or args.seed is not None or args.sigma is not None
    )

    if is_preparation_with_more:
        raise SystemExit("erro: --search e --rehearsal rodam sozinhos, cada um na sua semente")

    is_ambiguous = args.grade and args.seed is not None

    if is_ambiguous:
        raise SystemExit("erro: --grade roda as 30 sementes; nao combine com --seed")

    is_aimless = not args.grade and not is_preparation and args.seed is None

    if is_aimless:
        raise SystemExit(
            "erro: informe --seed N, --grade para as 330, --search ou --rehearsal"
        )

    is_contradictory = args.warmup and args.sigma is not None

    if is_contradictory:
        raise SystemExit(
            "erro: --warmup e o ramo da semente, que e anterior ao ataque\n"
            "       logo independente de sigma"
        )

    is_out_of_range = args.sigma is not None and not 0.0 <= args.sigma <= 1.0

    if is_out_of_range:
        raise SystemExit(f"erro: --sigma fora da faixa 0,0 a 1,0: {args.sigma}")


def run_whole_grid(args: Arguments, specifications: Specifications) -> None:
    """As 330 execucoes, com o indice agregado ao fim (D-085)."""
    rows = []

    for position, seed in enumerate(SEEDS, start=1):
        started = perf_counter()
        rows.extend(run_sweep(seed, SIGMAS, args.out, specifications))
        elapsed = perf_counter() - started

        print(f"  semente {seed:>2} de {len(SEEDS)}   "
              f"{len(SIGMAS)} condicoes   {elapsed:.1f}s")

    index = write_runs_index(rows, args.out)

    print(f"\n{args.out / RUNS_INDEX}   {len(index)} execucoes")
    print(f"  administradores comprometidos distintos: "
          f"{index['compromised_admin'].nunique()}")

    evaluation = write_evaluation(args.out)

    print(f"\navaliacao em {args.out}: metrics.csv ({len(evaluation.metrics)} linhas), "
          f"triviality.csv, comparison.csv, timing.csv")


def run_one_seed(args: Arguments, specifications: Specifications) -> None:
    """Uma semente: o aquecimento e, conforme os argumentos, uma ou onze condicoes."""
    started = perf_counter()
    branch = run_seed_branch(args.seed, args.out, specifications)

    print(f"  aquecimento   {len(branch.requests)} requisicoes   "
          f"{len(branch.profiles)} perfis   "
          f"{len(branch.thresholds)} limiares   "
          f"{perf_counter() - started:.1f}s")

    if args.warmup:
        return

    chosen = SIGMAS if args.sigma is None else (args.sigma,)

    for sigma in chosen:
        started = perf_counter()
        produced = run_sigma_branch(
            args.seed, sigma, args.out, branch, specifications
        )
        positives = int(produced.sessions["compromised"].sum())
        holdout_positives = int(produced.holdout["compromised"].sum())

        print(f"  sigma {sigma:.1f}     {len(produced.sessions)} sessoes   "
              f"{positives} positivas   "
              f"{positives / len(produced.sessions):.2%}   "
              f"holdout {len(produced.holdout)} com {holdout_positives}   "
              f"{perf_counter() - started:.1f}s")


def run_search(args: Arguments) -> None:
    """A preparatoria 902: gera o treino dela e avalia cada configuracao da grade.

    Mostra o F1 de validacao de cada uma, que e a nota da busca, calculada so no
    treino da 902. O holdout dela nao e consultado (D-045).
    """
    started = perf_counter()
    train = run_search_preparation(args.out, Specifications())

    print(f"  preparatoria 902   {len(train)} sessoes de treino   "
          f"{int(train['compromised'].sum())} positivas")

    scores = []

    for score in configuration_scores(HYPERPARAMETER_SEARCH_SEED, train):
        scores.append(score)
        print(f"  {score['model']:<14} {score['position'] + 1:>3}   "
              f"F1 {score['mean_f1']:.4f} +- {score['std_f1']:.4f}   "
              f"{perf_counter() - started:.0f}s")

    chosen = write_search(scores, args.out)

    print(f"\n{preparation_directory(args.out, HYPERPARAMETER_SEARCH_SEED) / CONFIGURATION}")
    print(chosen.to_string(index=False))


def run_rehearsal_and_show(args: Arguments, specifications: Specifications) -> None:
    """A 903 inteira, e o acerto dela na tela: a primeira olhada, numa semente reservada."""
    evaluation = run_rehearsal(args.out, specifications)

    print("  ensaio 903, sigma 0,5\n")
    print(evaluation.metrics.drop(columns=["seed", "sigma"]).to_string(index=False))
    print()
    print(evaluation.triviality.drop(columns=["seed", "sigma"]).to_string(index=False))
    print()
    print(evaluation.timing.to_string(index=False))


def models_configuration(args: Arguments) -> dict[str, dict] | None:
    """A configuracao da 902, quando a execucao vai treinar modelo."""
    if args.warmup:
        return None

    search = preparation_directory(args.out, HYPERPARAMETER_SEARCH_SEED)

    return read_configuration(search / CONFIGURATION)


def main() -> None:
    args = parse_args()
    check_request(args)

    if args.search:
        run_search(args)
        return

    specifications = Specifications(models=models_configuration(args))

    if args.rehearsal:
        run_rehearsal_and_show(args, specifications)
        return

    if args.grade:
        run_whole_grid(args, specifications)
        return

    run_one_seed(args, specifications)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as missing:
        raise SystemExit(f"erro: {missing}")
