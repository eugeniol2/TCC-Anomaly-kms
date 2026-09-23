"""Linha de comando do orquestrador: roda o pipeline na ordem certa.

    python -m src.pipeline --seed 1                # a varredura: aquecimento + 11 sigmas
    python -m src.pipeline --seed 1 --sigma 0.5    # so uma condicao
    python -m src.pipeline --warmup --seed 1       # so o ramo da semente
    python -m src.pipeline --grade                 # as 330, e o runs.csv

Os modulos individuais continuam rodando sozinhos, e devem continuar: a
fronteira entre eles e o arquivo, e inspecionar a saida de um antes do proximo
e como se confere o gerador. O que este comando acrescenta e **a ordem**, que
antes so existia em prosa.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

from src.globals.experiment import SEEDS, SIGMAS
from src.globals.layout import DEFAULT_ROOT, RUNS_INDEX
from src.pipeline.build import (
    Specifications,
    run_seed_branch,
    run_sigma_branch,
    run_sweep,
    write_runs_index,
)


class Arguments(argparse.Namespace):
    """Atributos que a linha de comando produz."""

    seed: int | None
    sigma: float | None
    warmup: bool
    grade: bool
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
        "--out", type=Path, default=DEFAULT_ROOT, help="raiz da pasta de dados"
    )

    return parser.parse_args(namespace=Arguments())


def check_request(args: Arguments) -> None:
    """Combinacoes que nao descrevem execucao nenhuma.

    A grade e a semente sao alternativas, e nao complementos: pedir as duas
    significaria coisas diferentes para quem escreve e para quem le.
    """
    is_ambiguous = args.grade and args.seed is not None

    if is_ambiguous:
        raise SystemExit("erro: --grade roda as 30 sementes; nao combine com --seed")

    is_aimless = not args.grade and args.seed is None

    if is_aimless:
        raise SystemExit("erro: informe --seed N, ou --grade para as 330")

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

        print(f"  sigma {sigma:.1f}     {len(produced.sessions)} sessoes   "
              f"{positives} positivas   "
              f"{positives / len(produced.sessions):.2%}   "
              f"{perf_counter() - started:.1f}s")


def main() -> None:
    args = parse_args()
    check_request(args)

    specifications = Specifications()

    if args.grade:
        run_whole_grid(args, specifications)
        return

    run_one_seed(args, specifications)


if __name__ == "__main__":
    main()
