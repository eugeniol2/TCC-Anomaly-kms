"""O comando unico do experimento: roda tudo, de ponta a ponta (D-121).

    python -m src.main                                   # tudo, na ordem do protocolo
    python -m src.main --sem-busca                       # reaproveita a busca anterior
    python -m src.main --sementes 1 --sigmas 0.5 --ate kms
    python -m src.main --saida C:\\tcc-data

Sem parametro, roda a busca de hiperparametros na 902, o ensaio na 903, as 330
execucoes, a avaliacao e as figuras. Os parametros so dizem **o que** rodar e
**onde** gravar. Os numeros do experimento moram no `parameters.py` de cada
entidade, cada um com a decisao que o fixou: se mudassem pela linha de comando,
um resultado poderia sair de valores que nenhuma decisao registra.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.pipeline.experiment import Options, run_experiment
from src.pipeline.stages import LAST_STAGE, STAGES
from src.shared.experiment import RESERVED_SEEDS, SEEDS, SIGMAS
from src.shared.layout import DEFAULT_ROOT

ALL_SEEDS = f"{SEEDS[0]}-{SEEDS[-1]}"
ALL_SIGMAS = ",".join(str(sigma) for sigma in SIGMAS)


def seeds_from(text: str) -> tuple[int, ...]:
    """Sementes escritas como faixa ("1-30") ou lista ("1,2,5")."""
    is_range = "-" in text

    if is_range:
        first, last = (int(bound) for bound in text.split("-"))
        seeds = tuple(range(first, last + 1))
    else:
        seeds = tuple(int(seed) for seed in text.split(","))

    reserved = sorted(set(seeds) & set(RESERVED_SEEDS))

    if reserved:
        raise argparse.ArgumentTypeError(
            f"sementes reservadas as preparatorias, fora das replicas: {reserved}"
        )

    return seeds


def sigmas_from(text: str) -> tuple[float, ...]:
    """Sigmas escritos como lista ("0.5" ou "0.0,0.5,1.0"), cada um entre 0 e 1."""
    sigmas = tuple(float(sigma) for sigma in text.split(","))
    outside = [sigma for sigma in sigmas if not 0.0 <= sigma <= 1.0]

    if outside:
        raise argparse.ArgumentTypeError(f"sigma fora da faixa 0,0 a 1,0: {outside}")

    return sigmas


def parse_options(arguments: list[str] | None = None) -> Options:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sementes", type=seeds_from, default=seeds_from(ALL_SEEDS),
                        help=f"faixa ou lista de sementes (padrao {ALL_SEEDS})")
    parser.add_argument("--sigmas", type=sigmas_from, default=sigmas_from(ALL_SIGMAS),
                        help="lista de sigmas (padrao: os onze, de 0.0 a 1.0)")
    parser.add_argument("--saida", type=Path, default=DEFAULT_ROOT,
                        help="raiz onde tudo e gravado (padrao data)")
    parser.add_argument("--ate", choices=STAGES, default=LAST_STAGE,
                        help="a entidade em que o pipeline para (padrao: ate as figuras)")
    parser.add_argument("--sem-busca", action="store_true",
                        help="reaproveita o config.csv de uma busca anterior na mesma saida")

    args = parser.parse_args(arguments)

    return Options(args.sementes, args.sigmas, args.saida, args.ate, args.sem_busca)


def main() -> None:
    try:
        run_experiment(parse_options(), report=print)
    except FileNotFoundError as missing:
        raise SystemExit(f"erro: {missing}")


if __name__ == "__main__":
    main()
