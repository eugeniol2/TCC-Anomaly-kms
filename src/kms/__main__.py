"""Linha de comando do M4: das requisicoes de uma fase aos desfechos.

    python -m src.kms --seed 1 --fase warmup
    python -m src.kms --seed 1 --fase evaluated --sigma 0.5

A fase e **obrigatoria e sem padrao** (D-060). Esquecer tem de falhar, nao
rodar o ramo errado em silencio gravando saida valida no lugar errado.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.globals.layout import DEFAULT_ROOT, run_directory, seed_directory
from src.globals.phases import EVALUATED, PHASES, WARMUP
from src.globals.tables import write_csv
from src.kms.build import build_outcomes
from src.kms.policy import OUTCOMES


class Arguments(argparse.Namespace):
    """Atributos que a linha de comando produz.

    O argparse monta o Namespace em tempo de execucao, entao sem estas
    anotacoes a IDE nao sabe que `seed` e inteiro nem que `out` e caminho.
    """

    seed: int
    fase: str
    sigma: float | None
    out: Path


def parse_args() -> Arguments:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, required=True, help="semente do experimento")
    parser.add_argument("--fase", choices=PHASES, required=True, help="qual fatia de tempo")
    parser.add_argument("--sigma", type=float, help="obrigatorio na fase evaluated")
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_ROOT, help="raiz da pasta de dados"
    )

    return parser.parse_args(namespace=Arguments())


def destination_of(args: Arguments) -> Path:
    """Onde a fase escreve, e de onde ela le o `requests.csv`.

    O aquecimento mora no ramo da semente, que roda 30 vezes; o periodo
    avaliado mora no ramo de sigma, que roda 330. Nenhum modulo monta caminho
    a mao — quem os conhece e o `layout.py`.
    """
    is_warmup = args.fase == WARMUP

    if is_warmup:
        return seed_directory(args.out, args.seed)

    return run_directory(args.out, args.seed, args.sigma)


def check_sigma(args: Arguments) -> None:
    """A fase avaliada mora numa pasta de sigma, entao precisa dele.

    A fase nunca e inferida de `--sigma` (D-049): e o contrario, a fase e
    declarada e o sigma e exigido quando ela o requer.
    """
    is_missing = args.fase == EVALUATED and args.sigma is None

    if is_missing:
        raise SystemExit("erro: --fase evaluated exige --sigma")

    is_pointless = args.fase == WARMUP and args.sigma is not None
    if is_pointless:
        raise SystemExit(
            "erro: --fase warmup nao aceita --sigma\n"
            "       o aquecimento e anterior ao ataque, logo independente de sigma"
        )


def read_inputs(
    destination: Path, tables: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """As requisicoes da fase e as duas tabelas estaticas da semente.

    As tabelas moram sempre no ramo da semente, mesmo quando as requisicoes
    vem de uma pasta de sigma: populacao e repositorio nao dependem de sigma.
    """
    needed = {
        "requests.csv": destination / "requests.csv",
        "keys.csv": tables / "keys.csv",
        "operators.csv": tables / "operators.csv",
    }
    missing = [str(path) for path in needed.values() if not path.exists()]

    if missing:
        raise FileNotFoundError(
            "faltam arquivos de entrada:\n       " + "\n       ".join(missing)
        )

    return tuple(pd.read_csv(path) for path in needed.values())


def report(destination: Path, outcomes: pd.DataFrame) -> None:
    """Resumo da execucao, para conferencia imediata na linha de comando."""
    counted = outcomes["outcome"].value_counts()

    print(f"{destination}")
    print(f"  outcomes.csv   {len(outcomes)} desfechos")

    for name in OUTCOMES:
        quantity = int(counted.get(name, 0))
        share = quantity / len(outcomes) if len(outcomes) else 0.0
        print(f"                 {name:<18} {quantity:>7}   {share:>7.3%}")


def main() -> None:
    args = parse_args()
    check_sigma(args)

    destination = destination_of(args)
    tables = seed_directory(args.out, args.seed)

    requests, keys, operators = read_inputs(destination, tables)
    outcomes = build_outcomes(requests, keys, operators, args.fase)

    write_csv(outcomes, destination / "outcomes.csv")
    report(destination, outcomes)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as missing:
        raise SystemExit(f"erro: {missing}")
