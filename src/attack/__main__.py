"""Linha de comando do M3: mescla a campanha ao trafego das semanas 5 a 8.

    python -m src.attack --seed 1 --sigma 0.5

Nao recebe `--fase`. O M3 roda num ramo so, o de sigma, porque o
aquecimento e anterior ao ataque (D-048). Fase obrigatoria existe nos modulos
que rodam nos dois ramos, e aqui ela nao teria segundo valor.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.attack.build import build_attack
from src.attack.parameters import AttackSpecification
from src.globals.layout import DEFAULT_ROOT, run_directory, seed_directory
from src.globals.tables import write_csv
from src.traffic.parameters import TrafficSpecification


class Arguments(argparse.Namespace):
    """Atributos que a linha de comando produz."""

    seed: int
    sigma: float
    out: Path


def parse_args() -> Arguments:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, required=True, help="semente do experimento")
    parser.add_argument(
        "--sigma", type=float, required=True, help="furtividade, de 0,0 a 1,0"
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_ROOT, help="raiz da pasta de dados"
    )

    return parser.parse_args(namespace=Arguments())


def check_sigma(args: Arguments) -> None:
    """Sigma fora de 0 a 1 nao interpola nada: extrapola."""
    is_out_of_range = not 0.0 <= args.sigma <= 1.0

    if is_out_of_range:
        raise SystemExit(f"erro: --sigma fora da faixa 0,0 a 1,0: {args.sigma}")


def read_inputs(tables: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """As duas tabelas estaticas e o trafego legitimo, todos do ramo da semente.

    Os tres moram em `seed-NN/`, mesmo que a saida va para uma pasta de sigma:
    populacao, repositorio e trafego legitimo nao dependem de sigma.
    """
    needed = {
        "operators.csv": tables / "operators.csv",
        "keys.csv": tables / "keys.csv",
        "requests.csv": tables / "requests.csv",
    }
    missing = [str(path) for path in needed.values() if not path.exists()]

    if missing:
        raise FileNotFoundError(
            "faltam arquivos de entrada:\n       " + "\n       ".join(missing)
        )

    return tuple(pd.read_csv(path) for path in needed.values())


def report(destination: Path, output) -> None:
    """Resumo da execucao, para conferencia imediata na linha de comando."""
    sessions = output.requests["session_id"].nunique()
    compromised = len(output.compromised)

    print(f"{destination}")
    print(f"  requests.csv   {len(output.requests)} requisicoes em {sessions} sessoes")
    print(f"  compromised_sessions.csv   {compromised} sessoes comprometidas")
    print(f"  run.csv        admin {output.run['compromised_admin'].iloc[0]}")
    print(f"                 {compromised / sessions:.2%} das sessoes do periodo")


def main() -> None:
    args = parse_args()
    check_sigma(args)

    tables = seed_directory(args.out, args.seed)
    destination = run_directory(args.out, args.seed, args.sigma)

    operators, keys, requests = read_inputs(tables)
    output = build_attack(
        args.seed, args.sigma, operators, keys, requests,
        TrafficSpecification(), AttackSpecification(),
    )

    write_csv(output.requests, destination / "requests.csv")
    write_csv(output.compromised, destination / "compromised_sessions.csv")
    write_csv(output.run, destination / "run.csv")
    report(destination, output)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as missing:
        raise SystemExit(f"erro: {missing}")
