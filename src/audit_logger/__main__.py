"""Linha de comando do M5: das tentativas e desfechos ao log de auditoria.

    python -m src.audit_logger --seed 1 --fase warmup
    python -m src.audit_logger --seed 1 --fase evaluated --sigma 0.5

A fase e **obrigatoria e sem padrao** (D-060), pela mesma razao do M4.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.audit_logger.build import build_log
from src.globals.layout import DEFAULT_ROOT, run_directory, seed_directory
from src.globals.phases import EVALUATED, PHASES, WARMUP
from src.globals.tables import write_csv


class Arguments(argparse.Namespace):
    """Atributos que a linha de comando produz."""

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
    """Onde a fase le e escreve. Quem conhece os caminhos e o `layout.py`."""
    is_warmup = args.fase == WARMUP

    if is_warmup:
        return seed_directory(args.out, args.seed)

    return run_directory(args.out, args.seed, args.sigma)


def check_sigma(args: Arguments) -> None:
    """A fase avaliada mora numa pasta de sigma, entao precisa dele."""
    is_missing = args.fase == EVALUATED and args.sigma is None

    if is_missing:
        raise SystemExit("erro: --fase evaluated exige --sigma")

    is_pointless = args.fase == WARMUP and args.sigma is not None

    if is_pointless:
        raise SystemExit(
            "erro: --fase warmup nao aceita --sigma\n"
            "       o aquecimento e anterior ao ataque, logo independente de sigma"
        )


def read_inputs(destination: Path, suggested: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """As tentativas e os desfechos, com erro claro quando faltam.

    Recebe o comando pronto em vez de monta-lo: assim esta funcao sabe **o que
    dizer** sem saber **como calcular**, e continua sendo sobre ler dois
    arquivos.
    """
    requests_path = destination / "requests.csv"
    outcomes_path = destination / "outcomes.csv"

    is_missing = not requests_path.exists() or not outcomes_path.exists()

    if is_missing:
        raise FileNotFoundError(
            f"faltam `requests.csv` ou `outcomes.csv` em {destination}"
            f"\n       rode antes:  {suggested}"
        )

    return pd.read_csv(requests_path), pd.read_csv(outcomes_path)


def kms_command(args: Arguments) -> str:
    """O comando que falta rodar, com os mesmos argumentos desta execucao."""
    elsewhere = "" if args.out == DEFAULT_ROOT else f" --out {args.out}"
    sigma = "" if args.sigma is None else f" --sigma {args.sigma}"

    return f"python -m src.kms --seed {args.seed} --fase {args.fase}{sigma}{elsewhere}"


def report(destination: Path, log: pd.DataFrame) -> None:
    """Resumo da execucao, para conferencia imediata na linha de comando."""
    first = log["timestamp"].iloc[0][:10]
    last = log["timestamp"].iloc[-1][:10]

    print(f"{destination}")
    print(f"  log.csv        {len(log)} eventos em {log['session_id'].nunique()} sessoes")
    print(f"                 {log['operator_id'].nunique()} operadores")
    print(f"                 de {first} a {last}")


def main() -> None:
    args = parse_args()
    check_sigma(args)

    destination = destination_of(args)

    requests, outcomes = read_inputs(destination, kms_command(args))
    log = build_log(requests, outcomes, args.fase)

    write_csv(log, destination / "log.csv")
    report(destination, log)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as missing:
        raise SystemExit(f"erro: {missing}")
