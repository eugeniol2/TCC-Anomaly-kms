"""Linha de comando do M6: do log de aquecimento aos perfis historicos.

    python -m src.historical_profiles --seed 1

Nao recebe `--fase` nem `--sigma`. O perfil sai das quatro semanas de
aquecimento, que moram no ramo da semente e sao anteriores ao ataque: nao ha
segundo ramo onde este modulo pudesse rodar (D-044, D-048, D-096).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.globals.layout import DEFAULT_ROOT, seed_directory
from src.globals.tables import MULTIVALUE_SEPARATOR, write_csv
from src.historical_profiles.build import build_profiles, window_width_hours


class Arguments(argparse.Namespace):
    """Atributos que a linha de comando produz."""

    seed: int
    out: Path


def parse_args() -> Arguments:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, required=True, help="semente do experimento")
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_ROOT, help="raiz da pasta de dados"
    )

    return parser.parse_args(namespace=Arguments())


def read_log(tables: Path) -> pd.DataFrame:
    """O log da fase de aquecimento, com erro claro quando falta."""
    path = tables / "log.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"falta `log.csv` em {tables}"
            f"\n       rode antes:  python -m src.audit_logger --seed N --fase warmup"
        )

    return pd.read_csv(path)


def report(destination: Path, profiles: pd.DataFrame) -> None:
    """Resumo da execucao, para conferencia imediata na linha de comando."""
    widths = window_width_hours(profiles)
    addresses = profiles["observed_ips"].str.split(MULTIVALUE_SEPARATOR).str.len()
    sessions = profiles["session_count"]

    print(f"{destination}")
    print(f"  historical_profiles.csv   {len(profiles)} operadores")
    print(f"     janela             mediana {widths.median():.1f} h   "
          f"[{widths.min():.1f} a {widths.max():.1f}]")
    print(f"     origens vistas     mediana {addresses.median():.0f}     "
          f"[{addresses.min()} a {addresses.max()}]")
    print(f"     sessoes no perfil  mediana {sessions.median():.0f}    "
          f"[{sessions.min()} a {sessions.max()}]")


def main() -> None:
    args = parse_args()

    destination = seed_directory(args.out, args.seed)
    profiles = build_profiles(read_log(destination))

    write_csv(profiles, destination / "historical_profiles.csv")
    report(destination, profiles)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as missing:
        raise SystemExit(f"erro: {missing}")
