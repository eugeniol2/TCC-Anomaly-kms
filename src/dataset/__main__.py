"""Linha de comando do M7: do log de auditoria ao conjunto de sessoes.

    python -m src.dataset --seed 1 --fase warmup
    python -m src.dataset --seed 1 --fase evaluated --sigma 0.5

A fase e **obrigatoria e sem padrao** (D-060), como no M4 e no M5. Aqui ela
decide mais do que o caminho: e ela que diz se o `sessions.csv` sai **com ou
sem rotulo** (D-063).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.dataset.build import ATTRIBUTES, LABEL, build_dataset
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


def destination_of(args: Arguments) -> Path:
    """Onde a fase le o log e escreve o conjunto."""
    is_warmup = args.fase == WARMUP

    if is_warmup:
        return seed_directory(args.out, args.seed)

    return run_directory(args.out, args.seed, args.sigma)


def read_inputs(
    destination: Path, tables: Path, phase: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """O log da fase, o perfil historico e, so na fase avaliada, o rotulo.

    O perfil mora sempre no ramo da semente, mesmo quando o log vem de uma
    pasta de sigma: ele sai das semanas 1 e 2 e nunca e recalculado (D-044).
    """
    needed = {
        "log.csv": destination / "log.csv",
        "historical_profiles.csv": tables / "historical_profiles.csv",
    }
    is_evaluated = phase == EVALUATED

    if is_evaluated:
        needed["compromised_sessions.csv"] = destination / "compromised_sessions.csv"

    missing = [str(path) for path in needed.values() if not path.exists()]

    if missing:
        raise FileNotFoundError(
            "faltam arquivos de entrada:\n       " + "\n       ".join(missing)
        )

    frames = {name: pd.read_csv(path) for name, path in needed.items()}

    return (
        frames["log.csv"],
        frames["historical_profiles.csv"],
        frames.get("compromised_sessions.csv"),
    )


def report(destination: Path, sessions: pd.DataFrame) -> None:
    """Resumo da execucao, com a proporcao de anomalias contada e nao afirmada."""
    print(f"{destination}")
    print(f"  sessions.csv   {len(sessions)} sessoes x {len(ATTRIBUTES)} atributos")

    has_label = LABEL in sessions.columns

    if not has_label:
        print("                 sem rotulo, como a fase de aquecimento exige")
        return

    positives = int(sessions[LABEL].sum())
    print(f"                 {positives} positivas   "
          f"{positives / len(sessions):.2%} do conjunto")


def main() -> None:
    args = parse_args()
    check_sigma(args)

    destination = destination_of(args)
    tables = seed_directory(args.out, args.seed)

    log, profiles, compromised = read_inputs(destination, tables, args.fase)
    sessions = build_dataset(log, profiles, args.fase, compromised)

    write_csv(sessions, destination / "sessions.csv")
    report(destination, sessions)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as missing:
        raise SystemExit(f"erro: {missing}")
