"""Onde cada arquivo do pipeline mora dentro de `data/`.

A hierarquia espelha a dependencia. O que depende so da semente fica no nivel da
semente; o que depende tambem de sigma fica um nivel abaixo.

Como o atacante age apenas nas semanas 4 e 5 (D-048, D-053), tudo que deriva das
semanas 1 a 3 e independente de sigma. O pipeline tem portanto **dois ramos**, e
M4, M5 e M7 aparecem nos dois: rodam 30 vezes sobre o aquecimento e 330 vezes
sobre o periodo avaliado (D-049).

    data/
      runs.csv                        indice das 330 execucoes (D-012)
      metrics.csv                     agregado final (M12)
      preparation/
        seed-902/                     busca de hiperparametros (D-047)
        seed-903/                     limiar X de exclusao por trivialidade (D-052)
      seed-01/                        ---- ramo da semente, 30 execucoes ----
        operators.csv                 M1
        keys.csv                      M1
        requests.csv                  M2, cinco semanas, so trafego legitimo
        outcomes.csv                  M4, semanas 1 a 3
        log.csv                       M5, semanas 1 a 3
        historical_profiles.csv       M6, das semanas 1 e 2
        sessions.csv                  M7, da semana 3
        thresholds.csv                M8, da semana 3
        sigma-0.0/                    ---- ramo de sigma, 330 execucoes ----
          requests.csv                M3, semanas 4 e 5, legitimo + ataque
          compromised_sessions.csv    M3
          outcomes.csv                M4, semanas 4 e 5
          log.csv                     M5, semanas 4 e 5
          sessions.csv                M7, semanas 4 e 5
          train.csv, holdout.csv      M9
          predictions_rules.csv       M10
          predictions_ml.csv          M11
        sigma-0.1/ ... sigma-1.0/
      seed-02/ ... seed-30/

Cada arquivo tem um unico modulo que o escreve. Os nomes se repetem entre os dois
niveis porque o modulo e o mesmo; o que distingue e o caminho, e o caminho diz de
que fatia de tempo aquele arquivo trata.

O `log.csv` no nivel da semente e a garantia fisica da D-048: sendo unico por
semente, nao existe lugar onde o atacante pudesse estar nele.
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_ROOT = Path("data")

RUNS_INDEX = "runs.csv"
"""Indice das execucoes: sigma, seed, compromised_admin.

Derivavel da semente, mas materializado em disco, porque derivavel nao e o mesmo
que inspecionavel. E por este arquivo que se confere, sem reexecutar o gerador,
que a mesma semente comprometeu o mesmo administrador nas 11 condicoes de sigma
(D-012).
"""

METRICS = "metrics.csv"
"""Saida do M12, agregando as 330 execucoes."""


def seed_directory(root: Path, seed: int) -> Path:
    """Ramo da semente: o aquecimento e tudo que dele deriva.

    Semanas 1 a 3, independentes de sigma porque o atacante nao age nelas. Aqui
    moram a populacao, o trafego legitimo das cinco semanas, o log do
    aquecimento, os perfis historicos e os limiares.
    """
    return root / f"seed-{seed:02d}"


def run_directory(root: Path, seed: int, sigma: float) -> Path:
    """Ramo de sigma: o periodo avaliado de uma execucao.

    Semanas 4 e 5, onde a campanha transcorre. Um decimal em sigma basta para os
    11 valores da grade, e mantem a ordem alfabetica igual a ordem numerica:
    sigma-0.0 ate sigma-1.0.
    """
    return seed_directory(root, seed) / f"sigma-{sigma:.1f}"


def preparation_directory(root: Path, seed: int) -> Path:
    """Onde fica uma execucao preparatoria, fora das 330 replicas.

    Sao duas, com sementes reservadas distintas: a **902** escolhe a configuracao
    de hiperparametros (D-047) e a **903** fixa o limiar X de exclusao por
    trivialidade (D-052). Ficam separadas porque servem a propositos diferentes e
    porque reaproveitar o holdout da 902 conflitaria com a D-045.

    Os limiares do baseline nao passam por aqui: saem da semana 3 de cada
    execucao, no ramo da semente (D-043).
    """
    return root / "preparation" / f"seed-{seed}"
