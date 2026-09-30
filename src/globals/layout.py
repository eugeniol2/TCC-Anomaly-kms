"""Onde cada arquivo do pipeline mora dentro de `data/`.

A hierarquia espelha a dependencia. O que depende so da semente fica no nivel da
semente; o que depende tambem de sigma fica um nivel abaixo.

Como o atacante age apenas nas semanas 5 a 8 (D-048, D-096), tudo que deriva do
aquecimento e independente de sigma. O pipeline tem portanto **dois ramos**, e
M4, M5 e M7 aparecem nos dois: rodam 30 vezes sobre o aquecimento e 330 vezes
sobre o periodo avaliado (D-049).

    data/
      runs.csv                        indice das 330 execucoes (D-012)
      metrics.csv                     M12, cada mecanismo em cada execucao
      triviality.csv                  M12, a arvore rasa e as duplicatas
      comparison.csv                  M12, Wilcoxon e Holm por sigma e modelo
      timing.csv                      M12, o tempo de cada mecanismo
      preparation/
        seed-902/                     busca de hiperparametros (D-047)
          config.csv                  M11, a configuracao escolhida (D-115)
          search_results.csv          M11, a nota de cada configuracao da grade
        seed-903/                     ensaio do pipeline antes das 330 (D-107)
      seed-01/                        ---- ramo da semente, 30 execucoes ----
        operators.csv                 M1
        keys.csv                      M1
        requests.csv                  M2, oito semanas, so trafego legitimo
        outcomes.csv                  M4, semanas 1 a 4
        log.csv                       M5, semanas 1 a 4
        historical_profiles.csv       M6, do aquecimento inteiro
        sessions.csv                  M7, semanas 1 a 4
        thresholds.csv                M8, do aquecimento inteiro
        sigma-0.0/                    ---- ramo de sigma, 330 execucoes ----
          requests.csv                M3, semanas 5 a 8, legitimo + ataque
          compromised_sessions.csv    M3
          outcomes.csv                M4, semanas 5 a 8
          log.csv                     M5, semanas 5 a 8
          sessions.csv                M7, semanas 5 a 8
          train.csv, holdout.csv      M9
          predictions_rules.csv       M10
          timing_rules.csv            M10, o tempo, unico arquivo nao deterministico
          predictions_ml.csv          M11
          timing_ml.csv               M11, o tempo dos dois modelos
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
"""Saida do M12: as metricas de cada mecanismo em cada execucao."""

TRIVIALITY = "triviality.csv"
"""Saida do M12: a arvore rasa e a procura de duplicata, por execucao (D-028, D-029)."""

COMPARISON = "comparison.csv"
"""Saida do M12: Wilcoxon e Holm, por sigma e modelo (D-111, D-116)."""

TIMING = "timing.csv"
"""Saida do M12: o tempo de decisao de cada mecanismo, resumido (D-106)."""

PREPARATION = "preparation"
"""A pasta das preparatorias, dentro da raiz de dados."""

CONFIGURATION = "config.csv"
"""A configuracao dos modelos que a busca da 902 escolheu (D-115)."""

SEARCH_RESULTS = "search_results.csv"
"""A nota de cada configuracao da grade, na 902 (D-115)."""


def seed_directory(root: Path, seed: int) -> Path:
    """Ramo da semente: o aquecimento e tudo que dele deriva.

    Semanas 1 a 4, independentes de sigma porque o atacante nao age nelas. Aqui
    moram a populacao, o trafego legitimo das oito semanas, o log do
    aquecimento, os perfis historicos e os limiares.
    """
    return root / f"seed-{seed:02d}"


def run_directory(root: Path, seed: int, sigma: float) -> Path:
    """Ramo de sigma: o periodo avaliado de uma execucao.

    Semanas 5 a 8, onde a campanha transcorre (D-096). Um decimal em sigma
    basta para os 11 valores da grade, e mantem a ordem alfabetica igual a
    ordem numerica: sigma-0.0 ate sigma-1.0.
    """
    return seed_directory(root, seed) / f"sigma-{sigma:.1f}"


def preparation_directory(root: Path, seed: int) -> Path:
    """Onde fica uma execucao preparatoria, fora das 330 replicas.

    Sao duas, com sementes reservadas distintas: a **902** escolhe a configuracao
    de hiperparametros (D-047) e a **903** e o ensaio do pipeline inteiro antes
    das 330 (D-107). Ficam separadas porque servem a propositos diferentes e
    porque reaproveitar o holdout da 902 conflitaria com a D-045.

    Os limiares do baseline nao passam por aqui: saem do aquecimento de cada
    execucao, no ramo da semente (D-043).

    Dentro dela a preparatoria tem o mesmo layout de uma replica: e o orquestrador
    rodando com `root / PREPARATION` como raiz.
    """
    return seed_directory(root / PREPARATION, seed)
