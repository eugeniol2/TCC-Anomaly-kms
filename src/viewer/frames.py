"""Os quadros do pipeline: o que entra, a entidade que processa, o que sai.

Um quadro por passo, com os arquivos que o comando unico gravou. Sao treze,
em tres fases: a preparacao dos dados, o aquecimento com a calibracao, e o
periodo avaliado, do ataque a avaliacao. O KMS e o Audit Logger sao quadros
separados no aquecimento, para nao confundir quem decide o desfecho com quem o
registra; e as regras e os modelos tambem, porque sao os dois lados da comparacao.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.dataset_generator.dataset.build import ATTRIBUTES, LABEL
from src.dataset_generator.partition.parameters import HOLDOUT_SHARE
from src.kms.repository.parameters import KeyRepositorySpecification
from src.policy_engine.baseline.build import MECHANISM as RULES
from src.policy_engine.calibration.parameters import PERCENTILE, THRESHOLD_ATTRIBUTES
from src.scenario_engine.attack.parameters import AttackSpecification
from src.scenario_engine.traffic.parameters import TrafficSpecification
from src.shared.phases import EVALUATED, WARMUP, belongs_to
from src.viewer.data import GridFiles, RunFiles, SeedFiles
from src.viewer.decisions import (
    Variavel,
    variaveis_do_audit_logger,
    variaveis_do_kms,
    variaveis_do_m1,
    variaveis_do_m10,
    variaveis_do_m11,
    variaveis_do_m12,
    variaveis_do_m2,
    variaveis_do_m3,
    variaveis_do_m4_m5,
    variaveis_do_m6,
    variaveis_do_m7,
    variaveis_do_m8,
    variaveis_do_m9,
)
from src.viewer.formatting import com_virgula, porcento


@dataclass(frozen=True)
class Fase:
    """Uma das tres fases, com os quadros que ela mostra."""

    nome: str
    quadros: tuple[int, ...]


FASES = (
    Fase("Fase 1 · Preparação dos dados", (1, 2)),
    Fase("Fase 2 · Aquecimento e calibração", (3, 4, 5, 6, 7)),
    Fase("Fase 3 · Ataque e comparação", (8, 9, 10, 11, 12, 13)),
)


@dataclass(frozen=True)
class Painel:
    """Um arquivo ou parametro que entra ou sai de uma entidade."""

    nome: str
    dado: Any
    legenda: str = ""

    def tamanho(self) -> str:
        is_tabela = isinstance(self.dado, pd.DataFrame)

        return f"{len(self.dado)} linhas x {len(self.dado.columns)} colunas" if is_tabela else ""


@dataclass(frozen=True)
class Quadro:
    """Um passo: o que entrou, a entidade que processou, o que saiu."""

    numero: int
    entidade: str
    modulos: str
    resumo: str
    entradas: tuple[Painel, ...]
    saidas: tuple[Painel, ...]
    variaveis: tuple[Variavel, ...] = field(default_factory=tuple)
    aviso: str = ""


@dataclass(frozen=True)
class Execucao:
    """O que os quadros de uma execucao leem: a semente, o sigma e os arquivos."""

    seed: int
    sigma: float
    semente: SeedFiles
    avaliado: RunFiles | None
    grade: GridFiles | None
    configuracao: dict[str, dict]


def recorte(inteiro: pd.DataFrame, parte: pd.DataFrame, periodo: str) -> str:
    """A legenda de um painel que mostra so um pedaco do arquivo."""
    return (f"{periodo}, {len(parte)} das {len(inteiro)} linhas do arquivo, "
            f"{porcento(len(parte) / len(inteiro), 0)}")


def lado(parte: pd.DataFrame) -> str:
    """Tamanho e proporcao de anomalias de um lado da particao, contados (D-023)."""
    positivas = int(parte[LABEL].sum())

    return (f"semanas 5 a 8, {len(parte)} sessões, {positivas} positivas "
            f"({porcento(positivas / len(parte), 2)})")


def do_mecanismo(tempo: pd.DataFrame, *mecanismos: str) -> pd.DataFrame:
    return tempo[tempo["mechanism"].isin(mecanismos)].reset_index(drop=True)


# Fase 1: a preparacao dos dados.


def populacao(execucao: Execucao) -> Quadro:
    semente = execucao.semente

    return Quadro(
        1, "População · Scenario Engine e KMS", "M1",
        "Constrói o **mundo estático**: os operadores, do Scenario Engine, e o "
        "repositório de chaves, do KMS. Nenhuma requisição ainda.",
        (Painel("semente", execucao.seed, "determina tudo que não é σ"),),
        (Painel("operators.csv", semente.operators, "44 operadores em três perfis"),
         Painel("keys.csv", semente.keys, "300 chaves em 12 escopos")),
        variaveis_do_m1(KeyRepositorySpecification(), semente.keys),
    )


def trafego_legitimo(execucao: Execucao) -> Quadro:
    semente = execucao.semente

    return Quadro(
        2, "Scenario Engine · tráfego legítimo", "M2",
        "Transforma as duas tabelas em **oito semanas de requisições**, cada operador "
        "no ritmo do seu regime.",
        (Painel("operators.csv", semente.operators, "quem age, e com que ritmo"),
         Painel("keys.csv", semente.keys, "o que pode ser pedido")),
        (Painel("requests.csv", semente.requests, "oito semanas, só legítimo, sem desfecho"),),
        variaveis_do_m2(TrafficSpecification()),
    )


# Fase 2: o aquecimento e a calibracao.


def pedidos_do_aquecimento(semente: SeedFiles) -> Painel:
    parte = semente.requests[belongs_to(WARMUP, semente.requests["timestamp"])]

    return Painel("requests.csv", parte,
                  recorte(semente.requests, parte, "o que foi pedido nas semanas 1 a 4"))


def kms(execucao: Execucao) -> Quadro:
    semente = execucao.semente

    return Quadro(
        3, "KMS", "M4",
        "O **KMS decide** o desfecho de cada requisição pela política da chave. "
        "Semanas 1 a 4, sem atacante.",
        (Painel("operators.csv", semente.operators, "quem pode pedir, e em que escopos"),
         Painel("keys.csv", semente.keys, "escopo e situação de cada chave"),
         pedidos_do_aquecimento(semente)),
        (Painel("outcomes.csv", semente.outcomes, "semanas 1 a 4: event_id e outcome"),),
        variaveis_do_kms(),
    )


def audit_logger(execucao: Execucao) -> Quadro:
    semente = execucao.semente

    return Quadro(
        4, "Audit Logger", "M5",
        "O **Audit Logger registra**: junta cada requisição ao desfecho e grava o log, "
        "com oito colunas. Semanas 1 a 4.",
        (pedidos_do_aquecimento(semente),
         Painel("outcomes.csv", semente.outcomes, "o desfecho que o KMS decidiu")),
        (Painel("log.csv", semente.log, "semanas 1 a 4, o log de auditoria"),),
        variaveis_do_audit_logger(),
    )


def perfis_historicos(execucao: Execucao) -> Quadro:
    semente = execucao.semente

    return Quadro(
        5, "Dataset Generator · perfis históricos", "M6",
        "Observa o que cada operador fez nas **quatro semanas de aquecimento**: a "
        "janela horária e as origens de rede.",
        (Painel("log.csv", semente.log, "semanas 1 a 4, inteiras"),),
        (Painel("historical_profiles.csv", semente.profiles, "a régua, uma linha por operador"),),
        variaveis_do_m6(),
    )


def sessoes_do_aquecimento(execucao: Execucao) -> Quadro:
    semente = execucao.semente

    return Quadro(
        6, "Dataset Generator · sessões", "M7",
        f"Agrupa o log **por sessão**: cada uma vira uma linha com {len(ATTRIBUTES)} "
        "atributos, comparada contra o perfil do operador.",
        (Painel("log.csv", semente.log, "semanas 1 a 4"),
         Painel("historical_profiles.csv", semente.profiles, "a régua")),
        (Painel("sessions.csv", semente.sessions, "semanas 1 a 4, sem rótulo"),),
        variaveis_do_m7(),
    )


def calibracao(execucao: Execucao) -> Quadro:
    semente = execucao.semente

    return Quadro(
        7, "Policy Engine · calibração", "M8",
        f"Percentil {PERCENTILE} de cada grandeza, sobre **todas** as sessões do "
        f"aquecimento: {len(THRESHOLD_ATTRIBUTES)} limiares, congelados daqui em diante.",
        (Painel("sessions.csv", semente.sessions, "semanas 1 a 4, inteiras"),),
        (Painel("thresholds.csv", semente.thresholds, "um conjunto por semente"),),
        variaveis_do_m8(),
    )


# Fase 3: o ataque e a comparacao.


def campanha(execucao: Execucao) -> Quadro:
    semente, avaliado = execucao.semente, execucao.avaliado
    parte = semente.requests[belongs_to(EVALUATED, semente.requests["timestamp"])]

    return Quadro(
        8, "Scenario Engine · campanha de ataque", "M3",
        f"Mescla **{AttackSpecification().campaign_sessions} sessões comprometidas** às "
        f"semanas 5 a 8, sob a credencial de um administrador. Em σ "
        f"**{com_virgula(execucao.sigma)}**.",
        (Painel("requests.csv", parte, recorte(semente.requests, parte, "as semanas 5 a 8")),
         Painel("sigma", execucao.sigma, "0,0 ostensivo, 1,0 indistinguível")),
        (Painel("requests.csv", avaliado.requests, "legítimo + ataque, renumerado"),
         Painel("compromised_sessions.csv", avaliado.compromised, "o rótulo, fora do log"),
         Painel("run.csv", avaliado.run, "qual administrador foi comprometido")),
        variaveis_do_m3(execucao.sigma, AttackSpecification(), TrafficSpecification()),
    )


def periodo_avaliado(execucao: Execucao) -> Quadro:
    avaliado = execucao.avaliado
    positivas = int(avaliado.sessions[LABEL].sum())

    return Quadro(
        9, "KMS · Audit Logger · Dataset Generator", "M4 · M5 · M7",
        "**As mesmas entidades do aquecimento**, agora sobre as semanas 5 a 8 com "
        "ataque. Ao fim, o rótulo é juntado.",
        (Painel("requests.csv", avaliado.requests, "semanas 5 a 8, com a campanha"),
         Painel("historical_profiles.csv", execucao.semente.profiles, "do aquecimento, nunca recalculado"),
         Painel("compromised_sessions.csv", avaliado.compromised, "o rótulo")),
        (Painel("log.csv", avaliado.log, "semanas 5 a 8"),
         Painel("sessions.csv", avaliado.sessions,
                f"{len(avaliado.sessions)} sessões, {positivas} positivas, já rotulado")),
        variaveis_do_m4_m5() + variaveis_do_m7(),
    )


def particao(execucao: Execucao) -> Quadro:
    avaliado = execucao.avaliado

    return Quadro(
        10, "Dataset Generator · partição", "M9",
        f"Divide o conjunto em treino e holdout, **por sessão** e por classe: "
        f"{porcento(1 - HOLDOUT_SHARE, 0)} e {porcento(HOLDOUT_SHARE, 0)}. A mesma "
        "divisão em todo σ da semente.",
        (Painel("sessions.csv", avaliado.sessions, "o conjunto rotulado"),
         Painel("semente", execucao.seed, "o sorteio depende só dela")),
        (Painel("train.csv", avaliado.train, lado(avaliado.train)),
         Painel("holdout.csv", avaliado.holdout, lado(avaliado.holdout))),
        variaveis_do_m9(avaliado.holdout),
    )


def regras(execucao: Execucao) -> Quadro:
    avaliado = execucao.avaliado
    alertas = int(avaliado.predictions_rules["predicted"].sum())

    return Quadro(
        11, "Policy Engine · as oito regras", "M10",
        "O baseline decide cada sessão do holdout: **alerta com duas ou mais regras "
        "disparadas**, contra os limiares do aquecimento.",
        (Painel("holdout.csv", avaliado.holdout, "as sessões a decidir"),
         Painel("thresholds.csv", execucao.semente.thresholds, "congelados desde o aquecimento")),
        (Painel("predictions_rules.csv", avaliado.predictions_rules,
                f"uma coluna por regra e a decisão; {alertas} alertas"),
         Painel("tempo", do_mecanismo(avaliado.timing, RULES), "a decisão, cronometrada")),
        variaveis_do_m10(),
    )


def modelos(execucao: Execucao) -> Quadro:
    avaliado = execucao.avaliado

    return Quadro(
        12, "Modelos supervisionados", "M11",
        "**Random Forest e XGBoost** treinam uma vez no treino e decidem o holdout, "
        "com a configuração que a busca na 902 escolheu.",
        (Painel("train.csv", avaliado.train, lado(avaliado.train)),
         Painel("holdout.csv", avaliado.holdout, lado(avaliado.holdout))),
        (Painel("predictions_ml.csv", avaliado.predictions_ml,
                "a decisão e o escore de cada modelo"),
         Painel("tempo", do_mecanismo(avaliado.timing, "random_forest", "xgboost"),
                "a decisão, cronometrada")),
        variaveis_do_m11(execucao.configuracao),
    )


def avaliacao(execucao: Execucao) -> Quadro:
    grade = execucao.grade
    entradas = (
        Painel("predictions_rules.csv", execucao.avaliado.predictions_rules, "as regras"),
        Painel("predictions_ml.csv", execucao.avaliado.predictions_ml, "os modelos"),
    )
    resumo = ("F1 com a matriz de confusão, no holdout inteiro e só entre "
              "administradores, e a árvore rasa da trivialidade, **desta execução**. "
              "A comparação entre as 30 sementes está na página Resultados.")

    if grade is None:
        return Quadro(13, "Avaliação", "M12", resumo, entradas, (), variaveis_do_m12(),
                      aviso="Faltam as tabelas da avaliação: rode python -m src.main.")

    def desta(tabela: pd.DataFrame) -> pd.DataFrame:
        linhas = (tabela["seed"] == execucao.seed) & (tabela["sigma"] == execucao.sigma)
        return tabela[linhas].drop(columns=["seed", "sigma"]).reset_index(drop=True)

    return Quadro(
        13, "Avaliação", "M12", resumo, entradas,
        (Painel("metrics.csv", desta(grade.metrics), "desta execução, por mecanismo e recorte"),
         Painel("triviality.csv", desta(grade.triviality), "a árvore rasa e as duplicatas")),
        variaveis_do_m12(),
    )


QUADROS = (
    populacao, trafego_legitimo,
    kms, audit_logger, perfis_historicos, sessoes_do_aquecimento, calibracao,
    campanha, periodo_avaliado, particao, regras, modelos, avaliacao,
)
"""Os treze quadros, na ordem de execucao. O numero de cada um e a posicao aqui."""


def quadros_da_fase(fase: Fase, execucao: Execucao) -> list[Quadro]:
    quadros = []

    for numero in fase.quadros:
        montar_quadro = QUADROS[numero - 1]
        quadros.append(montar_quadro(execucao))

    return quadros
