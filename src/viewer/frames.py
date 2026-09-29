"""Os quadros do pipeline: entrada, entidade, saida.

Este modulo existe porque o `steps.py` responde a pergunta errada para quem
esta conhecendo o trabalho. Ele mostra **funcoes** (`scope_pool`,
`holders_by_scope`, `split_keys_by_scope`), que e granularidade de
implementacao. Quem le a monografia precisa da granularidade da
**arquitetura**: que dados entraram, que entidade os processou, que dados
sairam.

Um quadro aqui corresponde a **um passo do diagrama**, com uma excecao: o
KMS e o Audit Logger, que o diagrama junta num passo so, sao dois quadros no
aquecimento, porque juntos confundiam quem decide o desfecho com quem o
registra. Por isso sao doze quadros para os onze passos do diagrama. A
numeracao e a ordem de execucao, que nao segue os numeros de modulo porque o
Scenario Engine e o par KMS · Audit Logger rodam duas vezes.

O detalhe por funcao nao se perde: cada quadro carrega os passos do `steps.py`
que lhe correspondem, e o app os mostra sob demanda. A ordem entre os dois e
deliberada: primeiro o que entrou e o que saiu, depois, para quem quiser,
como aquilo foi feito.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.attack.parameters import AttackSpecification
from src.calibration.parameters import PERCENTILE, THRESHOLD_ATTRIBUTES
from src.dataset.build import ATTRIBUTES, LABEL
from src.globals.experiment import SIGMAS
from src.globals.phases import EVALUATED, WARMUP, belongs_to
from src.partition.build import Partition
from src.partition.parameters import HOLDOUT_SHARE
from src.population.parameters import KeyRepositorySpecification
from src.traffic.parameters import TrafficSpecification
from src.viewer.decisions import (
    MODELOS,
    Variavel,
    variaveis_do_m1,
    variaveis_do_m2,
    variaveis_do_m3,
    variaveis_do_audit_logger,
    variaveis_do_kms,
    variaveis_do_m4_m5,
    variaveis_do_m6,
    variaveis_do_m7,
    variaveis_do_m8,
    variaveis_do_m9,
    variaveis_pendentes_do_m10_m11,
    variaveis_pendentes_do_m12,
)
from src.viewer.formatting import com_virgula, porcento
from src.viewer.steps import FASE_1, FASE_2, FASE_3, Step


@dataclass(frozen=True)
class Painel:
    """Um dado que entra ou sai de uma entidade.

    `dado` pode ser uma tabela, uma lista ou um numero. O app decide como
    mostrar; aqui so se declara o que e.
    """

    nome: str
    dado: Any
    legenda: str = ""

    def tamanho(self) -> str:
        """Quantas linhas, quando isso faz sentido."""
        is_tabela = isinstance(self.dado, pd.DataFrame)

        if is_tabela:
            return f"{len(self.dado)} linhas x {len(self.dado.columns)} colunas"

        is_contavel = isinstance(self.dado, (list, tuple, dict, set))

        if is_contavel:
            return f"{len(self.dado)} itens"

        return ""


@dataclass(frozen=True)
class Quadro:
    """Um passo do diagrama: o que entrou, quem processou, o que saiu."""

    numero: int
    fase: str
    entidade: str
    modulos: str
    resumo: str
    entradas: tuple[Painel, ...]
    saidas: tuple[Painel, ...]

    variaveis: tuple[Variavel, ...] = field(default_factory=tuple)
    """As variáveis de decisão que regem esta entidade.

    Vêm antes das tabelas na tela, e não depois, porque a pergunta que elas
    respondem (por que este dado é assim) precede a de que dado é este.
    """

    detalhes: tuple[Step, ...] = field(default_factory=tuple)
    """Os passos por funcao, para quem quiser abrir. Podem ser nenhum."""

    pendente: str = ""
    """Preenchido quando a entidade ainda não existe."""


def apenas(passos: list[Step], *modulos: str) -> tuple[Step, ...]:
    """Os passos daqueles módulos, na ordem em que rodaram."""
    return tuple(passo for passo in passos if passo.modulo in modulos)


def recorte(inteiro: pd.DataFrame, parte: pd.DataFrame, periodo: str) -> str:
    """A legenda de um painel que mostra so um pedaco do arquivo.

    Mostrar o tamanho do arquivo inteiro ao lado de uma legenda que fala de
    outro periodo engana: parece que as oito semanas inteiras entraram no KMS
    quando entrou so a fatia daquela fase. O painel passa a mostrar **o
    recorte**, e a legenda diz de onde ele saiu.
    """
    fatia = len(parte) / len(inteiro)

    return (f"{periodo}, {len(parte)} das {len(inteiro)} linhas do arquivo, "
            f"{porcento(fatia, 0)}")


def frames_da_fase_1(
    seed: int,
    operators: pd.DataFrame,
    keys: pd.DataFrame,
    requests: pd.DataFrame,
    detalhes: list[Step],
) -> list[Quadro]:
    """Passos 1 e 2 do diagrama: o mundo estático e o tráfego legítimo."""
    return [
        Quadro(
            numero=1,
            fase=FASE_1,
            entidade="População e repositório de chaves",
            modulos="M1",
            resumo=(
                "Constrói o **mundo estático**: quem existe e o que existe. Não "
                "gera comportamento nenhum: nenhuma requisição, nenhum evento."
            ),
            entradas=(
                Painel("semente", seed,
                       "parâmetro, sem período, que determina tudo que não é σ"),
            ),
            saidas=(
                Painel("operators.csv", operators,
                       "estático, sem período, 44 operadores em três perfis"),
                Painel("keys.csv", keys,
                       "estático, sem período, 300 chaves em 12 escopos"),
            ),
            variaveis=variaveis_do_m1(KeyRepositorySpecification(), keys),
            detalhes=apenas(detalhes, "M1"),
        ),
        Quadro(
            numero=2,
            fase=FASE_1,
            entidade="Scenario Engine",
            modulos="M2",
            resumo=(
                "Transforma as duas tabelas em **oito semanas de requisições**. "
                "Cada operador abre sessões conforme o ritmo do seu regime, e "
                "dentro de cada sessão pede chaves do seu alcance."
            ),
            entradas=(
                Painel("operators.csv", operators,
                       "estático: quem age, e com que ritmo"),
                Painel("keys.csv", keys, "estático: o que pode ser pedido"),
            ),
            saidas=(
                Painel("requests.csv", requests,
                       "oito semanas, só tráfego legítimo, sem desfecho"),
            ),
            variaveis=variaveis_do_m2(TrafficSpecification()),
            detalhes=apenas(detalhes, "M2"),
        ),
    ]


def frames_da_fase_2(
    requests: pd.DataFrame,
    operators: pd.DataFrame,
    keys: pd.DataFrame,
    outcomes: pd.DataFrame,
    log: pd.DataFrame,
    perfis: pd.DataFrame,
    sessoes: pd.DataFrame,
    limiares: pd.DataFrame,
    detalhes: list[Step],
) -> list[Quadro]:
    """Passos 3 a 7: o aquecimento e a calibração."""
    do_aquecimento = requests[belongs_to(WARMUP, requests["timestamp"])]

    return [
        Quadro(
            numero=3,
            fase=FASE_2,
            entidade="KMS",
            modulos="M4",
            resumo=(
                "O **KMS decide** o desfecho de cada requisição, consultando a "
                "política da chave: sucesso, negação por política, chave "
                "desabilitada ou chave inexistente. Semanas 1 a 4, sem "
                "atacante nenhum."
            ),
            entradas=(
                Painel("operators.csv", operators,
                       "estático: quem pode pedir, e em que escopos"),
                Painel("keys.csv", keys,
                       "estático: escopo e situação de cada chave"),
                Painel("requests.csv", do_aquecimento,
                       recorte(requests, do_aquecimento, "o que foi pedido nas semanas 1 a 4")),
            ),
            saidas=(
                Painel("outcomes.csv", outcomes,
                       "semanas 1 a 4, com duas colunas: event_id e outcome"),
            ),
            variaveis=variaveis_do_kms(),
            detalhes=apenas(detalhes, "M4"),
        ),
        Quadro(
            numero=4,
            fase=FASE_2,
            entidade="Audit Logger",
            modulos="M5",
            resumo=(
                "O **Audit Logger registra**: junta cada requisição ao desfecho "
                "que o KMS decidiu e grava o log de auditoria, com oito colunas. "
                "Semanas 1 a 4."
            ),
            entradas=(
                Painel("requests.csv", do_aquecimento,
                       recorte(requests, do_aquecimento, "o que foi pedido nas semanas 1 a 4")),
                Painel("outcomes.csv", outcomes,
                       "semanas 1 a 4, o desfecho que o KMS decidiu para cada evento"),
            ),
            saidas=(
                Painel("log.csv", log,
                       "semanas 1 a 4, as oito colunas do log de auditoria"),
            ),
            variaveis=variaveis_do_audit_logger(),
            detalhes=apenas(detalhes, "M5"),
        ),
        Quadro(
            numero=5,
            fase=FASE_2,
            entidade="Perfis históricos",
            modulos="M6",
            resumo=(
                "Observa o que cada operador fez nas **quatro semanas de "
                "aquecimento**: a janela horária em que abriu sessão e as "
                "origens de rede de onde veio."
            ),
            entradas=(
                Painel("log.csv", log,
                       "semanas 1 a 4, inteiras, porque a régua vê todo o aquecimento"),
            ),
            saidas=(
                Painel("historical_profiles.csv", perfis,
                       "**das semanas 1 a 4**, uma linha por operador: a "
                       "janela horária e as origens que ele usou ali"),
            ),
            variaveis=variaveis_do_m6(),
            detalhes=apenas(detalhes, "M6"),
        ),
        Quadro(
            numero=6,
            fase=FASE_2,
            entidade="Dataset Generator",
            modulos="M7",
            resumo=(
                "Agrupa o log por operador e, dentro dele, **por sessão**. Cada "
                f"sessão vira uma linha com {len(ATTRIBUTES)} atributos, comparada "
                "contra o "
                "perfil daquele operador."
            ),
            entradas=(
                Painel("log.csv", log,
                       "semanas 1 a 4, inteiras, que é o que se mede"),
                Painel("historical_profiles.csv", perfis,
                       "**das semanas 1 a 4**, a régua: contra ela "
                       "`atypical_hour` e `new_source_ip` são medidos, e é a "
                       "mesma que medirá as semanas 5 a 8"),
            ),
            saidas=(
                Painel("sessions.csv", sessoes,
                       f"semanas 1 a 4, uma linha por sessão, {len(ATTRIBUTES)} "
                       "atributos, sem rótulo; a Calibração lê o arquivo inteiro"),
            ),
            variaveis=variaveis_do_m7(),
            detalhes=apenas(detalhes, "M7"),
        ),
        Quadro(
            numero=7,
            fase=FASE_2,
            entidade="Calibração de limiares",
            modulos="M8",
            resumo=(
                f"Percentil {PERCENTILE} de cada grandeza, sobre **todas** as "
                f"sessões do aquecimento. {len(THRESHOLD_ATTRIBUTES)} limiares, um "
                "por regra de grandeza."
            ),
            entradas=(
                Painel("sessions.csv", sessoes,
                       "semanas 1 a 4, **o mesmo arquivo do Dataset Generator**, "
                       "inteiro"),
            ),
            saidas=(
                Painel("thresholds.csv", limiares,
                       f"**das semanas 1 a 4**, {len(THRESHOLD_ATTRIBUTES)} regras, um conjunto por "
                       "semente, congelado daqui em diante"),
            ),
            variaveis=variaveis_do_m8(),
            detalhes=apenas(detalhes, "M8"),
        ),
    ]


def lado(parte: pd.DataFrame) -> str:
    """A legenda de um lado da partição: tamanho e proporção, contados (D-023)."""
    positivas = int(parte[LABEL].sum())

    return (f"semanas 5 a 8, {len(parte)} sessões, {positivas} positivas "
            f"({porcento(positivas / len(parte), 2)})")


def frames_da_fase_3(
    seed: int,
    sigma: float,
    requests: pd.DataFrame,
    perfis: pd.DataFrame,
    campanha_requests: pd.DataFrame,
    compromised: pd.DataFrame,
    execucao: pd.DataFrame,
    log: pd.DataFrame,
    sessoes: pd.DataFrame,
    particao: Partition,
    detalhes: list[Step],
) -> list[Quadro]:
    """Passos 8 a 12. Os dois últimos ainda não existem."""
    do_avaliado = requests[belongs_to(EVALUATED, requests["timestamp"])]
    positivas = int(sessoes["compromised"].sum())

    return [
        Quadro(
            numero=8,
            fase=FASE_3,
            entidade="Scenario Engine · campanha de ataque",
            modulos="M3",
            resumo=(
                f"Mescla **{AttackSpecification().campaign_sessions} sessões "
                f"comprometidas** às semanas 5 a 8 do "
                f"tráfego legítimo, sob a credencial de um administrador. Em "
                f"σ **{com_virgula(sigma)}**."
            ),
            entradas=(
                Painel("requests.csv", do_avaliado,
                       recorte(requests, do_avaliado,
                               "as semanas 5 a 8, onde a campanha entra")),
                Painel("sigma", sigma,
                       "parâmetro, sem período: 0,0 ostensivo, 1,0 indistinguível"),
            ),
            saidas=(
                Painel("requests.csv", campanha_requests,
                       "semanas 5 a 8, legítimo + ataque, renumerado"),
                Painel("compromised_sessions.csv", compromised,
                       "semanas 5 a 8, o rótulo, fora do log"),
                Painel("run.csv", execucao,
                       "uma linha por execução: qual administrador foi "
                       "comprometido nesta semente"),
            ),
            variaveis=variaveis_do_m3(
                sigma, AttackSpecification(), TrafficSpecification()
            ),
            detalhes=apenas(detalhes, "M3"),
        ),
        Quadro(
            numero=9,
            fase=FASE_3,
            entidade="KMS · Audit Logger · Dataset Generator",
            modulos="M4 · M5 · M7",
            resumo=(
                "**As mesmas entidades do aquecimento** (KMS, Audit Logger e Dataset "
                "Generator), agora sobre as semanas "
                "5 a 8 com ataque. Ao fim, o rótulo é juntado."
            ),
            entradas=(
                Painel("requests.csv", campanha_requests,
                       "semanas 5 a 8, legítimo com a campanha dentro"),
                Painel("historical_profiles.csv", perfis,
                       "**do aquecimento**, o mesmo arquivo dos Perfis históricos, "
                       "nunca recalculado"),
                Painel("compromised_sessions.csv", compromised,
                       "semanas 5 a 8, o rótulo, que só é juntado aqui"),
            ),
            saidas=(
                Painel("log.csv", log, "semanas 5 a 8, o log do período avaliado"),
                Painel("sessions.csv", sessoes,
                       f"semanas 5 a 8, {len(sessoes)} sessões, {positivas} "
                       f"positivas ({porcento(positivas / len(sessoes), 2)}), já rotulado"),
            ),
            variaveis=variaveis_do_m4_m5() + variaveis_do_m7(),
            detalhes=apenas(detalhes, "M4 · M5", "M7"),
        ),
        Quadro(
            numero=10,
            fase=FASE_3,
            entidade="Partição experimental",
            modulos="M9",
            resumo=(
                f"Divide o conjunto em treino e holdout, **por sessão** e "
                f"estratificada por classe: {porcento(1 - HOLDOUT_SHARE, 0)} e "
                f"{porcento(HOLDOUT_SHARE, 0)}. **A mesma divisão em todo σ** da "
                f"semente."
            ),
            entradas=(
                Painel("sessions.csv", sessoes,
                       "semanas 5 a 8, o conjunto rotulado"),
                Painel("semente", seed,
                       "parâmetro: o sorteio depende só dela, e não de σ"),
            ),
            saidas=(
                Painel("train.csv", particao.train, lado(particao.train)),
                Painel("holdout.csv", particao.holdout, lado(particao.holdout)),
            ),
            variaveis=variaveis_do_m9(particao.holdout),
            detalhes=apenas(detalhes, "M9"),
        ),
        Quadro(
            numero=11,
            fase=FASE_3,
            entidade="Policy Engine · Modelos supervisionados",
            modulos="M10 · M11",
            resumo=(
                "O baseline de **oito regras** e os dois modelos (Random Forest "
                "e XGBoost) classificam o holdout, lado a lado."
            ),
            entradas=(),
            saidas=(),
            variaveis=variaveis_pendentes_do_m10_m11(),
            pendente="predictions_rules.csv e predictions_ml.csv ainda não existem.",
        ),
        Quadro(
            numero=12,
            fase=FASE_3,
            entidade="Avaliação e comparação",
            modulos="M12",
            resumo=(
                "F1 com a matriz de confusão completa, Wilcoxon pareado com "
                f"correção de Holm sobre até {len(SIGMAS) * len(MODELOS)} comparações, as das "
                "condições que a trivialidade mantiver. "
                "Aqui roda a verificação "
                "de trivialidade."
            ),
            entradas=(),
            saidas=(),
            variaveis=variaveis_pendentes_do_m12(),
            pendente="metrics.csv ainda não existe. Nenhum F1 foi calculado.",
        ),
    ]
