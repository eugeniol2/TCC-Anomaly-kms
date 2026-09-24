"""Os quadros do pipeline: entrada, entidade, saida.

Este modulo existe porque o `steps.py` responde a pergunta errada para quem
esta conhecendo o trabalho. Ele mostra **funcoes** — `scope_pool`,
`holders_by_scope`, `split_keys_by_scope` —, que e granularidade de
implementacao. Quem le a monografia precisa da granularidade da
**arquitetura**: que dados entraram, que entidade os processou, que dados
sairam.

Um quadro aqui corresponde a **um passo do diagrama**. Os onze passos do
diagrama sao os onze quadros, e a numeracao e a mesma — a ordem de execucao,
que nao segue os numeros de modulo porque o Scenario Engine e o par
KMS · Audit Logger rodam duas vezes.

O detalhe por funcao nao se perde: cada quadro carrega os passos do `steps.py`
que lhe correspondem, e o app os mostra sob demanda. A ordem entre os dois e
deliberada — primeiro o que entrou e o que saiu, depois, para quem quiser,
como aquilo foi feito.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.attack.parameters import AttackSpecification
from src.globals.phases import EVALUATED, WARMUP, belongs_to
from src.population.parameters import KeyRepositorySpecification
from src.traffic.parameters import TrafficSpecification
from src.viewer.decisions import (
    Variavel,
    variaveis_do_m1,
    variaveis_do_m2,
    variaveis_do_m3,
    variaveis_do_m4_m5,
    variaveis_do_m6,
    variaveis_do_m7,
    variaveis_do_m8,
    variaveis_pendentes_do_m9,
    variaveis_pendentes_do_m10_m11,
    variaveis_pendentes_do_m12,
)
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

    porque: str = ""
    """A decisao que da forma a esta entidade, quando ha uma que importe.

    Nao e comentario de codigo: e o que a monografia precisa justificar. Um
    quadro sem isto e um quadro que so move dado de um lado para o outro.
    """

    variaveis: tuple[Variavel, ...] = field(default_factory=tuple)
    """As variáveis de decisão que regem esta entidade.

    Vêm antes das tabelas na tela, e não depois, porque a pergunta que elas
    respondem — por que este dado é assim — precede a de que dado é este.
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

    return (f"{periodo} — {len(parte)} das {len(inteiro)} linhas do arquivo, "
            f"{fatia:.0%}")


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
                "gera comportamento nenhum — nenhuma requisição, nenhum evento."
            ),
            porque=(
                "Estas duas tabelas **são** a política de autorização. O KMS "
                "decide cada desfecho consultando-as, e é por isso que o rótulo "
                "do experimento acaba sendo derivado da política em vez de "
                "inventado pelo gerador (D-013)."
            ),
            entradas=(
                Painel("semente", seed,
                       "parâmetro, sem período — determina tudo que não é σ"),
            ),
            saidas=(
                Painel("operators.csv", operators,
                       "estático, sem período — 44 operadores em três perfis"),
                Painel("keys.csv", keys,
                       "estático, sem período — 300 chaves em 12 escopos"),
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
            porque=(
                "Ele emite **tentativas, nunca desfechos**. As sete colunas não "
                "incluem `outcome`: se o gerador escrevesse o resultado da "
                "autorização, o rótulo seria invenção dele, e a comparação entre "
                "regras e modelos mediria a ficção que a produziu (D-013).\n\n"
                "O ritmo vem do **regime**, nunca do rótulo do perfil — é o que "
                "faz o pico periódico do serviço automatizado aparecer no log "
                "sem que nada tenha consultado o nome do perfil.\n\n"
                "**A faixa de comprimento de cada regime é o típico, não um "
                "teto** (D-097). Uma sessão em vinte se estende por um excesso "
                "geométrico, e a que se estende também se alarga (D-098). Sem "
                "isso nenhuma sessão legítima passava de 40 eventos, o atacante "
                "ostensivo começa em 40, e as duas classes não se sobrepunham: "
                "`events` e `distinct_keys` separavam sozinhos, com F1 0,982 e "
                "0,879. A cauda é o que obriga a regra a medir comportamento em "
                "vez de ler a faixa."
            ),
            entradas=(
                Painel("operators.csv", operators,
                       "estático — quem age, e com que ritmo"),
                Painel("keys.csv", keys, "estático — o que pode ser pedido"),
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
    """Passos 3 a 6 do diagrama: o aquecimento e a calibração."""
    do_aquecimento = requests[belongs_to(WARMUP, requests["timestamp"])]

    return [
        Quadro(
            numero=3,
            fase=FASE_2,
            entidade="KMS · Audit Logger",
            modulos="M4 · M5",
            resumo=(
                "O **KMS decide** o desfecho de cada requisição consultando a "
                "política da chave; o **Audit Logger registra**. Semanas 1 a 4, "
                "sem atacante nenhum."
            ),
            porque=(
                "A ordem em que o KMS avalia é decisão, não detalhe: "
                "identificador inexistente, depois política, depois estado da "
                "chave (D-077). **Autorização antes de estado** — responder "
                "'chave desabilitada' a quem não detém o escopo confirmaria que "
                "a chave existe.\n\n"
                "O log tem **oito colunas e nenhuma a mais**. Sem escopo, sem "
                "perfil, sem proprietário: qualquer um deles deixaria o modelo "
                "reconstruir a fronteira de autorização e aprender a política em "
                "vez do comportamento (D-064). E sem rótulo (D-063)."
            ),
            entradas=(
                Painel("operators.csv", operators,
                       "estático — quem pode pedir, e em que escopos"),
                Painel("keys.csv", keys,
                       "estático — escopo e situação de cada chave"),
                Painel("requests.csv", do_aquecimento,
                       recorte(requests, do_aquecimento, "o que foi pedido nas semanas 1 a 4")),
            ),
            saidas=(
                Painel("outcomes.csv", outcomes,
                       "semanas 1 a 4 — duas colunas: event_id e outcome"),
                Painel("log.csv", log,
                       "semanas 1 a 4 — as oito colunas do log de auditoria"),
            ),
            variaveis=variaveis_do_m4_m5(),
            detalhes=apenas(detalhes, "M4", "M5"),
        ),
        Quadro(
            numero=4,
            fase=FASE_2,
            entidade="Perfis históricos",
            modulos="M6",
            resumo=(
                "Observa o que cada operador fez nas **quatro semanas de "
                "aquecimento**: a janela horária em que abriu sessão e as "
                "origens de rede de onde veio."
            ),
            porque=(
                "**Fica congelado.** Perfil móvel absorveria o comportamento do "
                "atacante durante o período avaliado: a atividade da campanha "
                "entraria na definição do que é habitual para o operador "
                "comprometido, e o desvio que deveria denunciá-lo viraria a "
                "normalidade dele (D-044).\n\n"
                "As origens observadas são um **subconjunto** das que o M1 "
                "sorteou, e a diferença entre as duas é o que produz origem "
                "inédita legítima depois. Sem essa folga, endereço novo "
                "significaria atacante (D-040).\n\n"
                "**O aquecimento inteiro entra aqui** (D-096). Até 23/09 a régua "
                "saía de duas semanas e a terceira ficava reservada para "
                "calibrar. Alargá-la para quatro **derruba pela metade** o "
                "alarme falso dos dois atributos de histórico: o que disparava "
                "não era comportamento anômalo, era perfil estimado de amostra "
                "pequena demais. Os números estão na seção 6.9 do "
                "`relatorio-fundacao.md`, medidos nas 30 sementes."
            ),
            entradas=(
                Painel("log.csv", log,
                       "semanas 1 a 4, inteiras — a régua vê todo o aquecimento"),
            ),
            saidas=(
                Painel("historical_profiles.csv", perfis,
                       "**das semanas 1 a 4** — uma linha por operador: a "
                       "janela horária e as origens que ele usou ali"),
            ),
            variaveis=variaveis_do_m6(),
            detalhes=apenas(detalhes, "M6"),
        ),
        Quadro(
            numero=5,
            fase=FASE_2,
            entidade="Dataset Generator",
            modulos="M7",
            resumo=(
                "Agrupa o log por operador e, dentro dele, **por sessão**. Cada "
                "sessão vira uma linha com oito atributos, comparada contra o "
                "perfil daquele operador."
            ),
            porque=(
                "**Não existe tamanho de janela.** A unidade é a sessão, e ela "
                "tem o comprimento que teve (D-061). O que se pede ao modelo é "
                "dizer se uma "
                "sessão é maliciosa, que é o que um analista investiga, e "
                "nenhuma linha tem rótulo misto.\n\n"
                "Consequência para a partição: como a linha já é a sessão, ela "
                "não pode atravessar fronteira nenhuma — o vazamento deixa de "
                "depender de uma regra ser seguida e passa a ser impossível por "
                "construção.\n\n"
                "Aqui o arquivo sai **sem coluna de rótulo**: ele alimenta só a "
                "calibração, e calibração por percentil não usa rótulo (D-063).\n\n"
                "**Este passo não mede o período avaliado — mede o aquecimento.** "
                "O perfil entra aqui como **régua**, e a mesma régua volta no "
                "passo 8 para medir as semanas 5 a 8. É isso que faz o limiar "
                "transferir: calibrado numa escala, aplicado na mesma. Se o "
                "aquecimento fosse medido contra outro perfil, o percentil 99 "
                "sairia de um instrumento e seria usado com outro.\n\n"
                "Repare que `atypical_hour` e `new_source_ip` valem **zero em "
                "todas estas linhas, por construção**: são as sessões que "
                "*definiram* a janela horária e o conjunto de origens, e nenhuma "
                "pode cair fora do que ela própria delimitou. Não é defeito — é "
                "a invariante que pegou o erro de arredondamento da D-090, e é "
                "conferida nas 30 sementes."
            ),
            entradas=(
                Painel("log.csv", log,
                       "semanas 1 a 4, inteiras — é o que se mede"),
                Painel("historical_profiles.csv", perfis,
                       "**das semanas 1 a 4** — é a régua: contra ela "
                       "`atypical_hour` e `new_source_ip` são medidos, e é a "
                       "mesma que medirá as semanas 5 a 8"),
            ),
            saidas=(
                Painel("sessions.csv", sessoes,
                       "semanas 1 a 4 — uma linha por sessão, oito atributos, "
                       "sem rótulo; o M8 lê o arquivo inteiro"),
            ),
            variaveis=variaveis_do_m7(),
            detalhes=apenas(detalhes, "M7"),
        ),
        Quadro(
            numero=6,
            fase=FASE_2,
            entidade="Calibração de limiares",
            modulos="M8",
            resumo=(
                "Percentil 99 de cada grandeza, sobre **todas** as sessões do "
                "aquecimento. Seis limiares, um por regra de grandeza."
            ),
            porque=(
                "O baseline **não recebe treino em momento nenhum** e chega ao "
                "período avaliado com estes números congelados (D-033, D-046).\n\n"
                "A assimetria com os modelos é deliberada e precisa ser "
                "declarada: calibrar limiar exige só comportamento normal, que "
                "um administrador teria antes de qualquer incidente; ajustar "
                "hiperparâmetro exige rótulo de ataque, que ele não teria.\n\n"
                "Seis e não oito porque `atypical_hour` e `new_source_ip` já vêm "
                "binárias do M7 — percentil sobre uma coluna de zeros e uns "
                "daria 0 ou 1 e não significaria nada (D-080).\n\n"
                "**O limiar sai do mesmo período que o perfil** (D-096). Até "
                "23/09 ele saía da semana 3 sozinha, reservada porque as semanas "
                "1 e 2 tinham construído o perfil. O argumento não se "
                "sustentava: as seis regras de grandeza **não consultam o "
                "perfil** — contam eventos, chaves e negações do log cru —, "
                "então não havia acoplamento a evitar. O que a divisão produzia "
                "era um limiar estimado de um terço dos dados disponíveis."
            ),
            entradas=(
                Painel("sessions.csv", sessoes,
                       "semanas 1 a 4 — **o mesmo arquivo do passo 5**, "
                       "inteiro"),
            ),
            saidas=(
                Painel("thresholds.csv", limiares,
                       "**das semanas 1 a 4** — seis regras, um conjunto por "
                       "semente, congelado daqui em diante"),
            ),
            variaveis=variaveis_do_m8(),
            detalhes=apenas(detalhes, "M8"),
        ),
    ]


def frames_da_fase_3(
    sigma: float,
    requests: pd.DataFrame,
    perfis: pd.DataFrame,
    campanha_requests: pd.DataFrame,
    compromised: pd.DataFrame,
    execucao: pd.DataFrame,
    log: pd.DataFrame,
    sessoes: pd.DataFrame,
    detalhes: list[Step],
) -> list[Quadro]:
    """Passos 7 a 11 do diagrama. Os três últimos ainda não existem."""
    do_avaliado = requests[belongs_to(EVALUATED, requests["timestamp"])]
    positivas = int(sessoes["compromised"].sum())

    return [
        Quadro(
            numero=7,
            fase=FASE_3,
            entidade="Scenario Engine — campanha de ataque",
            modulos="M3",
            resumo=(
                f"Mescla **58 sessões comprometidas** às semanas 5 a 8 do "
                f"tráfego legítimo, sob a credencial de um administrador. Em "
                f"σ **{sigma}**."
            ),
            porque=(
                "O atacante **não tem identidade própria**: age sob a credencial "
                "de um administrador real. Isso elimina detecção por controle de "
                "acesso e deixa o comportamento como único sinal.\n\n"
                "A campanha tem **o mesmo tamanho nas onze condições** (D-081). "
                "O que σ move é o comportamento dentro das sessões, nunca "
                "quantas são: se movesse as duas coisas, a proporção de anomalias "
                "mudaria junto e a comparação entre condições confundiria "
                "furtividade com desbalanceamento.\n\n"
                "Em **σ = 1** as cinco dimensões valem exatamente o que o M2 "
                "usaria para aquele administrador, e o M3 chama as mesmas "
                "funções: a sessão comprometida é indistinguível de uma "
                "legítima. E o piso declarado da varredura (D-082)."
            ),
            entradas=(
                Painel("requests.csv", do_avaliado,
                       recorte(requests, do_avaliado,
                               "as semanas 5 a 8, onde a campanha entra")),
                Painel("sigma", sigma,
                       "parâmetro, sem período — 0,0 ostensivo, 1,0 indistinguível"),
            ),
            saidas=(
                Painel("requests.csv", campanha_requests,
                       "semanas 5 a 8, legítimo + ataque, renumerado"),
                Painel("compromised_sessions.csv", compromised,
                       "semanas 5 a 8 — o rótulo, fora do log"),
                Painel("run.csv", execucao,
                       "uma linha por execução — qual administrador foi "
                       "comprometido nesta semente"),
            ),
            variaveis=variaveis_do_m3(
                sigma, AttackSpecification(), TrafficSpecification()
            ),
            detalhes=apenas(detalhes, "M3"),
        ),
        Quadro(
            numero=8,
            fase=FASE_3,
            entidade="KMS · Audit Logger · Dataset Generator",
            modulos="M4 · M5 · M7",
            resumo=(
                "**As mesmas entidades dos passos 3 e 5**, agora sobre as semanas "
                "5 a 8 com ataque. Ao fim, o rótulo é juntado."
            ),
            porque=(
                "O KMS **não sabe que há atacante**. Ele avalia a política da "
                "chave, e as requisições do atacante são negadas pelas mesmas "
                "regras que negam as legítimas. E isso que faz o rótulo ser "
                "derivado da política em vez de afirmado pelo gerador — o "
                "argumento central do trabalho.\n\n"
                "Cada sessão comprometida é **uma** positiva (D-061). A "
                "proporção abaixo é contada no dado, nunca herdada do parâmetro "
                "do gerador."
            ),
            entradas=(
                Painel("requests.csv", campanha_requests,
                       "semanas 5 a 8 — legítimo com a campanha dentro"),
                Painel("historical_profiles.csv", perfis,
                       "**do aquecimento** — o mesmo arquivo do passo 5, "
                       "nunca recalculado"),
                Painel("compromised_sessions.csv", compromised,
                       "semanas 5 a 8 — o rótulo, que só é juntado aqui"),
            ),
            saidas=(
                Painel("log.csv", log, "semanas 5 a 8 — o log do período avaliado"),
                Painel("sessions.csv", sessoes,
                       f"semanas 5 a 8 — {len(sessoes)} sessões, {positivas} "
                       f"positivas ({positivas / len(sessoes):.2%}), já rotulado"),
            ),
            variaveis=variaveis_do_m4_m5() + variaveis_do_m7(),
            detalhes=apenas(detalhes, "M4 · M5", "M7"),
        ),
        Quadro(
            numero=9,
            fase=FASE_3,
            entidade="Partição experimental",
            modulos="M9",
            resumo=(
                "Divide o conjunto em treino e holdout, **por sessão** e "
                "estratificada por classe: 60 % e 40 %."
            ),
            porque=(
                "Como a linha já é a sessão, não há o que separar: o vazamento "
                "entre partições é impossível por construção. Do holdout saem as "
                "**23 positivas** que a contagem absoluta exige (D-070)."
            ),
            entradas=(Painel("sessions.csv", sessoes,
                       "semanas 5 a 8 — o conjunto rotulado"),),
            saidas=(),
            variaveis=variaveis_pendentes_do_m9(),
            pendente="train.csv e holdout.csv ainda não existem.",
        ),
        Quadro(
            numero=10,
            fase=FASE_3,
            entidade="Policy Engine · Modelos supervisionados",
            modulos="M10 · M11",
            resumo=(
                "O baseline de **oito regras** e os dois modelos — Random Forest "
                "e XGBoost — classificam o holdout, lado a lado."
            ),
            porque=(
                "O baseline lê os limiares congelados no aquecimento e **não recebe "
                "treino**. Os modelos treinam, com configuração idêntica nas 330 "
                "execuções. **Sem reamostragem e sem SMOTE**: a proporção real é "
                "preservada (D-024)."
            ),
            entradas=(),
            saidas=(),
            variaveis=variaveis_pendentes_do_m10_m11(),
            pendente="predictions_rules.csv e predictions_ml.csv ainda não existem.",
        ),
        Quadro(
            numero=11,
            fase=FASE_3,
            entidade="Avaliação e comparação",
            modulos="M12",
            resumo=(
                "F1 com a matriz de confusão completa, Wilcoxon pareado com "
                "correção de Holm sobre 22 comparações. Aqui roda a verificação "
                "de trivialidade."
            ),
            porque=(
                "F1 **nunca** é reportado sem a matriz de confusão ao lado, e a "
                "especificidade entra porque ela mede alarme falso — a queixa "
                "clássica contra motor de regras, reportada por só 4 de 42 "
                "estudos lidos (D-021, D-022)."
            ),
            entradas=(),
            saidas=(),
            variaveis=variaveis_pendentes_do_m12(),
            pendente="metrics.csv ainda não existe. Nenhum F1 foi calculado.",
        ),
    ]
