"""A aba de comportamentos: os três regimes, um de cada vez, com o dado real.

Nas fases do pipeline os regimes aparecem aos pedaços, espalhados pelas
variáveis de decisão de cada quadro. Aqui eles são o assunto: quando cada um
abre sessão, quanto a sessão dura, o que ela pede e de onde vem, e por que cada
escolha é a que é.

**Cada número desta aba é lido do código ou medido no tráfego da semente.**
Nenhum está escrito na prosa: os parâmetros vêm de `REGIMES`, da
`TrafficSpecification` e da `AttackSpecification`, e as medições vêm do
`requests.csv` que o M2 produz para a semente escolhida. Mudado um parâmetro, a
aba muda junto, e é a mesma regra que o resto do viewer segue.

As curvas teóricas usam a mesma parametrização do gerador (`p = n / (n + m)`
para a Pascal, por exemplo), e ao lado delas vai sempre a frequência medida,
que é o que confirma que o gerador faz o que o texto diz.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy import stats

from src.scenario_engine.attack.parameters import AttackSpecification
from src.scenario_engine.attack.stealth import HourWindow, Stealth, stealth_of
from src.shared.phases import EVALUATED, belongs_to
from src.scenario_engine.traffic.calendar import business_days_among, simulated_days
from src.scenario_engine.traffic.operations import OPERATION_MIX, OPERATIONS
from src.scenario_engine.traffic.parameters import PRIMARY_ADDRESS_SHARE, TrafficSpecification
from src.scenario_engine.traffic.regimes import REGIMES, ArrivalRhythm, Regime, ScheduledRhythm
from src.scenario_engine.traffic.sessions import draw_request_count
from src.scenario_engine.population.profiles import PROFILES
from src.viewer.formatting import com_virgula, porcento
from src.viewer.theory import Teoria, teoria_da_geometrica

ORDEM = ("routine", "periodic_batch", "occasional_custody")
"""A ordem das abas e das séries dos gráficos.

O `routine` vem primeiro porque é a referência contra a qual o
`occasional_custody` se lê como irregular. A ordem das séries decide a cor, e
não se embaralha entre gráficos: o mesmo regime tem a mesma cor em todos.
"""

DIAS_DA_SEMANA = ("1 seg", "2 ter", "3 qua", "4 qui", "5 sex", "6 sáb", "7 dom")
# O número na frente mantém a ordem: o eixo de texto do Streamlit é ordenado
# alfabeticamente, e sem ele domingo apareceria antes de quarta.

RESUMOS = {
    "routine": (
        "**Trabalho em fluxo.** É a pessoa que usa o KMS como parte do "
        "expediente: chega em algum momento do horário comercial, cifra ou "
        "decifra o que precisa, e volta outro dia. Não há lote nem urgência, "
        "e um dia cheio não torna o seguinte mais vazio ou mais cheio. É o "
        "regime mais regular dos três, e existe também para ser a referência "
        "contra a qual o ritmo irregular do administrador aparece."
    ),
    "periodic_batch": (
        "**Máquina em horário marcado.** É a aplicação que roda por "
        "agendamento: acorda nas mesmas horas, todos os dias, inclusive fim de "
        "semana, dispara uma rajada de requisições em poucos segundos e para. "
        "Não há sorteio de quantas vezes ela roda, só um desvio de alguns "
        "minutos em torno da hora cheia. É o que produz os picos periódicos "
        "que o primeiro item da lista de conferência pede."
    ),
    "occasional_custody": (
        "**Custódia em rajadas.** É o administrador de chaves: passa dias sem "
        "abrir sessão e, quando o trabalho chega, faz várias numa tarde. As "
        "sessões são mais lentas, porque ele inspeciona entre uma requisição e "
        "outra, e pedem mais `DescribeKey` e `ExportKeyMaterial`. **É o regime "
        "que o atacante personifica**: a credencial comprometida é sempre de "
        "um administrador, e em σ 1,0 a sessão dele sai desta mesma "
        "distribuição."
    ),
}


@dataclass(frozen=True)
class Comportamento:
    """Um regime inteiro, na ordem em que a aba o apresenta."""

    regime: str
    perfil: str
    operadores: int
    resumo: str
    secoes: tuple[Teoria, ...]

    personificado: bool = False
    """Se é o regime que o atacante imita. Só ele ganha o slider de σ."""


@dataclass(frozen=True)
class Pagina:
    """Tudo que a aba mostra para uma semente."""

    quadro: pd.DataFrame
    gerais: tuple[Teoria, ...]
    comportamentos: tuple[Comportamento, ...]


# ── Escrita dos números ─────────────────────────────────────────────────


def hora_escrita(horas: float) -> str:
    """Hora fracionária do dia como `hh:mm`, truncada no minuto.

    Truncada, e não arredondada: 17h59min50s arredondado vira 18:00, que é a
    hora em que a janela fecha e em que nenhuma sessão abre.
    """
    minutos_no_dia = int(horas * 60)

    return f"{minutos_no_dia // 60:02d}:{minutos_no_dia % 60:02d}"


# ── Medições sobre o tráfego ────────────────────────────────────────────


def sessoes_do_trafego(requests: pd.DataFrame, operators: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por sessão: de quem, de que regime, quando abriu, quanto durou.

    A abertura é o instante do primeiro evento, que é o mesmo recorte que o M6
    usa para a janela horária habitual.
    """
    eventos = requests.assign(momento=pd.to_datetime(requests["timestamp"]))

    sessoes = eventos.groupby("session_id").agg(
        operator_id=("operator_id", "first"),
        abertura=("momento", "min"),
        fim=("momento", "max"),
        eventos=("momento", "size"),
        origem=("source_ip", "first"),
    )

    por_operador = operators.set_index("operator_id")
    principal = por_operador["usual_ips"].str.split("|").str[0]

    sessoes["regime"] = sessoes["operator_id"].map(por_operador["regime"])
    sessoes["minutos"] = (sessoes["fim"] - sessoes["abertura"]).dt.total_seconds() / 60
    sessoes["da_principal"] = sessoes["origem"] == sessoes["operator_id"].map(principal)

    return sessoes.reset_index()


def contagens_por_dia_util(
    sessoes: pd.DataFrame, operadores: pd.Series, dias_uteis: list
) -> pd.Series:
    """Sessões de cada operador em cada dia útil, **com os dias vazios**.

    Os dias sem sessão precisam entrar como zero. Sem eles a média sobe, a
    variância cai, e a contagem deixa de ser comparável com a Poisson ou a
    Pascal, que têm o zero no domínio.
    """
    dias = sessoes["abertura"].dt.date
    por_dia = sessoes.groupby([sessoes["operator_id"], dias]).size()

    grade = pd.MultiIndex.from_product([operadores, dias_uteis])

    return por_dia.reindex(grade, fill_value=0)


def frequencia_medida(valores: pd.Series, eixo: np.ndarray) -> np.ndarray:
    """Fração dos valores que cai em cada ponto do eixo."""
    return valores.value_counts(normalize=True).reindex(eixo, fill_value=0).to_numpy()


def horas_do_lote(ritmo: ScheduledRhythm) -> str:
    """As horas fixas do lote, escritas como `02h, 08h, 14h, 20h`."""
    horas = []

    for hora in ritmo.hours:
        horas.append(f"{hora:02d}h")

    return ", ".join(horas)


def descrever_ritmo(ritmo: ScheduledRhythm | ArrivalRhythm) -> tuple[str, str, str]:
    """Dias, horário e lei da contagem, como a tabela comparativa os mostra."""
    is_lote = isinstance(ritmo, ScheduledRhythm)

    if is_lote:
        return (
            "todos",
            f"{horas_do_lote(ritmo)} (±{ritmo.jitter_minutes} min)",
            f"fixa: {len(ritmo.hours)} por dia",
        )

    lei = "Poisson" if ritmo.dispersion is None else f"Pascal, n = {ritmo.dispersion}"
    media = com_virgula(ritmo.sessions_per_business_day)

    return (
        "úteis",
        f"{ritmo.opens_at:02d}h às {ritmo.closes_at:02d}h",
        f"{lei}, média {media} por dia",
    )


def quadro_comparativo(
    sessoes: pd.DataFrame, operators: pd.DataFrame, trafego: TrafficSpecification
) -> pd.DataFrame:
    """Os três regimes numa tabela: parâmetro de um lado, medição do outro."""
    linhas = []

    for nome in ORDEM:
        regime = REGIMES[nome]
        dias, horario, contagem = descrever_ritmo(regime.rhythm)
        menor, maior = regime.requests_range
        do_regime = sessoes[sessoes["regime"] == nome]
        operadores = operators[operators["regime"] == nome]

        linhas.append({
            "regime": nome,
            "perfil": operadores["profile"].iloc[0],
            "operadores": len(operadores),
            "dias": dias,
            "abre sessão": horario,
            "sessões": contagem,
            "requisições (típico)": f"{menor} a {maior}",
            "intervalo médio": f"{com_virgula(regime.seconds_between_requests)} s",
            "sessões medidas": len(do_regime),
            "duração mediana": f"{com_virgula(do_regime['minutos'].median())} min",
        })

    return pd.DataFrame(linhas)


def secao_do_que_e_regime() -> Teoria:
    """O vocabulário: o que o regime governa, e o que não."""
    return Teoria(
        titulo="Regime e perfil: quem governa o quê",
        texto=(
            "Todo operador tem um **perfil** e um **regime**, e cada coluna "
            "governa uma coisa. O **regime** decide o ritmo e o volume: em que "
            "dias e horas a sessão abre, quantas requisições ela tem e em que "
            "passo elas saem. O **perfil** decide a mistura de operações, isto "
            "é, que fração da sessão é `Decrypt`, `Encrypt`, `DescribeKey` ou "
            "`ExportKeyMaterial` (D-055).\n\n"
            "Hoje os dois andam um para um: todo `end_user` é `routine`, todo "
            "`automated_service` é `periodic_batch` e todo `administrator` é "
            "`occasional_custody`. O regime vem da coluna \"regime de "
            "exportação\" da Tabela 1 da proposta (D-007).\n\n"
            "Nenhum dos três é suspeito. Os três perfis são legítimos, e o que "
            "diz se uma sessão é maliciosa é a coluna `compromised` do "
            "`sessions.csv`, nunca o regime (D-092)."
        ),
    )


def secao_da_hora_do_dia(sessoes: pd.DataFrame) -> Teoria:
    """Em que hora cada regime abre sessão: picos contra platôs."""
    horas = sessoes["abertura"].dt.hour
    fracoes = pd.crosstab(horas, sessoes["regime"], normalize="columns")
    fracoes = fracoes.reindex(range(24), fill_value=0)[list(ORDEM)]

    dados = fracoes.rename_axis("hora").reset_index()

    return Teoria(
        titulo="Em que hora do dia cada regime abre sessão",
        texto=(
            "Cada barra é a fração das sessões **daquele regime** que abriu "
            "naquela hora, nas oito semanas desta semente. As três séries somam "
            "1 cada uma, então a altura compara formas, não volumes."
        ),
        dados=dados,
        rotulo_x="hora de abertura",
        rotulo_y="fração das sessões do regime",
        leitura=(
            "O `periodic_batch` é quatro espetos, um por hora de lote, e nada "
            "entre eles. Os outros dois são platôs: chegada espalhada por igual "
            "dentro do expediente, e nenhuma sessão fora dele. É a diferença "
            "entre máquina e pessoa, lida só pelo relógio."
        ),
    )


def secao_do_dia_da_semana(sessoes: pd.DataFrame) -> Teoria:
    """Em que dia da semana: o lote cobre os sete, a pessoa só os úteis."""
    dias = sessoes["abertura"].dt.weekday
    fracoes = pd.crosstab(dias, sessoes["regime"], normalize="columns")
    fracoes = fracoes.reindex(range(7), fill_value=0)[list(ORDEM)]
    fracoes.index = list(DIAS_DA_SEMANA)

    dados = fracoes.rename_axis("dia da semana").reset_index()

    return Teoria(
        titulo="Em que dia da semana",
        texto=(
            "Mesma leitura, por dia da semana. Pessoa não trabalha no fim de "
            "semana; o agendador não sabe que dia é."
        ),
        dados=dados,
        rotulo_x="dia da semana",
        rotulo_y="fração das sessões do regime",
        leitura=(
            "O `periodic_batch` é plano nos sete dias, porque roda o mesmo "
            "número de vezes todo dia. Os dois regimes de pessoa zeram no sábado "
            "e no domingo. A diferença entre os dias úteis é ruído do sorteio: "
            "nenhum regime trata segunda diferente de sexta."
        ),
    )


def secao_do_excesso(trafego: TrafficSpecification) -> Teoria:
    """Quantas requisições a mais a sessão ganha quando se estende (D-097).

    É a geométrica que `draw_request_count` sorteia, na mesma parametrização
    do numpy: `p = 1 / média`, com suporte a partir de 1. A curva é analítica,
    então descreve o gerador em vez de chamá-lo; o teste do viewer a compara
    com o sorteador de verdade.
    """
    chance = trafego.long_session_chance
    media = trafego.long_session_excess
    parada = 1.0 / media

    eixo = np.arange(1, int(4 * media) + 1)
    dados = pd.DataFrame({
        "requisições a mais": eixo,
        "chance": stats.geom.pmf(eixo, parada),
    })

    mediana = int(stats.geom.median(parada))

    return Teoria(
        titulo="Quanto a sessão cresce quando se estende",
        texto=(
            f"**Uma sessão em {1 / chance:.0f}** se estende; as outras ficam no "
            "tamanho típico. A que se estende ganha requisições a mais por uma "
            f"**geométrica** de média {com_virgula(media, 0)}: a cada requisição "
            f"extra, **1 chance em {com_virgula(media, 0)}** de parar ali.\n\n"
            "A mesma conta vale para as chaves distintas, sorteada à parte."
        ),
        dados=dados,
        rotulo_x="requisições a mais, na sessão que se estende",
        rotulo_y="chance",
        leitura=(
            "Ganhar pouco é o mais comum, e a chance cai um pouco a cada "
            f"requisição, sem um máximo fixo. Metade das sessões que se estendem "
            f"ganha até **{mediana}** a mais."
        ),
    )


# ── Quando a sessão abre ────────────────────────────────────────────────


def eixo_de_contagem(contagens: pd.Series) -> np.ndarray:
    """O eixo do grafico de sessoes por dia: de zero ao maior dia, pelo menos 8."""
    teto = max(8, int(contagens.max()))

    return np.arange(0, teto + 1)


def secao_contagem_poisson(ritmo: ArrivalRhythm, contagens: pd.Series) -> Teoria:
    """Quantas sessões por dia útil, no regime sem dispersão declarada."""
    media = ritmo.sessions_per_business_day
    eixo = eixo_de_contagem(contagens)
    teorica = stats.poisson.pmf(eixo, media)

    dados = pd.DataFrame({
        "sessões no dia": eixo,
        "Poisson": teorica,
        "medido nesta semente": frequencia_medida(contagens, eixo),
    })

    return Teoria(
        titulo="Quantas sessões por dia útil: Poisson",
        texto=(
            "O número de sessões de cada dia útil é sorteado de uma **Poisson** "
            f"de média **{com_virgula(media)}**. A Poisson é, por definição, a "
            "contagem de chegadas independentes a uma taxa constante, que é "
            "exatamente o trabalho em fluxo: o que aconteceu ontem não muda a "
            "chance de hoje."
        ),
        dados=dados,
        rotulo_x="sessões abertas num dia útil",
        rotulo_y="fração dos dias",
    )


def secao_contagem_pascal(ritmo: ArrivalRhythm, contagens: pd.Series) -> Teoria:
    """Quantas sessões por dia útil, no regime superdisperso."""
    media = ritmo.sessions_per_business_day
    dispersao = ritmo.dispersion
    sucesso = dispersao / (dispersao + media)
    eixo = eixo_de_contagem(contagens)

    variancia_teorica = media / sucesso
    pascal = stats.nbinom.pmf(eixo, dispersao, sucesso)
    poisson = stats.poisson.pmf(eixo, media)

    dados = pd.DataFrame({
        "sessões no dia": eixo,
        "Poisson de mesma média": poisson,
        f"Pascal, n = {dispersao}": pascal,
        "medido nesta semente": frequencia_medida(contagens, eixo),
    })

    return Teoria(
        titulo="Quantas sessões por dia útil: Pascal",
        texto=(
            "Pessoa em custódia não trabalha em fluxo. O trabalho chega em "
            "lote (uma auditoria, uma migração, um pedido de recuperação), e "
            "ela passa dias sem abrir sessão e depois abre várias numa tarde. "
            "Isso é **superdispersão**: a variância fica maior que a média, "
            "coisa que a Poisson não consegue produzir.\n\n"
            "A **binomial negativa**, ou Pascal, é a Poisson com um parâmetro a "
            "mais. Dá para lê-la como uma Poisson cuja taxa muda de um dia para "
            f"o outro, e o `n` controla quanto muda: quanto menor, mais "
            f"irregular. Com média **{com_virgula(media)}** e `n = {dispersao}`, "
            f"a variância teórica é **{com_virgula(variancia_teorica, 2)}**, ou "
            f"{com_virgula(variancia_teorica / media, 2)} vezes a média "
            "(D-071).\n\n"
            "A média é a que se escolheu; o outro parâmetro do sorteio, "
            f"`p = n / (n + m) = {com_virgula(sucesso, 2)}`, não é escolha: é o "
            "único valor que faz a média sair igual à pedida."
        ),
        dados=dados,
        rotulo_x="sessões abertas num dia útil",
        rotulo_y="fração dos dias",
    )


def descrever_janela(nome: str, ritmo: ScheduledRhythm | ArrivalRhythm) -> str:
    """A janela deste regime, numa frase.

    O texto usa o nome da entidade (Perfis históricos) e não o número do
    módulo: quem assiste a apresentação não sabe o que é M6.
    """
    is_lote = isinstance(ritmo, ScheduledRhythm)

    if is_lote:
        primeira = hora_escrita(min(ritmo.hours) - ritmo.jitter_minutes / 60)
        ultima = hora_escrita(max(ritmo.hours) + ritmo.jitter_minutes / 60)

        return (
            f"No `{nome}` a janela é larga: vai do primeiro lote menos o desvio "
            f"(perto de {primeira}) ao último mais o desvio (perto de {ultima})."
        )

    return (
        f"No `{nome}` a abertura cai uniforme entre {ritmo.opens_at:02d}h e "
        f"{ritmo.closes_at:02d}h, e a janela estreita é o que dá aos **Perfis "
        "históricos** uma faixa habitual para extrair."
    )


def secao_janela(nome: str, ritmo: ScheduledRhythm | ArrivalRhythm) -> Teoria:
    """A janela horária habitual deste regime."""
    return Teoria(titulo="A janela horária habitual", texto=descrever_janela(nome, ritmo))


def secao_processo_de_poisson(ritmo: ArrivalRhythm) -> Teoria:
    """Contagem Poisson e intervalo exponencial: o mesmo processo, em gancho."""
    return Teoria(
        titulo="Contagem e intervalo: dois jeitos de ver o mesmo processo",
        texto=(
            "Contar quantas sessões o dia tem (**Poisson**) e medir a espera "
            "entre uma e outra (**exponencial**) são dois jeitos de ver o mesmo "
            "processo. O gerador conta, e espalha as aberturas na janela.\n\n"
            "O `occasional_custody` não segue esse processo, e por isso é "
            "irregular."
        ),
    )


# ── Dentro da sessão ────────────────────────────────────────────────────


def secao_tamanho(
    regime: Regime, sessoes: pd.DataFrame, trafego: TrafficSpecification
) -> Teoria:
    """Quantas requisições: a faixa típica e a cauda (D-097), em gancho."""
    menor, maior = regime.requests_range
    eventos = sessoes["eventos"]

    eixo = np.arange(1, int(eventos.max()) + 1)
    dados = pd.DataFrame({
        "requisições na sessão": eixo,
        "medido nesta semente": frequencia_medida(eventos, eixo),
    })

    chance = trafego.long_session_chance
    excesso = trafego.long_session_excess

    return Teoria(
        titulo="Quantas requisições por sessão",
        texto=(
            f"Típico: uniforme entre **{menor} e {maior}**. **Uma sessão em "
            f"{1 / chance:.0f}** se estende, com excesso geométrico de média "
            f"{com_virgula(excesso, 0)}.\n\n"
            "A faixa **não é teto**: sem a cauda, o tamanho sozinho denunciava "
            "o atacante (D-097)."
        ),
        dados=dados,
        rotulo_x="requisições na sessão",
        rotulo_y="fração das sessões",
        leitura=f"Bloco plano de {menor} a {maior}, e uma cauda rala à direita.",
    )


LARGURAS_DE_FAIXA = (1, 2, 5, 10, 20, 30, 60)
FAIXAS_NO_GRAFICO = 20
# O grafico do intervalo cobre ate quatro medias em cerca de vinte barras, e a
# largura da faixa e a mais redonda perto disso: 10 s no routine, 20 s no
# occasional_custody, 1 s no periodic_batch.


def largura_de_faixa(media: float) -> int:
    """A largura redonda de faixa, em segundos, para ~20 barras até 4 médias."""
    alvo = 4 * media / FAIXAS_NO_GRAFICO

    return min(LARGURAS_DE_FAIXA, key=lambda largura: abs(largura - alvo))


def barras_da_exponencial(media: float) -> pd.DataFrame:
    """A chance de o intervalo cair em cada faixa de segundos.

    Analítica: a exponencial de média `m` põe `exp(-a/m) - exp(-b/m)` na faixa
    de `a` a `b`. O teste do viewer confere contra `request_instants`.
    """
    largura = largura_de_faixa(media)
    inicios = np.arange(0, 4 * media, largura)

    chance = np.exp(-inicios / media) - np.exp(-(inicios + largura) / media)

    return pd.DataFrame({"segundos até a próxima": inicios, "chance": chance})


def secao_intervalo(regime: Regime, sessoes: pd.DataFrame) -> Teoria:
    """O passo dentro da sessão, em barras, como o gráfico da geométrica.

    Até 28/09 era uma curva acumulada ("chance de o intervalo passar disto"),
    medida e teórica lado a lado, e não havia gráfico nenhum no
    `periodic_batch`, porque o log grava segundos inteiros. A barra por faixa
    se lê de uma vez e serve aos três regimes.
    """
    media = regime.seconds_between_requests
    largura = largura_de_faixa(media)
    mediana = media * np.log(2)
    casas = 0 if mediana >= 10 else 1

    return Teoria(
        titulo="Quando sai cada requisição",
        texto=(
            "Entre uma requisição e a seguinte passa um tempo sorteado de uma "
            f"**exponencial** de média **{com_virgula(media, 0 if media >= 10 else 1)} s**: "
            "intervalos curtos são os mais comuns, e de vez em quando vem uma "
            "pausa longa. É a versão em tempo contínuo da geométrica.\n\n"
            "A duração da sessão é a soma desses intervalos: nesta semente, "
            f"mediana de **{com_virgula(sessoes['minutos'].median())} min**."
        ),
        dados=barras_da_exponencial(media),
        rotulo_x=f"segundos até a próxima requisição (faixas de {largura} s)",
        rotulo_y="chance",
        leitura=(
            "A primeira barra é a mais alta: a próxima requisição costuma vir "
            "logo. Cada barra cai na mesma proporção da anterior, e metade dos "
            f"intervalos é menor que **{com_virgula(mediana, casas)} s**."
        ),
    )


def secao_mistura(perfil: str, eventos: pd.DataFrame) -> Teoria:
    """O que a sessão pede: a mistura do perfil, declarada e medida."""
    declarada = OPERATION_MIX[perfil]
    medida = eventos["operation"].value_counts(normalize=True).reindex(OPERATIONS, fill_value=0)

    dados = pd.DataFrame({
        "operação": list(OPERATIONS),
        "declarado (D-055)": declarada,
        "medido nesta semente": medida.to_numpy(),
    })

    maior_distancia = np.abs(np.array(declarada) - medida.to_numpy()).max()

    return Teoria(
        titulo="O que a sessão pede",
        texto=(
            f"A mistura de operações **não é do regime, é do perfil** "
            f"(`{perfil}`, D-055). Cada requisição sorteia a operação pela "
            "linha do perfil, e **nenhuma célula é zero**: operação que só um "
            "perfil fizesse marcaria o perfil por construção.\n\n"
            "As operações que mudariam o estado do repositório (criar, "
            "rotacionar, habilitar, desabilitar, apagar chave) ficaram de fora "
            "do tráfego: `keys.csv` tem um único módulo que o escreve."
        ),
        dados=dados,
        rotulo_x="operação",
        rotulo_y="fração das requisições",
        horizontal=True,
        leitura=(
            "As barras medidas repetem as declaradas: a maior distância entre "
            f"as duas, nesta semente, é de {com_virgula(maior_distancia * 100, 2)} "
            "ponto percentual."
        ),
    )


def hora_da_janela(janela: HourWindow) -> str:
    """Uma janela de horas como `09h às 19h`."""
    return f"{janela.opens_at:02d}h às {janela.closes_at:02d}h"


def faixa_escrita(faixa: tuple[int, int]) -> str:
    """Uma faixa inteira como `8 a 25`."""
    return f"{faixa[0]} a {faixa[1]}"


def linhas_do_atacante(stealth: Stealth) -> list[tuple[str, str]]:
    """Cada dimensão de σ e o valor dela neste `stealth`, já escrito."""
    madrugada = hora_da_janela(stealth.atypical_window)

    return [
        ("intervalo entre requisições", f"{com_virgula(stealth.seconds_between_requests)} s"),
        # "(típico)" porque as duas faixas tem cauda (D-097, D-098): lidas sem
        # o rotulo, 8 a 25 e 3 a 12 pareciam teto.
        ("requisições por sessão (típico)", faixa_escrita(stealth.requests_range)),
        ("chaves distintas (típico)", faixa_escrita(stealth.distinct_keys_range)),
        (f"chance de abrir fora do expediente ({madrugada}, qualquer dia)",
         porcento(stealth.atypical_hour_chance, 0)),
        ("chance de vir de origem inédita", porcento(stealth.novel_address_chance, 0)),
        ("requisições a escopo obsoleto", porcento(stealth.stale_scope_rate)),
        ("requisições a chave inexistente", porcento(stealth.absent_identifier_rate)),
    ]


def tabela_do_atacante(
    sigma: float, regime: Regime, trafego: TrafficSpecification, ataque: AttackSpecification
) -> pd.DataFrame:
    """As dimensões em σ 0,0, no σ escolhido e em σ 1,0, lado a lado.

    Os três vêm de `stealth_of`, a função que a campanha de ataque usa para
    resolver σ: a coluna do meio é o que o atacante faz de fato naquela
    condição, e não uma interpolação refeita aqui.
    """
    colunas = {
        "σ 0,0 (ostensivo)": 0.0,
        f"σ {com_virgula(sigma)} (escolhido)": sigma,
        "σ 1,0 (legítimo)": 1.0,
    }
    ostensivo = stealth_of(0.0, regime, trafego, ataque)
    dimensoes = []

    for nome, _ in linhas_do_atacante(ostensivo):
        dimensoes.append(nome)

    tabela = {"dimensão": dimensoes}

    for rotulo, valor in colunas.items():
        stealth = stealth_of(valor, regime, trafego, ataque)
        escritos = []

        for _, escrito in linhas_do_atacante(stealth):
            escritos.append(escrito)

        tabela[rotulo] = escritos

    return pd.DataFrame(tabela)


AMOSTRAS_DO_GRAFICO = 20_000
# Sessoes sorteadas de cada lado para o grafico de tamanho. Vinte mil bastam
# para a forma ficar lisa, e custam uma fracao de segundo.

SEMENTE_DO_GRAFICO = 20260927
# Fixa para o grafico nao tremer a cada interacao. Fora das replicas: estes
# sorteios nao pertencem ao experimento, so ilustram a tela.


def tamanhos_da_sessao(
    sigma: float, regime: Regime, trafego: TrafficSpecification, ataque: AttackSpecification
) -> pd.DataFrame:
    """Quantas requisições tem a sessão do atacante neste σ, e a legítima.

    As duas saem de `draw_request_count`, a função que a campanha e o tráfego
    legítimo usam de fato; o atacante com a faixa que `stealth_of` resolve
    para este σ, o legítimo com a do regime. Em σ 1,0 as duas faixas são a
    mesma, e as barras se sobrepõem.
    """
    rng = np.random.default_rng(SEMENTE_DO_GRAFICO)
    faixa_do_atacante = stealth_of(sigma, regime, trafego, ataque).requests_range

    def sortear(faixa: tuple[int, int]) -> pd.Series:
        tamanhos = []

        for _ in range(AMOSTRAS_DO_GRAFICO):
            tamanhos.append(draw_request_count(rng, faixa, trafego))

        return pd.Series(tamanhos)

    atacante = sortear(faixa_do_atacante)
    legitima = sortear(regime.requests_range)

    eixo = np.arange(1, int(max(atacante.max(), legitima.max())) + 1)

    return pd.DataFrame({
        "requisições na sessão": eixo,
        # A ordem das colunas decide a cor: a primeira pega o azul e a segunda
        # o vermelho-alaranjado. Legitimo em azul, atacante em vermelho.
        "sessão legítima": frequencia_medida(legitima, eixo),
        "sessão do atacante": frequencia_medida(atacante, eixo),
    })


def secao_atacante(
    regime: Regime,
    sessoes: pd.DataFrame,
    trafego: TrafficSpecification,
    ataque: AttackSpecification,
) -> Teoria:
    """As dimensões de σ, do ostensivo até este regime.

    Os valores vêm de `stealth_of`, a função que o M3 usa para resolver σ, e
    não de uma releitura dos parâmetros. Reescrever aqui quais são as
    dimensões e que forma cada uma tem foi o erro da primeira versão desta
    seção: ela mostrava o horário como faixa interpolada, quando ele é uma
    chance, e deixava a origem inédita de fora.
    """
    furtivo = stealth_of(1.0, regime, trafego, ataque)

    no_periodo = sessoes[belongs_to(EVALUATED, sessoes["abertura"])]
    legitimas = no_periodo.groupby("operator_id").size().median()

    return Teoria(
        titulo="O atacante, que personifica este regime",
        texto=(
            "O atacante não tem regime próprio. Ele age sob a credencial de um "
            "administrador, e cada dimensão do comportamento dele vai do valor "
            "**ostensivo**, em σ 0,0, ao valor **deste regime**, em σ 1,0, em "
            "linha reta entre os dois. Os valores de σ 1,0 não são copiados: a "
            "campanha de ataque os lê do **Scenario Engine**, então em σ 1,0 a "
            "sessão comprometida sai da "
            "mesma distribuição que uma legítima (D-082). Mova o σ abaixo para "
            "ver cada dimensão mudar.\n\n"
            "O horário e a origem são **chances**, e não valores: sessão "
            "fora do expediente ou origem inédita é uma marca que a sessão tem "
            "ou não tem, e o que σ regula é quantas sessões a carregam. Quando "
            "não carrega, a sessão abre em dia útil dentro da janela do "
            f"regime ({hora_da_janela(furtivo.usual_window)}) e vem de um dos "
            "endereços habituais, com a mesma escolha geométrica do tráfego "
            "legítimo.\n\n"
            "A mistura de operações **não está na tabela** porque não "
            "varia: o atacante sorteia sempre pela linha do `administrator`. "
            "Um perfil próprio de exfiltração carregaria sinal que σ não "
            "controla e continuaria detectável em σ 1,0.\n\n"
            f"A campanha tem **{ataque.campaign_sessions} sessões** nas semanas "
            "5 a 8, fixas em todas as condições de σ (D-081). O tamanho foi "
            "medido contra este regime: nesta semente, cada administrador tem "
            f"mediana de **{legitimas:.0f}** sessões legítimas no mesmo período."
        ),
    )


# ── Os métodos de sorteio de cada regime ────────────────────────────────


class Metodo(NamedTuple):
    """Um sorteio do gerador: o que decide, com que lei, e onde."""

    etapa: str
    metodo: str
    parametro: str
    funcao: str


def metodos_do_lote(ritmo: ScheduledRhythm) -> list[Metodo]:
    """O calendário do `periodic_batch`: nada é sorteado, só o desvio."""
    horas = horas_do_lote(ritmo)

    return [
        Metodo("quantas sessões no dia", "nenhum: contagem fixa",
               f"{len(ritmo.hours)} por dia, todos os dias", "scheduled_starts"),
        Metodo("em que instante", "hora fixa + **uniforme** contínua",
               f"{horas}, desvio de ±{ritmo.jitter_minutes} min", "scheduled_starts"),
    ]


def metodo_do_instante(ritmo: ArrivalRhythm) -> Metodo:
    """Em que instante do dia a pessoa abre sessão."""
    return Metodo("em que instante", "**uniforme** contínua na janela",
                  f"{ritmo.opens_at:02d}h às {ritmo.closes_at:02d}h", "arrival_starts")


def metodos_da_rotina(ritmo: ArrivalRhythm) -> list[Metodo]:
    """O calendário do `routine`: Poisson e instante uniforme."""
    media = com_virgula(ritmo.sessions_per_business_day)

    return [
        Metodo("quantas sessões no dia útil", "**Poisson**",
               f"média {media}", "daily_session_count"),
        metodo_do_instante(ritmo),
    ]


def metodos_da_custodia(ritmo: ArrivalRhythm) -> list[Metodo]:
    """O calendário do `occasional_custody`: Pascal e instante uniforme."""
    media = com_virgula(ritmo.sessions_per_business_day)

    return [
        Metodo("quantas sessões no dia útil", "**Pascal** (binomial negativa)",
               f"média {media}, n = {ritmo.dispersion}", "daily_session_count"),
        metodo_do_instante(ritmo),
    ]


def metodos_do_calendario(ritmo: ScheduledRhythm | ArrivalRhythm) -> list[Metodo]:
    """Os sorteios que situam a sessão no calendário, conforme o ritmo."""
    is_lote = isinstance(ritmo, ScheduledRhythm)

    if is_lote:
        return metodos_do_lote(ritmo)

    is_superdisperso = ritmo.dispersion is not None

    if is_superdisperso:
        return metodos_da_custodia(ritmo)

    return metodos_da_rotina(ritmo)


def metodos_da_sessao_do_regime(regime: Regime, perfil: str) -> list[Metodo]:
    """Os sorteios dentro da sessão cujo valor muda de um regime para outro."""
    menor, maior = regime.requests_range

    return [
        Metodo("quantas requisições (típico)", "**uniforme** inteira",
               f"{menor} a {maior}", "draw_request_count"),
        Metodo("quando sai cada requisição", "**exponencial**",
               f"média {com_virgula(regime.seconds_between_requests)} s",
               "request_instants"),
        Metodo("que operação", "**categórica**, uma por requisição",
               f"mistura do perfil `{perfil}`", "draw_operations"),
    ]


def metodos_comuns(trafego: TrafficSpecification) -> list[Metodo]:
    """Os sorteios iguais nos três regimes, com os mesmos valores.

    Todos leem da `TrafficSpecification` ou de parâmetro global, e nada do
    regime: por isso não mudam de uma aba para outra, e ficam na visão geral.
    """
    chaves_menor, chaves_maior = trafego.distinct_keys_range
    chance = porcento(trafego.long_session_chance, 0)
    excesso = com_virgula(trafego.long_session_excess, 0)
    obsoleto = porcento(trafego.stale_scope_rate)
    inexistente = porcento(trafego.absent_identifier_rate)

    return [
        Metodo("de que origem de rede", "**geométrica** truncada",
               f"principal com {porcento(PRIMARY_ADDRESS_SHARE, 0)}",
               "choose_source_address"),
        Metodo("a sessão se estende?", "**Bernoulli**, e se sim **geométrica**",
               f"chance {chance}, excesso de média {excesso}", "draw_request_count"),
        Metodo("quantas chaves distintas", "**uniforme** inteira, com a mesma cauda",
               f"{chaves_menor} a {chaves_maior}, até o que couber",
               "distinct_key_count"),
        Metodo("quais chaves", "sorteio **sem reposição** no alcance",
               "depois repetição uniforme e permutação", "session_targets"),
        Metodo("a requisição falha?", "**Bernoulli** por requisição",
               f"escopo obsoleto {obsoleto}, chave inexistente {inexistente}",
               "apply_deviations"),
    ]


def tabela_de_metodos(metodos: list[Metodo], coluna_do_valor: str) -> str:
    """Os sorteios numa tabela markdown, numerados."""
    cabecalho = (
        f"| # | o que se decide | método | {coluna_do_valor} | função |\n"
        "|---|---|---|---|---|\n"
    )
    linhas = []

    for numero, metodo in enumerate(metodos, start=1):
        linhas.append(
            f"| {numero} | {metodo.etapa} | {metodo.metodo} | {metodo.parametro} "
            f"| `{metodo.funcao}` |\n"
        )

    return cabecalho + "".join(linhas)


def secao_metodos(regime: Regime, perfil: str) -> Teoria:
    """Os sorteios que mudam de um regime para outro, numa tabela.

    Até 27/09 a tabela trazia os dez sorteios, e cinco deles eram idênticos
    nas três abas. Os idênticos foram para a visão geral
    (`secao_metodos_comuns`), e aqui fica só o que distingue o regime.
    """
    metodos = metodos_do_calendario(regime.rhythm) + metodos_da_sessao_do_regime(regime, perfil)

    return Teoria(
        titulo="Os métodos deste regime",
        texto=(
            "Só os sorteios que **mudam** de um regime para outro. Os que são "
            "iguais nos três estão na **Visão geral**.\n\n"
            + tabela_de_metodos(metodos, "neste regime")
        ),
    )


def secao_metodos_comuns(trafego: TrafficSpecification) -> Teoria:
    """Os sorteios iguais nos três regimes, numa tabela só."""
    return Teoria(
        titulo="Os métodos comuns aos três regimes",
        texto=(
            "Estes sorteios são **iguais nos três regimes**, com os mesmos "
            "valores. O que muda de um regime para outro está na aba de cada "
            "um.\n\n"
            + tabela_de_metodos(metodos_comuns(trafego), "valor")
        ),
    )


# ── Montagem ────────────────────────────────────────────────────────────


def secoes_do_ritmo(
    nome: str, regime: Regime, sessoes: pd.DataFrame, operadores: pd.Series,
    trafego: TrafficSpecification,
) -> list[Teoria]:
    """As seções de "quando abre", que dependem do tipo de ritmo."""
    ritmo = regime.rhythm
    dias = simulated_days(trafego)

    is_lote = isinstance(ritmo, ScheduledRhythm)

    # O lote nao ganha secao de calendario: horas e desvio ja estao na tabela
    # de metodos, e o grafico do desvio, plano de ponta a ponta, so repetia
    # o que o numero diz.
    if is_lote:
        return [secao_janela(nome, ritmo)]

    contagens = contagens_por_dia_util(sessoes, operadores, business_days_among(dias))
    is_superdisperso = ritmo.dispersion is not None

    if is_superdisperso:
        return [secao_contagem_pascal(ritmo, contagens), secao_janela(nome, ritmo)]

    return [
        secao_contagem_poisson(ritmo, contagens),
        secao_processo_de_poisson(ritmo),
        secao_janela(nome, ritmo),
    ]


def montar_comportamento(
    nome: str,
    requests: pd.DataFrame,
    operators: pd.DataFrame,
    sessoes: pd.DataFrame,
    especificacoes: tuple[TrafficSpecification, AttackSpecification],
) -> Comportamento:
    """Um regime, das sessões no calendário até o que acontece dentro delas."""
    trafego, ataque = especificacoes
    regime = REGIMES[nome]

    operadores = operators[operators["regime"] == nome]
    perfil = str(operadores["profile"].iloc[0])
    do_regime = sessoes[sessoes["regime"] == nome]
    eventos = requests[requests["operator_id"].isin(operadores["operator_id"])]

    secoes = [secao_metodos(regime, perfil)]
    secoes += secoes_do_ritmo(nome, regime, do_regime, operadores["operator_id"], trafego)
    secoes += [
        secao_tamanho(regime, do_regime, trafego),
        secao_intervalo(regime, do_regime),
        secao_mistura(perfil, eventos),
    ]

    is_personificado = perfil == "administrator"

    if is_personificado:
        secoes.append(secao_atacante(regime, do_regime, trafego, ataque))

    return Comportamento(
        nome, perfil, len(operadores), RESUMOS[nome], tuple(secoes), is_personificado
    )


def montar_pagina(
    requests: pd.DataFrame,
    operators: pd.DataFrame,
    trafego: TrafficSpecification,
    ataque: AttackSpecification,
) -> Pagina:
    """A aba inteira, para o tráfego de uma semente."""
    sessoes = sessoes_do_trafego(requests, operators)

    comportamentos = []

    for nome in ORDEM:
        comportamentos.append(
            montar_comportamento(nome, requests, operators, sessoes, (trafego, ataque))
        )

    # A origem de rede fecha a visao geral, e nao cada aba de regime: os tres
    # escolhem o endereco pela mesma geometrica, so muda quantos ele tem.
    mais_enderecos = 0

    for perfil in PROFILES:
        mais_enderecos = max(mais_enderecos, perfil.addresses_range[1])

    gerais = (
        secao_do_que_e_regime(),
        secao_metodos_comuns(trafego),
        secao_da_hora_do_dia(sessoes),
        secao_do_dia_da_semana(sessoes),
        teoria_da_geometrica(PRIMARY_ADDRESS_SHARE, mais_enderecos),
        secao_do_excesso(trafego),
    )

    return Pagina(
        quadro_comparativo(sessoes, operators, trafego), gerais, tuple(comportamentos)
    )
