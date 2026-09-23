"""Visualizador do pipeline: um passo do diagrama por vez.

    streamlit run src/viewer/app.py

Nao e simulacao nem mock. Cada quadro chama as funcoes de verdade do modulo,
com a semente escolhida, e mostra o que entrou e o que saiu.

A unidade da tela e o **passo do diagrama**, nao a funcao. Quem esta
conhecendo o trabalho precisa ver que dados entraram, que entidade os
processou e que dados sairam — `scope_pool` e `holders_by_scope` sao
granularidade de implementacao e ficam guardados atras de "por dentro", para
quem quiser.
"""

from __future__ import annotations

import sys
from pathlib import Path

# O streamlit poe a pasta do script no sys.path, nao a raiz do projeto, entao
# `import src...` falharia. O pytest resolve isso pelo pythonpath do
# pyproject.toml, que so vale para ele. Aqui a raiz entra a mao, antes dos
# imports do projeto.
RAIZ_DO_PROJETO = Path(__file__).resolve().parents[2]

if str(RAIZ_DO_PROJETO) not in sys.path:
    sys.path.insert(0, str(RAIZ_DO_PROJETO))

import pandas as pd
import streamlit as st

from src.attack.build import build_attack
from src.attack.parameters import AttackSpecification
from src.audit_logger.build import build_log
from src.calibration.build import build_thresholds, calibration_period
from src.dataset.build import build_dataset
from src.globals.experiment import SIGMAS
from src.globals.phases import EVALUATED, WARMUP
from src.historical_profiles.build import build_profiles
from src.kms.build import build_outcomes
from src.population.build import build_population
from src.population.parameters import KeyRepositorySpecification
from src.traffic.build import build_traffic
from src.traffic.parameters import TrafficSpecification
from src.viewer.decisions import como_tabela
from src.viewer.frames import (
    Painel,
    Quadro,
    frames_da_fase_1,
    frames_da_fase_2,
    frames_da_fase_3,
)
from src.viewer.theory import Teoria
from src.viewer.steps import (
    FASES,
    Fase,
    Step,
    attack_steps,
    population_steps,
    traffic_steps,
    warmup_steps,
)

TITULO = "Pipeline em tres fases"

# ATENCAO: nada de docstring de constante neste arquivo. O streamlit tem
# "magic" — uma string solta no nivel do modulo e renderizada como conteudo
# da pagina, e as explicacoes apareciam acima do titulo. Aqui, comentario.

# Altura em pixels das tabelas, para caberem duas lado a lado sem rolar a
# pagina. As tabelas sao mostradas INTEIRAS: o `st.dataframe` virtualiza as
# linhas, entao 66 mil custam o mesmo que seis, e e mostrar tudo que permite
# ordenar por uma coluna e procurar um operador para explicar um conceito.
ALTURA_DA_TABELA = 320

# Quantas linhas os passos do "por dentro" mostram. Ali a amostra basta: o
# que se quer ver e a forma do resultado de cada funcao, e o arquivo inteiro
# ja esta no painel de saida do quadro.
LINHAS_NO_DETALHE = 12

# Altura do expansor das variaveis de decisao. Maior que a das tabelas
# porque a coluna de significado e longa e quebra em varias linhas.
ALTURA_DAS_VARIAVEIS = 420

# Altura dos graficos de teoria. Baixa: eles ilustram uma forma, nao
# servem para ler valor — quem quiser o numero passa o mouse.
ALTURA_DO_GRAFICO = 260

# Os tres primeiros slots da paleta categorica de referencia, um par por
# tema. Nao e a mesma cor clareada: sao passos escolhidos para cada fundo e
# verificados como conjunto. A versao clara reprova sobre fundo escuro, na
# faixa de luminosidade, entao usar uma so nos dois modos nao era opcao.
# A ORDEM e o mecanismo de seguranca para daltonismo: nao se embaralha, e
# um quarto slot exigiria rodar a verificacao de novo.
SLOTS_CLARO = ("#2a78d6", "#eb6834", "#1baf7a")
SLOTS_ESCURO = ("#3987e5", "#d95926", "#199e70")


@st.cache_data(show_spinner="rodando o M1...")
def tabelas(seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    populacao = build_population(seed, KeyRepositorySpecification())

    return populacao.operators, populacao.keys


@st.cache_data(show_spinner="rodando o M2...")
def trafego(seed: int) -> pd.DataFrame:
    operators, keys = tabelas(seed)

    return build_traffic(seed, operators, keys, TrafficSpecification())


@st.cache_data(show_spinner="rodando o aquecimento...")
def aquecimento(seed: int) -> dict[str, pd.DataFrame]:
    """Tudo que a fase 2 produz, de uma vez.

    Num dicionario porque os quadros da fase 2 e da fase 3 consomem pedacos
    diferentes, e recalcular o perfil na fase 3 violaria a D-044 — que manda
    usar o mesmo, nunca um recalculado.
    """
    operators, keys = tabelas(seed)
    requests = trafego(seed)

    outcomes = build_outcomes(requests, keys, operators, WARMUP)
    log = build_log(requests, outcomes, WARMUP)
    perfis = build_profiles(log)
    sessoes = build_dataset(log, perfis, WARMUP)

    return {
        "outcomes": outcomes,
        "log": log,
        "perfis": perfis,
        "sessoes": sessoes,
        "semana_3": calibration_period(sessoes),
        "limiares": build_thresholds(sessoes),
    }


@st.cache_data(show_spinner="rodando a campanha...")
def avaliado(seed: int, sigma: float) -> dict[str, pd.DataFrame]:
    operators, keys = tabelas(seed)
    perfis = aquecimento(seed)["perfis"]

    campanha = build_attack(
        seed, sigma, operators, keys, trafego(seed),
        TrafficSpecification(), AttackSpecification(),
    )
    outcomes = build_outcomes(campanha.requests, keys, operators, EVALUATED)
    log = build_log(campanha.requests, outcomes, EVALUATED)

    return {
        "requests": campanha.requests,
        "compromised": campanha.compromised,
        "run": campanha.run,
        "log": log,
        "sessoes": build_dataset(log, perfis, EVALUATED, campanha.compromised),
    }


@st.cache_data(show_spinner="detalhando por funcao...")
def detalhes_da_fase(fase_nome: str, seed: int, foco: str, sigma: float) -> list[Step]:
    """Os passos por funcao, que cada quadro guarda sob demanda."""
    operators, keys = tabelas(seed)

    if fase_nome == FASES[0].nome:
        return population_steps(seed, KeyRepositorySpecification()) + traffic_steps(
            seed, operators, keys, TrafficSpecification(), foco
        )

    if fase_nome == FASES[1].nome:
        return warmup_steps(seed, operators, keys, trafego(seed), foco)

    return attack_steps(
        seed, sigma, operators, keys, trafego(seed), aquecimento(seed)["perfis"]
    )


def quadros_da_fase(fase: Fase, seed: int, foco: str, sigma: float) -> list[Quadro]:
    """Os quadros daquela fase, ja com os dados reais dentro."""
    operators, keys = tabelas(seed)
    detalhes = detalhes_da_fase(fase.nome, seed, foco, sigma)

    if fase.nome == FASES[0].nome:
        return frames_da_fase_1(seed, operators, keys, trafego(seed), detalhes)

    warmup = aquecimento(seed)

    if fase.nome == FASES[1].nome:
        return frames_da_fase_2(
            trafego(seed), operators, keys,
            warmup["outcomes"], warmup["log"], warmup["perfis"],
            warmup["sessoes"], warmup["semana_3"], warmup["limiares"],
            detalhes,
        )

    campanha = avaliado(seed, sigma)

    return frames_da_fase_3(
        sigma, trafego(seed), warmup["perfis"],
        campanha["requests"], campanha["compromised"], campanha["run"],
        campanha["log"], campanha["sessoes"], detalhes,
    )


def mostrar_painel(painel: Painel) -> None:
    """Um arquivo ou parametro, com previa quando e tabela."""
    tamanho = painel.tamanho()
    rodape = f"  ·  {tamanho}" if tamanho else ""

    st.markdown(f"**`{painel.nome}`**{rodape}")

    if painel.legenda:
        st.caption(painel.legenda)

    is_tabela = isinstance(painel.dado, pd.DataFrame)

    if is_tabela:
        st.dataframe(painel.dado, use_container_width=True, hide_index=True,
                     height=ALTURA_DA_TABELA)
        return

    is_lista = isinstance(painel.dado, (list, tuple))

    if is_lista:
        st.code("\n".join(str(item) for item in painel.dado))
        return

    st.code(str(painel.dado))


def mostrar_lado(titulo: str, paineis: tuple[Painel, ...], vazio: str) -> None:
    """As entradas ou as saidas, empilhadas e em largura cheia.

    Em largura cheia, e nao em duas colunas lado a lado, porque as tabelas tem
    muitas colunas — `log.csv` tem oito, `sessions.csv` tem doze — e metade da
    tela cortava as ultimas. Quem precisa comparar entrada com saida rola a
    pagina; quem precisa filtrar uma coluna nao consegue se ela estiver
    escondida.
    """
    st.subheader(titulo)

    if not paineis:
        st.caption(vazio)
        return

    for painel in paineis:
        mostrar_painel(painel)


def mostrar_variaveis(quadro: Quadro) -> None:
    """As variaveis de decisao, num expansor acima das tabelas.

    Vem antes da entrada e da saida porque respondem a pergunta anterior: nao
    que dado e este, mas **por que ele e assim**. Cada linha traz a entrada do
    registro que fixou o valor, que e onde esta escrito o porque.
    """
    if not quadro.variaveis:
        return

    rotulo = f"Variáveis de decisão: {len(quadro.variaveis)} que regem esta entidade"

    with st.expander(rotulo):
        st.dataframe(
            como_tabela(quadro.variaveis),
            use_container_width=True, hide_index=True,
            height=ALTURA_DAS_VARIAVEIS,
            column_config={
                "variável": st.column_config.TextColumn(width="medium"),
                "valor": st.column_config.TextColumn(width="medium"),
                "o que é": st.column_config.TextColumn(width="large"),
            },
        )

        for variavel in quadro.variaveis:
            if variavel.teoria:
                mostrar_teoria(variavel.nome, variavel.teoria)


def slots_do_tema() -> tuple[str, ...]:
    """Os passos da paleta que servem ao tema em vigor.

    O streamlit resolve o tema no navegador, entao ele so e conhecido em tempo
    de execucao. Faltando a informacao, o claro e o padrao — e o padrao do
    proprio streamlit.
    """
    is_escuro = st.context.theme.get("type") == "dark"

    return SLOTS_ESCURO if is_escuro else SLOTS_CLARO


def mostrar_grafico(teoria: Teoria) -> None:
    """O grafico da distribuicao, na forma que o dominio pede.

    Barras para dominio discreto — nao existe meia sessao —, linha para
    continuo. Trocar os dois faria o grafico afirmar algo falso sobre o dado:
    linha entre inteiros sugere que os valores entre eles existem.
    """
    eixo_x = teoria.dados.columns[0]
    series = list(teoria.dados.columns[1:])

    comum = {
        "data": teoria.dados, "x": eixo_x, "y": series,
        "color": list(slots_do_tema()[:len(series)]),
        "x_label": teoria.rotulo_x, "y_label": teoria.rotulo_y,
        "height": ALTURA_DO_GRAFICO, "use_container_width": True,
    }
    is_discreto = teoria.forma == "barras"

    if is_discreto:
        st.bar_chart(stack=False, **comum)
        return

    st.line_chart(**comum)


def mostrar_teoria(nome: str, teoria: Teoria) -> None:
    """A teoria de uma variavel, no expansor dela.

    Aninhado dentro do expansor das variaveis, entao usa um `popover`: o
    streamlit nao permite expansor dentro de expansor, e o popover da o mesmo
    'clique para abrir' sem essa restricao.
    """
    with st.popover(f"Teoria · {nome}", use_container_width=True):
        st.markdown(f"### {teoria.titulo}")
        st.markdown(teoria.texto)

        mostrar_grafico(teoria)

        if teoria.leitura:
            st.caption(f"**O que ler no gráfico.** {teoria.leitura}")


def mostrar_detalhes(quadro: Quadro) -> None:
    """Os passos por funcao, fechados por padrao.

    Fechados porque sao granularidade de implementacao: quem esta conhecendo
    o trabalho precisa primeiro do que entrou e do que saiu.
    """
    if not quadro.detalhes:
        return

    rotulo = f"Por dentro: {len(quadro.detalhes)} chamadas de funcao"

    with st.expander(rotulo):
        for passo in quadro.detalhes:
            st.markdown(f"**`{passo.funcao}`**")
            st.caption(passo.legenda)
            st.markdown(passo.explicacao)

            is_tabela = isinstance(passo.saida, pd.DataFrame)

            if is_tabela:
                st.dataframe(passo.saida.head(LINHAS_NO_DETALHE),
                             use_container_width=True, hide_index=True)
            else:
                st.code(str(passo.saida)[:600])

            st.divider()


def mostrar_quadro(quadro: Quadro, total: int, titular: bool) -> None:
    """Entrada a esquerda, entidade no meio, saida a direita.

    `titular` diz se este quadro leva o titulo grande da pagina. So o primeiro
    da tela leva: quando os anteriores ficam empilhados, os de cima viram
    secoes.
    """
    if titular:
        st.title(quadro.entidade)
    else:
        st.markdown(f"## {quadro.entidade}")

    st.caption(
        f"Passo {quadro.numero} de {total}  ·  `{quadro.modulos}`  ·  {quadro.fase}"
    )

    st.markdown(quadro.resumo)

    if quadro.pendente:
        st.warning(quadro.pendente, icon=":material/schedule:")

    if quadro.porque:
        with st.container(border=True):
            st.markdown("**Por que e assim**")
            st.markdown(quadro.porque)

    mostrar_variaveis(quadro)

    mostrar_lado("Entra", quadro.entradas, "Nada: esta entidade abre a sequência.")
    mostrar_lado("Sai", quadro.saidas, "Nada ainda: a entidade não foi escrita.")

    mostrar_detalhes(quadro)


def escolher_fase() -> Fase:
    """Seletor das tres fases da arquitetura, no topo da barra lateral."""
    st.sidebar.title("Fase")

    rotulos = [
        f"{fase.nome}  (parcial)" if fase.pendencia else fase.nome
        for fase in FASES
    ]
    escolhido = st.sidebar.radio("Fase", rotulos, index=0, label_visibility="collapsed")

    return FASES[rotulos.index(escolhido)]


def barra_lateral(fase: Fase) -> tuple[int, str, float]:
    """Semente, operador em foco e — so na fase 3 — sigma."""
    st.sidebar.divider()
    st.sidebar.subheader("Controles")

    seed = int(st.sidebar.number_input("Semente", min_value=1, max_value=30, value=1))

    operators, _ = tabelas(seed)
    nomes = sorted(operators["operator_id"])
    padrao = nomes.index("admin_02") if "admin_02" in nomes else 0
    foco = str(st.sidebar.selectbox("Operador em foco", nomes, index=padrao))

    st.sidebar.caption(
        "O operador em foco so afeta o 'por dentro' de cada quadro, que segue "
        "um sujeito so. Os arquivos sao sempre os inteiros."
    )

    sigma = 0.5

    if fase.nome == FASES[2].nome:
        sigma = float(st.sidebar.select_slider(
            "Furtividade σ", options=list(SIGMAS), value=0.5,
            help="0,0 e ostensivo; 1,0 e indistinguivel de uma sessao legitima.",
        ))

    st.sidebar.caption(
        "A semente determina tudo que nao e sigma, e a mesma semente sempre "
        "devolve os mesmos dados."
    )

    return seed, foco, sigma


# Chave do seletor de passo, que e a unica fonte da verdade da navegacao. Os
# botoes nao guardam estado proprio: escrevem nesta chave, e o seletor a le.
# Com duas fontes — um contador e um seletor — elas se dessincronizavam, e
# era dai que vinha o botao que sumia.
PASSO_ESCOLHIDO = "passo_escolhido"


def rotulo_do_passo(quadro: Quadro, total: int) -> str:
    """Como o passo aparece no seletor: numero, total e entidade."""
    return f"{quadro.numero} de {total}  ·  {quadro.entidade}"


def andar(rotulos: list[str], passos: int) -> None:
    """Move a selecao, sem deixar sair da faixa.

    Callback de `on_click`, e nao codigo no corpo da funcao, de proposito. O
    callback roda **antes** do redesenho, entao o seletor e os botoes ja veem
    o valor novo. Incrementando no corpo, o botao era desenhado com o valor
    velho e so se corrigia no clique seguinte — que era o bug de o `Avancar`
    aparecer desabilitado depois de voltar.
    """
    atual = rotulos.index(st.session_state[PASSO_ESCOLHIDO])
    destino = min(max(atual + passos, 0), len(rotulos) - 1)

    st.session_state[PASSO_ESCOLHIDO] = rotulos[destino]


def navegacao(quadros: list[Quadro]) -> tuple[int, bool]:
    """Seletor de passo, botoes e progresso, na barra lateral."""
    total = len(quadros)
    rotulos = [rotulo_do_passo(quadro, total) for quadro in quadros]

    st.sidebar.divider()
    st.sidebar.subheader("Passo")

    # Trocar de fase muda a lista inteira, e o rotulo guardado deixa de
    # existir. Sem isto o seletor levantaria erro em vez de recomecar.
    is_desconhecido = st.session_state.get(PASSO_ESCOLHIDO) not in rotulos

    if is_desconhecido:
        st.session_state[PASSO_ESCOLHIDO] = rotulos[0]

    escolhido = st.sidebar.selectbox(
        "Ir para", rotulos, key=PASSO_ESCOLHIDO, label_visibility="collapsed"
    )
    atual = rotulos.index(escolhido) + 1

    atras, adiante = st.sidebar.columns(2)

    atras.button(
        "◀ Voltar", key="voltar", use_container_width=True,
        disabled=atual <= 1, on_click=andar, args=(rotulos, -1),
    )
    adiante.button(
        "Avancar ▶", key="avancar", use_container_width=True,
        disabled=atual >= total, on_click=andar, args=(rotulos, +1),
    )

    st.sidebar.progress(atual / total, text=f"Passo {atual} de {total}")

    quadro = quadros[atual - 1]
    st.sidebar.caption(f"**{quadro.entidade}**  ·  `{quadro.modulos}`")

    empilhar = st.sidebar.checkbox("Manter os anteriores na tela", value=False)

    return atual, empilhar


def descrever_fase(fase: Fase) -> None:
    """A fase e o objetivo dela, no topo da pagina, antes dos quadros.

    Estiveram na barra lateral por uma versao, e foi erro: sem saber o que a
    fase 2 quer, "o KMS decide o desfecho e o Audit Logger registra" e uma
    frase sobre encanamento. Com o objetivo a vista, o mesmo passo vira um meio
    para alguma coisa.

    Aparece **uma vez por tela**, e nao dentro de cada quadro, porque e
    contexto da fase inteira. O quadro traz o titulo da entidade, que e o que
    muda de passo para passo.
    """
    st.caption(f"{fase.nome.upper()}   ·   `{fase.modulos}`   ·   {fase.execucoes}")

    with st.container(border=True):
        st.markdown(f"**Objetivo.** {fase.objetivo}")

    if fase.pendencia:
        st.info(fase.pendencia, icon=":material/schedule:")

    with st.expander("Mais sobre esta fase"):
        st.markdown(fase.descricao)

    st.divider()


def main() -> None:
    st.set_page_config(page_title=TITULO, layout="wide")

    fase = escolher_fase()
    seed, foco, sigma = barra_lateral(fase)

    quadros = quadros_da_fase(fase, seed, foco, sigma)
    atual, empilhar = navegacao(quadros)

    descrever_fase(fase)

    primeiro = 1 if empilhar else atual

    for numero in range(primeiro, atual + 1):
        mostrar_quadro(quadros[numero - 1], len(quadros), numero == primeiro)

        if numero < atual:
            st.divider()


main()
