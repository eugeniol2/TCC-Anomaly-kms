"""Visualizador do pipeline: um passo do diagrama por vez.

    streamlit run src/viewer/app.py

Nao e simulacao nem mock. Cada quadro chama as funcoes de verdade do modulo,
com a semente escolhida, e mostra o que entrou e o que saiu.

A unidade da tela e o **passo do diagrama**, nao a funcao. Quem esta
conhecendo o trabalho precisa ver que dados entraram, que entidade os
processou e que dados sairam: `scope_pool` e `split_keys_by_scope` sao
granularidade de implementacao e ficam guardados atras de "por dentro", para
quem quiser.

A pagina **Comportamentos**, no topo, trata os tres regimes como assunto
proprio (`behaviors.py`). Eles aparecem aos pedacos nas variaveis de decisao
de cada quadro, e o que cada um e so se entende visto inteiro.
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
from src.calibration.build import build_thresholds
from src.dataset.build import build_dataset
from src.globals.experiment import SIGMAS
from src.globals.phases import EVALUATED, WARMUP
from src.historical_profiles.build import build_profiles
from src.kms.build import build_outcomes
from src.partition.build import Partition, build_partition
from src.population.build import build_population
from src.population.parameters import KeyRepositorySpecification
from src.traffic.build import build_traffic
from src.traffic.parameters import TrafficSpecification
from src.traffic.regimes import REGIMES
from src.viewer.behaviors import (
    Comportamento,
    Pagina,
    montar_pagina,
    tabela_do_atacante,
    tamanhos_da_sessao,
)
from src.viewer.decisions import como_tabela
from src.viewer.frames import (
    Painel,
    Quadro,
    frames_da_fase_1,
    frames_da_fase_2,
    frames_da_fase_3,
)
from src.viewer.formatting import porcento
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
# "magic": uma string solta no nivel do modulo e renderizada como conteudo
# da pagina, e as explicacoes apareciam acima do titulo. Aqui, comentario.

# Altura em pixels das tabelas, para caberem duas lado a lado sem rolar a
# pagina. As tabelas sao mostradas INTEIRAS: o `st.dataframe` virtualiza as
# linhas, entao dezenas de milhares custam o mesmo que seis, e e mostrar tudo
# ordenar por uma coluna e procurar um operador para explicar um conceito.
ALTURA_DA_TABELA = 320

# Quantas linhas os passos do "por dentro" mostram. Ali a amostra basta: o
# que se quer ver e a forma do resultado de cada funcao, e o arquivo inteiro
# ja esta no painel de saida do quadro.
LINHAS_NO_DETALHE = 12

# Altura do expansor das variaveis de decisao. Maior que a das tabelas
# porque a coluna de significado e longa e quebra em varias linhas.
TODOS = "todos"
EXEMPLO_DE_FILTRO = 'outcome != "success"'

ALTURA_DAS_VARIAVEIS = 420

# Altura dos graficos de teoria. Baixa: eles ilustram uma forma, nao
# servem para ler valor, e quem quiser o numero passa o mouse.
ALTURA_DO_GRAFICO = 260

# Os tres primeiros slots da paleta categorica de referencia, um par por
# tema. Nao e a mesma cor clareada: sao passos escolhidos para cada fundo e
# verificados como conjunto. A versao clara reprova sobre fundo escuro, na
# faixa de luminosidade, entao usar uma so nos dois modos nao era opcao.
# A ORDEM e o mecanismo de seguranca para daltonismo: nao se embaralha, e
# um quarto slot exigiria rodar a verificacao de novo.
SLOTS_CLARO = ("#2a78d6", "#eb6834", "#1baf7a")
SLOTS_ESCURO = ("#3987e5", "#d95926", "#199e70")


@st.cache_data(show_spinner="rodando a População...")
def tabelas(seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    populacao = build_population(seed, KeyRepositorySpecification())

    return populacao.operators, populacao.keys


@st.cache_data(show_spinner="rodando o Scenario Engine...")
def trafego(seed: int) -> pd.DataFrame:
    operators, keys = tabelas(seed)

    return build_traffic(seed, operators, keys, TrafficSpecification())


@st.cache_data(show_spinner="rodando o aquecimento...")
def aquecimento(seed: int) -> dict[str, pd.DataFrame]:
    """Tudo que a fase 2 produz, de uma vez.

    Num dicionario porque os quadros da fase 2 e da fase 3 consomem pedacos
    diferentes, e recalcular o perfil na fase 3 violaria a D-044, que manda
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
        "limiares": build_thresholds(sessoes),
    }


@st.cache_data(show_spinner="rodando a campanha...")
def avaliado(seed: int, sigma: float) -> dict[str, pd.DataFrame | Partition]:
    operators, keys = tabelas(seed)
    perfis = aquecimento(seed)["perfis"]

    campanha = build_attack(
        seed, sigma, operators, keys, trafego(seed),
        TrafficSpecification(), AttackSpecification(),
    )
    outcomes = build_outcomes(campanha.requests, keys, operators, EVALUATED)
    log = build_log(campanha.requests, outcomes, EVALUATED)
    sessoes = build_dataset(log, perfis, EVALUATED, campanha.compromised)

    return {
        "requests": campanha.requests,
        "compromised": campanha.compromised,
        "run": campanha.run,
        "log": log,
        "sessoes": sessoes,
        "particao": build_partition(seed, sessoes),
    }


@st.cache_data(show_spinner="medindo os regimes...")
def comportamentos(seed: int) -> Pagina:
    operators, _ = tabelas(seed)

    return montar_pagina(
        trafego(seed), operators, TrafficSpecification(), AttackSpecification()
    )


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
            warmup["sessoes"], warmup["limiares"],
            detalhes,
        )

    campanha = avaliado(seed, sigma)

    return frames_da_fase_3(
        seed, sigma, trafego(seed), warmup["perfis"],
        campanha["requests"], campanha["compromised"], campanha["run"],
        campanha["log"], campanha["sessoes"], campanha["particao"], detalhes,
    )


def mostrar_painel(painel: Painel, chave: str) -> None:
    """Um arquivo ou parametro, com filtro quando e tabela."""
    tamanho = painel.tamanho()
    rodape = f"  ·  {tamanho}" if tamanho else ""

    st.markdown(f"**`{painel.nome}`**{rodape}")

    if painel.legenda:
        st.caption(painel.legenda)

    is_tabela = isinstance(painel.dado, pd.DataFrame)

    if is_tabela:
        st.dataframe(filtrar(painel.dado, chave), use_container_width=True,
                     hide_index=True, height=ALTURA_DA_TABELA)
        return

    is_lista = isinstance(painel.dado, (list, tuple))

    if is_lista:
        st.code("\n".join(str(item) for item in painel.dado))
        return

    st.code(str(painel.dado))


def filtrar(quadro: pd.DataFrame, chave: str) -> pd.DataFrame:
    """Um operador e uma expressao, acima da tabela.

    A busca da barra do `st.dataframe` procura texto solto em qualquer coluna:
    ela nao expressa "este operador **e** origem fora destas duas", que e a
    pergunta que se faz ao explicar `new_source_ip` ou `atypical_hour`.

    A expressao vai para o `DataFrame.query` do pandas, entao aceita `and`,
    `or`, `not in` e comparacao. Expressao invalida vira aviso, nunca tela
    quebrada: quem esta apresentando nao pode perder a tela por um parentese.
    """
    escolhido, expressao = controles_do_filtro(quadro, chave)

    filtrado = quadro

    if escolhido != TODOS:
        filtrado = filtrado[filtrado["operator_id"] == escolhido]

    if not expressao.strip():
        return relatar_filtro(quadro, filtrado)

    try:
        filtrado = filtrado.query(expressao)
    except Exception as erro:
        st.warning(f"expressao invalida: {erro}", icon=":material/error:")

        return filtrado

    return relatar_filtro(quadro, filtrado)


def controles_do_filtro(quadro: pd.DataFrame, chave: str) -> tuple[str, str]:
    """O seletor de operador e a caixa de expressao, lado a lado."""
    tem_operador = "operator_id" in quadro.columns

    coluna_operador, coluna_expressao = st.columns([1, 3])

    escolhido = TODOS

    if tem_operador:
        operadores = [TODOS] + sorted(quadro["operator_id"].unique())
        escolhido = coluna_operador.selectbox(
            "Operador", operadores, key=f"op_{chave}",
        )

    expressao = coluna_expressao.text_input(
        "Filtro",
        key=f"expr_{chave}",
        placeholder=EXEMPLO_DE_FILTRO,
        help=(
            "Expressao do pandas. Aceita `and`, `or`, `not in`, `>`, `==`. "
            "Exemplos: `outcome != \"success\"` · "
            "`source_ip not in [\"10.1.2.3\", \"10.4.5.6\"]` · "
            "`distinct_keys > 12 and events < 20`"
        ),
    )

    return escolhido, expressao


def relatar_filtro(inteiro: pd.DataFrame, filtrado: pd.DataFrame) -> pd.DataFrame:
    """Diz quanto sobrou, para o numero na tela nunca enganar."""
    mudou = len(filtrado) != len(inteiro)

    if mudou:
        st.caption(
            f"**{len(filtrado)}** de {len(inteiro)} linhas "
            f"({porcento(len(filtrado) / len(inteiro))})"
        )

    return filtrado


def mostrar_lado(
    titulo: str, paineis: tuple[Painel, ...], vazio: str, chave: str
) -> None:
    """As entradas ou as saidas, empilhadas e em largura cheia.

    Em largura cheia, e nao em duas colunas lado a lado, porque as tabelas tem
    muitas colunas (`log.csv` tem oito, `sessions.csv` tem doze), e metade da
    tela cortava as ultimas. Quem precisa comparar entrada com saida rola a
    pagina; quem precisa filtrar uma coluna nao consegue se ela estiver
    escondida.
    """
    st.subheader(titulo)

    if not paineis:
        st.caption(vazio)
        return

    for posicao, painel in enumerate(paineis):
        mostrar_painel(painel, f"{chave}_{posicao}")


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
    de execucao. Faltando a informacao, o claro e o padrao, que e tambem o do
    proprio streamlit.
    """
    is_escuro = st.context.theme.get("type") == "dark"

    return SLOTS_ESCURO if is_escuro else SLOTS_CLARO


def mostrar_grafico(teoria: Teoria) -> None:
    """O grafico da distribuicao, na forma que o dominio pede.

    Barras para dominio discreto (nao existe meia sessao), linha para
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
        st.bar_chart(stack=False, horizontal=teoria.horizontal, **comum)
        return

    st.line_chart(**comum)


def mostrar_secao(teoria: Teoria) -> None:
    """Titulo, texto, grafico quando houver, e o que ler nele."""
    st.markdown(f"### {teoria.titulo}")
    st.markdown(teoria.texto)

    tem_grafico = teoria.dados is not None

    if tem_grafico:
        mostrar_grafico(teoria)

    if teoria.leitura:
        st.caption(f"**O que ler no gráfico.** {teoria.leitura}")


def mostrar_teoria(nome: str, teoria: Teoria) -> None:
    """A teoria de uma variavel, no expansor dela.

    Aninhado dentro do expansor das variaveis, entao usa um `popover`: o
    streamlit nao permite expansor dentro de expansor, e o popover da o mesmo
    'clique para abrir' sem essa restricao.
    """
    with st.popover(f"Teoria · {nome}", use_container_width=True):
        mostrar_secao(teoria)


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


def mostrar_quadro(quadro: Quadro, titular: bool) -> None:
    """Titulo, uma descricao curta, e depois as tabelas.

    So isso de texto, de proposito. O quadro tinha ainda uma legenda de passo,
    uma caixa "Por que e assim" e, acima dele, o objetivo e a descricao da
    fase: prosa demais para uma tela de apresentacao, e o porque de cada
    decisao ja mora no registro. Numero do passo e modulo estao na barra
    lateral.

    `titular` diz se este quadro leva o titulo grande da pagina. So o primeiro
    da tela leva: quando os anteriores ficam empilhados, os de cima viram
    secoes.
    """
    if titular:
        st.title(quadro.entidade)
    else:
        st.markdown(f"## {quadro.entidade}")

    st.markdown(quadro.resumo)

    if quadro.pendente:
        st.warning(quadro.pendente, icon=":material/schedule:")

    mostrar_variaveis(quadro)

    mostrar_lado("Input", quadro.entradas,
                 "Nada: esta entidade abre a sequência.",
                 f"entra{quadro.numero}")
    mostrar_lado("Output", quadro.saidas,
                 "Nada ainda: a entidade não foi escrita.",
                 f"sai{quadro.numero}")

    mostrar_detalhes(quadro)


# Chaves dos controles da barra lateral. Cada uma e a unica fonte do valor
# daquele controle: o widget a le e escreve, e ninguem mais guarda copia.
FASE_ESCOLHIDA = "fase_escolhida"
SEMENTE_ESCOLHIDA = "semente_escolhida"
FOCO_ESCOLHIDO = "foco_escolhido"
SIGMA_ESCOLHIDO = "sigma_escolhido"
PASSO_ESCOLHIDO = "passo_escolhido"
EMPILHAR = "empilhar"
SIGMA_DO_ATACANTE = "sigma_do_atacante"

# Todos os controles usam `persist_state="session"`. Sem isso o Streamlit
# apaga o valor de um widget na execucao em que ele nao e desenhado, e trocar
# de pagina e exatamente isso: ao voltar de Comportamentos, tudo recomecava
# da fase 1, passo 1. A primeira correcao guardava copias a mao, e o seletor
# de passo continuava voltando ao inicio no navegador; o mecanismo nativo e o
# que o proprio Streamlit restaura.
PERSISTE = "session"

SIGMA_PADRAO = 0.5
FOCO_PADRAO = "admin_02"


def escolher_fase() -> Fase:
    """Seletor das tres fases da arquitetura, no topo da barra lateral."""
    st.sidebar.title("Fase")

    rotulos = [
        f"{fase.nome}  (parcial)" if fase.pendencia else fase.nome
        for fase in FASES
    ]
    st.session_state.setdefault(FASE_ESCOLHIDA, rotulos[0])

    escolhido = st.sidebar.radio(
        "Fase", rotulos, key=FASE_ESCOLHIDA, label_visibility="collapsed",
        persist_state=PERSISTE,
    )

    return FASES[rotulos.index(escolhido)]


def escolher_semente() -> int:
    """O seletor de semente, o mesmo nas duas paginas."""
    st.session_state.setdefault(SEMENTE_ESCOLHIDA, 1)

    return int(st.sidebar.number_input(
        "Semente", min_value=1, max_value=30, key=SEMENTE_ESCOLHIDA,
        persist_state=PERSISTE,
    ))


def escolher_foco(nomes: list[str]) -> str:
    """O operador que o 'por dentro' segue.

    Se o guardado nao existir nesta semente, volta ao padrao em vez de
    levantar erro: o seletor nao aceita valor fora das opcoes.
    """
    padrao = FOCO_PADRAO if FOCO_PADRAO in nomes else nomes[0]
    is_invalido = st.session_state.get(FOCO_ESCOLHIDO) not in nomes

    if is_invalido:
        st.session_state[FOCO_ESCOLHIDO] = padrao

    return str(st.sidebar.selectbox(
        "Operador em foco", nomes, key=FOCO_ESCOLHIDO, persist_state=PERSISTE
    ))


def barra_lateral(fase: Fase) -> tuple[int, str, float]:
    """Semente, operador em foco e, so na fase 3, sigma."""
    st.sidebar.divider()
    st.sidebar.subheader("Controles")

    seed = escolher_semente()

    operators, _ = tabelas(seed)
    foco = escolher_foco(sorted(operators["operator_id"]))

    st.sidebar.caption(
        "O operador em foco so afeta o 'por dentro' de cada quadro, que segue "
        "um sujeito so. Os arquivos sao sempre os inteiros."
    )

    sigma = SIGMA_PADRAO

    if fase.nome == FASES[2].nome:
        st.session_state.setdefault(SIGMA_ESCOLHIDO, SIGMA_PADRAO)
        sigma = float(st.sidebar.select_slider(
            "Furtividade σ", options=list(SIGMAS), key=SIGMA_ESCOLHIDO,
            persist_state=PERSISTE,
            help="0,0 e ostensivo; 1,0 e indistinguivel de uma sessao legitima.",
        ))

    st.sidebar.caption(
        "A semente determina tudo que nao e sigma, e a mesma semente sempre "
        "devolve os mesmos dados."
    )

    return seed, foco, sigma


# O seletor de passo (PASSO_ESCOLHIDO) e a unica fonte da verdade da
# navegacao. Os botoes nao guardam estado proprio: escrevem nesta chave, e o
# seletor a le. Com duas fontes (um contador e um seletor) elas se
# dessincronizavam, e era dai que vinha o botao que sumia.


def rotulo_do_passo(posicao: int, total: int, quadro: Quadro) -> str:
    """Como o passo aparece no seletor: posicao na fase, total e entidade.

    A posicao e contada dentro da fase, e nao pelo numero do passo no
    diagrama: com o numero do diagrama o seletor dizia "6 de 4", porque o
    total e o da fase.
    """
    return f"{posicao} de {total}  ·  {quadro.entidade}"


def andar(rotulos: list[str], passos: int) -> None:
    """Move a selecao, sem deixar sair da faixa.

    Callback de `on_click`, e nao codigo no corpo da funcao, de proposito. O
    callback roda **antes** do redesenho, entao o seletor e os botoes ja veem
    o valor novo. Incrementando no corpo, o botao era desenhado com o valor
    velho e so se corrigia no clique seguinte, que era o bug de o `Avancar`
    aparecer desabilitado depois de voltar.
    """
    atual = rotulos.index(st.session_state[PASSO_ESCOLHIDO])
    destino = min(max(atual + passos, 0), len(rotulos) - 1)

    st.session_state[PASSO_ESCOLHIDO] = rotulos[destino]


def navegacao(quadros: list[Quadro]) -> tuple[int, bool]:
    """Seletor de passo, botoes e progresso, na barra lateral."""
    total = len(quadros)
    rotulos = [
        rotulo_do_passo(posicao, total, quadro)
        for posicao, quadro in enumerate(quadros, start=1)
    ]

    st.sidebar.divider()
    st.sidebar.subheader("Passo")

    # Trocar de fase muda a lista inteira, e o rotulo guardado deixa de
    # existir. Sem isto o seletor levantaria erro em vez de recomecar.
    is_desconhecido = st.session_state.get(PASSO_ESCOLHIDO) not in rotulos

    if is_desconhecido:
        st.session_state[PASSO_ESCOLHIDO] = rotulos[0]

    escolhido = st.sidebar.selectbox(
        "Ir para", rotulos, key=PASSO_ESCOLHIDO, label_visibility="collapsed",
        persist_state=PERSISTE,
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

    st.session_state.setdefault(EMPILHAR, False)
    empilhar = st.sidebar.checkbox(
        "Manter os anteriores na tela", key=EMPILHAR, persist_state=PERSISTE
    )

    return atual, empilhar


@st.cache_data(show_spinner="sorteando sessoes...")
def tamanhos_no_sigma(regime: str, sigma: float) -> pd.DataFrame:
    return tamanhos_da_sessao(
        sigma, REGIMES[regime], TrafficSpecification(), AttackSpecification()
    )


def mostrar_sigma_do_atacante(regime: str) -> None:
    """O slider de σ, e o que o atacante faz naquela condição.

    A tabela e o grafico sao recalculados a cada posicao do slider, pelas
    mesmas funcoes que a campanha de ataque usa. O grafico mostra so o
    tamanho da sessao porque e a dimensao em que a sobreposicao com o
    legitimo se ve melhor: em sigma 0 as duas nuvens nao se tocam, em sigma 1
    sao a mesma.
    """
    st.markdown("### O atacante em cada σ")

    st.session_state.setdefault(SIGMA_DO_ATACANTE, SIGMA_PADRAO)
    sigma = float(st.select_slider(
        "Furtividade σ", options=list(SIGMAS), key=SIGMA_DO_ATACANTE,
        persist_state=PERSISTE,
        format_func=lambda valor: f"{valor:.1f}".replace(".", ","),
        help="0,0 e ostensivo; 1,0 e indistinguivel de uma sessao legitima.",
    ))

    tabela = tabela_do_atacante(
        sigma, REGIMES[regime], TrafficSpecification(), AttackSpecification()
    )
    st.dataframe(tabela, use_container_width=True, hide_index=True)

    tamanhos = tamanhos_no_sigma(regime, sigma)
    mostrar_grafico(Teoria(
        titulo="", texto="", dados=tamanhos,
        rotulo_x="requisições na sessão", rotulo_y="fração das sessões",
    ))
    st.caption(
        "**O que ler no gráfico.** Quanto mais as duas cores se sobrepõem, "
        "menos o tamanho da sessão sozinho denuncia o atacante."
    )


def mostrar_comportamento(comportamento: Comportamento) -> None:
    """Um regime, numa aba: quem usa, o resumo, e as secoes em ordem."""
    st.caption(
        f"regime `{comportamento.regime}`  ·  perfil `{comportamento.perfil}`  ·  "
        f"{comportamento.operadores} operadores nesta semente"
    )

    with st.container(border=True):
        st.markdown(comportamento.resumo)

    for secao in comportamento.secoes:
        mostrar_secao(secao)
        st.divider()

    if comportamento.personificado:
        mostrar_sigma_do_atacante(comportamento.regime)


def pagina_dos_comportamentos() -> None:
    """Os tres regimes em detalhe, com o trafego da semente escolhida."""
    st.sidebar.subheader("Controles")
    seed = escolher_semente()

    st.sidebar.caption(
        "Tudo nesta página é medido no tráfego legítimo desta semente (as oito "
        "semanas do Scenario Engine) ou lido dos parâmetros do gerador."
    )

    st.title("Comportamento dos operadores")
    st.markdown(
        "Como cada um dos três regimes usa o KMS: quando abre sessão, quanto "
        "ela dura, o que pede e de onde vem. Os gráficos mostram a distribuição "
        "que o gerador usa e, em alguns, a frequência medida nesta semente."
    )

    pagina = comportamentos(seed)
    nomes = ["Visão geral"] + [comportamento.regime for comportamento in pagina.comportamentos]
    abas = st.tabs(nomes)

    with abas[0]:
        st.dataframe(pagina.quadro, use_container_width=True, hide_index=True)

        for secao in pagina.gerais:
            mostrar_secao(secao)

    for aba, comportamento in zip(abas[1:], pagina.comportamentos):
        with aba:
            mostrar_comportamento(comportamento)


def pagina_do_pipeline() -> None:
    """As tres fases, um passo do diagrama por vez."""
    fase = escolher_fase()
    seed, foco, sigma = barra_lateral(fase)

    quadros = quadros_da_fase(fase, seed, foco, sigma)
    atual, empilhar = navegacao(quadros)

    primeiro = 1 if empilhar else atual

    for numero in range(primeiro, atual + 1):
        mostrar_quadro(quadros[numero - 1], numero == primeiro)

        if numero < atual:
            st.divider()


def main() -> None:
    """Duas paginas, escolhidas no topo da tela."""
    st.set_page_config(page_title=TITULO, layout="wide")

    paginas = [
        st.Page(pagina_do_pipeline, title="Pipeline", default=True),
        st.Page(pagina_dos_comportamentos, title="Comportamentos"),
    ]

    st.navigation(paginas, position="top").run()


main()
