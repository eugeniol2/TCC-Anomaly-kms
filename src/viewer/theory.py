"""A teoria por trás das variáveis que são distribuições, com gráfico.

Algumas variáveis do gerador não são um número escolhido: são a **forma de uma
distribuição**. `PRIMARY_ADDRESS_SHARE = 0.80` não diz só quanto pesa a
principal: diz que os endereços decaem geometricamente, e é o decaimento que
faz o mais raro faltar no aquecimento.

Isso não cabe numa célula de tabela, e não se demonstra em prosa. Cada função
aqui devolve o texto **e os dados de um gráfico**, calculados na hora a partir
da mesma distribuição que o gerador usa.

**Só as distribuições globais moram aqui**: as que valem para o tráfego
inteiro, qualquer que seja o regime. As de um regime só (a Poisson do
`routine`, a Pascal do `occasional_custody`, o passo exponencial de cada um)
estão na página Comportamentos, em `behaviors.py`, com a frequência medida ao
lado da curva.

Os gráficos são do Streamlit e não do matplotlib, de propósito: eles herdam o
tema claro ou escuro da página e trazem tooltip ao passar o mouse, que numa
tela de apresentação vale mais que controle fino sobre a marca.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.entities.scenario_engine.traffic.sessions import address_weights
from src.formulas.distributions import chance_never_drawn
from src.viewer.formatting import com_virgula, porcento

# A cor não mora aqui. Os passos da paleta mudam entre o tema claro e o
# escuro (a versão clara reprovou na verificação contra fundo escuro), e
# quem sabe o tema é o app. Aqui só se declara quantas séries existem; o app
# toma os primeiros N slots, na ordem fixa que é o mecanismo de segurança
# para daltonismo.


@dataclass(frozen=True)
class Teoria:
    """A explicação de uma distribuição, com o gráfico que a sustenta.

    O gráfico é opcional porque a aba de comportamentos usa o mesmo formato
    para seções que se explicam só em texto. Nas variáveis de decisão ele está
    sempre presente.
    """

    titulo: str
    texto: str
    dados: pd.DataFrame | None = None
    rotulo_x: str = ""
    rotulo_y: str = ""

    forma: str = "barras"
    """`barras` para domínio discreto, `linha` para contínuo."""

    leitura: str = ""
    """O que se deve enxergar no gráfico. Sem isto, ele é decoração."""

    horizontal: bool = False
    """Barras deitadas, para quando o eixo tem nomes longos.

    Com `ExportKeyMaterial` no eixo, o rótulo em pé ocupava quase toda a altura
    do gráfico e as barras ficavam achatadas numa faixa de poucos pixels.
    """


def teoria_da_geometrica(principal: float, maximo_de_enderecos: int) -> Teoria:
    """Como a origem de rede é escolhida, e por que alguma precisa faltar.

    **Os pesos vêm de `address_weights`, a função que o M2 de fato usa.** Esta
    teoria reimplementava a fórmula, e a duplicação era um risco silencioso:
    mudada a forma do decaimento no gerador, o gráfico seguiria desenhando a
    antiga, plausível e errada, numa apresentação. É a mesma regra que o
    `steps.py` declara para si: o observador se adapta ao código, nunca o
    contrário.

    Só o `principal` continua chegando por parâmetro, porque ele é a variável
    de decisão que a tabela mostra ao lado do gráfico.
    """
    pesos = address_weights(maximo_de_enderecos)
    enderecos = []

    for posicao in range(maximo_de_enderecos):
        enderecos.append(f"{posicao + 1}º")

    dados = pd.DataFrame({
        "endereço": enderecos,
        "chance de ser usado": pesos,
    })

    quarto = pesos[-1]
    # Sessoes na regua, mediana nas 30 sementes. E o expoente que decide se o
    # endereco raro falta: regua mais longa o derruba, e foi o custo da D-096.
    SESSOES_NA_REGUA = 32
    ausencia = chance_never_drawn(quarto, SESSOES_NA_REGUA)

    return Teoria(
        titulo="Por que a origem de rede decai geometricamente",
        texto=(
            "Cada operador tem de duas a quatro origens habituais, e a sessão "
            "escolhe uma delas por uma **geométrica truncada**: a principal leva "
            f"{porcento(principal, 0)}, e cada endereço seguinte leva "
            f"{porcento(principal, 0)} do "
            "que sobrou.\n\n"
            "A razão não é realismo pelo realismo. O atributo `new_source_ip` "
            "pergunta se a origem da sessão **apareceu na régua**, que são as "
            "quatro semanas de aquecimento. Se todos os endereços de um operador "
            "aparecessem sempre, nenhuma sessão legítima teria origem inédita no "
            "período avaliado, e origem inédita viraria marcador perfeito do "
            "atacante.\n\n"
            f"Com o decaimento, o quarto endereço fica em **{porcento(quarto, 2)}** de "
            f"chance por sessão. Numa régua de cerca de {SESSOES_NA_REGUA} "
            f"sessões, ele tem ~{porcento(ausencia, 0)} de chance de **não aparecer nenhuma vez**, e é "
            "essa ausência que produz origem inédita legítima depois.\n\n"
            "**A régua de quatro semanas apertou essa folga**, e é o custo "
            "declarado da D-096: com mais semanas observadas, a maior parte dos "
            "perfis passa a ter visto todos os endereços do operador, e sobra "
            "menos para ser inédito depois. Nenhuma das 30 sementes chega a "
            "zero, que é a condição mínima, mas a margem ficou fina: a "
            "contagem por semente está na seção 6.9 do `relatorio-fundacao.md`."
        ),
        dados=dados,
        rotulo_x="posição do endereço na lista do operador",
        rotulo_y="chance de a sessão vir dele",
        leitura=(
            "A queda é acentuada de propósito. O que importa não é a altura da "
            "primeira barra: é a **última ser baixa o bastante** para o endereço "
            "faltar no aquecimento com frequência."
        ),
    )


def teoria_da_dirichlet(chaves: pd.DataFrame, concentracao: float) -> Teoria:
    """Como as chaves se repartem entre escopos, medido no dado real."""
    por_escopo = chaves["scope"].value_counts().sort_values(ascending=False)
    uniforme = len(chaves) / len(por_escopo)

    dados = pd.DataFrame({
        "escopo": por_escopo.index,
        "chaves": por_escopo.to_numpy(),
    })

    return Teoria(
        titulo="Por que os escopos têm tamanhos desiguais",
        texto=(
            "As chaves se repartem entre os escopos por uma **Dirichlet**, que "
            "é a distribuição de proporções que somam 1, que é o sorteio natural "
            "quando se quer dividir um total em partes. O parâmetro de "
            f"concentração, aqui **{com_virgula(concentracao)}**, controla quão desigual: "
            "valores altos aproximam a divisão de partes iguais, valores baixos "
            "produzem uns poucos escopos grandes e muitos pequenos.\n\n"
            "A desigualdade é **deliberada**. O alcance de um operador é a soma "
            "das chaves dos escopos que ele detém. Se todos os escopos tivessem "
            "o mesmo tamanho, dois operadores do mesmo perfil teriam alcances "
            "quase idênticos, e `distinct_keys` variaria pouco entre eles, e o "
            "que tornaria o atributo mais fácil de separar por um limiar global "
            "do que pelo histórico de cada um.\n\n"
            "O gráfico é medido no `keys.csv` desta semente, não simulado."
        ),
        dados=dados,
        rotulo_x="escopo, do maior para o menor",
        rotulo_y="chaves",
        leitura=(
            f"Se a divisão fosse igual, todo escopo teria **{uniforme:.0f} "
            f"chaves**. O maior aqui tem {por_escopo.iloc[0]} e o menor "
            f"{por_escopo.iloc[-1]}, uma diferença de "
            f"{por_escopo.iloc[0] / por_escopo.iloc[-1]:.1f} vezes. É essa "
            "distância que faz dois operadores do mesmo perfil terem alcances "
            "diferentes."
        ),
    )
