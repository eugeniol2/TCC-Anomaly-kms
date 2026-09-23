"""A teoria por trás das variáveis que são distribuições, com gráfico.

Algumas variáveis do gerador não são um número escolhido: são a **forma de uma
distribuição**. `CUSTODY_DISPERSION = 2` não quer dizer "dois" de nada — quer
dizer que as sessões do administrador chegam por uma Pascal em vez de uma
Poisson, e a diferença entre as duas é o que produz o ritmo irregular que a
lista de conferência exige.

Isso não cabe numa célula de tabela, e não se demonstra em prosa. Cada função
aqui devolve o texto **e os dados de um gráfico**, calculados na hora a partir
da mesma distribuição que o gerador usa.

Os gráficos são do Streamlit e não do matplotlib, de propósito: eles herdam o
tema claro ou escuro da página e trazem tooltip ao passar o mouse, que numa
tela de apresentação vale mais que controle fino sobre a marca.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

# A cor não mora aqui. Os passos da paleta mudam entre o tema claro e o
# escuro — a versão clara reprovou na verificação contra fundo escuro —, e
# quem sabe o tema é o app. Aqui só se declara quantas séries existem; o app
# toma os primeiros N slots, na ordem fixa que é o mecanismo de segurança
# para daltonismo.


@dataclass(frozen=True)
class Teoria:
    """A explicação de uma distribuição, com o gráfico que a sustenta."""

    titulo: str
    texto: str
    dados: pd.DataFrame
    rotulo_x: str
    rotulo_y: str

    forma: str = "barras"
    """`barras` para domínio discreto, `linha` para contínuo."""

    leitura: str = ""
    """O que se deve enxergar no gráfico. Sem isto, ele é decoração."""


def teoria_da_pascal(media: float, dispersao: int) -> Teoria:
    """Por que o administrador usa binomial negativa e não Poisson.

    As duas têm a mesma média. O que muda é a variância, e é a variância que
    produz o "ritmo irregular, padrão humano" que a lista de conferência pede.
    """
    contagens = np.arange(0, 9)
    sucesso = dispersao / (dispersao + media)

    poisson = stats.poisson.pmf(contagens, media)
    pascal = stats.nbinom.pmf(contagens, dispersao, sucesso)

    dados = pd.DataFrame({
        "sessões no dia": contagens,
        "Poisson": poisson,
        f"Pascal, n = {dispersao}": pascal,
    })

    variancia_pascal = media / sucesso

    return Teoria(
        titulo="Poisson ou binomial negativa (Pascal)?",
        texto=(
            "Uma **Poisson** descreve chegadas independentes a uma taxa "
            "constante — o caso de quem trabalha sempre no mesmo ritmo. Ela tem "
            "uma propriedade rígida: a **variância é igual à média**. Com média "
            f"{media}, a variância também é {media}.\n\n"
            "Pessoas não trabalham assim. Um custodiante de chaves passa dias "
            "sem abrir uma sessão e então faz seis numa tarde, porque o trabalho "
            "chega em lote e não em fluxo. Isso é **superdispersão**: variância "
            "maior que a média.\n\n"
            "A **binomial negativa**, também chamada de Pascal, é a Poisson com "
            "um parâmetro a mais. Ela se lê como uma Poisson cuja taxa varia de "
            "dia para dia, e o parâmetro `n` controla quanto varia — quanto "
            f"menor, mais irregular. Com `n = {dispersao}` e a mesma média "
            f"{media}, a variância sobe para **{variancia_pascal:.2f}**, ou "
            f"{variancia_pascal / media:.2f} vez a média."
        ),
        dados=dados,
        rotulo_x="sessões abertas num dia útil",
        rotulo_y="probabilidade",
        leitura=(
            "As duas curvas têm o mesmo centro de massa — a média é a mesma. O "
            "que muda são as pontas: a Pascal põe **mais peso no zero** e "
            "**mais peso na cauda** de 5 ou mais. É exatamente isso que produz "
            "dias vazios e dias cheios, e é isso que se lê no log como ritmo "
            "humano."
        ),
    )


def teoria_da_poisson(media: float) -> Teoria:
    """Onde a Poisson continua sendo a escolha certa."""
    contagens = np.arange(0, 8)

    dados = pd.DataFrame({
        "sessões no dia": contagens,
        "Poisson": stats.poisson.pmf(contagens, media),
    })

    return Teoria(
        titulo="Por que Poisson basta para o usuário legítimo",
        texto=(
            "A Poisson descreve chegadas independentes a uma taxa constante, e "
            "tem variância igual à média — nenhum parâmetro extra para ajustar.\n\n"
            "Para o `end_user` isso é suficiente, e é suficiente de "
            "propósito. Ele é o **contraste** contra o qual o ritmo irregular do "
            "`administrator` aparece: se os dois fossem superdispersos, o item "
            "'ritmo irregular, padrão humano' da lista de conferência não teria "
            "contra o que ser medido.\n\n"
            f"Com média {media} sessões por dia útil, o usuário abre de uma a "
            "duas sessões na maior parte dos dias, e a cauda longa é rara."
        ),
        dados=dados,
        rotulo_x="sessões abertas num dia útil",
        rotulo_y="probabilidade",
        leitura=(
            "Uma corcova só, estreita, centrada perto da média. Compare com a "
            "Pascal do `administrator`: lá o zero e a cauda carregam muito mais "
            "peso."
        ),
    )


def teoria_da_exponencial(intervalos: dict[str, float]) -> Teoria:
    """O intervalo entre requisições dentro de uma sessão."""
    segundos = np.linspace(0, 240, 120)

    dados = pd.DataFrame({"segundos": segundos})

    for perfil, media in intervalos.items():
        dados[perfil] = stats.expon.pdf(segundos, scale=media)

    return Teoria(
        titulo="Por que o intervalo entre requisições é exponencial",
        texto=(
            "Dentro de uma sessão, o tempo até a próxima requisição é sorteado "
            "de uma **exponencial**. Ela é a distribuição do intervalo entre "
            "eventos de um processo sem memória: o tempo já decorrido não muda "
            "a chance de o próximo evento ocorrer agora.\n\n"
            "É a escolha natural aqui porque **a duração da sessão não é "
            "escrita em lugar nenhum** — ela emerge da soma desses intervalos. "
            "E é da duração que sai `requests_per_minute`, que é uma das cinco "
            "dimensões que σ interpola.\n\n"
            "Cada perfil tem sua média. O `automated_service` dispara "
            "requisições quase em rajada; o `administrator` trabalha devagar, "
            "inspecionando entre uma e outra."
        ),
        dados=dados,
        rotulo_x="segundos até a próxima requisição",
        rotulo_y="densidade",
        forma="linha",
        leitura=(
            "Quanto mais alta a curva na origem, mais curto o intervalo típico. "
            "A do `automated_service` despenca quase de imediato — quase toda a "
            "massa está nos primeiros segundos. A do `administrator` é rasa e "
            "longa, o que dá sessões de dezenas de minutos."
        ),
    )


def teoria_da_geometrica(principal: float, maximo_de_enderecos: int) -> Teoria:
    """Como a origem de rede é escolhida, e por que alguma precisa faltar."""
    posicoes = np.arange(maximo_de_enderecos)
    pesos = principal * (1 - principal) ** posicoes
    pesos = pesos / pesos.sum()

    dados = pd.DataFrame({
        "endereço": [f"{posicao + 1}º" for posicao in posicoes],
        "chance de ser usado": pesos,
    })

    quarto = pesos[-1] if maximo_de_enderecos >= 4 else pesos[-1]
    ausencia = (1 - quarto) ** 40

    return Teoria(
        titulo="Por que a origem de rede decai geometricamente",
        texto=(
            "Cada operador tem de duas a quatro origens habituais, e a sessão "
            "escolhe uma delas por uma **geométrica truncada**: a principal leva "
            f"{principal:.0%}, e cada endereço seguinte leva {principal:.0%} do "
            "que sobrou.\n\n"
            "A razão não é realismo pelo realismo. O atributo `novel_source` "
            "pergunta se a origem da sessão **apareceu no perfil das semanas 1 "
            "e 2**. Se todos os endereços de um operador aparecessem sempre, "
            "nenhuma sessão legítima teria origem inédita no período avaliado — "
            "e origem inédita viraria marcador perfeito do atacante.\n\n"
            f"Com o decaimento, o quarto endereço fica em **{quarto:.2%}** de "
            "chance por sessão. Num aquecimento de cerca de 40 sessões, ele tem "
            f"~{ausencia:.0%} de chance de **não aparecer nenhuma vez** — e é "
            "essa ausência que produz origem inédita legítima depois."
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
            "é a distribuição de proporções que somam 1 — o sorteio natural "
            "quando se quer dividir um total em partes. O parâmetro de "
            f"concentração, aqui **{concentracao}**, controla quão desigual: "
            "valores altos aproximam a divisão de partes iguais, valores baixos "
            "produzem uns poucos escopos grandes e muitos pequenos.\n\n"
            "A desigualdade é **deliberada**. O alcance de um operador é a soma "
            "das chaves dos escopos que ele detém. Se todos os escopos tivessem "
            "o mesmo tamanho, dois operadores do mesmo perfil teriam alcances "
            "quase idênticos, e `distinct_keys` variaria pouco entre eles — o "
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
            f"{por_escopo.iloc[-1]} — uma diferença de "
            f"{por_escopo.iloc[0] / por_escopo.iloc[-1]:.1f} vezes. É essa "
            "distância que faz dois operadores do mesmo perfil terem alcances "
            "diferentes."
        ),
    )
