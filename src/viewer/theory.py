from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.entities.scenario_engine.traffic.sessions import address_weights
from src.formulas.distributions import chance_never_drawn
from src.viewer.formatting import com_virgula, porcento


@dataclass(frozen=True)
class Teoria:
    titulo: str
    texto: str
    dados: pd.DataFrame | None = None
    rotulo_x: str = ""
    rotulo_y: str = ""

    forma: str = "barras"

    leitura: str = ""

    horizontal: bool = False


def teoria_da_geometrica(principal: float, maximo_de_enderecos: int) -> Teoria:
    """Como a origem de rede é escolhida, e por que alguma precisa faltar.

    Os pesos vêm de `address_weights`, a função que o gerador usa. Só o `principal`
    chega por parâmetro.
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
    # Sessoes na regua: a mediana nas 30 sementes.
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
