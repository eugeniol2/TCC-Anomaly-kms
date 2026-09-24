"""A teoria por trás das variáveis que são distribuições, com gráfico.

Algumas variáveis do gerador não são um número escolhido: são a **forma de uma
distribuição**. `CUSTODY_DISPERSION = 2` não quer dizer "dois" de nada: quer
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

from src.traffic.sessions import address_weights

# A cor não mora aqui. Os passos da paleta mudam entre o tema claro e o
# escuro (a versão clara reprovou na verificação contra fundo escuro), e
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
            "constante, que é o caso de quem trabalha sempre no mesmo ritmo. Ela tem "
            "uma propriedade rígida: a **variância é igual à média**. Com média "
            f"{media}, a variância também é {media}.\n\n"
            "Pessoas não trabalham assim. Um custodiante de chaves passa dias "
            "sem abrir uma sessão e então faz seis numa tarde, porque o trabalho "
            "chega em lote e não em fluxo. Isso é **superdispersão**: variância "
            "maior que a média.\n\n"
            "A **binomial negativa**, também chamada de Pascal, é a Poisson com "
            "um parâmetro a mais. Ela se lê como uma Poisson cuja taxa varia de "
            "dia para dia, e o parâmetro `n` controla quanto varia: quanto "
            f"menor, mais irregular. Com `n = {dispersao}` e a mesma média "
            f"{media}, a variância sobe para **{variancia_pascal:.2f}**, ou "
            f"{variancia_pascal / media:.2f} vez a média."
        ),
        dados=dados,
        rotulo_x="sessões abertas num dia útil",
        rotulo_y="probabilidade",
        leitura=(
            "As duas curvas têm o mesmo centro de massa, porque a média é a mesma. O "
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
            "tem variância igual à média, sem nenhum parâmetro extra para ajustar.\n\n"
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
            "escrita em lugar nenhum**: ela emerge da soma desses intervalos. "
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
            "A do `automated_service` despenca quase de imediato, e quase toda a "
            "massa está nos primeiros segundos. A do `administrator` é rasa e "
            "longa, o que dá sessões de dezenas de minutos."
        ),
    )


def teoria_da_cauda(
    faixa: tuple[int, int], chance: float, excesso: float, teto_ostensivo: int
) -> Teoria:
    """Por que a faixa de comprimento da sessão precisou de cauda (D-097).

    **O eixo comeca perto do teto, e nao no inicio da faixa.** Desenhado de
    ponta a ponta, o grafico gasta tres quartos da largura com as duas curvas
    sobrepostas, onde nao ha nada a ver, e espreme no canto direito a unica
    coisa que importa: que uma delas continua acima de zero e a outra nao.
    Recortar nao esconde nada: a esquerda do recorte as duas valem o mesmo.
    """
    menor, maior = faixa

    # A curva mostrada e a **cumulativa invertida**: "qual a chance de a sessao
    # passar de N eventos?". A densidade tem um degrau enorme em `maior`, de
    # 4,7 % para 0,18 % de um evento para o outro, e em escala linear a cauda
    # depois dele vira uma linha rente ao eixo, que se le como "acaba aqui".
    # E o contrario do que o grafico existe para mostrar.
    #
    # Calculada **analiticamente**, e nao somando uma densidade truncada: a
    # soma daria zero no ultimo ponto do eixo, que e o proprio artefato que
    # esta curva veio corrigir.
    tipicos = np.arange(menor, maior + 1)
    quantos = len(tipicos)
    sobrevive = 1.0 - 1.0 / excesso

    # Comeca poucos eventos antes do teto, onde as curvas ainda coincidem, para
    # que o leitor veja as duas juntas e depois se separarem.
    inicio = maior - 4
    comprimentos = np.arange(inicio, teto_ostensivo + 30)

    def passa_de(n: int, com_cauda: bool) -> float:
        """Chance de a sessao ter mais de `n` eventos."""
        maiores = (tipicos > n).sum() / quantos

        if not com_cauda:
            return maiores

        # Tipico acima de n ja passa, com ou sem excesso. Tipico abaixo so
        # passa se a sessao se estendeu o bastante: a geometrica sobrevive
        # a `n - t` com (1 - p) elevado a essa diferenca.
        alcancados = [
            chance * sobrevive ** (n - t) for t in tipicos if t <= n
        ]

        return maiores + sum(alcancados) / quantos

    passa_de_fechada = np.array([passa_de(n, False) for n in comprimentos])
    passa_de_com_cauda = np.array([passa_de(n, True) for n in comprimentos])

    dados = pd.DataFrame({
        "eventos na sessão": comprimentos,
        "faixa fechada": passa_de_fechada,
        "com cauda": passa_de_com_cauda,
    })

    acima = passa_de(maior, com_cauda=True)
    alcanca = passa_de(teto_ostensivo, com_cauda=True)

    return Teoria(
        titulo="Por que a faixa de comprimento precisou de cauda",
        texto=(
            f"A faixa de cada regime (aqui {faixa}) dizia o comprimento da "
            "sessão por **sorteio uniforme numa faixa fechada**. Isso a tornava "
            "um **teto rígido**: nenhuma sessão legítima podia ter mais de "
            f"{maior} eventos, nunca.\n\n"
            f"O atacante ostensivo sorteia a partir de {teto_ostensivo}. As duas "
            "classes **não se sobrepunham**, e o percentil 99 do baseline caía "
            f"exatamente em {maior}. A regra `events` passava a ter **falso "
            "positivo zero por construção** e separava as classes sozinha, com "
            "F1 **0,982** em σ 0,0, por aritmética de faixa e não por "
            "comportamento. A condição σ 0,0 seria excluída da comparação pelo "
            "critério de trivialidade, e pela razão errada (D-097).\n\n"
            f"A correção: **uma sessão em {1 / chance:.0f}** se estende por um "
            f"excesso geométrico de média {excesso:.0f}. Tráfego real de KMS tem "
            "cauda (migração em lote, reprocessagem, job que repete), e o teto "
            "era artefato do sorteio, não propriedade do domínio.\n\n"
            f"Hoje **{acima:.1%}** das sessões passam do típico, e a mais longa "
            "chega à faixa do atacante. O mesmo mecanismo vale para a amplitude: "
            "a sessão que se estende também se alarga (D-098)."
        ),
        dados=dados,
        rotulo_x="eventos na sessão",
        rotulo_y="chance de a sessão passar deste tamanho",
        forma="linha",
        leitura=(
            f"Cada ponto responde: **qual a chance de a sessão passar de N "
            f"eventos?** O eixo começa em {inicio}, e não no início da faixa, "
            "porque à esquerda daqui as duas curvas valem quase o mesmo e não "
            "há o que comparar.\n\n"
            f"Elas descem juntas até {maior - 1}. Em **{maior}** se separam, e "
            "é este o gráfico inteiro: a **faixa fechada** cai a zero e fica "
            "ali, porque à direita do teto não existe sessão legítima nenhuma. "
            f"A **com cauda** vale **{alcanca:.1%}** nesse ponto e segue "
            "descendo devagar, sem nunca tocar o eixo.\n\n"
            "É pouco, e é o suficiente: enquanto for maior que zero, sessão "
            "longa é explicação possível para um tamanho grande, e a regra "
            "precisa medir comportamento em vez de ler a faixa."
        ),
    )


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
    posicoes = np.arange(maximo_de_enderecos)

    dados = pd.DataFrame({
        "endereço": [f"{posicao + 1}º" for posicao in posicoes],
        "chance de ser usado": pesos,
    })

    quarto = pesos[-1]
    # Sessoes na regua, mediana nas 30 sementes. E o expoente que decide se o
    # endereco raro falta: regua mais longa o derruba, e foi o custo da D-096.
    SESSOES_NA_REGUA = 32
    ausencia = (1 - quarto) ** SESSOES_NA_REGUA

    return Teoria(
        titulo="Por que a origem de rede decai geometricamente",
        texto=(
            "Cada operador tem de duas a quatro origens habituais, e a sessão "
            "escolhe uma delas por uma **geométrica truncada**: a principal leva "
            f"{principal:.0%}, e cada endereço seguinte leva {principal:.0%} do "
            "que sobrou.\n\n"
            "A razão não é realismo pelo realismo. O atributo `new_source_ip` "
            "pergunta se a origem da sessão **apareceu na régua**, que são as "
            "quatro semanas de aquecimento. Se todos os endereços de um operador "
            "aparecessem sempre, nenhuma sessão legítima teria origem inédita no "
            "período avaliado, e origem inédita viraria marcador perfeito do "
            "atacante.\n\n"
            f"Com o decaimento, o quarto endereço fica em **{quarto:.2%}** de "
            "chance por sessão. Numa régua de cerca de 32 sessões, ele tem "
            f"~{ausencia:.0%} de chance de **não aparecer nenhuma vez**, e é "
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
            f"concentração, aqui **{concentracao}**, controla quão desigual: "
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
