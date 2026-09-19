"""Parametros do trafego legitimo e seus valores de referencia.

Contrato entre a linha de comando, que os recebe, e a construcao das
requisicoes, que os consome.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

FIRST_DAY = date(2026, 1, 5)
"""
## Por que existe um `FIRST_DAY`

**Pergunta.** Por que precisamos de um valor chamado *first day*?

A data em si é arbitrária: 05/01/2026 não tem razão nenhuma, poderia ser qualquer outra.
Ser segunda-feira é praticidade — alinha cada semana simulada com uma semana civil, o que
deixa o log conferível a olho e a semana descritível no texto.

E não é a data do relógio. Com `date.today()`, a mesma semente daria timestamps
diferentes em dias diferentes, o teste de referência falharia toda vez e a reprodução
byte a byte deixaria de valer.
"""

WEEK_COUNT = 7
"""Semanas simuladas (D-069, que substituiu as cinco da D-053).

1 e 2 constroem o perfil historico, 3 calibra os limiares, **4 a 7 sao o periodo
avaliado**. O M2 gera as sete de uma vez, so com trafego legitimo; o M3 e que
mescla a campanha nas quatro ultimas.

O periodo avaliado passou de duas para quatro semanas porque cada sessao
comprometida e **uma** positiva (D-061), e so um administrador e comprometido
por semente (D-036): com duas semanas, o holdout ficava com cerca de doze
positivas, contagem absoluta na faixa da armadilha de Athapaththu et al.
Dobrar o periodo dobra as positivas sem tocar no modelo de ameaca.

Para o M2 isto e so um numero: ele nao conhece os papeis das semanas, gera
`week_count * 7` dias iguais e deixa a divisao para os modulos seguintes.
"""

DAYS_PER_WEEK = 7
BUSINESS_WEEKDAYS = frozenset({0, 1, 2, 3, 4})
"""Segunda a sexta, na numeracao de `date.weekday`."""

PRINCIPAL_ADDRESS_SHARE = 0.80
"""Peso da origem de rede principal; as demais decaem geometricamente (D-058).

Num operador de quatro enderecos isto poe o quarto em torno de 0,64 %. Nas duas
semanas de perfil um usuario esporadico abre cerca de 15 sessoes, entao esse
endereco tem por volta de 91 % de chance de **nao aparecer no aquecimento** — e
de produzir origem inedita legitima nas semanas 4 a 7.

E o que a D-040 encomendou: origem nova sem regra de excecao no gerador, so
pela distribuicao.
"""

IDENTIFIER_SPACE = 2**48
IDENTIFIER_PREFIX = "k_"
IDENTIFIER_DIGITS = 12
"""Formato do identificador de chave, espelhado de `keys.csv`.

Duplicado de proposito: a fronteira entre modulos e o arquivo, nao a chamada de
funcao, entao o M2 nao importa do M1. O teste
`test_absent_identifier_matches_the_repository_format` guarda a duplicacao.

O M2 precisa do formato para forjar o identificador inexistente da D-056, que
tem de ser indistinguivel de um real ate o M4 procura-lo e nao achar.
"""


@dataclass(frozen=True)
class ScheduledRhythm:
    """Lotes em horas fixas, todos os dias do calendario.

    O regime do servico automatizado. Horas fixas e cobertura de fim de semana
    sao o que produz os picos periodicos de volume, e o que separa maquina de
    humano sem que nada consulte o rotulo do perfil.
    """

    hours: tuple[int, ...]
    jitter_minutes: int


@dataclass(frozen=True)
class ArrivalRhythm:
    """Chegadas aleatorias em dias uteis, dentro de uma janela de horario.

    `dispersion` ausente significa Poisson, onde a variancia iguala a media.
    Presente, e o parametro `n` da binomial negativa, que produz variancia
    maior — o ritmo irregular exigido do administrador. Poisson sairia regular
    demais para passar por padrao humano.
    """

    sessions_per_business_day: float
    opens_at: int
    closes_at: int
    dispersion: int | None = None


@dataclass(frozen=True)
class Regime:
    """Como um regime abre sessoes e o que acontece dentro delas (D-058)."""

    rhythm: ScheduledRhythm | ArrivalRhythm
    requests_range: tuple[int, int]
    seconds_between_requests: float


REGIMES: dict[str, Regime] = {
    "periodic_batch": Regime(
        rhythm=ScheduledRhythm(hours=(2, 8, 14, 20), jitter_minutes=10),
        requests_range=(20, 40),
        seconds_between_requests=2.0,
    ),
    "sporadic": Regime(
        rhythm=ArrivalRhythm(sessions_per_business_day=1.5, opens_at=8, closes_at=18),
        requests_range=(6, 20),
        seconds_between_requests=45.0,
    ),
    "occasional_custody": Regime(
        rhythm=ArrivalRhythm(
            sessions_per_business_day=2.0, opens_at=9, closes_at=19, dispersion=2
        ),
        requests_range=(8, 25),
        seconds_between_requests=90.0,
    ),
}
"""O comportamento de cada regime, e nao de cada perfil.

A coluna `regime` existe para que o M2 dependa do comportamento e nao do rotulo
do perfil. A correspondencia e um para um, mas a indirecao e o que impede o
gerador de consultar `profile` para decidir ritmo.

A media entre requisicoes fixa a duracao tipica da sessao: o lote descarrega em
cerca de um minuto, a sessao esporadica dura entre cinco e quinze minutos, e a
de custodia entre doze e quarenta. E dessa duracao que sai a taxa, uma das cinco
dimensoes que sigma interpola.
"""

DEFAULT_STALE_SCOPE_RATE = 0.005
"""Fracao das requisicoes que endereca chave fora dos escopos atuais (D-056).

O M4 as nega por politica. Sem elas, nenhuma requisicao legitima poderia ser
negada — a D-042 manda o operador pedir so chaves dos seus escopos e o M4
autoriza por escopo — e `denied > 0` passaria a significar atacante, que e o
separador trivial que a D-028 procura.

A tentativa e comum, sem marca: quem a distingue e o M4, ao avaliar a politica.
Valor provisorio ate a calibracao sobre o log de aquecimento.
"""

DEFAULT_MISTYPED_RATE = 0.003
"""Fracao das requisicoes que endereca identificador inexistente (D-056).

Digitacao errada ou referencia velha. Produz o desfecho de identificador
inexistente em trafego legitimo, que de outro modo tambem so viria do atacante.
Valor provisorio ate a calibracao.
"""

DEFAULT_DISTINCT_KEYS_RANGE = (3, 12)
"""Chaves distintas que uma sessao legitima toca (D-058).

Lido contra o alcance medido nas 30 sementes: o administrador alcanca 96,5
chaves na mediana, entao uma sessao legitima usa no maximo cerca de 12 % do que
a credencial permite. E esse contraste que o atributo de chaves distintas mede
contra a varredura ampla do atacante.

Limitado pelo tamanho da sessao e pelo alcance do operador, os dois menores em
alguns casos: usuario de escopo pequeno alcanca so 4 chaves.
"""


@dataclass(frozen=True)
class TrafficSpecification:
    """Como o trafego legitimo deve ser gerado."""

    first_day: date = FIRST_DAY
    week_count: int = WEEK_COUNT
    stale_scope_rate: float = DEFAULT_STALE_SCOPE_RATE
    mistyped_rate: float = DEFAULT_MISTYPED_RATE
    distinct_keys_range: tuple[int, int] = DEFAULT_DISTINCT_KEYS_RANGE

    @property
    def day_count(self) -> int:
        """Dias corridos do periodo simulado."""
        return self.week_count * DAYS_PER_WEEK
