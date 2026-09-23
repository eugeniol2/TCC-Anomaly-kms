"""Como sigma interpola as cinco dimensoes do comportamento do atacante (D-082).

Este arquivo e o vocabulario: quais sao as dimensoes e que forma cada uma tem.
Os extremos ostensivos moram em `parameters.py` e os furtivos vem do M2, pela
mesma separacao que ha entre `traffic/regimes.py` e `traffic/parameters.py`.

Duas dimensoes sao **probabilidades** e tres sao **grandezas**, e a diferenca
nao e arbitraria. Horario atipico e origem inedita sao atributos binarios da
sessao: nao existe meia origem inedita, entao o que sigma pode regular e a
**chance** de a sessao ter aquela marca. Taxa, chaves distintas e falhas sao
contagens, e ali sigma regula o **valor**.

A propriedade que este arquivo garante, e que o teste confere, e que em
**sigma 1 toda dimensao vale exatamente o que o M2 usaria** para aquele
administrador. A sessao comprometida passa a sair da mesma distribuicao da
legitima, e nenhum mecanismo pode separa-las — e o piso declarado da varredura.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.attack.parameters import AttackSpecification
from src.traffic.parameters import TrafficSpecification
from src.traffic.regimes import ArrivalRhythm, Regime


@dataclass(frozen=True)
class HourWindow:
    """Faixa de horas em que a sessao pode abrir."""

    opens_at: int
    closes_at: int


@dataclass(frozen=True)
class Stealth:
    """As cinco dimensoes ja resolvidas para um valor de sigma.

    Quem consome isto nao sabe o que sigma vale nem como a interpolacao e
    feita: recebe valores prontos e sorteia com eles. E o que permite testar a
    convergencia comparando este objeto contra os parametros do M2, sem
    executar sessao nenhuma.
    """

    seconds_between_requests: float
    requests_range: tuple[int, int]
    distinct_keys_range: tuple[int, int]
    atypical_hour_chance: float
    novel_address_chance: float
    stale_scope_rate: float
    absent_identifier_rate: float
    usual_window: HourWindow
    atypical_window: HourWindow


def between(sigma: float, ostensive: float, furtive: float) -> float:
    """Interpolacao linear do extremo ostensivo ao furtivo.

    Em sigma 0 devolve o ostensivo, em sigma 1 o furtivo. Linear porque a
    varredura tem onze pontos igualmente espacados (D-004): curva com joelho
    concentraria a mudanca em poucas condicoes e deixaria as outras quase
    iguais entre si, desperdicando pontos da grade.

    A forma com os dois pesos, e nao `ostensive + sigma * (furtive -
    ostensive)`, e deliberada. A segunda e algebricamente identica mas nao
    devolve o extremo **exato** em ponto flutuante: com 0,30 e 0,005 ela da
    0,005000000000000004 em sigma 1. A diferenca nao muda sorteio nenhum, mas
    desfaz a igualdade exata contra os parametros do M2 — que e a propriedade
    que o teste de convergencia verifica, e a unica prova barata de que
    sigma 1 e mesmo o piso.
    """
    return (1.0 - sigma) * ostensive + sigma * furtive


def range_between(
    sigma: float, ostensive: tuple[int, int], furtive: tuple[int, int]
) -> tuple[int, int]:
    """Interpola as duas pontas de uma faixa inteira, arredondando cada uma."""
    lowest = round(between(sigma, ostensive[0], furtive[0]))
    highest = round(between(sigma, ostensive[1], furtive[1]))

    return lowest, highest


def window_of(rhythm: ArrivalRhythm) -> HourWindow:
    """A janela horaria do regime que o atacante personifica."""
    return HourWindow(rhythm.opens_at, rhythm.closes_at)


def stealth_of(
    sigma: float,
    regime: Regime,
    traffic: TrafficSpecification,
    attack: AttackSpecification,
) -> Stealth:
    """As cinco dimensoes resolvidas para este sigma e este regime.

    O extremo furtivo sai de `regime` e de `traffic`, que sao os mesmos
    objetos que o M2 consome. Nao ha numero repetido entre os dois modulos, e
    por isso recalibrar o trafego legitimo move o extremo furtivo junto.
    """
    is_scheduled = not isinstance(regime.rhythm, ArrivalRhythm)

    if is_scheduled:
        raise ValueError(
            "o administrador comprometido tem ritmo de chegada, nao de lote; "
            f"recebido: {type(regime.rhythm).__name__}"
        )

    return Stealth(
        seconds_between_requests=between(
            sigma, attack.ostensive_request_interval, regime.seconds_between_requests
        ),
        requests_range=range_between(
            sigma, attack.ostensive_requests_range, regime.requests_range
        ),
        distinct_keys_range=range_between(
            sigma, attack.ostensive_distinct_keys_range, traffic.distinct_keys_range
        ),
        atypical_hour_chance=1.0 - sigma,
        novel_address_chance=1.0 - sigma,
        stale_scope_rate=between(
            sigma, attack.ostensive_stale_scope_rate, traffic.stale_scope_rate
        ),
        absent_identifier_rate=between(
            sigma,
            attack.ostensive_absent_identifier_rate,
            traffic.absent_identifier_rate,
        ),
        usual_window=window_of(regime.rhythm),
        atypical_window=HourWindow(
            attack.ostensive_opens_at, attack.ostensive_closes_at
        ),
    )
