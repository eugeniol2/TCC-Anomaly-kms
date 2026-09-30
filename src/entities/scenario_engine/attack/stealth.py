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
legitima, e nenhum mecanismo pode separa-las. E o piso declarado da varredura.

A interpolacao e **linear** porque a varredura tem onze pontos igualmente
espacados (D-004): curva com joelho concentraria a mudanca em poucas condicoes e
deixaria as outras quase iguais entre si, desperdicando pontos da grade. A conta
mora em `src/formulas/interpolation.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.formulas.interpolation import interpolate, interpolate_range
from src.entities.scenario_engine.attack.parameters import AttackSpecification
from src.entities.scenario_engine.traffic.parameters import TrafficSpecification
from src.entities.scenario_engine.traffic.regimes import ArrivalRhythm, Regime


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
        seconds_between_requests=interpolate(
            sigma, attack.ostensive_request_interval, regime.seconds_between_requests
        ),
        requests_range=interpolate_range(
            sigma, attack.ostensive_requests_range, regime.requests_range
        ),
        distinct_keys_range=interpolate_range(
            sigma, attack.ostensive_distinct_keys_range, traffic.distinct_keys_range
        ),
        atypical_hour_chance=1.0 - sigma,
        novel_address_chance=1.0 - sigma,
        stale_scope_rate=interpolate(
            sigma, attack.ostensive_stale_scope_rate, traffic.stale_scope_rate
        ),
        absent_identifier_rate=interpolate(
            sigma,
            attack.ostensive_absent_identifier_rate,
            traffic.absent_identifier_rate,
        ),
        usual_window=window_of(regime.rhythm),
        atypical_window=HourWindow(
            attack.ostensive_opens_at, attack.ostensive_closes_at
        ),
    )
