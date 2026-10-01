from __future__ import annotations

from dataclasses import dataclass

from src.formulas.interpolation import interpolate, interpolate_range
from src.entities.scenario_engine.attack.parameters import AttackSpecification
from src.entities.scenario_engine.traffic.parameters import TrafficSpecification
from src.entities.scenario_engine.traffic.regimes import ArrivalRhythm, Regime


@dataclass(frozen=True)
class HourWindow:
    opens_at_hour: int
    closes_at_hour: int


@dataclass(frozen=True)
class Stealth:
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
    return HourWindow(rhythm.opens_at_hour, rhythm.closes_at_hour)


def stealth_of(
    sigma: float,
    regime: Regime,
    traffic: TrafficSpecification,
    attack: AttackSpecification,
) -> Stealth:
    """As cinco dimensoes resolvidas para este sigma e este regime.

    O extremo furtivo sai de `regime` e `traffic`, os mesmos objetos do trafego
    legitimo: mudar um muda o outro.
    """
    is_scheduled = not isinstance(regime.rhythm, ArrivalRhythm)

    if is_scheduled:
        raise ValueError(
            "o administrador comprometido tem ritmo de chegada, nao de lote; "
            f"recebido: {type(regime.rhythm).__name__}"
        )

    return Stealth(
        seconds_between_requests=interpolate(
            sigma, attack.ostensive_seconds_between_requests, regime.seconds_between_requests
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
            attack.ostensive_opens_at_hour, attack.ostensive_closes_at_hour
        ),
    )
