"""Os numeros que governam a calibracao dos limiares do baseline (M8)."""

from __future__ import annotations

PERCENTILE = 99
"""Percentil do aquecimento que vira limiar de cada regra de grandeza (D-031, D-043).

Escolhido antes de qualquer dado e mantido desde entao. Ele fixa, por
construcao, que cerca de **1 % das sessoes do aquecimento** dispararia cada
regra — e o aquecimento e limpo, entao esse 1 % e alarme falso por definicao.
E o orcamento de triagem que a D-075 depois transformou em "duas ou mais regras
disparadas".

Um percentil mais baixo compraria revocacao com alarme falso; mais alto deixaria
o limiar encostar no maximo observado e a regra nasceria quase morta — que foi
o que aconteceu com `chaves_por_evento` e obrigou a D-080 a troca-la.
"""

THRESHOLD_ATTRIBUTES = (
    "events",
    "duration_minutes",
    "requests_per_minute",
    "distinct_keys",
    "failures_per_event",
    "denials_per_event",
)
"""As seis regras de grandeza (D-080).

Os outros dois atributos dos oito — `atypical_hour` e `new_source_ip` — sao
**regras de historico** e nao tem limiar: eles ja sao binarios e disparam quando
valem 1. Calibrar percentil sobre uma coluna de zeros e uns daria 0 ou 1 e nao
significaria nada.
"""
