"""Os numeros que governam o baseline de regras (M10).

Sao poucos porque o baseline **nao tem nada para ajustar**: os limiares chegam
prontos do M8, calibrados no aquecimento, e ficam congelados (D-033, D-046). O
que mora aqui e so a forma de combinar as regras.
"""

from __future__ import annotations

MINIMUM_RULES_FIRED = 2
"""Quantas regras precisam disparar para a sessao virar alerta (D-075).

Qualquer combinacao de duas serve: duas de grandeza, duas de historico ou uma de
cada. Escolhido por **orcamento de triagem**, olhando so trafego limpo: com uma
regra bastando, o baseline marcava cerca de 13 % das sessoes legitimas; com duas,
menos de 1 %. Escolher olhando o F1 numa preparatoria rotulada daria ao baseline
o treino que a D-033 lhe nega.
"""

HISTORY_RULE_FIRES = 1
"""O valor em que uma regra de historico dispara (D-080).

`atypical_hour` e `new_source_ip` ja chegam do M7 como 0 ou 1, lidos contra o
perfil de cada operador. Nao tem limiar: disparam quando valem 1.
"""
