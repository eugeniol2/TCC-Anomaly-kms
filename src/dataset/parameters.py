"""Os numeros que governam o M7.

Sao dois, e os dois existem para o mesmo tipo de caso degenerado: sessao em que
todos os eventos caem no mesmo segundo. E raro, mas nao impossivel, e sem piso
a taxa de requisicoes seria divisao por zero.
"""

from __future__ import annotations

SHORTEST_MEASURABLE_MINUTES = 1 / 60
"""Piso da duracao ao calcular a taxa, equivalente a um segundo.

Uma sessao cujos eventos caem todos no mesmo segundo tem duracao zero, e
`eventos / 0` e infinito. Um segundo e a menor duracao que o `timestamp`
consegue representar (ele e ISO com segundos, D-066), entao o piso nao
inventa resolucao que o log nao tem: ele apenas recusa afirmar que o intervalo
foi menor do que o formato sabe medir.

A alternativa, deixar a taxa como infinito ou ausente, poria um valor especial
numa coluna numerica que arvores e ensembles teriam de tratar. Prefere-se um
numero grande e finito.
"""

LABEL_ABSENT = 0
LABEL_PRESENT = 1
"""O rotulo como inteiro, nao como booleano.

CSV nao tem tipo booleano, e `True`/`False` viram texto que cada leitor
interpreta a seu modo. 0 e 1 atravessam a escrita e a leitura sem ambiguidade,
e e o que scikit-learn espera em `y`.
"""
