"""Os numeros que governam o M9.

E um so. O resto da particao e forma, e mora nas decisoes: por sessao e nunca por
evento (D-018, D-061), estratificada por classe (D-018), e com o mesmo sorteio
nas onze condicoes de sigma de uma semente (D-102).
"""

from __future__ import annotations

HOLDOUT_SHARE = 0.4
"""Fracao de cada classe que vai para o holdout (D-070).

Era 0,3 na D-020. Dez pontos transferidos para o holdout aumentam as positivas em
um terco sem custar execucao nenhuma, e o treino continua folgado para arvores.
Com as 58 sessoes da campanha (D-081), sao 23 positivas no holdout, a contagem
absoluta que a armadilha de Athapaththu et al. cobra.

Aplicada **dentro de cada classe**, e nao sobre o conjunto: e o que torna a
estratificacao exata em vez de aproximada. O numero de cada lado e o inteiro mais
proximo; a regra do meio esta em `holdout_count`.
"""
