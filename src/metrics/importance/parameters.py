"""Os numeros que governam a importancia por permutacao (D-124)."""

from __future__ import annotations

REPEATS = 10
"""Quantas vezes cada atributo e embaralhado em cada execucao (D-124).

A importancia da execucao e a media das quedas de F1 nas dez permutacoes, e a que
se reporta por sigma e a mediana das 30 sementes. Arbitrario, pelo custo: dez
repeticoes levam cerca de 1,3 s por execucao, e com o retreino dos modelos a
importancia inteira leva uns 11 minutos nas 330 (680 s em 01/10).
"""
