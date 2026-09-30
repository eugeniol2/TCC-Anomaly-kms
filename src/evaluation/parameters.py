"""Os numeros que governam a avaliacao (M12)."""

from __future__ import annotations

EXCLUSION_F1 = 0.95
"""F1 mediano da arvore rasa a partir do qual a condicao sai da comparacao principal (D-028)."""

STUMP_DEPTH = 1
"""A arvore da verificacao de trivialidade: um corte so, sem peso por classe (D-028, D-116)."""

STUMP_RANDOM_STATE = 0
"""So desempata dois atributos de ganho igual; nao muda a arvore em mais nada."""

ALTERNATIVE = "two-sided"
"""Wilcoxon bilateral: acusa diferenca nos dois sentidos (D-116)."""

ZERO_METHOD = "wilcox"
"""Diferencas exatamente zero saem da conta, como no teste classico (D-116)."""

SIGNIFICANCE = 0.05
"""O alfa, aplicado ao p-valor ja corrigido por Holm (D-116)."""
