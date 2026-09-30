"""As entidades da arquitetura da proposta (Figura 1), uma pasta cada.

Scenario Engine, KMS, Audit Logger, Dataset Generator, Policy Engine e os modelos
supervisionados (a caixa "Pipeline de ML"). Cada modulo mora na entidade a que pertence
e grava o seu arquivo (D-121, D-123).

Nenhuma entidade importa de `src.metrics`: a avaliacao le o que elas gravaram, e nunca
o contrario.
"""
