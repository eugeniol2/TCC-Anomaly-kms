"""Composicao do M5: da tentativa mais o desfecho ao registro de auditoria.

O KMS **decide**, o Audit Logger **registra**. A separacao e de dominio, nao
de codigo: por isso o M4 entrega so `event_id` e `outcome` (D-078), e a
montagem do registro acontece aqui.

Mora separado da linha de comando para que o teste exercite exatamente o que
a execucao real exercita.
"""

from __future__ import annotations

import pandas as pd

from src.shared.phases import belongs_to

COLUMNS = (
    "event_id",
    "session_id",
    "operator_id",
    "timestamp",
    "source_ip",
    "operation",
    "key_id",
    "outcome",
)
"""As oito colunas de `log.csv` (D-064).

Sao as sete do `requests.csv` mais `outcome`, sem renomear nada, e por isso o
identificador ja nasce `event_id` no M2, e nao `request_id`.

**Nada de escopo, perfil ou proprietario.** Poriam o modelo em condicao de
reconstruir a fronteira de autorizacao, e ele passaria a aprender a politica
em vez do comportamento. O sinal tem de vir do ritmo e do desvio.

**E nada de rotulo** (D-063). Ele viaja em `compromised_sessions.csv` e o M7
o junta so na fase `evaluated`. Um log de auditoria que carrega verdade de
fundo deixa de ser um log.

O `key_id` e o **requisitado**, nao o resolvido, para que identificador
inexistente seja observavel no arquivo.
"""


def build_log(
    requests: pd.DataFrame, outcomes: pd.DataFrame, phase: str
) -> pd.DataFrame:
    """Junta as tentativas da fase aos desfechos, por `event_id`.

    A juncao e por igualdade de chave e nao por posicao de linha. Por posicao
    funcionaria hoje, porque o M4 preserva a ordem, mas passaria a depender de
    uma garantia que nenhuma entrada do registro da.

    Junta por dentro: evento sem desfecho, ou desfecho sem evento, some em vez
    de virar linha com celula vazia. O teste confere que nao sumiu nenhum.
    """
    of_phase = requests[belongs_to(phase, requests["timestamp"])]

    joined = of_phase.merge(outcomes, on="event_id", how="inner")

    return joined[list(COLUMNS)].reset_index(drop=True)
