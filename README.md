# anomaly-detection-kms

Pipeline experimental de um TCC de Sistemas de Informação (UFRPE). Compara um baseline
de regras contra modelos supervisionados na detecção de anomalias em metadados de
auditoria de um KMS experimental.

O modelo de ataque é credencial legítima de administrador comprometida: o atacante age
sob a credencial de um administrador real, o que elimina detecção por controle de acesso
e deixa o comportamento como único sinal.

Não há log real de KMS disponível. Os dados são gerados pelo próprio pipeline, e a
documentação do gerador é parte do trabalho.

## Estrutura

```
src/
  shared/          auxiliares comuns a M1..M12
    rng.py         fluxos de aleatoriedade derivados da semente
    tables.py      escrita de CSV e embaralhamento de linhas
    layout.py      onde cada arquivo mora dentro de data/
    experiment.py  a grade: sementes 1 a 30, sigma de 0,0 a 1,0
  population/      M1, um arquivo por conceito
    __main__.py    linha de comando e fluxo principal
    specification.py  parametros do repositorio de chaves
    scopes.py      escopos, compartilhados por operadores e chaves
    profiles.py    perfis comportamentais da populacao
    operators.py   construcao de operators.csv
    keys.py        construcao de keys.csv
data/              saida CSV de todos os modulos (nao versionada)
tests/             verificacao de determinismo e de formato
```

A pasta `data/` espelha a dependencia dos modulos. Como o atacante age apenas nas
semanas 4 e 5, tudo que deriva das semanas 1 a 3 e independente de sigma — e o
pipeline tem dois ramos. M4, M5 e M7 aparecem nos dois.

```
data/
  runs.csv                        indice das 330 execucoes
  metrics.csv                     agregado final (M12)
  preparation/
    seed-902/                     busca de hiperparametros
    seed-903/                     limiar X de exclusao por trivialidade
  seed-01/                        ---- ramo da semente, 30 execucoes ----
    operators.csv  keys.csv       M1
    requests.csv                  M2, cinco semanas, so legitimo
    outcomes.csv  log.csv         M4, M5 — semanas 1 a 3
    historical_profiles.csv       M6, das semanas 1 e 2
    sessions.csv                  M7, da semana 3
    thresholds.csv                M8, da semana 3
    sigma-0.0/                    ---- ramo de sigma, 330 execucoes ----
      requests.csv                M3, semanas 4 e 5, legitimo + ataque
      compromised_sessions.csv    M3
      outcomes.csv  log.csv       M4, M5 — semanas 4 e 5
      sessions.csv                M7, semanas 4 e 5
      train.csv  holdout.csv      M9
      predictions_rules.csv       M10
      predictions_ml.csv          M11
    sigma-0.1/ ... sigma-1.0/
  seed-02/ ... seed-30/
```

O `seed-NN/log.csv` sendo unico por semente e a garantia fisica de que o atacante
nao toca o aquecimento: nao ha lugar onde ele pudesse estar.

Cada modulo M1..M12 e um pacote sob `src/`, com o fluxo principal em `__main__.py`
e um arquivo por conceito. A fronteira entre modulos continua sendo o arquivo CSV,
nao a chamada de funcao.

## Módulos

Cada módulo lê arquivo e escreve arquivo. A fronteira entre módulos é o arquivo, não a
chamada de função: cada um roda isolado e a saída é inspecionável antes do próximo.

| | Módulo | Entrada | Saída |
|---|---|---|---|
| M1 | `population` | seed | `operators.csv`, `keys.csv` |
| M2 | `traffic` | seed, tabelas | `requests.csv` |
| M3 | `attack` | seed, sigma, tabelas, `requests.csv` | `requests.csv` (semanas 4 e 5, legítimo + ataque), `compromised_sessions.csv` |
| M4 | `kms` | fase, `requests.csv`, `keys.csv` | `outcomes.csv` |
| M5 | `audit_logger` | fase, requests, outcomes | `log.csv` |
| M6 | `historical_profiles` | `log.csv` (semanas 1 e 2) | `historical_profiles.csv` |
| M7 | `dataset` | fase, `log.csv`, profiles, `compromised_sessions.csv` (só em `evaluated`) | `sessions.csv` |
| M8 | `calibration` | `sessions.csv` (semana 3) | `thresholds.csv` |
| M9 | `partition` | `sessions.csv` (semanas 4 e 5) | `train.csv`, `holdout.csv` |
| M10 | `baseline` | `holdout.csv`, `thresholds.csv` | `predictions_rules.csv` |
| M11 | `models` | `train.csv`, `holdout.csv`, config | `predictions_ml.csv` |
| M12 | `evaluation` | predictions | `metrics.csv` |

As cinco semanas simuladas têm papéis distintos: as semanas 1 e 2 constroem o perfil
histórico, a semana 3 calibra os limiares do baseline, e as semanas 4 e 5 são o
período avaliado, o único em que o atacante age.

O M3 lê o `requests.csv` do M2 e escreve outro, mesclando a campanha às semanas 4 e 5
do tráfego legítimo — nunca acrescentando linhas ao arquivo do M2. Já `fase` não é
arquivo: é o parâmetro obrigatório de M4, M5 e M7, que diz qual arquivo o módulo lê e
em qual dos dois ramos escreve.

O rótulo de sessão comprometida **não é coluna do `log.csv`**: ele viaja em
`compromised_sessions.csv` e o M7 o junta só na fase `evaluated`. Um log de auditoria
que carrega verdade de fundo deixa de ser um log, e coluna que não existe no arquivo
não pode vazar. Por isso o `sessions.csv` do aquecimento não tem rótulo — ele alimenta
só a calibração, que não usa rótulo.

Nomes de código e de arquivo em inglês; o texto da monografia é em português e traz uma
tabela de correspondência entre os dois.

## Ambiente

Python 3.11.

```
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`requirements.txt` fixa as dependências diretas. `requirements-lock.txt` tem o
congelamento completo, para recriar o ambiente exatamente.

## Execução

Cada módulo roda sozinho pela linha de comando e recebe a semente como parâmetro
explícito:

```
python -m src.population --seed 42
```

## Testes

```
python -m pytest
```

Cobrem determinismo e as invariantes de que os modulos seguintes dependem:
toda chave tem proprietario que detem seu escopo, todo escopo tem detentor,
identificadores de chave nunca sequenciais. As invariantes rodam nas 30
sementes da grade, nao numa so.

O `test_reference_output_has_not_changed` e detector de mudanca, nao teste de
correcao: falha sempre que o gerador mudar, inclusive de proposito. Quando
falhar, confirme se a mudanca era intencional, registre a decisao e atualize o
valor de referencia.

## Reprodutibilidade

Semente mais código determinam a saída inteira. É por isso que `data/` não é versionada:
apagar a pasta e reexecutar reproduz os CSVs byte a byte, e é assim que o determinismo é
conferido.

Fluxos de aleatoriedade são separados por subsistema — população e chaves, tráfego
legítimo, campanha de ataque — todos derivados da mesma semente. A grade experimental é
de 11 condições de sigma (0,0 a 1,0) por 30 réplicas, totalizando 330 execuções.

## Estado

Em construção. **M1 implementado**, M2 a M12 pendentes. A Meta 1 é M1, M2, M4 e M5 sem
atacante, gerando as cinco semanas de uma execução limpa — é dela que sairão o perfil
histórico das semanas 1 e 2 e os limiares da semana 3.
