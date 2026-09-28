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
  globals/              auxiliares comuns a todos os módulos
    rng.py              fluxos de aleatoriedade derivados da semente
    tables.py           escrita de CSV e embaralhamento de linhas
    layout.py           onde cada arquivo mora dentro de data/
    phases.py           o calendário: âncora, semanas e as duas fases
    experiment.py       a grade: sementes 1 a 30, sigma de 0,0 a 1,0
  population/           M1  operadores, escopos e chaves
  traffic/              M2  o tráfego legítimo das oito semanas
  attack/               M3  a campanha de ataque, interpolada por sigma
  kms/                  M4  o desfecho de cada requisição, pela política
  audit_logger/         M5  o log de auditoria
  historical_profiles/  M6  o perfil histórico de cada operador, da régua
  dataset/              M7  uma linha por sessão, com os oito atributos
  calibration/          M8  os limiares do baseline, da régua
  pipeline/             o orquestrador: a ordem de execução, em código
  viewer/               a tela do Streamlit que mostra o pipeline por dentro
  examples/             demonstração dos fluxos de aleatoriedade
data/                   saída CSV de todos os módulos (não versionada)
tests/                  determinismo, formato e invariantes, nas 30 sementes
```

Cada módulo é um pacote com o fluxo principal em `__main__.py`, os números que o
governam em `parameters.py` quando os tem, e um arquivo por conceito.

A pasta `data/` espelha a dependencia dos modulos. Como o atacante age apenas nas
semanas 5 a 8, tudo que deriva do aquecimento e independente de sigma, e o
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
    requests.csv                  M2, oito semanas, so legitimo
    outcomes.csv  log.csv         M4, M5, semanas 1 a 4
    historical_profiles.csv       M6, do aquecimento inteiro
    sessions.csv                  M7, semanas 1 a 4
    thresholds.csv                M8, do aquecimento inteiro
    sigma-0.0/                    ---- ramo de sigma, 330 execucoes ----
      requests.csv                M3, semanas 5 a 8, legitimo + ataque
      compromised_sessions.csv    M3
      run.csv                     M3, qual administrador foi comprometido
      outcomes.csv  log.csv       M4, M5, semanas 5 a 8
      sessions.csv                M7, semanas 5 a 8
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
| M3 | `attack` | seed, sigma, tabelas, `requests.csv` | `requests.csv` (semanas 5 a 8, legítimo + ataque), `compromised_sessions.csv`, `run.csv` |
| M4 | `kms` | fase, `requests.csv`, `keys.csv`, `operators.csv` | `outcomes.csv` |
| M5 | `audit_logger` | fase, requests, outcomes | `log.csv` |
| M6 | `historical_profiles` | `log.csv` (aquecimento inteiro) | `historical_profiles.csv` |
| M7 | `dataset` | fase, `log.csv`, profiles, `compromised_sessions.csv` (só em `evaluated`) | `sessions.csv` |
| M8 | `calibration` | `sessions.csv` (aquecimento inteiro) | `thresholds.csv` |
| M9 | `partition` | `sessions.csv` (semanas 5 a 8) | `train.csv`, `holdout.csv` |
| M10 | `baseline` | `holdout.csv`, `thresholds.csv` | `predictions_rules.csv` |
| M11 | `models` | `train.csv`, `holdout.csv`, config | `predictions_ml.csv` |
| M12 | `evaluation` | predictions | `metrics.csv` |

As oito semanas simuladas têm dois papéis. As semanas 1 a 4 constroem a **régua** (o
perfil histórico de cada operador **e** os limiares do baseline, do mesmo período) e as
semanas 5 a 8 são o período avaliado, o único em que o atacante age. A história é a de
uma implantação: uma empresa com quatro semanas de log constrói o baseline e o põe em
produção.

O M3 lê o `requests.csv` do M2 e escreve outro, mesclando a campanha às semanas 5 a 8
do tráfego legítimo, nunca acrescentando linhas ao arquivo do M2. Já `fase` não é
arquivo: é o parâmetro obrigatório de M4, M5 e M7, que diz qual arquivo o módulo lê e
em qual dos dois ramos escreve.

O rótulo de sessão comprometida **não é coluna do `log.csv`**: ele viaja em
`compromised_sessions.csv` e o M7 o junta só na fase `evaluated`. Um log de auditoria
que carrega verdade de fundo deixa de ser um log, e coluna que não existe no arquivo
não pode vazar. Por isso o `sessions.csv` do aquecimento não tem rótulo: ele alimenta
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

O orquestrador roda os módulos na ordem certa, que mora em `src/pipeline/build.py`:

```
python -m src.pipeline --seed 1                # varredura: aquecimento + 11 sigmas
python -m src.pipeline --seed 1 --sigma 0.5    # só uma condição
python -m src.pipeline --warmup --seed 1       # só o ramo da semente
python -m src.pipeline --grade                 # as 330, e o runs.csv
```

`--out` aceita outra raiz para `data/`. A grade inteira ocupa cerca de 3,4 GB.

Cada módulo também roda sozinho, com a semente como parâmetro explícito, e os que rodam
nos dois ramos (M4, M5, M7) exigem `--fase warmup` ou `--fase evaluated`:

```
python -m src.population --seed 7
python -m src.traffic    --seed 7
python -m src.kms        --seed 7 --fase warmup
```

O módulo lê os arquivos que o anterior escreveu naquela semente, então a ordem importa, e
ele falha com mensagem clara se eles não existirem.

A tela que mostra o pipeline por dentro, passo a passo e com os dados de verdade:

```
streamlit run src/viewer/app.py
```

## Testes

```
python -m pytest
```

São 1527 testes, em cerca de sete minutos. Cobrem determinismo e as invariantes de que os
módulos seguintes dependem, e rodam nas 30 sementes da grade, não numa só, porque falha
específica de semente é o que passa despercebido.

| Arquivo | Testes | O que garante |
|---|---|---|
| `test_population.py` | 334 | todo escopo tem detentor, a chave não tem dono, identificador nunca sequencial |
| `test_traffic.py` | 547 | nenhuma coluna carrega o desfecho, origem sempre habitual, as duas falhas legítimas ocorrem, o ritmo de cada regime |
| `test_kms.py` | 307 | a ordem de avaliação dos desfechos, o log com oito colunas e sem rótulo, os quatro desfechos no tráfego limpo |
| `test_attack.py` | 147 | sigma 0 e sigma 1 reproduzem os dois extremos, a campanha só nas semanas 5 a 8, o mesmo administrador em todo sigma |
| `test_dataset.py` | 140 | perfis, sessões e limiares, e o rótulo só no período avaliado |
| `test_viewer.py` | 39 | a tela mostra o que o pipeline produz, e as curvas batem com o gerador |
| `test_pipeline.py` | 8 | o orquestrador grava o mesmo que os módulos gravariam, e sempre os mesmos bytes |
| `test_experiment.py` | 5 | a grade de sementes e de sigma, e as sementes reservadas fora dela |

O `test_reference_output_has_not_changed` e detector de mudanca, nao teste de
correcao: falha sempre que o gerador mudar, inclusive de proposito. Quando
falhar, confirme se a mudanca era intencional, registre a decisao e atualize o
valor de referencia. O do M1 compara o CSV inteiro; o do M2 compara um resumo
SHA-256, porque o arquivo tem cerca de 77 mil linhas e versiona-lo pesaria mais
que o codigo.

## Reprodutibilidade

Semente mais código determinam a saída inteira. É por isso que `data/` não é versionada:
apagar a pasta e reexecutar reproduz os CSVs byte a byte, e é assim que o determinismo é
conferido.

Fluxos de aleatoriedade são separados por subsistema (população e chaves, tráfego
legítimo, campanha de ataque), todos derivados da mesma semente. A grade experimental é
de 11 condições de sigma (0,0 a 1,0) por 30 réplicas, totalizando 330 execuções.

## Estado

Em construção. **M1 a M8 implementados, e o orquestrador**; M9 a M12 pendentes. O ramo da
semente já produz a régua inteira (o perfil histórico e os limiares do baseline, das
quatro semanas de aquecimento) e o ramo de sigma já produz o conjunto rotulado das
semanas 5 a 8.

O M2 produz cerca de 77 mil requisições em 3,8 mil sessões por semente. Os três itens que
as convenções do projeto mandam conferir antes do M3 estão cobertos por teste: o serviço
automatizado abre lote nas quatro horas fixas, o administrador tem ritmo mais disperso que
o regime `routine`, e os dois caminhos de falha legítima ocorrem nas 30 sementes.
