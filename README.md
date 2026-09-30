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

As pastas de `src/` são as **entidades da arquitetura da proposta** (Figura 1), e cada
módulo mora na entidade a que pertence.

```
src/
  main.py                    o comando único: roda tudo, de ponta a ponta
  scenario_engine/           o gerador de cenários
    population/              M1  os operadores, com perfil, regime e escopos
    traffic/                 M2  o tráfego legítimo das oito semanas
    attack/                  M3  a campanha de ataque, interpolada por sigma
  kms/                       o Experimental KMS
    repository/              M1  o repositório de chaves e os escopos
    build.py, policy.py      M4  o desfecho de cada requisição, pela política
  audit_logger/              M5  o log de auditoria
  dataset_generator/         da auditoria ao conjunto de dados
    historical_profiles/     M6  o perfil histórico de cada operador, da régua
    dataset/                 M7  uma linha por sessão, com os oito atributos
    partition/               M9  treino e holdout, a mesma divisão em todo sigma
  policy_engine/             o baseline de regras
    calibration/             M8  os limiares, da régua
    baseline/                M10 as oito regras decidindo sobre o holdout
  models/                    M11 Random Forest e XGBoost, e a busca de hiperparâmetros
  evaluation/                M12 métricas, trivialidade, Wilcoxon com Holm e tempo
    figures/                 as figuras da monografia
  pipeline/                  a ordem: os passos de cada ramo e o experimento
  shared/                    sementes, calendário, pastas, CSV e cronômetro
  viewer/                    a tela do Streamlit que mostra o pipeline por dentro
  examples/                  demonstração dos fluxos de aleatoriedade
data/                        saída CSV de todos os módulos (não versionada)
tests/                       determinismo, formato e invariantes, nas 30 sementes
```

Cada módulo é um pacote com a lógica em `build.py`, os números que o governam em
`parameters.py` quando os tem, e um arquivo por conceito. O M1 se divide entre duas
entidades, porque os operadores e as chaves saem do mesmo sorteio; quem os monta, nessa
ordem, é o `src/pipeline/population.py`.

A pasta `data/` espelha a dependencia dos modulos. Como o atacante age apenas nas
semanas 5 a 8, tudo que deriva do aquecimento e independente de sigma, e o
pipeline tem dois ramos. M4, M5 e M7 aparecem nos dois.

```
data/
  runs.csv                        indice das 330 execucoes
  metrics.csv                     M12, cada mecanismo em cada execucao
  triviality.csv                  M12, a arvore rasa e as duplicatas
  comparison.csv                  M12, Wilcoxon e Holm por sigma e modelo
  timing.csv                      M12, o tempo de cada mecanismo
  preparation/
    seed-902/                     busca de hiperparametros, com config.csv
    seed-903/                     ensaio do pipeline antes das 330
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
      timing_rules.csv            M10, o tempo, único arquivo não determinístico
      predictions_ml.csv          M11
      timing_ml.csv               M11, o tempo dos dois modelos
    sigma-0.1/ ... sigma-1.0/
  seed-02/ ... seed-30/
```

O `seed-NN/log.csv` sendo unico por semente e a garantia fisica de que o atacante
nao toca o aquecimento: nao ha lugar onde ele pudesse estar.

## Módulos

Cada módulo grava o seu arquivo, e o arquivo é a fronteira: a saída de cada um é
inspecionável antes do próximo.

| | Entidade | Módulo | Entrada | Saída |
|---|---|---|---|---|
| M1 | Scenario Engine e KMS | `population`, `repository` | semente | `operators.csv`, `keys.csv` |
| M2 | Scenario Engine | `traffic` | semente, tabelas | `requests.csv` |
| M3 | Scenario Engine | `attack` | semente, sigma, tabelas, `requests.csv` | `requests.csv` (semanas 5 a 8, legítimo + ataque), `compromised_sessions.csv`, `run.csv` |
| M4 | KMS | `kms` | fase, `requests.csv`, `keys.csv`, `operators.csv` | `outcomes.csv` |
| M5 | Audit Logger | `audit_logger` | fase, requests, outcomes | `log.csv` |
| M6 | Dataset Generator | `historical_profiles` | `log.csv` (aquecimento inteiro) | `historical_profiles.csv` |
| M7 | Dataset Generator | `dataset` | fase, `log.csv`, perfis, `compromised_sessions.csv` (só em `evaluated`) | `sessions.csv` |
| M8 | Policy Engine | `calibration` | `sessions.csv` (aquecimento inteiro) | `thresholds.csv` |
| M9 | Dataset Generator | `partition` | `sessions.csv` (semanas 5 a 8) | `train.csv`, `holdout.csv` |
| M10 | Policy Engine | `baseline` | `holdout.csv`, `thresholds.csv` | `predictions_rules.csv`, `timing_rules.csv` |
| M11 | Modelos | `models` | `train.csv`, `holdout.csv`, `config.csv` da 902 | `predictions_ml.csv`, `timing_ml.csv` |
| M12 | Avaliação | `evaluation` | as 330 execuções | `metrics.csv`, `triviality.csv`, `comparison.csv`, `timing.csv`, `figures/` |

As oito semanas simuladas têm dois papéis. As semanas 1 a 4 constroem a **régua** (o
perfil histórico de cada operador **e** os limiares do baseline, do mesmo período) e as
semanas 5 a 8 são o período avaliado, o único em que o atacante age. A história é a de
uma implantação: uma empresa com quatro semanas de log constrói o baseline e o põe em
produção.

O M3 lê o `requests.csv` do M2 e escreve outro, mesclando a campanha às semanas 5 a 8
do tráfego legítimo, nunca acrescentando linhas ao arquivo do M2. Já `fase` não é
arquivo: é o parâmetro obrigatório das funções do M4, do M5 e do M7, que o orquestrador
passa em cada ramo.

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

Um comando só roda o experimento inteiro:

```
python -m src.main
```

Ele roda, nesta ordem, a busca de hiperparâmetros na semente 902 (uns 9 minutos), o
ensaio na 903, as 330 execuções (uns 12 minutos), a avaliação e as figuras. No fim, os
resultados estão em `data/comparison.csv`, `metrics.csv`, `triviality.csv` e
`timing.csv`, e as figuras em `data/figures/`.

Os parâmetros dizem **o que** rodar e **onde** gravar:

| Parâmetro | O que faz | Padrão |
|---|---|---|
| `--sementes` | faixa (`1-30`) ou lista (`1,5,9`) | `1-30` |
| `--sigmas` | lista (`0.5` ou `0.0,0.5,1.0`) | os onze, de 0,0 a 1,0 |
| `--saida` | a raiz onde tudo é gravado | `data` |
| `--ate` | a entidade em que o pipeline para: `scenario_engine`, `kms`, `audit_logger`, `dataset_generator`, `policy_engine`, `models`, `evaluation` ou `figures` | `figures` |
| `--sem-busca` | reaproveita o `config.csv` de uma busca anterior na mesma saída | desligado |

```
python -m src.main --sem-busca                        # sem refazer os 9 minutos da busca
python -m src.main --sementes 1 --sigmas 0.5 --ate kms  # uma execução, até o KMS
python -m src.main --saida C:\tcc-data                # fora do OneDrive
```

Os números do experimento (58 sessões de ataque, 44 operadores, percentil 99...) **não**
são parâmetros do comando: moram no `parameters.py` de cada entidade, cada um com a
decisão que o fixou. A grade inteira ocupa cerca de 3,4 GB.

A tela que mostra o pipeline por dentro, passo a passo e com os dados de verdade:

```
streamlit run src/viewer/app.py
```

## Testes

```
python -m pytest
```

São 1675 testes, em cinco a oito minutos. Cobrem determinismo e as invariantes de que os
módulos seguintes dependem, e rodam nas 30 sementes da grade, não numa só, porque falha
específica de semente é o que passa despercebido.

| Arquivo | Testes | O que garante |
|---|---|---|
| `test_population.py` | 334 | todo escopo tem detentor, a chave não tem dono, identificador nunca sequencial |
| `test_traffic.py` | 547 | nenhuma coluna carrega o desfecho, origem sempre habitual, as duas falhas legítimas ocorrem, o ritmo de cada regime |
| `test_kms.py` | 307 | a ordem de avaliação dos desfechos, o log com oito colunas e sem rótulo, os quatro desfechos no tráfego limpo |
| `test_attack.py` | 147 | sigma 0 e sigma 1 reproduzem os dois extremos, a campanha só nas semanas 5 a 8, o mesmo administrador em todo sigma |
| `test_dataset.py` | 140 | perfis, sessões e limiares, e o rótulo só no período avaliado |
| `test_partition.py` | 40 | cada sessão de um lado só, 23 positivas no holdout, a mesma divisão em todo sigma |
| `test_baseline.py` | 63 | cada regra dispara onde a D-080 diz, qualquer par alerta e nenhuma regra sozinha, o rótulo não decide |
| `test_viewer.py` | 39 | a tela mostra o que o pipeline produz, e as curvas batem com o gerador |
| `test_models.py` | 15 | o rótulo do holdout não decide, a mesma semente treina os mesmos modelos, a busca escolhe pela regra de empate |
| `test_evaluation.py` | 19 | as métricas de uma matriz conhecida, o recorte dos administradores, a AUC, a árvore rasa, as duplicatas, Holm só sobre as condições mantidas |
| `test_pipeline.py` | 11 | o orquestrador grava o mesmo que os módulos gravariam, e sempre os mesmos bytes |
| `test_main.py` | 8 | o comando único: os parâmetros, as sementes reservadas recusadas, e o `--ate` parando na entidade certa |
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
conferido. A exceção declarada são os arquivos de tempo (`timing_rules.csv`,
`timing_ml.csv` e o `timing.csv` da avaliação): tempo de inferência muda a cada execução,
e por isso mora em arquivo próprio, fora dessa conferência.

Fluxos de aleatoriedade são separados por subsistema (população e chaves, tráfego
legítimo, campanha de ataque, partição e modelos), todos derivados da mesma semente. A
grade experimental é
de 11 condições de sigma (0,0 a 1,0) por 30 réplicas, totalizando 330 execuções.

## Estado

**Os doze módulos estão implementados, e o orquestrador.** A busca da 902, a grade das 330
e o ensaio da 903 já rodaram; os resultados estão nas quatro tabelas da raiz de `data/`.

O ramo da semente produz a régua inteira (o perfil histórico e os limiares do baseline,
das quatro semanas de aquecimento). O ramo de sigma produz o conjunto rotulado das
semanas 5 a 8, dividido em treino e holdout, com 23 das 58 sessões do atacante no
holdout e as mesmas sessões legítimas de cada lado nas onze condições de uma semente; e
sobre ele decidem o baseline de regras e os dois modelos, gravando o tempo que levaram.

O M2 produz cerca de 77 mil requisições em 3,8 mil sessões por semente. Os três itens que
as convenções do projeto mandam conferir antes do M3 estão cobertos por teste: o serviço
automatizado abre lote nas quatro horas fixas, o administrador tem ritmo mais disperso que
o regime `routine`, e os dois caminhos de falha legítima ocorrem nas 30 sementes.
