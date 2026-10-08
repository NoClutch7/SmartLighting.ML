# Histórico de desenvolvimento do SmartLighting.ML

## 1. Finalidade deste documento

Este documento consolida o estado técnico do `SmartLighting.ML` para permitir a
continuidade do TCC por outro desenvolvedor ou por uma IA. O conteúdo foi
reconstruído a partir do código-fonte, documentação, testes, metadados, modelos e
relatórios presentes no repositório local.

Não existe uma pasta `.git` neste diretório. Consequentemente, commits, autores,
branches, tags, mensagens e datas de alteração não puderam ser consultados. Onde
uma decisão ou ordem exata não pode ser comprovada pelos artefatos, este texto
usa a indicação **“Não identificado nos artefatos disponíveis”**.

Referências principais:

- visão operacional: `README.md`;
- planejamento: `docs/PLANEJAMENTO.md`;
- cenário sintético inicial: `docs/DADOS_SINTETICOS.md`;
- preparação: `docs/PREPARACAO_DADOS.md`;
- experimento v1: `docs/MODELO_ISOLATION_FOREST.md`;
- metodologia v2: `docs/EXPERIMENTOS_V2.md`;
- cenário ampliado v2: `docs/DADOS_SINTETICOS_V2.md`;
- pacote Python: `src/smartlighting_ml/`;
- testes: `tests/`;
- artefatos experimentais: `artifacts/`.

## 2. Visão geral do projeto

O `SmartLighting.ML` é o componente de Machine Learning de um TCC sobre
iluminação pública inteligente em uma maquete de condomínio residencial. O
condomínio é representado como uma área quadrada, com dez postes de iluminação
no perímetro da quadra e seis sensores PIR distribuídos pelas calçadas. Não há
postes dentro da quadra. As posições exatas dos sensores ainda não foram
mapeadas; por isso, os identificadores `PIR-01` a `PIR-06` são neutros e não
devem ser interpretados como entrada, rua ou área esportiva.

O problema de ML é aprender o padrão histórico individual de cada sensor e
atribuir um escore de anomalia a janelas de atividade. O objetivo não é contar
pessoas nem concluir que ocorreu uma invasão. Cada evento representa uma borda
elétrica de ativação `HIGH → LOW` do PIR. Uma pessoa pode não gerar pulso, gerar
um pulso ou gerar vários pulsos.

O firmware considerado mantém o sinal ativo por um período físico ainda não
medido com precisão. O simulador impede novos eventos enquanto o pulso está
ativo e durante o rearme. O retorno da iluminação ao brilho de repouso de
aproximadamente 8% após sete segundos sem sinal, seguido por transição gradual,
é comportamento do atuador e não gera linhas adicionais de telemetria.

O relacionamento planejado com o restante do TCC é:

1. ESP32 e PIR produzem telemetria;
2. o módulo Python valida e agrega os eventos;
3. características de cinco minutos são calculadas;
4. um detector por sensor atribui escore e decisão;
5. futuramente, a `SmartLighting.Api` poderá consumir a resposta.

A integração com a API C#, firmware e frontend não está implementada neste
repositório. A pasta `src/smartlighting_ml/contracts/` contém apenas o pacote
reservado para contratos futuros.

## 3. Arquitetura e estrutura do repositório

### 3.1 Configuração do pacote

`pyproject.toml` define o pacote `smartlighting-ml`, versão `0.1.0`, com Python
3.11 ou superior e layout `src/`. As dependências declaradas são:

- NumPy;
- Pandas;
- Scikit-learn;
- tzdata;
- Pytest como dependência opcional de teste e também em `requirements.txt`.

Os comandos instaláveis são:

| Comando | Responsabilidade |
| --- | --- |
| `smartlighting-generate` | gera o cenário sintético inicial de 30 dias |
| `smartlighting-prepare` | prepara e divide os dados v1 |
| `smartlighting-train` | treina e avalia o Isolation Forest v1 |
| `smartlighting-generate-v2` | gera o cenário de desenvolvimento de 120 dias |
| `smartlighting-prepare-v2` | prepara os 120 dias sem criar teste |
| `smartlighting-experiments-v2` | executa folds progressivos de desenvolvimento |

`.gitignore` exclui ambientes, caches, dados gerados e artefatos treinados. Isso
significa que os CSVs e modelos atualmente existentes podem não ser versionados
quando um repositório Git for criado; uma estratégia de armazenamento e
reprodução deverá ser definida.

### 3.2 Pacotes Python

#### `src/smartlighting_ml/data_processing/`

- `synthetic_generator.py`: cenário reproduzível de 30 dias usado pela v1;
- `preparation_pipeline.py`: valida eventos, cria janelas, qualidade,
  características e divisões cronológicas;
- `synthetic_generator_v2.py`: cenário exclusivo de desenvolvimento com 120
  dias, episódios independentes e hard negatives;
- `prepare_v2_development.py`: prepara todos os 120 dias como desenvolvimento,
  sem criar conjunto de teste.

#### `src/smartlighting_ml/training/`

- `train_isolation_forest.py`: auditoria de entrada, seis pipelines
  `StandardScaler + IsolationForest`, calibração por validação, baseline
  estatístico, métricas, serialização e relatórios da v1.

#### `src/smartlighting_ml/inference/`

- `predictor.py`: classe `AnomalyPredictor`, responsável por selecionar o modelo
  do sensor, validar uma janela processada e retornar o contrato de inferência.

#### `src/smartlighting_ml/experiments/`

- `v2.py`: ambiente separado de comparação metodológica com feature sets,
  hiperparâmetros e validação temporal progressiva.

#### `src/smartlighting_ml/contracts/`

Somente o arquivo `__init__.py` e uma descrição de propósito. Contratos formais,
DTOs, serialização de entrada e transporte com a API ainda não foram iniciados.

### 3.3 Dados

| Caminho | Conteúdo |
| --- | --- |
| `data/raw/synthetic/` | cenário v1 bruto de 30 dias |
| `data/processed/` | janelas e divisões v1 |
| `data/raw/synthetic_v2_development/` | cenário v2 bruto de 120 dias |
| `data/processed/v2_development/` | janelas v2 somente de desenvolvimento |

O contrato de evento bruto permanece compatível entre versões:

- `EventIdentifier`;
- `SensorId`;
- `DeviceId`;
- `OccurredAtUtc`.

Rótulos ficam em arquivos separados e não integram as características do
modelo. No cenário v2, `EpisodeIdentifier` identifica o fenômeno independente
que pode produzir vários pulsos e janelas.

### 3.4 Artefatos

| Caminho | Conteúdo |
| --- | --- |
| `artifacts/isolation_forest_v1/` | seis modelos, limiares, metadados e resultados finais v1 |
| `artifacts/experiments_v2/` | relatório estrutural original com 30 dias |
| `artifacts/experiments_v2/development_120d/` | execução rápida do cenário ampliado |

## 4. Funcionalidades implementadas

### 4.1 Gerador sintético v1

`synthetic_generator.py` usa semente `20261008` e gera 30 dias para seis sensores
fixos, com dispositivos `ESP32-SENSOR-01` a `ESP32-SENSOR-06`. Os perfis de taxa
são abstratos e diferentes por sensor; não representam posições físicas.

O gerador implementa:

- amostragem Poisson em janelas de cinco minutos;
- taxas distintas para madrugada, manhã, tarde e noite;
- fatores de fim de semana;
- pulsos normais com duração uniforme de 2 a 6 segundos;
- rearme normal de 1,5 segundo;
- oscilações com pulsos de 0,15 a 0,45 segundo e rearme de 0,25 segundo;
- rejeição de candidatas durante pulso ativo ou rearme;
- rajadas `out_of_pattern_burst`;
- oscilações `electrical_oscillation`;
- indisponibilidades de telemetria;
- UUIDs determinísticos e timestamps UTC.

Os dados v1 atualmente presentes registram 22.090 eventos, 540 pulsos rotulados,
48 janelas anômalas e seis indisponibilidades. Esses valores estão documentados
nos artefatos e documentos produzidos durante o desenvolvimento; não foram
recalculados neste levantamento.

### 4.2 Preparação de dados

`preparation_pipeline.py`:

1. exige os campos do contrato;
2. valida strings obrigatórias e timestamps UTC;
3. remove duplicatas por `EventIdentifier`;
4. ordena cronologicamente;
5. verifica que cada sensor pertence a um único dispositivo;
6. valida indisponibilidades e referências dos rótulos;
7. constrói dias civis completos no fuso `America/Sao_Paulo`;
8. agrega eventos por sensor em janelas fixas de cinco minutos;
9. atribui zero somente a janelas confiáveis sem evento;
10. atribui valor ausente à contagem quando uma indisponibilidade intersecta a
    janela;
11. calcula características;
12. separa cronologicamente treino, validação e teste na proporção 70/15/15.

As nove características são:

1. `event_count_5m`;
2. `hour_sin`;
3. `hour_cos`;
4. `weekday_sin`;
5. `weekday_cos`;
6. `is_weekend`;
7. `events_last_15m`;
8. `events_last_60m`;
9. `rolling_mean_1h`.

As características históricas usam estritamente janelas anteriores. Os últimos
15 minutos correspondem a três janelas anteriores; a hora corresponde a doze.
Quando falta histórico completo ou existe indisponibilidade no intervalo, a
característica fica ausente e a linha não entra nos datasets de modelagem.

Para impedir contaminação do treino, o pipeline exclui a própria janela anômala
e as doze janelas seguintes do mesmo sensor. A exclusão posterior é necessária
porque o histórico de uma hora ainda conteria a anomalia.

Saídas v1:

- `all_windows.csv`;
- `train.csv`;
- `validation.csv`;
- `test.csv`;
- `anomaly_windows_evaluation.csv`.

### 4.3 Treinamento v1

`train_isolation_forest.py` cria um modelo independente por sensor. Cada modelo é
um pipeline Scikit-learn composto por:

- `StandardScaler` ajustado no sensor;
- `IsolationForest`.

Hiperparâmetros confirmados em
`artifacts/isolation_forest_v1/metadata.json`:

| Parâmetro | Valor |
| --- | --- |
| `n_estimators` | 300 |
| `max_samples` | `auto` — 256 amostras efetivas por árvore |
| `max_features` | 1,0, padrão do estimador |
| `contamination` | `auto` |
| `bootstrap` | `False` |
| `random_state` | 20261008 |
| `n_jobs` | 1 |

O escore do projeto é `-score_samples`: quanto maior, mais anômala a janela. O
limiar interno decorrente de `contamination` não determina a decisão final. Para
cada sensor, o limiar externo é escolhido na validação maximizando F1; empates
são resolvidos por recall, precisão e especificidade.

A função de calibração possui implementação eficiente baseada em ordenação e
contagens acumuladas. Seu critério é equivalente à avaliação exaustiva dos
limiares distintos. A otimização foi introduzida para tornar os folds v2
praticáveis sem alterar os artefatos v1.

### 4.4 Detector estatístico

O baseline calcula média e desvio-padrão de `event_count_5m` no treino do sensor.
Seu escore é a padronização de cauda superior. O limiar também maximiza F1 na
validação. O teste é aplicado apenas depois da calibração.

Esse detector foi mantido na v2 e usa os mesmos blocos temporais do Isolation
Forest.

### 4.5 Inferência

`AnomalyPredictor` carrega `metadata.json`, o limiar e o `.joblib` específico do
sensor. Recebe uma `Mapping` ou `pandas.Series`, valida timestamp e finitude das
nove características e retorna:

- `SensorId`;
- `WindowStartUtc`;
- `anomaly_score`;
- `is_anomaly`;
- `model_version`.

O módulo funciona localmente com artefatos v1. Serviço HTTP, fila, chamada pela
API C# e monitoramento operacional não estão implementados.

### 4.6 Cenário sintético v2

`synthetic_generator_v2.py` cria 120 dias exclusivos de desenvolvimento, de 1º
de fevereiro a 31 de maio de 2026 no fuso de São Paulo. Não cria teste.

Além dos perfis individuais, acrescenta:

- fator diário log-normal por sensor, sigma 0,18;
- variação senoidal de 28 dias, amplitude 0,12;
- um episódio anômalo planejado por sensor e dia;
- alternância entre rajada e oscilação;
- horários distribuídos pelas 24 horas e sete dias da semana;
- períodos intensos legítimos, de 10 a 20 minutos, sem rótulo;
- indisponibilidades;
- `EpisodeIdentifier`.

Os hard negatives legítimos não se sobrepõem aos episódios anômalos do mesmo
sensor. `legitimate_activity_episodes.csv` serve apenas para auditoria e não é
característica nem rótulo do modelo.

Volumes registrados em `docs/DADOS_SINTETICOS_V2.md` e
`scenario_metadata.json`:

| Item | Quantidade |
| --- | ---: |
| Duração | 120 dias |
| Eventos observados | 99.404 |
| Pulsos rotulados | 8.256 |
| Episódios anômalos observados | 718 |
| Períodos legítimos intensos | 102 |
| Indisponibilidades | 30 |
| Janelas totais | 207.360 |
| Janelas elegíveis para desenvolvimento | 206.422 |
| Janelas não confiáveis | 506 |

Foram planejados 720 episódios; duas oscilações foram suprimidas pela interação
entre indisponibilidade e rearme. A distribuição observada é:

| Sensor | Rajadas | Oscilações | Total |
| --- | ---: | ---: | ---: |
| PIR-01 | 60 | 60 | 120 |
| PIR-02 | 60 | 59 | 119 |
| PIR-03 | 60 | 60 | 120 |
| PIR-04 | 60 | 60 | 120 |
| PIR-05 | 60 | 60 | 120 |
| PIR-06 | 60 | 59 | 119 |

### 4.7 Preparação v2

`prepare_v2_development.py` reutiliza validação, janela, confiabilidade e
características da preparação v1, mas grava:

- `development_windows.csv`;
- `anomaly_windows_evaluation.csv`.

Todas as linhas recebem `split="development"`. Não são criados `train.csv`,
`validation.csv` ou `test.csv` para o cenário de 120 dias.

### 4.8 Executor experimental v2

`experiments/v2.py` aceita explicitamente `processed_dir` e `output_dir`. Quando
existe `development_windows.csv`, usa somente esse arquivo. O modo de
compatibilidade com o cenário antigo lê `all_windows.csv`, mas filtra apenas os
splits `train` e `validation`.

Conjuntos configuráveis:

| Nome | Características |
| --- | --- |
| `original_9` | todas as nove |
| `without_rolling_mean_1h` | remove a média redundante |
| `count_with_history` | contagem atual, 15 min e 60 min |
| `count_with_temporal` | contagem atual e características temporais |

Grade padrão disponível, mas ainda não executada integralmente:

- `n_estimators`: 100 e 300;
- `max_samples`: 128 e `auto`;
- `max_features`: 0,5 e 1,0;
- `contamination="auto"`;
- `random_state=20261008`;
- `n_jobs=1`.

Os folds possuem treino expansivo, calibração e avaliação, nessa ordem, sem
embaralhamento. Proporções padrão: 40% inicial para treino, 20% para calibração,
20% para avaliação e avanço de 10% por fold.

Rótulos são usados para:

- remover do treino anomalias e as doze janelas contaminadas seguintes;
- calibrar o limiar;
- calcular métricas.

Não são usados como característica ou como alvo supervisionado de treinamento.
O relatório inclui TP, FP, TN, FN, precisão, recall, F1, métricas por tipo e taxa
de falsos positivos entre janelas normais com contagem zero.

## 5. Histórico de desenvolvimento reconstruído

Sem commits, a evolução abaixo é uma reconstrução lógica baseada em arquivos e
dependências, não uma cronologia Git verificável.

### Etapa 1 — estrutura inicial

Foram criados o layout `src/`, pacotes de processamento, treinamento, inferência,
contratos e testes, além de `README.md`, `.gitignore`, `requirements.txt` e
`pyproject.toml`. O desenho separou responsabilidades visando futura integração
com `SmartLighting.Api`.

Data e commit: **Não identificado nos artefatos disponíveis**.

### Etapa 2 — correção do domínio físico

A documentação passou a representar dez postes no perímetro, nenhum poste no
interior da quadra e seis sensores nas calçadas. Nomes antigos baseados em áreas
específicas foram substituídos por `PIR-01` a `PIR-06`.

Evidências: `README.md`, `docs/PLANEJAMENTO.md`, `docs/DADOS_SINTETICOS.md` e
constante `SENSORS` em `synthetic_generator.py`.

### Etapa 3 — gerador sintético de 30 dias

Foi implementada a simulação elétrica de pulsos, rearme, perfis probabilísticos,
anomalias, indisponibilidades e reprodução por semente. Os rótulos foram
separados para evitar vazamento.

### Etapa 4 — preparação em cinco minutos

Foi criada a grade completa por sensor, tratamento explícito de confiabilidade,
características cíclicas e históricas, divisão cronológica e exclusão de
contaminação no treino.

### Etapa 5 — Isolation Forest v1

Foram treinados seis modelos independentes e comparados ao baseline estatístico.
Modelos, limiares, previsões e métricas foram persistidos. A inferência local foi
implementada.

O primeiro desenho de anomalias produzia distribuição inadequada: nove janelas
positivas, duas na validação e nenhuma no teste. O cenário de 30 dias foi então
revisto para seis sensores e anomalias distribuídas, resultando em 2–3 janelas
positivas por sensor na validação e no teste. Esta evolução é comprovada pelos
documentos e artefatos finais; commits intermediários não estão disponíveis.

### Etapa 6 — auditoria do v1

A auditoria identificou baixa sensibilidade, falsos positivos, suporte positivo
insuficiente, redundância perfeita entre `events_last_60m` e
`rolling_mean_1h`, teste v1 sem fins de semana e forte favorecimento do baseline
pelas anomalias de alta frequência.

O teste v1 passou a ser considerado observado e impróprio para seleção v2.

### Etapa 7 — estrutura experimental v2 inicial

Foram implementados feature sets, grade, folds progressivos, métricas por tipo,
hard check de acesso ao teste e preservação byte a byte dos artefatos v1. Com os
30 dias, todos os 18 pares fold/sensor foram insuficientes: calibração com 0–1
positivo e avaliação com 0–3.

Evidência: `artifacts/experiments_v2/development_report.json`.

### Etapa 8 — cenário ampliado v2

Foram criados 120 dias exclusivamente de desenvolvimento, episódios
independentes, variação natural e períodos legítimos intensos. Os folds passaram
a possuir 23–25 episódios por sensor na calibração e 23–24 na avaliação, com ao
menos dez episódios de cada tipo e nenhum episódio atravessando fronteiras.

O modo rápido foi reduzido para uma representação e dez árvores para validar
apenas a estrutura. A grade completa não foi executada.

## 6. Experimento v1

### 6.1 Dados e divisão

O cenário v1 contém 51.840 janelas totais e 112 não confiáveis. Os conjuntos de
modelagem registrados são:

| Conjunto | Linhas | Intervalo UTC |
| --- | ---: | --- |
| Treino | 35.905 | 2026-01-01 04:00 a 2026-01-22 02:55 |
| Validação | 7.743 | 2026-01-22 03:00 a 2026-01-26 14:55 |
| Teste | 7.776 | 2026-01-26 15:00 a 2026-01-31 02:55 |

O treino exclui anomalias e histórico contaminado. A validação calibra os
limiares. O teste é aplicado somente após essa escolha.

### 6.2 Positivos

| Sensor | Validação | Teste |
| --- | ---: | ---: |
| PIR-01 | 2 | 2 |
| PIR-02 | 3 | 3 |
| PIR-03 | 3 | 3 |
| PIR-04 | 3 | 3 |
| PIR-05 | 3 | 3 |
| PIR-06 | 2 | 2 |

Essa quantidade é insuficiente para métricas estáveis; uma única janela altera
substancialmente recall e F1.

### 6.3 Resultados finais do teste v1

Fonte: `artifacts/isolation_forest_v1/results/test_metrics.csv`.

| Sensor | Detector | TP | FP | TN | FN | Precisão | Recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PIR-01 | Isolation Forest | 1 | 31 | 1263 | 1 | 0,0313 | 0,5000 | 0,0588 |
| PIR-01 | Estatístico | 2 | 0 | 1294 | 0 | 1,0000 | 1,0000 | 1,0000 |
| PIR-02 | Isolation Forest | 0 | 0 | 1293 | 3 | 0,0000 | 0,0000 | 0,0000 |
| PIR-02 | Estatístico | 3 | 0 | 1293 | 0 | 1,0000 | 1,0000 | 1,0000 |
| PIR-03 | Isolation Forest | 1 | 17 | 1276 | 2 | 0,0556 | 0,3333 | 0,0952 |
| PIR-03 | Estatístico | 2 | 0 | 1293 | 1 | 1,0000 | 0,6667 | 0,8000 |
| PIR-04 | Isolation Forest | 0 | 0 | 1293 | 3 | 0,0000 | 0,0000 | 0,0000 |
| PIR-04 | Estatístico | 2 | 0 | 1293 | 1 | 1,0000 | 0,6667 | 0,8000 |
| PIR-05 | Isolation Forest | 0 | 1 | 1292 | 3 | 0,0000 | 0,0000 | 0,0000 |
| PIR-05 | Estatístico | 2 | 0 | 1293 | 1 | 1,0000 | 0,6667 | 0,8000 |
| PIR-06 | Isolation Forest | 1 | 38 | 1256 | 1 | 0,0256 | 0,5000 | 0,0488 |
| PIR-06 | Estatístico | 2 | 0 | 1294 | 0 | 1,0000 | 1,0000 | 1,0000 |
| **Total** | **Isolation Forest** | **3** | **87** | **7673** | **13** | **0,0333** | **0,1875** | **0,0566** |
| **Total** | **Estatístico** | **13** | **0** | **7760** | **3** | **1,0000** | **0,8125** | **0,8966** |

Esses são resultados experimentais sintéticos, não resultados validados em
hardware ou condomínio real. O baseline é diretamente favorecido porque as
anomalias v1 são principalmente aumentos de frequência.

### 6.4 Auditoria de características e comportamento

Evidências registradas durante a auditoria:

- `event_count_5m` separou positivos de validação por aproximadamente 15,6
  desvios-padrão do treino;
- `events_last_60m` e `rolling_mean_1h` têm correlação 1,0;
- cerca de 68% das janelas possuem contagem zero;
- 48 dos 87 falsos positivos tinham contagem atual zero, mas histórico recente
  não nulo;
- falsos positivos concentraram-se especialmente às 02h–03h e 17h–22h;
- o teste v1 contém apenas dias úteis.

O modelo não diferencia semanticamente rajadas de oscilações; produz apenas um
escore binário. No teste, detectou 2 de 10 janelas de rajada e 1 de 6 janelas de
oscilação.

## 7. Experimento v2

### 7.1 Objetivo

Investigar se a representação e os hiperparâmetros podem melhorar o detector
sem reutilizar o teste v1. A v2 é uma estrutura de desenvolvimento, não um novo
modelo final.

### 7.2 Isolamento metodológico

Medidas implementadas:

- não há caminho de leitura de `test.csv` ou `test_predictions.csv` no executor;
- o carregador usa somente desenvolvimento explícito ou treino+validação legados;
- artefatos v1 são gravados em diretório diferente;
- teste automatizado calcula hashes v1 antes e depois de uma execução v2;
- treino de cada fold exclui anomalias e doze janelas posteriores;
- calibração precede avaliação;
- episódios que cruzariam fronteiras são proibidos pelo teste do cenário;
- um novo teste independente foi deliberadamente adiado.

### 7.3 Folds de 120 dias

| Fold | Calibração por sensor | Avaliação por sensor |
| --- | ---: | ---: |
| 0 | 24–25 episódios | 23–24 episódios |
| 1 | 23–24 episódios | 24 episódios |
| 2 | 23–24 episódios | 23–24 episódios |

O mínimo operacional configurado é dez episódios independentes. Ele foi
atingido, mas não garante estabilidade estatística.

### 7.4 Execução rápida existente

`artifacts/experiments_v2/development_120d/development_report.json` registra:

- `status="ready_for_development_comparison"`;
- `test_data_accessed=false`;
- apenas `original_9`;
- `n_estimators=10`;
- `max_samples=128`;
- `max_features=1.0`;
- 108 linhas de métricas;
- 36 linhas gerais elegíveis, correspondentes aos dois detectores, seis sensores
  e três folds.

Essa execução é somente smoke/validação estrutural. Seus valores de F1 não devem
ser tratados como comparação conclusiva ou seleção de modelo. A grade completa
não foi executada.

### 7.5 Diferenças entre v1 e v2

| Aspecto | V1 | V2 |
| --- | --- | --- |
| Duração sintética | 30 dias | 120 dias de desenvolvimento |
| Teste | teste final já observado | nenhum teste criado |
| Unidade positiva | janela | episódio independente e janela |
| Validação | uma divisão fixa | três folds progressivos |
| Feature sets | nove fixas | quatro configuráveis |
| Hiperparâmetros | uma configuração | grade configurável |
| Hard negatives | não documentados como episódios | 102 períodos legítimos intensos |
| Métrica por tipo | análise posterior | parte do relatório |
| Proteção do v1 | não aplicável | hashes e diretórios separados |

## 8. Testes e validações

A suíte atual possui 32 testes e a última execução registrada durante o
desenvolvimento terminou com `32 passed`. Os arquivos de teste são:

### `tests/test_synthetic_generator.py`

- esquema, sensores, período e identificadores;
- separação dos rótulos;
- pulso e rearme;
- indisponibilidades opcionais;
- perfis individuais;
- ausência de eventos observados em outage;
- reprodução byte a byte.

### `tests/test_preparation_pipeline.py`

- validação, deduplicação, ordenação;
- janelas vazias confiáveis e janelas não confiáveis nulas;
- UTC para São Paulo e características cíclicas;
- históricos estritamente anteriores;
- divisões cronológicas e características completas;
- exclusão de anomalias e histórico contaminado do treino;
- geração de todos os arquivos.

### `tests/test_isolation_forest_training.py`

- seis modelos independentes;
- ordem das características e hiperparâmetros persistidos;
- calibração reproduzível pela validação;
- consistência da matriz de confusão;
- contrato de inferência;
- reprodução de métricas e previsões;
- ausência de rótulos nos datasets e cronologia.

### `tests/test_experiments_v2.py`

- quatro feature sets;
- nenhuma leitura de arquivos de teste v1;
- folds progressivos sem sobreposição;
- métricas por tipo e falsos positivos com zero;
- registro de insuficiência no cenário antigo;
- preservação por hash dos artefatos v1.

### `tests/test_synthetic_v2_development.py`

- 120 dias e seis sensores fixos;
- consistência dos rótulos e independência dos episódios;
- cobertura das 24 horas e sete dias;
- mínimo de dez episódios de cada tipo por sensor/bloco;
- nenhum episódio atravessando fronteiras;
- períodos legítimos sem rótulo e sem sobreposição;
- indisponibilidade com contagem nula;
- deduplicação preservando `EpisodeIdentifier`.

### Limitações dos testes

- não há teste integrado com ESP32 real;
- não há teste da futura API C#;
- não há teste de carga, latência ou consumo de memória;
- não há validação estatística com dados reais;
- não há CI/CD identificado;
- cobertura percentual: **Não identificado nos artefatos disponíveis**;
- os testes dependem de CSVs gerados localmente, que são ignorados pelo Git.

## 9. Decisões técnicas e justificativas sustentadas por evidência

### Modelo por sensor

Há seis modelos independentes porque os sensores podem ter distribuições
distintas e suas posições ainda não estão semanticamente mapeadas. Isso evita
forçar uma única distribuição global.

### Isolation Forest

Foi escolhido por ser um detector não supervisionado, compatível com a ausência
esperada de muitos rótulos reais. Os rótulos sintéticos servem para avaliação e
calibração, não como alvo de ajuste do estimador.

### Janelas de cinco minutos

São a unidade analítica documentada desde o planejamento. Permitem representar
silêncio confiável, frequência atual e histórico móvel.

### Características cíclicas

Seno e cosseno evitam descontinuidade artificial entre 23h e 0h e entre fim e
início da semana.

### Histórico estritamente anterior

As somas e média são deslocadas uma janela antes do rolling. Essa escolha impede
que a própria janela ou dados futuros entrem no histórico.

### Outages como ausência, não zero

Contagem zero significa telemetria confiável sem pulso. Outage significa dado
desconhecido e recebe valor ausente. Essa distinção está no pipeline e nos
testes.

### Rótulos separados

O CSV de eventos mantém somente o contrato operacional. Rótulos separados
reduzem risco de entrada acidental no modelo e facilitam avaliação cega.

### Baseline estatístico

Foi mantido para verificar se a complexidade multivariada traz benefício sobre
uma regra de frequência ajustada no treino e calibrada na validação.

### Semente fixa

`20261008` é usada na geração e treinamento para reprodução. A razão específica
para a escolha desse número é **Não identificado nos artefatos disponíveis**.

### Padronização

`StandardScaler` é aplicado por sensor. O Isolation Forest não depende de escala
da mesma forma que métodos baseados em distância, mas a decisão mantém uma
interface uniforme e registra as transformações junto com o modelo.

## 10. Problemas, riscos e limitações conhecidos

### Limitações do domínio e hardware

- posições exatas dos sensores não mapeadas;
- duração e rearme reais do PIR não medidos;
- dados representam pulsos, não pessoas, direção ou velocidade;
- clima, animais, veículos, feriados e obstruções não são modelados de forma
  explícita;
- relação espacial sensor–poste não é simulada;
- não há dados reais da maquete no repositório.

### Limitações metodológicas v1

- somente 2–3 positivos por sensor em validação e teste;
- teste composto apenas por dias úteis;
- redundância perfeita entre duas características;
- anomalias de frequência favorecem o baseline;
- F1 do Isolation Forest 0,0566 e 87 falsos positivos;
- teste v1 já observado, portanto não reutilizável para seleção v2.

### Limitações metodológicas v2

- 718 episódios são sintéticos e não garantem validade externa;
- um episódio por sensor/dia é uma hipótese forte e regular;
- mínimo de dez episódios por bloco não garante estabilidade;
- nenhuma grade completa foi executada;
- nenhuma configuração v2 foi congelada;
- novo teste independente ainda não existe;
- o relatório rápido usa apenas 10 árvores e não é resultado final;
- o texto `interpretation` do relatório `development_120d` ainda contém a frase
  genérica de que nenhuma configuração pode ser declarada superior “quando os
  folds têm positivos insuficientes”, apesar de `status` indicar prontidão e a
  lista `limitations` estar vazia. A regra de não declarar superioridade continua
  correta, mas a justificativa textual está desatualizada.

### Inconsistências documentais

- `README.md` ainda afirma, em trecho introdutório, que modelo e treinamento não
  foram implementados, embora ambos existam;
- `docs/PLANEJAMENTO.md` ainda descreve treinamento como futuro e lista
  implementação do modelo fora do escopo daquela etapa;
- a árvore do README não lista todos os novos módulos e documentos v2;
- artefatos e dados são ignorados por `.gitignore`, embora atualmente existam no
  workspace.

### Riscos de engenharia

- nenhum controle de esquema versionado para a futura API;
- nenhuma estratégia de migração de modelo;
- nenhum registro central de execução além de JSON/CSV locais;
- nenhum pipeline de CI/CD identificado;
- nenhuma assinatura ou checksum publicada para datasets de referência;
- nenhuma observabilidade de inferência;
- nenhum tratamento operacional para drift ou retreinamento;
- artefatos Joblib devem ser carregados somente de origem confiável.

## 11. Estado atual

### Implementado e testado

- pacote Python instalável;
- gerador v1 de 30 dias;
- pipeline v1 de preparação;
- seis Isolation Forest v1;
- baseline estatístico;
- calibração por validação;
- avaliação final v1;
- persistência de modelos, limiares, metadados, previsões e métricas;
- inferência local por janela;
- estrutura experimental v2;
- gerador v2 de 120 dias;
- preparação v2 somente de desenvolvimento;
- folds temporais progressivos;
- métricas por tipo e falsos positivos com contagem zero;
- 32 testes automatizados.

### Implementado parcialmente ou apenas como estrutura

- comparação v2: executor pronto, mas grade completa não executada;
- seleção de características: alternativas implementadas, nenhuma escolhida;
- inferência: módulo local pronto, sem serviço ou integração;
- contratos: pacote reservado, sem modelos de dados formais;
- cenário físico: quantidade de sensores/postes conhecida, posições dos sensores
  ainda não mapeadas.

### Não iniciado ou não identificado

- novo teste independente pós-congelamento;
- treinamento de configuração v2 final;
- coleta e validação com hardware real;
- integração `SmartLighting.Api`;
- alterações de firmware ou frontend;
- deployment, monitoramento e drift;
- versionamento Git disponível neste diretório;
- CI/CD;
- critérios de aceitação operacional e custo de falso alerta;
- revisão acadêmica externa dos protocolos.

## 12. Próximas etapas recomendadas

1. Corrigir as inconsistências do README, planejamento e mensagem genérica do
   relatório v2, sem alterar resultados históricos.
2. Registrar o repositório em Git e decidir como versionar datasets grandes e
   artefatos, possivelmente com armazenamento externo e checksums.
3. Congelar formalmente o protocolo de comparação v2: métrica primária,
   agregação entre sensores/folds, tolerância a falsos positivos e regra de
   desempate.
4. Executar a grade v2 somente nos folds de desenvolvimento de 120 dias.
5. Comparar os quatro feature sets, com atenção à remoção de
   `rolling_mean_1h` redundante.
6. Relatar média, dispersão e números absolutos entre folds; não escolher por uma
   única execução.
7. Avaliar rajadas, oscilações e hard negatives separadamente.
8. Realizar análise de sensibilidade à regularidade artificial de um episódio
   por sensor/dia.
9. Congelar feature set, hiperparâmetros e método de limiar antes de qualquer
   novo teste.
10. Criar ou coletar um novo período cronologicamente posterior, com dias úteis,
    fins de semana, outages e atividade legítima intensa.
11. Executar uma única avaliação final no novo teste e registrar intervalos de
    confiança e contagens absolutas.
12. Iniciar coleta real na maquete, medir pulsos/rearme e recalibrar as hipóteses.
13. Definir contrato de integração com a API e critérios de falha segura.
14. Planejar monitoramento, versionamento e rollback de modelos.

## 13. Guia de continuidade

### 13.1 Preparar o ambiente

No PowerShell, a partir da raiz do projeto:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
```

O projeto declara Python 3.11 ou superior. Os artefatos v1 registram que a
execução histórica usou Python 3.14.0, NumPy 2.5.3, Pandas 3.0.6,
Scikit-learn 1.9.1 e Joblib 1.6.0.

### 13.2 Reproduzir o cenário v1

```powershell
smartlighting-generate
smartlighting-prepare
smartlighting-train
```

Atenção: esses comandos reescrevem dados e artefatos v1 nos caminhos padrão. Não
devem ser executados se o objetivo for preservar os resultados históricos sem
antes copiar ou versionar os artefatos.

### 13.3 Executar inferência local v1

```python
import pandas as pd

from smartlighting_ml.inference.predictor import AnomalyPredictor

window = pd.read_csv("data/processed/test.csv").iloc[0]
result = AnomalyPredictor().predict(window)
print(result)
```

Esse exemplo é operacional, mas o teste v1 não deve ser usado para seleção ou
ajuste da v2.

### 13.4 Reproduzir o desenvolvimento v2

```powershell
smartlighting-generate-v2
smartlighting-prepare-v2
smartlighting-experiments-v2 --quick `
  --processed-dir data/processed/v2_development `
  --output-dir artifacts/experiments_v2/development_120d
```

O modo `--quick` usa somente `original_9`, 10 árvores, `max_samples=128` e
`max_features=1.0`. É um smoke test, não uma seleção.

### 13.5 Grade de desenvolvimento

O comando abaixo disponibiliza a grade completa configurada no código:

```powershell
smartlighting-experiments-v2 `
  --processed-dir data/processed/v2_development `
  --output-dir artifacts/experiments_v2/grid_120d
```

Antes de executá-lo, definir por escrito o protocolo de seleção e o custo
computacional esperado. O comando não acessa o teste v1, mas pode demorar e gerar
muitos ajustes.

### 13.6 Executar testes

```powershell
pytest
```

Ou, de forma explícita:

```powershell
python -m pytest -q
```

Resultado mais recente registrado no workspace durante o desenvolvimento deste
histórico: `32 passed`.

### 13.7 Cuidados para continuidade

- não usar `data/processed/test.csv` ou resultados v1 para escolher a v2;
- não criar repetidamente novos testes sintéticos;
- não interpretar pulso como pessoa;
- não converter outage em zero;
- não interpretar anomalia como ameaça;
- não contar pulsos do mesmo episódio como positivos independentes;
- não sobrescrever `artifacts/isolation_forest_v1` sem preservar o histórico;
- manter `random_state=20261008` quando a comparação exigir reprodução;
- registrar qualquer mudança de feature set, fold, limiar ou gerador;
- atualizar este documento quando uma etapa for concluída.

## 14. Conclusão

O projeto possui uma cadeia funcional completa para dados sintéticos: geração,
preparação, treinamento v1, inferência local, avaliação e estrutura experimental
v2. O resultado v1 demonstrou que o Isolation Forest atual não supera uma regra
estatística simples no cenário de 30 dias, mas esse resultado tem alta incerteza
e forte dependência do desenho sintético.

A principal evolução metodológica é a v2: dados exclusivamente de
desenvolvimento, episódios independentes, hard negatives, folds progressivos e
isolamento do teste observado. O projeto ainda não possui um modelo v2 final nem
evidência em dados reais. A continuidade deve priorizar protocolo pré-registrado,
comparação somente em desenvolvimento, congelamento da configuração e posterior
avaliação única em um novo teste independente.
