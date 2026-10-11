# Avaliação inicial do Isolation Forest

## Protocolo

Foram treinados seis modelos independentes, um para cada `SensorId`. Cada modelo
usa somente as linhas do respectivo sensor em `train.csv` e as nove
características, na ordem registrada em `metadata.json`.

O experimento usa:

- `StandardScaler` ajustado separadamente por sensor;
- `IsolationForest` com 300 árvores;
- `max_samples="auto"`;
- `contamination="auto"`;
- `random_state=20261008`;
- `n_jobs=1` para execução determinística.

O limiar de cada sensor maximiza o F1 na validação. Empates são resolvidos por
recall, precisão e especificidade, nessa ordem. O teste não participa do ajuste
do modelo, hiperparâmetros ou limiar e somente é avaliado após a calibração.

Como referência, foi usado um detector estatístico de frequência. Ele transforma
`event_count_5m` em um escore padronizado de cauda superior usando média e desvio
padrão calculados no treino. Seu limiar também é calibrado exclusivamente na
validação.

## Auditoria anterior ao treinamento

- treino: 35.905 janelas;
- validação: 7.743 janelas;
- teste: 7.776 janelas;
- seis sensores presentes nos três conjuntos;
- nenhum valor ausente ou infinito nas características;
- nenhuma sobreposição cronológica entre os conjuntos;
- nenhuma anomalia conhecida ou janela com histórico contaminado no treino;
- duas ou três janelas positivas por sensor na validação e no teste.

`events_last_60m` e `rolling_mean_1h` têm correlação absoluta 1, pois a segunda é
a primeira dividida por 12. As duas foram mantidas para cumprir a especificação
do experimento, e essa redundância está registrada nos metadados.

## Resultados no teste reservado

### Isolation Forest

| Sensor | TP | FP | TN | FN | Precisão | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SENSOR-01 | 1 | 31 | 1263 | 1 | 0,0313 | 0,5000 | 0,0588 |
| SENSOR-02 | 0 | 0 | 1293 | 3 | 0,0000 | 0,0000 | 0,0000 |
| SENSOR-03 | 1 | 17 | 1276 | 2 | 0,0556 | 0,3333 | 0,0952 |
| SENSOR-04 | 0 | 0 | 1293 | 3 | 0,0000 | 0,0000 | 0,0000 |
| SENSOR-05 | 0 | 1 | 1292 | 3 | 0,0000 | 0,0000 | 0,0000 |
| SENSOR-06 | 1 | 38 | 1256 | 1 | 0,0256 | 0,5000 | 0,0488 |
| **Total** | **3** | **87** | **7673** | **13** | **0,0333** | **0,1875** | **0,0566** |

### Detector estatístico de frequência

| Sensor | TP | FP | TN | FN | Precisão | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SENSOR-01 | 2 | 0 | 1294 | 0 | 1,0000 | 1,0000 | 1,0000 |
| SENSOR-02 | 3 | 0 | 1293 | 0 | 1,0000 | 1,0000 | 1,0000 |
| SENSOR-03 | 2 | 0 | 1293 | 1 | 1,0000 | 0,6667 | 0,8000 |
| SENSOR-04 | 2 | 0 | 1293 | 1 | 1,0000 | 0,6667 | 0,8000 |
| SENSOR-05 | 2 | 0 | 1293 | 1 | 1,0000 | 0,6667 | 0,8000 |
| SENSOR-06 | 2 | 0 | 1294 | 0 | 1,0000 | 1,0000 | 1,0000 |
| **Total** | **13** | **0** | **7760** | **3** | **1,0000** | **0,8125** | **0,8966** |

## Inferência

`AnomalyPredictor` carrega automaticamente o modelo correspondente ao
`SensorId`. A entrada deve ser uma janela já processada e conter as nove
características finitas.

```python
import pandas as pd

from smartlighting_ml.inference.predictor import AnomalyPredictor

window = pd.read_csv("data/processed/test.csv").iloc[0]
result = AnomalyPredictor().predict(window)
```

A resposta contém `SensorId`, `WindowStartUtc`, `anomaly_score`, `is_anomaly` e
`model_version`.

## Limitações

Os resultados não demonstram que o detector estatístico será superior em dados
reais. As anomalias sintéticas foram construídas principalmente como aumentos de
frequência, favorecendo diretamente esse baseline. O Isolation Forest recebeu
apenas duas ou três janelas positivas por sensor para calibrar e avaliar o
limiar, de modo que uma única janela altera fortemente recall e F1.

Os eventos representam pulsos elétricos `HIGH → LOW`, não pessoas. Falhas de
telemetria são excluídas pelo pipeline e não podem ser interpretadas como ausência
de movimento. Uma classificação anômala indica somente diferença estatística em
relação ao histórico sintético; não comprova invasão, ameaça ou incidente de
segurança.
