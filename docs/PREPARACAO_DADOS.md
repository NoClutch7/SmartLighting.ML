# Pipeline de preparação dos dados

## Entrada e validação

O pipeline lê `pir_events.csv`, `telemetry_outages.csv` e
`anomaly_labels.csv` de `data/raw/synthetic`. Ele exige os campos documentados,
valida timestamps UTC, rejeita identificadores vazios, remove eventos duplicados
por `EventIdentifier` e ordena os registros cronologicamente.

Os rótulos não são unidos às características. Eles são carregados somente para
proteger o treinamento inicial e produzir o arquivo separado de avaliação.

## Janelas e confiabilidade

Para cada sensor, o pipeline cria uma grade contínua de janelas de cinco minutos
que cobre dias civis completos em `America/Sao_Paulo`. Uma janela confiável sem
eventos recebe `event_count_5m = 0`.

Uma janela é não confiável quando seu intervalo se sobrepõe, mesmo parcialmente,
a uma indisponibilidade. Nesse caso, `event_count_5m` fica vazio e a coluna
`telemetry_reliable` recebe `False`. Essas janelas não entram nos conjuntos de
modelagem e não são interpretadas como ausência de movimento.

## Características

Cada conjunto de modelagem mantém `SensorId`, `DeviceId`, `WindowStartUtc` e
`WindowStartLocal`, além de:

- `event_count_5m`;
- `hour_sin` e `hour_cos`;
- `weekday_sin` e `weekday_cos`;
- `is_weekend`;
- `events_last_15m`;
- `events_last_60m`;
- `rolling_mean_1h`.

As características cíclicas são calculadas no horário local. As três
características históricas usam exclusivamente janelas anteriores à janela da
linha: três anteriores para 15 minutos e doze anteriores para uma hora. Elas
ficam vazias quando não existe histórico completo ou quando o intervalo contém
telemetria não confiável. Linhas assim permanecem em `all_windows.csv`, mas não
entram nos datasets de modelagem.

## Separação cronológica

As janelas são divididas globalmente por horário, sem embaralhamento:

- 70% do período para treinamento;
- 15% para validação;
- 15% para teste.

Janelas com anomalias conhecidas são identificadas por meio do arquivo separado
de rótulos. No treinamento, são excluídas a própria janela anômala e as doze
janelas posteriores do mesmo sensor, pois suas características de uma hora ainda
conteriam aquela contagem. Validação e teste preservam anomalias para avaliação
posterior. Nenhum campo de rótulo aparece nos datasets do modelo.

## Saídas

O comando `smartlighting-prepare` grava em `data/processed`:

- `all_windows.csv`: auditoria de todas as janelas, incluindo qualidade e split;
- `train.csv`: linhas elegíveis do período de treinamento;
- `validation.csv`: linhas elegíveis do período de validação;
- `test.csv`: linhas elegíveis do período de teste;
- `anomaly_windows_evaluation.csv`: relação separada entre evento anômalo e
  janela, destinada somente à avaliação.

Nenhum desses passos implementa ou treina o Isolation Forest.
