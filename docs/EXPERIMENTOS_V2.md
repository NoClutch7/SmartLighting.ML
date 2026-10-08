# Estrutura experimental v2

## Objetivo e isolamento

O módulo `smartlighting_ml.experiments.v2` recebe explicitamente um diretório de
desenvolvimento. Para compatibilidade, ele pode usar as janelas `train` e
`validation` de `all_windows.csv`; no cenário ampliado, lê somente
`development_windows.csv`. Ele não lê `test.csv`, `test_predictions.csv` nem os
modelos v1. Suas saídas ficam em `artifacts/experiments_v2`, separadas de
`artifacts/isolation_forest_v1`.

O teste v1 já foi observado e não pode ser usado para escolher características,
hiperparâmetros ou limiares da versão 2.

## Conjuntos de características

Quatro alternativas são configuradas:

- `original_9`: todas as nove características da v1;
- `without_rolling_mean_1h`: remove a característica redundante;
- `count_with_history`: contagem atual e históricos de 15 e 60 minutos;
- `count_with_temporal`: contagem atual e características cíclicas de hora e dia.

## Validação temporal progressiva

Cada fold possui três blocos contíguos, sem embaralhamento:

1. treino expansivo;
2. calibração do limiar;
3. avaliação de desenvolvimento posterior à calibração.

O treino exclui janelas anômalas conhecidas e as doze janelas posteriores do
mesmo sensor. Os rótulos não entram nas características nem no ajuste do
Isolation Forest; são usados para remover contaminação, calibrar o limiar e
calcular métricas.

## Grade de hiperparâmetros

A grade padrão combina:

- `n_estimators`: 100 e 300;
- `max_samples`: 128 e `auto`;
- `max_features`: 0,5 e 1,0.

Todos os modelos usam `contamination="auto"`, `random_state=20261008` e
`n_jobs=1`. O modo `--quick` usa uma única configuração pequena apenas para
validar o fluxo; ele não seleciona um modelo.

## Relatórios

`development_fold_metrics.csv` contém, por fold, sensor, detector e tipo de
anomalia:

- TP, FP, TN e FN;
- precisão, recall e F1;
- positivos e negativos;
- falsos positivos em janelas com contagem zero;
- taxa de falso positivo entre janelas normais com contagem zero;
- indicador `sufficient_for_comparison`.

As métricas são calculadas para todas as anomalias e separadamente para rajadas e
oscilações elétricas. O detector estatístico original usa exatamente os mesmos
blocos temporais.

`development_report.json` registra configuração, grade, limites temporais e
limitações. Nenhuma rotina escolhe ou publica automaticamente uma configuração
vencedora.

## Resultado anterior com 30 dias

O modo rápido encontrou três folds e marcou todos os 18 pares fold/sensor como
insuficientes. Dependendo do bloco, havia:

- zero ou uma janela positiva na calibração;
- de zero a três janelas positivas na avaliação.

Nenhuma linha geral ficou elegível para comparação estável. Isso é um resultado
metodológico, não uma derrota de uma configuração específica. A grade completa
não deve ser usada para declarar superioridade com os dados atuais.

## Cenário ampliado de 120 dias

O novo cenário fica em `data/processed/v2_development` e não contém um conjunto
de teste. No modo rápido, os três folds apresentaram:

- 23 a 25 episódios independentes por sensor na calibração;
- 23 a 24 episódios independentes por sensor na avaliação;
- pelo menos 10 episódios de cada tipo por sensor em todos esses blocos;
- nenhum episódio dividido entre treino, calibração e avaliação;
- nenhuma leitura do teste v1.

O relatório ficou com status `ready_for_development_comparison`. Esse status
significa apenas que o mínimo operacional foi alcançado. Ele não garante
estabilidade estatística e não autoriza declarar uma configuração superior sem
analisar múltiplos folds e incerteza.

## Protocolo para ampliação

Antes da comparação v2:

1. ampliar o período de desenvolvimento sem consultar um novo teste;
2. distribuir episódios independentes de rajada e oscilação ao longo dos blocos;
3. garantir pelo menos vários episódios de cada tipo por sensor em calibração e
   avaliação, contando episódios/janelas e não pulsos individuais;
4. manter indisponibilidades fora da modelagem e explicitamente marcadas;
5. definir previamente o mínimo de positivos e o critério de seleção;
6. executar a grade somente nos folds de desenvolvimento;
7. congelar configuração, características e limiares;
8. criar ou coletar um novo período cronologicamente posterior e avaliá-lo uma
   única vez como teste independente.

Com os dados atuais, os relatórios podem validar código, ausência de vazamento,
formato das métricas e comportamento dos folds, mas não sustentar que uma
configuração é superior.
