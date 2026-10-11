# Segundo experimento de detecção de anomalias

## Pergunta e dados

O primeiro experimento teve poucos intervalos anormais no teste e muitos
alarmes falsos com Isolation Forest. Esta segunda versão verifica se um conjunto
menor de características melhora o comportamento sem escolher parâmetros a
partir do teste já observado da versão 1.

O cenário de desenvolvimento contém 120 dias simulados, de 1º de fevereiro a
31 de maio de 2026. Ele inclui seis perfis de sensores, variação diária,
rajadas atípicas, oscilações elétricas, atividade intensa legítima e períodos
de telemetria indisponível. Os intervalos indisponíveis são excluídos.

O teste final usa outros 30 dias simulados, iniciados em 1º de junho de 2026,
gerados com a semente `20261011`. Nenhum intervalo do teste final foi usado
para escolher características, parâmetros ou limiares. Os dois períodos seguem
as mesmas hipóteses do gerador; são separados cronologicamente, mas **não são
evidência independente do comportamento de sensores físicos**.

## Escolha antes do teste final

Foram comparados em três blocos cronológicos de desenvolvimento:

- Isolation Forest com as nove características originais;
- Isolation Forest com contagem atual e informações de horário e dia;
- detector estatístico de frequência como referência.

As duas configurações de Isolation Forest usaram 100 árvores, 128 amostras por
árvore, `max_features=1.0` e semente `20261008`. Os limiares foram ajustados
somente no bloco de calibração de cada comparação. Os três blocos tinham
episódios positivos suficientes segundo o critério definido no experimento.

| Método | F1 em cada bloco de desenvolvimento | Alarmes falsos por bloco |
| --- | --- | --- |
| Isolation Forest, nove características | 0,067; 0,060; 0,071 | 2.600; 3.355; 2.980 |
| Isolation Forest, contagem e horário | 0,449; 0,580; 0,616 | 48; 42; 86 |
| Estatístico de frequência | 0,893; 0,920; 0,913 | 23; 11; 7 |

Os blocos progressivos se sobrepõem no tempo; seus números não devem ser
somados como se fossem testes independentes. A configuração simplificada do
Isolation Forest foi escolhida antes de gerar e analisar o teste final. O
detector estatístico permaneceu como comparação.

## Treinamento e teste final

Dentro dos 120 dias de desenvolvimento, os primeiros 80% dos horários foram
usados para ajustar seis modelos, um por sensor. Janelas com anomalia conhecida
ou histórico recente contaminado foram retiradas desse treino. Os últimos 20%
foram usados para ajustar o limite de cada modelo. Esse processamento produziu
157.698 janelas de treino e 41.286 de calibração.

O teste posterior continha 51.610 janelas confiáveis, das quais 179 tinham
anomalias simuladas conhecidas:

| Método | Detectadas | Não detectadas | Alarmes falsos | Normais corretas |
| --- | ---: | ---: | ---: | ---: |
| Isolation Forest v2 | 82 | 97 | 70 | 51.361 |
| Estatístico de frequência | 169 | 10 | 24 | 51.407 |

No teste final, o Isolation Forest v2 obteve precisão de 0,539, recall de
0,458 e F1 de 0,495. O detector estatístico obteve precisão de 0,876, recall
de 0,944 e F1 de 0,909. O método estatístico foi melhor nesse cenário. O
gerador criou anomalias sobretudo como aumentos de frequência, favorecendo
um método que mede diretamente a contagem de registros. Não há base para
afirmar que qualquer um dos dois terá essas taxas na maquete.

Os resultados da versão 1 estão em
[`MODELO_ISOLATION_FOREST.md`](MODELO_ISOLATION_FOREST.md). Suas métricas não
podem ser comparadas diretamente com as desta versão porque os cenários de
teste são diferentes. A melhora da configuração simplificada foi observada
na comparação **dentro dos mesmos blocos** de desenvolvimento da versão 2.

## Reproduzir

Na raiz do projeto, com o ambiente Python instalado:

```powershell
smartlighting-generate-v2
smartlighting-prepare-v2
smartlighting-experiments-v2 --focused --processed-dir data/processed/v2_development --output-dir artifacts/experiments_v2/focused_120d
smartlighting-generate-v2 --output-dir data/raw/synthetic_v2_holdout --scenario synthetic_v2_holdout --days 30 --seed 20261011 --start-local 2026-06-01
smartlighting-prepare-v2 --raw-dir data/raw/synthetic_v2_holdout --output-dir data/processed/v2_holdout
smartlighting-train-v2
```

O código da comparação entre configurações está em `experiments/v2.py`. O
relatório do teste final é gerado em
`artifacts/isolation_forest_v2/results/test_metrics.csv`; as respostas de cada
intervalo ficam em `test_predictions.csv`. Esses arquivos são ignorados pelo
Git e podem ser recriados pelos comandos acima.

## Limite para uso real

O modelo ainda não recebe os `SensorEvents` do SQL Server. Um alerta real
exigiria histórico representativo, tratamento de oscilações e falhas de
telemetria, associação aos GUIDs dos sensores reais, avaliação com dados não
usados no treino e integração com a API. A versão 2 demonstra treinamento,
avaliação e integração de **resultados simulados** com a página Web.
