# SmartLighting.ML

Projeto de Machine Learning de um TCC sobre iluminação pública inteligente em um
condomínio residencial de formato quadrado. A maquete possui 10 postes de
iluminação distribuídos ao redor da quadra, no perímetro do condomínio, e 6
sensores espalhados pelas calçadas e controlados por dispositivos ESP32. Não
existem postes instalados dentro da quadra.

O objetivo deste módulo é estudar padrões de atividade de cada sensor e
identificar intervalos atípicos. As posições exatas dos sensores ainda não
foram mapeadas; por isso, os perfis simulados usam identificadores neutros e
não são associados a regiões específicas nem aos GUIDs cadastrados na API.

Esta versão gera dados simulados, prepara intervalos de cinco minutos, treina
seis modelos Isolation Forest e compara seus resultados com um método
estatístico. A versão 2 foi treinada e avaliada em um período simulado posterior
ao desenvolvimento. Um serviço Python disponibiliza os resultados para
a API e para a página **IA** da Web. O treinamento e a avaliação ainda não usam
eventos reais do SQL Server e não produzem alertas operacionais.

## Tecnologias previstas

- Python
- Pandas
- NumPy
- Scikit-learn
- Isolation Forest (disponível no Scikit-learn)
- Pytest

## Estrutura

```text
SmartLighting.ML/
├── artifacts/                 # Modelos, previsões e métricas gerados localmente
├── data/
│   ├── raw/                   # Dados recebidos sem transformação
│   └── processed/             # Dados preparados para treinamento/inferência
├── src/
│   └── smartlighting_ml/
│       ├── data_processing/   # Limpeza e transformação de dados dos sensores
│       │   ├── synthetic_generator.py
│       │   └── preparation_pipeline.py
│       ├── training/          # Treinamento e persistência dos modelos
│       │   ├── train_isolation_forest.py
│       │   └── train_v2.py
│       ├── inference/         # Inferência e serviço HTTP experimental
│       │   ├── predictor.py
│       │   └── demo_server.py
│       ├── experiments/       # Experimentos metodológicos isolados
│       │   └── v2.py
├── tests/                     # Testes automatizados
├── docs/
│   ├── PLANEJAMENTO.md        # Escopo e etapas planejadas do módulo de ML
│   ├── DADOS_SINTETICOS.md    # Hipóteses e uso do gerador
│   ├── PREPARACAO_DADOS.md    # Regras do pipeline de características
│   ├── MODELO_ISOLATION_FOREST.md
│   ├── MODELO_V2.md
│   └── EXPERIMENTOS_V2.md
├── .gitignore
├── README.md
└── requirements.txt
```

Os arquivos `.gitkeep` mantêm no Git as pastas que, inicialmente, não possuem
conteúdo. Arquivos de dados, modelos treinados e configurações locais são
ignorados para evitar versionar artefatos grandes ou informações sensíveis.

## Preparação do ambiente

Requer Python 3.11 ou superior.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Gerador de dados sintéticos

Após preparar o ambiente, gere o conjunto padrão de 30 dias com:

```powershell
smartlighting-generate
```

O comando cria em `data/raw/synthetic`:

- `sensor_events.csv`: pulsos observados, sem qualquer rótulo;
- `anomaly_labels.csv`: rótulos separados, exclusivamente para avaliação;
- `telemetry_outages.csv`: intervalos em que a ausência de eventos não é
  confiável.

A execução usa a semente fixa `20261008`. É possível escolher outro diretório,
semente ou desativar as indisponibilidades:

```powershell
smartlighting-generate --output-dir data/raw/meu_cenario --seed 42 --no-include-outages
```

As hipóteses elétricas, os padrões simulados e as limitações estão detalhados em
[`docs/DADOS_SINTETICOS.md`](docs/DADOS_SINTETICOS.md).

Execute os testes com:

```powershell
pytest
```

## Preparação dos dados

Com os arquivos sintéticos disponíveis, execute:

```powershell
smartlighting-prepare
```

O pipeline valida e deduplica os eventos, cria janelas completas de cinco
minutos, marca indisponibilidades, calcula características temporais sem usar o
futuro e grava os conjuntos cronológicos em `data/processed`. A especificação
detalhada está em
[`docs/PREPARACAO_DADOS.md`](docs/PREPARACAO_DADOS.md).

## Treinamento e avaliação reproduzíveis

Execute estes comandos, nesta ordem, a partir da raiz do projeto:

```powershell
smartlighting-generate
smartlighting-prepare
smartlighting-train
```

O conjunto padrão simula 30 dias de atividade de seis sensores. As linhas
preparadas são divididas em ordem cronológica: treino para ajustar os modelos,
validação para definir os limiares e teste posterior para medir o resultado.
As situações anormais criadas pelo gerador têm rótulos separados; esses rótulos
não são características fornecidas ao Isolation Forest no treinamento.

As métricas completas ficam em
`artifacts/isolation_forest_v1/results/test_metrics.csv` e as respostas por
intervalo em `artifacts/isolation_forest_v1/results/test_predictions.csv`.
Esses arquivos são gerados localmente e ignorados pelo Git; o procedimento e os
resultados avaliados estão descritos em
[`docs/MODELO_ISOLATION_FOREST.md`](docs/MODELO_ISOLATION_FOREST.md).

## Demonstração experimental na API e na Web

Depois de gerar, preparar e treinar a versão 2, inicie o serviço Python na raiz
deste projeto:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m smartlighting_ml.inference.demo_server
```

O serviço escuta somente em `127.0.0.1:8765`. Por padrão, ele lê os resultados
do teste final sintético da versão 2, carrega os modelos locais e verifica
novamente quatro exemplos: um
acerto de anomalia, um alarme falso, uma anomalia não detectada e um acerto de
período normal. A `SmartLighting.Api` consulta `GET /demo` desse serviço por
meio da rota `GET /ml/demo`. A `SmartLighting.Web` apresenta os totais de
treino, validação e teste, as métricas dos dois métodos e os quatro exemplos
na página `/ia`, acessível pelo menu **IA**. Caso o Python não esteja em
execução, a página informa a indisponibilidade.

A API e a Web devem estar em execução para visualizar a página. O serviço
Python precisa ser iniciado novamente após um novo treinamento, pois carrega
os resultados ao iniciar.

Esta demonstração **não** lê `SensorEvents` do SQL Server, não grava anomalias
no banco e não emite alertas operacionais. Os identificadores `SENSOR-01` a
`SENSOR-06` representam perfis simulados; não são os GUIDs dos sensores da
maquete. Os resultados medem apenas o comportamento do modelo nesse cenário
sintético. O processamento de eventos reais permanece como etapa futura.

## Abordagem implementada

O **Isolation Forest** é aplicado separadamente aos padrões simulados de cada
sensor. Os registros são agregados em janelas de cinco minutos e analisados
com informações de horário, dia da semana e atividade recente. O método
estatístico de comparação considera a frequência de registros e foi avaliado
nos mesmos intervalos de teste. Os resultados atuais não demonstram detecção
confiável nos sensores reais.

O planejamento detalhado está em [`docs/PLANEJAMENTO.md`](docs/PLANEJAMENTO.md).

## Resultado do primeiro experimento

No teste sintético reservado, o Isolation Forest detectou 3 de 16 intervalos
anormais e marcou 87 intervalos normais como atípicos. O método estatístico
detectou 13 de 16 e não marcou intervalos normais nesse mesmo conjunto. Como
as anomalias simuladas são principalmente aumentos de frequência, o cenário
favorece o método estatístico. Consulte
[`docs/MODELO_ISOLATION_FOREST.md`](docs/MODELO_ISOLATION_FOREST.md) para a
metodologia, a tabela completa e as limitações.

## Experimentos v2

A estrutura de desenvolvimento v2 usa somente os períodos originalmente
destinados a treino e validação. O teste v1 não participa da comparação.

Valide rapidamente o fluxo com:

```powershell
smartlighting-experiments-v2 --quick
```

O cenário ampliado de desenvolvimento é criado e preparado separadamente:

```powershell
smartlighting-generate-v2
smartlighting-prepare-v2
smartlighting-experiments-v2 --quick --processed-dir data/processed/v2_development --output-dir artifacts/experiments_v2/development_120d
```

Uma comparação focada nos blocos de desenvolvimento levou à escolha da versão
2 do Isolation Forest com contagem atual e informações de horário e dia. Para
recriar o **novo teste**, posterior aos 120 dias de desenvolvimento:

```powershell
smartlighting-experiments-v2 --focused --processed-dir data/processed/v2_development --output-dir artifacts/experiments_v2/focused_120d
smartlighting-generate-v2 --output-dir data/raw/synthetic_v2_holdout --scenario synthetic_v2_holdout --days 30 --seed 20261011 --start-local 2026-06-01
smartlighting-prepare-v2 --raw-dir data/raw/synthetic_v2_holdout --output-dir data/processed/v2_holdout
smartlighting-train-v2
```

No novo teste sintético, o Isolation Forest v2 detectou 82 de 179 intervalos
anormais, deixou passar 97 e produziu 70 alarmes falsos. O método estatístico
detectou 169 de 179 e produziu 24 alarmes falsos. Esses resultados **não**
comprovam desempenho com sensores físicos. Os detalhes, os comandos completos
e as limitações estão em [`docs/MODELO_V2.md`](docs/MODELO_V2.md). Consulte
também [`docs/EXPERIMENTOS_V2.md`](docs/EXPERIMENTOS_V2.md) e
[`docs/DADOS_SINTETICOS_V2.md`](docs/DADOS_SINTETICOS_V2.md).
