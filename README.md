# SmartLighting.ML

Projeto de Machine Learning de um TCC sobre iluminação pública inteligente em um
condomínio residencial de formato quadrado. A maquete possui 10 postes de
iluminação distribuídos ao redor da quadra, no perímetro do condomínio, e 6
sensores PIR espalhados pelas calçadas e controlados por dispositivos ESP32. Não
existem postes instalados dentro da quadra.

O objetivo deste módulo é aprender o comportamento histórico de circulação de
cada sensor e, futuramente, identificar movimentações atípicas. As posições
exatas dos sensores ainda não foram mapeadas; por isso, o projeto usa apenas
identificadores neutros e não atribui sensores a regiões específicas.

Esta versão contém o gerador de telemetria sintética e o pipeline de preparação
dos dados. O modelo, o treinamento e a integração com a API C#
`SmartLighting.Api` ainda não foram implementados.

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
├── artifacts/                 # Modelos e metadados gerados futuramente
├── data/
│   ├── raw/                   # Dados recebidos sem transformação
│   └── processed/             # Dados preparados para treinamento/inferência
├── src/
│   └── smartlighting_ml/
│       ├── data_processing/   # Limpeza e transformação de dados dos sensores
│       │   ├── synthetic_generator.py
│       │   └── preparation_pipeline.py
│       ├── training/          # Treinamento e persistência do modelo
│       │   └── train_isolation_forest.py
│       ├── inference/         # Detecção de anomalias com modelo treinado
│       │   └── predictor.py
│       ├── experiments/       # Experimentos metodológicos isolados
│       │   └── v2.py
│       └── contracts/         # Contratos de entrada e saída para a API C#
├── tests/                     # Testes automatizados
├── docs/
│   ├── PLANEJAMENTO.md        # Escopo e etapas planejadas do módulo de ML
│   ├── DADOS_SINTETICOS.md    # Hipóteses e uso do gerador
│   ├── PREPARACAO_DADOS.md    # Regras do pipeline de características
│   ├── MODELO_ISOLATION_FOREST.md
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

- `pir_events.csv`: pulsos observados, sem qualquer rótulo;
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

## Integração futura

A pasta `contracts` será responsável pelas estruturas de dados trocadas com a
`SmartLighting.Api`. Essa separação permitirá manter o formato das requisições e
respostas independente da lógica de processamento, treinamento e inferência.

Uma futura integração poderá executar o serviço Python separadamente e trocar
dados serializados com a API C#, sem acoplar o código do modelo ao projeto .NET.

## Abordagem planejada

O modelo previsto é o **Isolation Forest**, aplicado separadamente aos padrões de
cada sensor. Os eventos serão agregados em janelas de cinco minutos e analisados
em conjunto com informações de horário e dia da semana. Dessa forma, uma
movimentação não será considerada anormal apenas por ocorrer com baixa frequência:
ela será comparada ao comportamento histórico esperado para aquele sensor e
contexto temporal.

O planejamento detalhado está em [`docs/PLANEJAMENTO.md`](docs/PLANEJAMENTO.md).

## Treinamento e avaliação

Para treinar os seis modelos, calibrar os limiares somente com validação e então
avaliar o teste reservado:

```powershell
smartlighting-train
```

Os modelos, limiares, metadados, previsões e métricas são gravados em
`artifacts/isolation_forest_v1`. A metodologia e os resultados iniciais estão em
[`docs/MODELO_ISOLATION_FOREST.md`](docs/MODELO_ISOLATION_FOREST.md).

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

Execute a grade completa somente depois de ampliar os positivos dos blocos de
desenvolvimento:

```powershell
smartlighting-experiments-v2
```

Consulte [`docs/EXPERIMENTOS_V2.md`](docs/EXPERIMENTOS_V2.md) antes de interpretar
qualquer resultado. As hipóteses do cenário de 120 dias estão em
[`docs/DADOS_SINTETICOS_V2.md`](docs/DADOS_SINTETICOS_V2.md).
