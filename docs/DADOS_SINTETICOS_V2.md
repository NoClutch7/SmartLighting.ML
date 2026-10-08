# Cenário sintético de desenvolvimento v2

## Escopo

O cenário v2 simula 120 dias consecutivos, de 1º de fevereiro a 31 de maio de
2026 no horário `America/Sao_Paulo`, para os sensores `PIR-01` a `PIR-06`. Ele é
exclusivamente de desenvolvimento: nenhum novo teste foi criado.

Os arquivos brutos ficam em `data/raw/synthetic_v2_development` e os processados
em `data/processed/v2_development`. Os contratos de evento permanecem:
`EventIdentifier`, `SensorId`, `DeviceId` e `OccurredAtUtc`.

## Hipóteses de atividade normal

Cada sensor conserva seu perfil probabilístico individual para madrugada,
manhã, tarde, noite e fins de semana. Para evitar semanas artificialmente
idênticas, são acrescentados:

- fator diário log-normal independente por sensor, com sigma 0,18;
- variação senoidal lenta de 28 dias, com amplitude 0,12;
- amostragem Poisson em cada janela de cinco minutos.

Esses valores são hipóteses de simulação, não estimativas de moradores reais.
Eventos continuam representando pulsos `HIGH → LOW`, não pessoas.

## Episódios anômalos

É planejado um episódio independente por sensor e por dia. Os tipos alternam
entre:

- `out_of_pattern_burst`: rajada de até três minutos;
- `electrical_oscillation`: sequência rápida de doze pulsos.

Horário e minuto variam deterministicamente por dia e sensor, cobrindo todas as
24 horas e os sete dias da semana. Cada episódio recebe um
`EpisodeIdentifier`; pulsos do mesmo fenômeno não são contados como episódios
independentes.

Dos 720 episódios planejados, 718 foram observados. Duas oscilações foram
suprimidas pela indisponibilidade/rearme, comportamento coerente com a simulação.

| Sensor | Rajadas | Oscilações | Total |
| --- | ---: | ---: | ---: |
| PIR-01 | 60 | 60 | 120 |
| PIR-02 | 60 | 59 | 119 |
| PIR-03 | 60 | 60 | 120 |
| PIR-04 | 60 | 60 | 120 |
| PIR-05 | 60 | 60 | 120 |
| PIR-06 | 60 | 59 | 119 |

Os rótulos permanecem separados em `anomaly_labels.csv`.

## Movimento intenso legítimo

Foram criados 102 períodos legítimos de atividade intensa, com duração entre 10
e 20 minutos e pulsos mais frequentes. Eles não recebem rótulo de anomalia e não
se sobrepõem temporalmente aos episódios anômalos do mesmo sensor. Funcionam como
hard negatives para evitar que toda contagem alta seja artificialmente anômala.

O arquivo `legitimate_activity_episodes.csv` existe apenas para auditoria do
cenário e não entra como característica ou rótulo do modelo.

## Indisponibilidades

Foram simulados 30 intervalos de indisponibilidade. Eventos dentro deles não são
observados e 506 janelas de cinco minutos foram marcadas como não confiáveis,
com contagem ausente. Elas não são convertidas em zero nem utilizadas pelos
experimentos.

## Volumes finais

- duração: 120 dias;
- eventos observados: 99.404;
- pulsos rotulados: 8.256;
- episódios anômalos independentes: 718;
- períodos legítimos intensos: 102;
- janelas totais: 207.360;
- janelas elegíveis para desenvolvimento: 206.422;
- janelas não confiáveis: 506.

## Cobertura dos folds

Os folds progressivos apresentam 23–25 episódios por sensor na calibração e
23–24 na avaliação. Cada combinação sensor/bloco contém pelo menos 10 episódios
de cada tipo. Nenhum episódio cruza uma fronteira de treino, calibração ou
avaliação.

Esse objetivo melhora a viabilidade metodológica, mas não garante estabilidade
estatística. A versão 2 ainda deverá reportar variação entre folds, métricas por
tipo e números absolutos. O novo teste independente somente poderá ser criado ou
coletado depois que características, hiperparâmetros e critérios de seleção
forem congelados.
