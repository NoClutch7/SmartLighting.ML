# Planejamento do SmartLighting.ML

## 1. Contexto

O `SmartLighting.ML` integra um TCC de iluminação pública inteligente aplicado a
um condomínio residencial de formato quadrado. A maquete do projeto contém:

- 10 postes de iluminação distribuídos ao redor da quadra, no perímetro do
  condomínio;
- nenhum poste instalado dentro da quadra;
- 6 sensores PIR espalhados pelas calçadas e controlados por ESP32;
- ruas internas, calçadas, casas e a quadra central representadas fisicamente.

As posições exatas dos seis sensores ainda não foram mapeadas. Assim, nenhum
`SensorId` representa por enquanto uma entrada, rua, área esportiva ou outra
região específica. Seus registros serão utilizados para caracterizar o padrão
individual de cada sensor sem inferir uma localização não confirmada.

## 2. Objetivo do módulo de Machine Learning

O módulo deverá aprender automaticamente o padrão individual de cada sensor, sem
depender de uma lista fixa de horários permitidos. O resultado esperado será uma
indicação de normalidade ou anomalia acompanhada de uma pontuação, preservando a
identificação do sensor e o intervalo analisado.

Exemplos do contexto residencial incluem diferenças de circulação entre sensores
em horários e dias distintos e eventos incomuns para um sensor considerando seu
próprio histórico. A interpretação física dessas diferenças somente será feita
depois do mapeamento das posições.

Esses exemplos descrevem o domínio e não constituem regras fixas de detecção.

## 3. Estratégia planejada

### 3.1 Unidade de análise

Os eventos de movimento serão agrupados em janelas consecutivas de cinco
minutos. Cada janela deverá permanecer associada ao sensor que originou os dados.

### 3.2 Características temporais

O estudo inicial considerará, no mínimo:

- quantidade ou frequência de eventos na janela de cinco minutos;
- horário da janela;
- dia da semana;
- identificação do sensor ou treinamento individual por sensor.

Outras características somente serão incluídas após análise dos dados coletados.

### 3.3 Modelo

O algoritmo planejado é o **Isolation Forest**, da biblioteca Scikit-learn. Por
ser um método não supervisionado de detecção de anomalias, ele é adequado ao
cenário inicial, no qual não se espera dispor de um conjunto amplo de eventos
previamente rotulados como normais ou anormais.

O treinamento, os hiperparâmetros, os critérios de avaliação e os limiares de
decisão serão definidos em uma etapa posterior. Nenhum modelo está implementado
nesta fase.

## 4. Fluxo futuro de dados

O fluxo conceitual previsto é:

1. os sensores registram movimento nos pontos da maquete;
2. os eventos são disponibilizados para processamento;
3. o módulo agrupa os eventos por sensor em janelas de cinco minutos;
4. as características temporais são preparadas;
5. o Isolation Forest avalia cada janela em relação ao histórico do sensor;
6. o resultado poderá ser consumido pela `SmartLighting.Api`.

O formato de comunicação e a tecnologia de transporte entre Python e C# ainda
serão definidos. A separação em `contracts`, `data_processing`, `training` e
`inference` prepara o projeto para essa integração sem criar acoplamento agora.

## 5. Fases previstas

1. definir o formato dos registros produzidos pelos sensores;
2. mapear a posição física de cada `SensorId` na maquete;
3. coletar e explorar dados representativos da maquete residencial;
4. validar qualidade, continuidade e identificação dos eventos;
5. construir o processamento em janelas de cinco minutos — concluído nesta
   etapa;
6. treinar e avaliar o Isolation Forest por sensor;
7. definir critérios de alerta e monitoramento;
8. especificar e implementar a integração com a `SmartLighting.Api`;
9. criar testes e documentar os resultados do TCC.

## 6. Fora do escopo desta etapa

- implementação ou treinamento do modelo;
- definição definitiva de limiares de anomalia;
- alterações na `SmartLighting.Api`;
- alterações no firmware dos dispositivos ESP32;
- acionamento automático dos postes a partir das anomalias.

O gerador de dados sintéticos desta fase está documentado em
[`DADOS_SINTETICOS.md`](DADOS_SINTETICOS.md). Ele apoia o desenvolvimento e os
testes, mas não substitui a coleta real da maquete.
