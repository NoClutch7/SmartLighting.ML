# Dados sintéticos de sensores PIR

## Finalidade

O gerador produz 30 dias de telemetria reproduzível para seis sensores PIR
espalhados pelas calçadas da maquete. Os identificadores fixos são `PIR-01` a
`PIR-06`, associados respectivamente a `ESP32-SENSOR-01` a
`ESP32-SENSOR-06`. Como as posições ainda não foram mapeadas, nenhum identificador
indica uma região física específica.

A maquete possui 10 postes distribuídos ao redor da quadra, no perímetro do
condomínio, e nenhum poste dentro da quadra. O gerador modela a telemetria dos
sensores, não a posição ou o acionamento individual desses postes.

O CSV principal representa **pulsos elétricos de ativação**, e não pessoas. Uma
pessoa pode produzir nenhum, um ou vários pulsos, conforme o sensor, seu tempo de
retenção e o movimento observado.

## Correspondência com o firmware

O firmware registra uma ativação somente na transição do sinal inativo para o
ativo. Como o circuito é ativo em nível baixo, essa borda é `HIGH → LOW`.
Enquanto o sinal continua em `LOW`, o gerador não aceita outra ativação para o
mesmo sensor. Uma nova ativação somente pode ocorrer depois do fim do pulso e do
intervalo de rearme.

O retorno da iluminação ao brilho de repouso de aproximadamente 8% ocorre após
sete segundos sem sinal ativo e é seguido por uma transição gradual. Esse
comportamento pertence ao atuador de iluminação: ele não gera eventos adicionais
e, portanto, não aparece como uma linha no CSV.

## Parâmetros assumidos

A duração real dos pulsos e o rearme do hardware ainda não foram medidos. Para
que essa incerteza fique explícita, o cenário padrão usa:

| Parâmetro | Valor sintético |
| --- | ---: |
| Duração de um pulso PIR normal | uniforme entre 2 e 6 segundos |
| Rearme após um pulso normal | 1,5 segundo |
| Duração durante oscilação elétrica | uniforme entre 0,15 e 0,45 segundo |
| Rearme durante oscilação elétrica | 0,25 segundo |
| Retenção da iluminação pelo firmware | 7 segundos sem sinal ativo |
| Brilho de repouso | aproximadamente 8% |
| Janela analítica futura | 5 minutos |

Os parâmetros estão centralizados em `GeneratorConfig` e devem ser recalibrados
quando medições do hardware estiverem disponíveis.

## Padrões simulados

As taxas probabilísticas variam individualmente para cada um dos seis sensores,
por hora local de São Paulo e tipo de dia. Cada perfil possui taxas distintas
para madrugada, manhã, tarde e noite, além de um fator próprio para fins de
semana. Os perfis são abstratos: servem para produzir diversidade estatística e
não representam locais específicos da maquete.

Os horários locais são usados para formar o comportamento residencial, mas os
timestamps são convertidos e gravados em UTC no formato ISO 8601.

## Anomalias e indisponibilidade

São inseridas duas classes controladas de anomalia:

- `out_of_pattern_burst`: rajada de pulsos na madrugada, fora do padrão usual;
- `electrical_oscillation`: alternâncias elétricas rápidas que respeitam os
  estados ativo, inativo e de rearme da simulação.

Os rótulos ficam somente em `anomaly_labels.csv`, relacionados ao evento por
`EventIdentifier`. O arquivo `pir_events.csv` não revela quais linhas são
anômalas e pode ser usado como entrada de uma avaliação sem vazamento do rótulo.

Quando habilitadas, as indisponibilidades removem da telemetria observável os
eventos ocorridos nos respectivos intervalos. Esses intervalos são registrados
em `telemetry_outages.csv`, permitindo diferenciar silêncio observado de ausência
de dados confiáveis. Use `--no-include-outages` para gerar um cenário sem falhas
de telemetria.

## Arquivos e esquema

`pir_events.csv` contém exatamente:

- `EventIdentifier`;
- `SensorId`;
- `DeviceId`;
- `OccurredAtUtc`.

`anomaly_labels.csv` contém `EventIdentifier` e `AnomalyType`.
`telemetry_outages.csv` contém sensor, dispositivo, início, fim e motivo da
indisponibilidade.

## Limitações

- As taxas foram definidas por hipótese, não estimadas de moradores reais.
- Os perfis não podem ser interpretados como regiões físicas até que os sensores
  sejam mapeados na maquete.
- A simulação não representa a relação espacial entre os 6 sensores e os 10
  postes do perímetro.
- Não existe modelagem de indivíduos, direção, velocidade ou quantidade de
  pessoas.
- Clima, feriados, eventos do condomínio, animais, veículos e obstruções não são
  representados explicitamente.
- A duração e o rearme do PIR são suposições e podem divergir do componente real.
- Oscilações elétricas reais podem ter formas e frequências diferentes.
- Falhas de rede são representadas como intervalos completos, sem atrasos,
  duplicações ou reordenação de mensagens.
- As anomalias controladas não cobrem todas as situações relevantes e não devem
  ser usadas como prova isolada de desempenho no mundo real.

Na avaliação do TCC, os resultados sintéticos devem ser apresentados como
evidência de funcionamento do pipeline. A validade prática do detector exige
posterior coleta na maquete, calibração dos parâmetros e comparação com eventos
reais revisados.
