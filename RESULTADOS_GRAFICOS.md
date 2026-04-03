# Interpretação dos Gráficos — Algoritmo WTA (Winner-Takes-All)

> **Contexto:** Este documento explica os gráficos gerados pelo algoritmo WTA implementado em `main.py` e `test.py`, aplicado ao dataset Iris com 3 neurônios e até 4 features. Cada seção corresponde a uma chamada de `plot_*` no código.

---

## 1. Curvas de Aprendizado — `plot_curva_aprendizado()`

Estes dois gráficos juntos respondem à pergunta: **"O modelo aprendeu bem e na velocidade certa?"**

---

### 1.1 Taxa de Aprendizado ao Longo das Épocas (Decaimento Linear)

**O que é plotado:** A evolução do valor de `learning_rate` a cada época.

**Como é calculado:**

$$LR_t = LR_0 \times \left(1 - \frac{t}{T}\right)$$

Onde $LR_0$ é a taxa inicial, $t$ é a época atual e $T$ é o total de épocas configuradas.

**O que o gráfico mostra:**
- Uma **reta decrescente** que parte de $LR_0$ (ex: 0.1) e converge para zero na última época.
- Este decaimento é **determinístico e baseado no tempo**, garantindo que o treino finalize independentemente dos dados.

**Como interpretar:**
| Comportamento | Significado |
|---|---|
| Reta decaindo suavemente | Comportamento esperado e saudável |
| Reta termina antes da última época | O critério de parada foi acionado (convergência por QE) |

**Por que esse método?** O decaimento linear garante que nas épocas iniciais o neurônio dá "saltos maiores" para explorar o espaço, e nas épocas finais faz "ajustes finos" com passos menores — sem depender do erro para isso.

---

### 1.2 Erro de Quantização ao Longo das Épocas (Saúde do Treino)

**O que é plotado:** O Erro de Quantização médio (QE) calculado após cada época completa.

**Como é calculado:**

$$QE_t = \frac{1}{N} \sum_{i=1}^{N} \| x_i - w_{vencedor(x_i)} \|$$

Onde $N$ é o número de amostras, $x_i$ é a amostra e $w_{vencedor}$ é o peso do neurônio mais próximo.

**O que o gráfico mostra:**
- Uma **curva decrescente** que indica que os neurônios estão se aproximando das amostras.
- O preenchimento sob a curva evidencia a área total de erro ao longo do treino.

**Como interpretar:**
| Comportamento | Significado |
|---|---|
| Queda acentuada nas primeiras épocas | Neurônios saindo do centróide inicial e se especializando |
| Queda gradual e suave | Ajuste fino convergindo bem |
| Platô (curva ficou horizontal) | Convergência — os neurônios pararam de se mover significativamente |
| O critério $\Delta QE < 10^{-6}$ foi atingido entre duas épocas | O loop de treino foi interrompido com `break` (Early Stopping) |

---

## 2. Métricas de Treinamento — `plot_metricas()`

Três gráficos que detalham a dinâmica interna do treinamento.

---

### 2.1 Erro de Quantização (QE) por Época

Igual ao descrito na Seção 1.2, mas agora apresentado junto com os outros dois gráficos de métricas para análise comparativa.

**Ponto de atenção:** Este é o termômetro do modelo. Se ao final o QE ainda estiver alto (longe de zero), significa que os neurônios não representam bem os dados — pode ser necessário aumentar as épocas ou ajustar a taxa inicial.

---

### 2.2 Taxa de Variância (VR) por Época

**O que é plotado:** A variância média das posições dos neurônios em relação ao centróide deles mesmos.

**Como é calculado:**

$$\bar{w} = \frac{1}{K} \sum_{k=1}^{K} w_k, \qquad VR = \frac{1}{K} \sum_{k=1}^{K} \| w_k - \bar{w} \|^2$$

Onde $K$ é o número de neurônios.

**O que o gráfico mostra:**
- A **separação** entre os neurônios ao longo do tempo.
- Uma VR crescente indica os neurônios se espalhando e se especializando em regiões distintas do espaço.

**Como interpretar:**
| Comportamento | Significado |
|---|---|
| VR cresce rapidamente nas 1ªs épocas | Neurônios saindo do centróide central e divergindo para clusters distintos |
| VR estabiliza | Neurônios se fixaram em suas posições finais |
| VR muito baixa ao final | Neurônios muito próximos — possível colapso (não se separaram bem) |
| VR muito alta | Neurônios podem ter divergido para regiões sem dados relevantes |

---

### 2.3 Taxa de Decaimento (DR) entre Épocas

**O que é plotado:** A variação relativa do QE entre duas épocas consecutivas — exibida como gráfico de barras.

**Como é calculado:**

$$DR_t = \frac{QE_{t-1} - QE_t}{QE_{t-1}}, \quad \text{se } QE_{t-1} \neq 0$$

**O que o gráfico mostra:**
- **Barras verdes** (DR ≥ 0): o erro reduziu em relação à época anterior — melhora positiva.
- **Barras vermelhas** (DR < 0): o erro aumentou — instabilidade ou oscilação.

**Como interpretar:**
| Comportamento | Significado |
|---|---|
| Barras verdes altas no início, pequenas ao final | Curva clássica de convergência saudável |
| Barras oscilando entre verde e vermelho | Instabilidade — learning rate inicial pode ser alta demais |
| Todas as barras próximas de zero | Convergência rápida ou modelo estagnado desde o início |

---

## 3. Erro Quadrático por Amostra — `plot_erro_quadratico()`

Dois painéis que respondem: **"Quais amostras o modelo representa mal?"**

---

### 3.1 Dispersão do Erro Quadrático por Índice

**O que é plotado:** Para cada amostra $i$, seu erro quadrático individual em relação ao neurônio vencedor, colorido pelo neurônio que a capturou.

**Como é calculado:**

$$EQ_i = \| x_i - w_{vencedor(x_i)} \|^2 = \sum_j (x_{ij} - w_{kj})^2$$

A linha vermelha tracejada representa o **MSE médio** $= \frac{1}{N}\sum EQ_i$.

**Como interpretar:**
| Comportamento | Significado |
|---|---|
| Pontos concentrados abaixo da linha vermelha | A maioria das amostras está bem representada |
| Pontos isolados com EQ muito alto | Amostras outlier ou na fronteira de dois clusters |
| Um neurônio (cor) com erros sistematicamente maiores | Aquele neurônio está cobrindo uma região muito ampla ou heterogênea |

**Relação com o terminal:**
- `MSE (médio)`: média de todos os $EQ_i$ — quanto menor, melhor.
- `SSE (total)`: soma de todos os $EQ_i$ — útil para comparar modelos com mesmo número de amostras.

---

### 3.2 Histograma da Distribuição dos Erros Quadráticos

**O que é plotado:** A distribuição de frequência de todos os $EQ_i$.

**Como interpretar:**
| Comportamento | Significado |
|---|---|
| Histograma concentrado próximo a zero | A maior parte das amostras está muito perto de seus neurônios — boa representação |
| Distribuição espalhada / cauda longa à direita | Há amostras distantes dos neurônios — possível problema na fronteira entre classes |
| Dois "picos" (bimodal) | Dois grupos distintos de erros — neurônios especializados vs. zona de transição |

---

## 4. Movimento dos Neurônios — `plot_wta_movement()`

**O que é plotado:** As trajetórias dos neurônios ao longo das épocas, desde a posição inicial (centróide geral dos dados) até a posição final.

**Como interpretar:**
- Cada linha conecta as posições do neurônio a cada época salvas em `wta.history`.
- Linhas longas = neurônio precisou se deslocar muito para encontrar seu cluster.
- Linhas curtas = neurônio já estava próximo do cluster desde o início.

**Resultado esperado no Iris:** Os 3 neurônios devem se dispersar para as regiões de *Setosa*, *Versicolor* e *Virginica*.

---

## 5. WTA vs K-Means — `plot_comparison()`

**O que é plotado:** Posições finais dos neurônios do WTA (marcador `X` preto) e dos centróides do K-Means (marcador `◆` amarelo) sobre os dados de treino.

**Como interpretar:**
- Se os marcadores WTA e K-Means estiverem **sobrepostos ou muito próximos**, o WTA convergiu para soluções equivalentes às do K-Means clássico — excelente resultado.
- Se estiverem **distantes**, o WTA pode ter ficado preso em um mínimo local ou convergido de forma diferente por causa da aleatoriedade da ordem das amostras.

**Dado numérico complementar** (saída no terminal):
```
Distância euclidiana 4D: X.XXXX
```
- Valores abaixo de `0.1` indicam convergência praticamente idêntica ao K-Means.
- Valores acima de `0.5` sugerem divergência relevante entre os dois algoritmos.

---

## Sumário Rápido de Diagnóstico

| Gráfico | Ideal | Problema |
|---|---|---|
| **LR** | Reta suave decrescendo de $LR_0$ a ~0 | Termina muito cedo (early stopping precoce) |
| **QE (Curva)** | Queda e platô abaixo de 0.5 | Ainda alto ao final → mais épocas |
| **QE (Métricas)** | Queda contínua | Oscilações → instabilidade |
| **VR** | Cresce e estabiliza | Fica baixa → neurônios não separaram |
| **DR** | Barras verdes decrescentes | Barras vermelhas → instabilidade |
| **EQ Dispersão** | Pontos abaixo do MSE | Pontos muito altos → outliers / fronteiras |
| **EQ Histograma** | Pico concentrado em zero | Cauda longa → má representação |
| **WTA vs K-Means** | Marcadores sobrepostos | Distantes → convergência diferente |

---

## Observações sobre o Dataset Iris

- **Iris-setosa** tende a ser perfeitamente separável, com pureza 100% e EQ muito baixo.
- **Iris-versicolor** e **Iris-virginica** têm fronteira de decisão difusa, causando os erros de classificação observados (~4 a 6 amostras erradas nas execuções típicas).
- A acurácia observada de **86–91%** é consistente com o esperado para WTA não supervisionado aplicado ao Iris com 4 features, sem qualquer ajuste de hiperparâmetros por validação cruzada.
