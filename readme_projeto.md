# Documentação do Projeto: Redes Neurais Competitivas (WTA) com Dataset Iris

Este documento descreve o funcionamento e os objetivos do script `main.py`, que implementa uma rede neural competitiva do tipo **Winner-Takes-All (WTA)** aplicada ao clássico conjunto de dados Iris.

## 📋 Objetivo do Script

O objetivo principal deste código é demonstrar a capacidade de **aprendizado não supervisionado** de uma rede WTA. O script busca agrupar (clusterizar) as amostras do dataset Iris baseando-se em suas características físicas (`sepal_length` e `petal_width`), sem conhecer previamente os rótulos das espécies.

Ao final, o modelo é validado para verificar se os grupos formados pelos neurônios correspondem às espécies reais (**Setosa, Versicolor e Virginica**).

---

## 🛠️ Como o Código Funciona

O fluxo de execução é dividido em quatro etapas principais:

### 1. Processamento de Dados (`DataProcessor`)
O script carrega o arquivo `Iris.csv` e organiza os dados.
- **Divisão Configurável**: Você pode definir a proporção de treino e teste (ex: 70% para treino e 30% para teste).
- **Separação por Classe**: O sistema garante que a proporção de cada espécie seja mantida em ambos os conjuntos, evitando vieses.

### 2. Inicialização dos Neurônios
Diferente de inicializações aleatórias comuns, este script utiliza uma estratégia de **Centro Geométrico**:
- Todos os neurônios começam exatamente na mesma coordenada: a **média global** de todas as amostras do dataset.
- Isso coloca os neurônios no "coração" dos dados desde o início.

### 3. Treinamento Competitivo (WTA)
Durante o treinamento, as amostras são apresentadas aos neurônios uma a uma:
- **Competição**: Para cada amostra, calcula-se a distância euclidiana entre ela e todos os neurônios.
- **Vincit Omnia (O Vencedor leva tudo)**: Apenas o neurônio mais próximo da amostra (o vencedor) tem seus pesos atualizados, movendo-se ligeiramente na direção daquela amostra.
- **Desempate Aleatório**: Como todos começam no mesmo ponto, o primeiro desempate é aleatório. Uma vez que um neurônio se move para "capturar" uma região, os outros tendem a ganhar as próximas disputas em outras direções.

### 5. Análise de Domínio por Neurônio (Rótulos de Classe)
Como o WTA é um algoritmo **não supervisionado**, os neurônios não sabem os nomes das espécies. A análise para dizer que um neurônio "ganhou uma classe" é feita de forma estatística após o treino:

1.  **Fase de Coleta**: O sistema passa todas as amostras de treinamento pela rede já treinada.
2.  **Mapeamento de Vencedores**: Registra-se para cada amostra qual neurônio ganhou. Por exemplo: *"Amostra de Setosa 1 foi ganha pelo Neurônio 3"*, *"Amostra de Setosa 2 foi ganha pelo Neurônio 3"*.
3.  **Votação Majoritária**: O sistema olha para o Neurônio 3 e pergunta: *"Quais classes caíram aqui?"*. Se caíram 30 Setosas e 2 Versicolors, pela regra da maioria, o **Neurônio 3 é rotulado como 'Especialista em Setosa'**.

---

## 📊 Por que comparar com K-Means?

O script realiza uma comparação final entre o **WTA (Rede Neural)** e o **K-Means (Algoritmo Estatístico Clássico)**. Essa comparação é fundamental por três motivos:

1.  **Validar centros de massa**: O K-Means é conhecido por encontrar matematicamente os centros de gravidade ideais de grupos de pontos. Se a posição final dos neurônios do WTA for próxima aos centroides do K-Means, isso prova que a rede neural "aprendeu" corretamente a estrutura estatística do dataset.
2.  **Aprendizado Online vs. Offline**:
    *   O **WTA** aprende de forma "viva" (online), reagindo a cada amostra individualmente e movendo-se no gráfico.
    *   O **K-Means** geralmente processa o lote todo repetidamente (estratégico/offline).
    Compará-los permite ver se a abordagem biológica do WTA atinge resultados tão precisos quanto o cálculo estatístico bruto.
3.  **Baseline de Eficiência**: O K-Means serve como o *padrão-ouro* (baseline) para agrupamento. Se o WTA atingir uma acurácia próxima ou superior à dele, demonstra que a rede neural está bem calibrada (Taxa de Aprendizado e Épocas corretas).

---

## 🎯 Análise dos Resultados

### Como o resultado é buscado?
O script busca a convergência onde cada um dos 3 neurônios se torna um "protótipo" ou centro de gravidade de uma região do gráfico. Como o dataset Iris possui nuvens de pontos razoavelmente separadas, a rede naturalmente divide os neurônios para que cada um "viva" no centro de uma dessas nuvens.

### Explicando o Resultado Final (Output)
Ao final da execução, o script imprime uma tabela de teste. Veja como interpretar:

1.  **Mapeamento de Neurônios (Visto no Passo 1 do Output)**: O sistema identifica a classe dominante de cada neurônio. Isso é o que permite traduzir o número do neurônio (ex: N1) para um nome legível (ex: Iris-setosa).
2.  **Validação de Teste (Passo 2 do Output)**: O script passa o conjunto de teste (amostras inéditas). 
    *   Se uma planta real de *Iris-virginica* é atraída pelo **Neurônio 1** (que foi anteriormente rotulado como especialista em *Virginica*), o sistema marca como **✅ ACERTOU**.
    *   Se a planta é atraída por um neurônio que se especializou em outra espécie, ele marca como **❌ ERROU**.
3.  **Acurácia**: Indica a porcentagem do conjunto de teste que foi corretamente atraída pelo seu "neurônio especialista" correspondente. 
    > [!IMPORTANT]
    > Uma acurácia alta (geralmente acima de 80%) prova que o WTA conseguiu descobrir sozinho as fronteiras entre as classes, apenas olhando para as distâncias e sem nunca ver os rótulos de espécie durante o ajuste de pesos.

---

## 🚀 Como Executar

Certifique-se de ter as bibliotecas `numpy` e `matplotlib` instaladas e o arquivo `Iris.csv` no mesmo diretório.

```bash
python main.py
```

Você verá a animação dos neurônios se movendo e, ao fechar a janela, o relatório de acerto aparecerá no seu terminal.
