import csv
import json
import random
import matplotlib.pyplot as plt
import math
import numpy as np


row_data = []

with open("Iris.csv", "r") as file:
    reader = csv.reader(file)
    for row in reader:
        row_json = json.loads(json.dumps(row))
        row_data.append(row_json)


def calculate_euclidean_distance(vector_a, vector_b):
    """
    Calcula a distância euclidiana entre dois vetores de tamanho variável (no seu caso, tamanho 2).
    """
    sum_sq = 0.0
    for a, b in zip(vector_a, vector_b):
        sum_sq += (a - b) ** 2
    return math.sqrt(sum_sq)


def get_winning_neuron(input_vector, neurons_weights):
    """
    Aplica um Winner-Takes-All (WTA) determinando o neurônio vencedor.
    input_vector: ex: [sepal_length, petal_width]
    neurons_weights: lista de vetores de pesos dos neurônios (ex: [[w1_x, w1_y], [w2_x, w2_y], ...])
    """
    min_distance = float("inf")
    winning_index = -1

    for index, weights in enumerate(neurons_weights):
        distance = calculate_euclidean_distance(input_vector, weights)
        if distance < min_distance:
            min_distance = distance
            winning_index = index

    return winning_index


def update_winning_neuron(input_vector, winning_weights, learning_rate=0.1):
    """
    Atualiza APENAS os pesos do neurônio vencedor (WTA puro, sem vizinhança).
    Fórmula: W(t+1) = W(t) + alpha * (X - W(t))
    """
    for i in range(len(winning_weights)):
        winning_weights[i] = winning_weights[i] + learning_rate * (
            input_vector[i] - winning_weights[i]
        )
    return winning_weights


class DataProcessor:
    def __init__(self, data, train_ratio=0.7):
        self.data = data
        self.train_ratio = train_ratio
        self.processed_data = []
        self.group_especies = []
        self.training_data = []
        self.test_data = []

    def convert_to_data(self, row):
        json = {
            "index": row[0],
            "sepal_length": row[1],
            "sepal_width": row[2],
            "petal_length": row[3],
            "petal_width": row[4],
            "species": row[5],
        }
        return json

    def save_data_partition(self, data_save):
        data_save = data_save[1:]
        for x in data_save:
            processed_row = self.convert_to_data(x)
            self.processed_data.append(processed_row)

    def get_knowledge_class(self):
        setosa = []
        versicolor = []
        virginica = []
        for x in self.processed_data:
            if x["species"] == "Iris-setosa":
                setosa.append(x)
            elif x["species"] == "Iris-versicolor":
                versicolor.append(x)
            elif x["species"] == "Iris-virginica":
                virginica.append(x)
        self.group_especies = [setosa, versicolor, virginica]
        return self.group_especies

    def split_data(self):
        self.training_data = []
        self.test_data = []

        for group in self.group_especies:
            temp_group = group[:]
            random.shuffle(temp_group)

            # Calcula a quantidade de amostras de treino baseado na proporção
            split_idx = int(len(temp_group) * self.train_ratio)

            self.training_data.extend(temp_group[:split_idx])
            self.test_data.extend(temp_group[split_idx:])

        random.shuffle(self.training_data)
        random.shuffle(self.test_data)

    def plot_data(self, data_used):
        colors = {
            "Iris-setosa": "red",
            "Iris-versicolor": "green",
            "Iris-virginica": "blue",
        }

        for data_point in data_used:
            x = float(data_point["sepal_length"])
            y = float(data_point["petal_width"])
            species = data_point["species"]
            plt.scatter(x, y, color=colors.get(species, "black"))

        plt.xlabel("Sepal Length")
        plt.ylabel("Petal Width")
        plt.title("Sepal Length vs Petal Width")
        plt.show()

    def call_functions(self):
        self.save_data_partition(self.data)
        self.get_knowledge_class()
        self.split_data()
        
        # Informativo no terminal
        print(f">> Dataset processado: {len(self.training_data)} amostras para Treino e {len(self.test_data)} para Teste.")


class Winner_take_all:

    def __init__(
        self, data, features=None, num_neurons=3, learning_rate=0.1, epochs=20
    ):
        self.data = data
        self.features = features if features else ["sepal_length", "petal_width"]
        self.learning_rate = learning_rate
        self.num_neurons = num_neurons
        self.epochs = epochs

        centro = self._calcular_centro(self.data)

        # Todos os neurônios partem exatamente juntos do centro de todas as amostras
        self.weights = [[c for c in centro] for _ in range(self.num_neurons)]
        self.history = []

    def _calcular_centro(self, data):
        """Calcula o centro vetorial geral percorrendo as amostras."""
        num_amostras = len(data)
        num_features = len(self.features)

        # Ponto de partida na origem (0, 0, ...)
        soma_vetorial = [0.0] * num_features

        # Itera pelo dataset calculando o somatório das amostras como vetores
        for ponto in data:
            for i, f in enumerate(self.features):
                soma_vetorial[i] += float(ponto[f])

        # O centro de todas as amostras é a divisão da soma vetorial pelo total de amostras
        centro_amostras = [soma / num_amostras for soma in soma_vetorial]
        return centro_amostras

    def _euclidean_distance(self, x, w):
        soma = 0
        for j in range(len(x)):
            diferenca = x[j] - w[j]
            soma += diferenca**2
        return math.sqrt(soma)

    def _winner(self, x):
        """Retorna o índice do neurônio vencedor.
        Em caso de empate (distâncias iguais), sorteia aleatoriamente entre os candidatos.
        Isso evita que apenas o neurônio 0 vença quando todos partem do mesmo ponto.
        """
        distancias = [self._euclidean_distance(x, w) for w in self.weights]
        min_distance = min(distancias)

        # Coleta todos os índices com a mesma distância mínima (com tolerância numérica)
        candidatos = [
            i for i, d in enumerate(distancias) if abs(d - min_distance) < 1e-9
        ]
        return random.choice(candidatos)

    def _update_weights(self, x, w):
        new_w = []
        for j in range(len(w)):
            updated = w[j] + self.learning_rate * (x[j] - w[j])
            new_w.append(updated)
        return new_w

    def train(self):
        for epoch in range(self.epochs):
            dados_epoca = self.data[:]
            random.shuffle(dados_epoca)
            for ponto in dados_epoca:
                vetor_entrada = [float(ponto[f]) for f in self.features]
                vencedor_idx = self._winner(vetor_entrada)
                self.weights[vencedor_idx] = self._update_weights(
                    vetor_entrada, self.weights[vencedor_idx]
                )
            # salva posição dos neurônios após cada época
            self.history.append([w[:] for w in self.weights])

    def train_live(self, training_data, pause=0.1):
        """
        Treina o WTA e exibe a posição dos neurônios ao vivo a cada época.
        pause: tempo (em segundos) de pausa entre épocas para visualização.
        """
        colors_data = {
            "Iris-setosa": "red",
            "Iris-versicolor": "green",
            "Iris-virginica": "blue",
        }
        neuron_colors = ["black", "orange", "purple", "cyan"]

        plt.ion()
        fig, ax = plt.subplots()

        def draw_state(title_suffix, prev_weights=None):
            ax.clear()

            # Plota o dataset
            for p in training_data:
                ax.scatter(
                    float(p[self.features[0]]),
                    float(p[self.features[1]]) if len(self.features) > 1 else 0,
                    color=colors_data.get(p["species"], "black"),
                    alpha=0.3,
                    s=20,
                )

            # Plota a posição inicial (se houver prev_weights) com linhas indicando movimento
            if prev_weights:
                for i, w_prev in enumerate(prev_weights):
                    color = neuron_colors[i % len(neuron_colors)]
                    # Posição inicial (círculo vazado)
                    ax.scatter(
                        w_prev[0],
                        w_prev[1] if len(w_prev) > 1 else 0,
                        marker="o",
                        facecolors="none",
                        edgecolors=color,
                        s=150,
                        linewidth=2,
                        zorder=4,
                        label=f"Início N{i+1}" if i < len(neuron_colors) else "",
                    )

                    # Linha conectando início ao fim
                    w_atual = self.weights[i]
                    ax.plot(
                        [w_prev[0], w_atual[0]],
                        [
                            w_prev[1] if len(w_prev) > 1 else 0,
                            w_atual[1] if len(w_atual) > 1 else 0,
                        ],
                        color=color,
                        linestyle="--",
                        alpha=0.5,
                        zorder=3,
                    )

            # Plota posição atual (final ou de partida se não tiver prev_weights)
            for i, w in enumerate(self.weights):
                color = neuron_colors[i % len(neuron_colors)]
                ax.scatter(
                    w[0],
                    w[1] if len(w) > 1 else 0,
                    marker="X",
                    s=200,
                    color=color,
                    label=f"N{i+1}" if prev_weights is None else f"Fim N{i+1}",
                    zorder=5,
                )

            ax.set_title(f"WTA — {title_suffix}")
            ax.set_xlabel(self.features[0])
            if len(self.features) > 1:
                ax.set_ylabel(self.features[1])

            # Ajuste de legendas sem duplicidades
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc="upper left")
            plt.pause(pause)

        # Histórico original do tempo 0
        self.history.append([w[:] for w in self.weights])

        for epoch in range(self.epochs):
            # 1) Mostra gráfico ANTES do treinamento desta época
            pesos_iniciais_epoca = [w[:] for w in self.weights]
            draw_state(f"Início da Época {epoch + 1}/{self.epochs}")

            # 2) Treinamento da época
            dados_epoca = self.data[:]
            random.shuffle(dados_epoca)
            for ponto in dados_epoca:
                vetor_entrada = [float(ponto[f]) for f in self.features]
                vencedor_idx = self._winner(vetor_entrada)
                self.weights[vencedor_idx] = self._update_weights(
                    vetor_entrada, self.weights[vencedor_idx]
                )
            self.history.append([w[:] for w in self.weights])

            # 3) Mostra gráfico DEPOIS do treinamento da época (mostrando rastro de movimento)
            draw_state(
                f"Fim da Época {epoch + 1}/{self.epochs}",
                prev_weights=pesos_iniciais_epoca,
            )

        plt.ioff()
        plt.show()

    def train_live_by_sample(self, training_data, pause=0.1):
        """
        Treina o WTA exibindo o movimento a cada única interação de amostra.
        pause: tempo (em segundos) de pausa. Reduza se quiser mais rápido.
        """
        colors_data = {
            "Iris-setosa": "red",
            "Iris-versicolor": "green",
            "Iris-virginica": "blue",
        }
        neuron_colors = ["black", "orange", "purple", "cyan"]

        plt.ion()
        fig, ax = plt.subplots()

        def draw_state(title_suffix, prev_weights=None, current_sample=None):
            ax.clear()

            # Plota o dataset (amostras de fundo mais transparentes)
            for p in training_data:
                ax.scatter(
                    float(p[self.features[0]]),
                    float(p[self.features[1]]) if len(self.features) > 1 else 0,
                    color=colors_data.get(p["species"], "black"),
                    alpha=0.1,  # Reduzido pra destacar a amostra atual
                    s=20,
                )

            # Destaca a amostra sendo avaliada agora
            if current_sample:
                ax.scatter(
                    current_sample[0],
                    current_sample[1] if len(current_sample) > 1 else 0,
                    marker="*",
                    color="magenta",
                    s=300,
                    edgecolors="black",
                    zorder=6,
                    label="Amostra da Vez",
                )

            # Plota a posição inicial desta amostra se houver
            if prev_weights:
                for i, w_prev in enumerate(prev_weights):
                    color = neuron_colors[i % len(neuron_colors)]
                    ax.scatter(
                        w_prev[0],
                        w_prev[1] if len(w_prev) > 1 else 0,
                        marker="o",
                        facecolors="none",
                        edgecolors=color,
                        s=150,
                        linewidth=2,
                        zorder=4,
                        label=f"Origem N{i+1}" if i < len(neuron_colors) else "",
                    )

                    w_atual = self.weights[i]
                    ax.plot(
                        [w_prev[0], w_atual[0]],
                        [
                            w_prev[1] if len(w_prev) > 1 else 0,
                            w_atual[1] if len(w_atual) > 1 else 0,
                        ],
                        color=color,
                        linestyle="--",
                        alpha=0.5,
                        zorder=3,
                    )

            # Plota posição final
            for i, w in enumerate(self.weights):
                color = neuron_colors[i % len(neuron_colors)]
                ax.scatter(
                    w[0],
                    w[1] if len(w) > 1 else 0,
                    marker="X",
                    s=200,
                    color=color,
                    label=f"N{i+1}",
                    zorder=5,
                )

            ax.set_title(f"WTA — {title_suffix}")
            ax.set_xlabel(self.features[0])
            if len(self.features) > 1:
                ax.set_ylabel(self.features[1])

            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc="upper left")
            plt.pause(pause)

        self.history.append([w[:] for w in self.weights])
        draw_state(f"Início Global do Treino")

        for epoch in range(self.epochs):
            dados_epoca = self.data[:]
            random.shuffle(dados_epoca)

            for iter_idx, ponto in enumerate(dados_epoca):
                pesos_antes = [w[:] for w in self.weights]
                vetor_entrada = [float(ponto[f]) for f in self.features]

                vencedor_idx = self._winner(vetor_entrada)
                self.weights[vencedor_idx] = self._update_weights(
                    vetor_entrada, self.weights[vencedor_idx]
                )

                draw_state(
                    f"Época {epoch + 1} | Amostra {iter_idx + 1}/{len(dados_epoca)} | N{vencedor_idx+1} Venceu!",
                    prev_weights=pesos_antes,
                    current_sample=vetor_entrada,
                )

            self.history.append([w[:] for w in self.weights])

        plt.ioff()
        plt.show()

    def predict(self, input_vector):

        return self._winner(input_vector)

    def test(self, test_data, train_data=None):
        """
        Classifica os dados de teste e avalia a qualidade do agrupamento.
        """
        print("\n" + "=" * 60)
        print("          APLICAÇÃO DO TESTE NO MODELO WTA          ")
        print("=" * 60)

        neuron_labels = {}
        if train_data:
            print("\n>> PASSO 1: Identificar qual classe dominou cada neurônio <<")
            neuron_classes = {i: [] for i in range(self.num_neurons)}

            # Passa o dado de treino pra ver onde caem
            for p in train_data:
                vetor = [float(p[f]) for f in self.features]
                vencedor = self.predict(vetor)
                neuron_classes[vencedor].append(p["species"])

            for i in range(self.num_neurons):
                if neuron_classes[i]:
                    most_common = max(
                        set(neuron_classes[i]), key=neuron_classes[i].count
                    )
                    neuron_labels[i] = most_common
                    print(
                        f"   Neurônio {i+1} : especializou em -> {most_common} ({len(neuron_classes[i])} amostras associadas)"
                    )
                else:
                    neuron_labels[i] = "Desconhecido"
                    print(
                        f"   Neurônio {i+1} : não capturou amostras de treino (inativo)"
                    )

        print("\n>> PASSO 2: Validar o conjunto de teste <<")
        acertos = 0
        total = len(test_data)

        for idx, p in enumerate(test_data):
            vetor = [float(p[f]) for f in self.features]
            vencedor = self.predict(vetor)
            real = p["species"]

            if train_data:
                predicao = neuron_labels[vencedor]
                status = "✅ ACERTOU" if predicao == real else "❌ ERROU  "
                if predicao == real:
                    acertos += 1

                print(
                    f"   Teste #{idx+1:02d} | Categoria Real: {real:<15} | Caiu no N{vencedor+1} | Rotulo Assumido: {predicao:<15} -> {status}"
                )
            else:
                print(
                    f"   Teste #{idx+1:02d} | Categoria Real: {real:<15} | -> Caiu no Neurônio {vencedor+1}"
                )

        if train_data:
            acuracia = (acertos / total) * 100
            print("-" * 60)
            print(f"🏆 Resultado Final: Acertou {acertos} de {total} ({acuracia:.2f}%)")
        print("=" * 60 + "\n")

        # --- PLOTAGEM DOS DADOS DE TESTE ---
        plt.figure(figsize=(8, 6))
        colors_data = {
            "Iris-setosa": "red",
            "Iris-versicolor": "green",
            "Iris-virginica": "blue",
        }
        neuron_colors = ["black", "orange", "purple", "cyan"]

        # 1. Plota os pontos reais de teste
        for p in test_data:
            x = float(p[self.features[0]])
            y = float(p[self.features[1]]) if len(self.features) > 1 else 0
            plt.scatter(
                x, y, 
                color=colors_data.get(p["species"], "gray"), 
                alpha=0.6,
                s=40
            )

        # 2. Plota as posições finais dos neurônios com seus rótulos assumidos
        for i, w in enumerate(self.weights):
            color = neuron_colors[i % len(neuron_colors)]
            rotulo = neuron_labels.get(i, f"N{i+1}")
            plt.scatter(
                w[0],
                w[1] if len(w) > 1 else 0,
                marker="X",
                s=250,
                color=color,
                edgecolors="white",
                label=f"N{i+1} : {rotulo}",
                zorder=5,
            )

        plt.title(f"Amostras de Teste vs Posição Final WTA (Acurácia: {acuracia:.2f}%)" if train_data else "Amostras de Teste")
        plt.xlabel(self.features[0])
        if len(self.features) > 1:
            plt.ylabel(self.features[1])
            
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc="best")
        
        plt.show()


class SimpleKMeans:

    def __init__(self, k=3, epochs=50):

        self.k = k
        self.epochs = epochs
        self.centroids = []

    def _distance(self, a, b):

        soma = 0

        for i in range(len(a)):
            soma += (a[i] - b[i]) ** 2

        return soma**0.5

    def fit(self, data):
        self.centroids = random.sample(data, self.k)
        for epoch in range(self.epochs):
            clusters = [[] for _ in range(self.k)]
            # atribuir pontos
            for point in data:
                min_dist = float("inf")
                idx = 0
                for i, centroid in enumerate(self.centroids):
                    d = self._distance(point, centroid)
                    if d < min_dist:
                        min_dist = d
                        idx = i
                clusters[idx].append(point)
            # atualizar centroides
            new_centroids = []
            for cluster in clusters:
                if len(cluster) == 0:
                    new_centroids.append(random.choice(data))
                    continue
                mean = [0] * len(cluster[0])
                for point in cluster:
                    for j in range(len(point)):
                        mean[j] += point[j]
                for j in range(len(mean)):
                    mean[j] /= len(cluster)
                new_centroids.append(mean)
            self.centroids = new_centroids
        return clusters


def plot_wta_movement(processor, wta):
    colors = {
        "Iris-setosa": "red",
        "Iris-versicolor": "green",
        "Iris-virginica": "blue",
    }
    plt.figure()
    # plota dataset
    for p in processor.training_data:
        x = float(p["sepal_length"])
        y = float(p["petal_width"])
        plt.scatter(x, y, color=colors[p["species"]], alpha=0.4)

    # plota trajetória dos neurônios
    for neuron in range(len(wta.weights)):
        traj_x = []
        traj_y = []
        for epoch in wta.history:
            traj_x.append(epoch[neuron][0])
            traj_y.append(epoch[neuron][1])

        plt.plot(traj_x, traj_y, marker="o", linewidth=2)
    plt.title("Movimento dos neurônios WTA")
    plt.xlabel("Sepal Length")
    plt.ylabel("Petal Width")
    plt.show()


def plot_comparison(processor, wta, kmeans):
    plt.figure()
    colors = {
        "Iris-setosa": "red",
        "Iris-versicolor": "green",
        "Iris-virginica": "blue",
    }
    # dataset
    for p in processor.training_data:
        x = float(p["sepal_length"])
        y = float(p["petal_width"])
        plt.scatter(x, y, color=colors[p["species"]], alpha=0.3)
    # WTA
    for w in wta.weights:
        plt.scatter(w[0], w[1], marker="X", s=200, color="black", label="WTA")
    # KMeans
    for c in kmeans.centroids:
        plt.scatter(c[0], c[1], marker="D", s=200, color="yellow", label="KMeans")
    plt.title("WTA vs KMeans")
    plt.show()


def extract_vectors(data):
    vectors = []
    for point in data:
        x = float(point["sepal_length"])
        y = float(point["petal_width"])
        vectors.append([x, y])
    return vectors


# | ---- MAIN -- --|
# Aqui você pode definir a proporção: 0.7 = 70% treino, 0.3 = 30% teste
processor = DataProcessor(row_data, train_ratio=0.7)
processor.call_functions()

train_vectors = extract_vectors(processor.training_data)

# WTA
wta = Winner_take_all(processor.training_data, num_neurons=3, epochs=8)
# Você pode trocar para wta.train_live() se quiser ver passo-por-epoca
# wta.train()
wta.train_live(processor.training_data, pause=0.1)
# wta.train_live_by_sample(processor.training_data, pause=0.1)

# Passa o teste
wta.test(test_data=processor.test_data, train_data=processor.training_data)

# KMeans
kmeans = SimpleKMeans(k=3)
kmeans.fit(train_vectors)

plot_wta_movement(processor, wta)

plot_comparison(processor, wta, kmeans)
