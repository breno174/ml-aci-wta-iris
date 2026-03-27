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
    def __init__(self, data):
        self.data = data
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

            self.training_data.extend(temp_group[:40])
            self.test_data.extend(temp_group[40:])

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
        # self.plot_data(self.test_data)


class Winner_take_all:

    def __init__(self, data, num_neurons=4, learning_rate=0.3, epochs=100):
        self.data = data
        self.learning_rate = learning_rate
        self.num_neurons = num_neurons
        self.epochs = epochs
        amostras_iniciais = random.sample(self.data, self.num_neurons)
        self.weights = [
            [float(p["sepal_length"]), float(p["petal_width"])]
            for p in amostras_iniciais
        ]
        self.history = []

    def _euclidean_distance(self, x, w):
        soma = 0
        for j in range(len(x)):
            diferenca = x[j] - w[j]
            soma += diferenca**2
        return math.sqrt(soma)

    def _winner(self, x):
        min_distance = float("inf")
        winner_index = -1
        for i, w in enumerate(self.weights):
            distance = self._euclidean_distance(x, w)
            if distance < min_distance:
                min_distance = distance
                winner_index = i
        return winner_index

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
                vetor_entrada = [
                    float(ponto["sepal_length"]),
                    float(ponto["petal_width"]),
                ]
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

        for epoch in range(self.epochs):
            dados_epoca = self.data[:]
            random.shuffle(dados_epoca)
            for ponto in dados_epoca:
                vetor_entrada = [
                    float(ponto["sepal_length"]),
                    float(ponto["petal_width"]),
                ]
                vencedor_idx = self._winner(vetor_entrada)
                self.weights[vencedor_idx] = self._update_weights(
                    vetor_entrada, self.weights[vencedor_idx]
                )
            self.history.append([w[:] for w in self.weights])

            # --- Atualiza o gráfico ---
            ax.clear()

            # Plota o dataset
            for p in training_data:
                ax.scatter(
                    float(p["sepal_length"]),
                    float(p["petal_width"]),
                    color=colors_data[p["species"]],
                    alpha=0.3,
                    s=20,
                )

            # Plota posição atual dos neurônios
            for i, w in enumerate(self.weights):
                color = neuron_colors[i % len(neuron_colors)]
                ax.scatter(
                    w[0],
                    w[1],
                    marker="X",
                    s=200,
                    color=color,
                    label=f"Neurônio {i+1}",
                    zorder=5,
                )

            ax.set_title(f"WTA — Época {epoch + 1}/{self.epochs}")
            ax.set_xlabel("Sepal Length")
            ax.set_ylabel("Petal Width")
            ax.legend(loc="upper left")
            plt.pause(pause)

        plt.ioff()
        plt.show()

    def predict(self, input_vector):

        return self._winner(input_vector)


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
processor = DataProcessor(row_data)
processor.call_functions()

train_vectors = extract_vectors(processor.training_data)

# WTA
wta = Winner_take_all(processor.training_data, num_neurons=3)
# wta.train()
wta.train_live(processor.training_data, pause=0.1)

# KMeans
kmeans = SimpleKMeans(k=3)
kmeans.fit(train_vectors)

plot_wta_movement(processor, wta)

plot_comparison(processor, wta, kmeans)
