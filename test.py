import csv
import json
import math
import random

from main import DataProcessor, Winner_take_all, SimpleKMeans


# ──────────────────────────────────────────────────────────────────────────────
# Subclass de Winner_take_all adaptada para 4 features (sem matplotlib)
# Sobrescreve apenas o método test() para saída puramente em logs.
# ──────────────────────────────────────────────────────────────────────────────
class WTA_4D(Winner_take_all):
    """WTA configurado para trabalhar com N features sem plotagem."""

    def test(self, test_data, train_data=None):
        """
        Classifica os dados de teste e exibe os resultados em logs.
        Retorna a acurácia (%) se train_data for fornecido.
        """
        print("\n" + "=" * 65)
        print("        AVALIAÇÃO DO MODELO WTA — 4 FEATURES (IRIS)        ")
        print("=" * 65)

        neuron_labels = {}

        # PASSO 1: identificar a classe dominante de cada neurônio via treino
        if train_data:
            print("\n>> PASSO 1: Mapeamento Neurônio → Classe (dados de treino) <<")
            neuron_classes = {i: [] for i in range(self.num_neurons)}

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
                    total_n = len(neuron_classes[i])
                    dominant_count = neuron_classes[i].count(most_common)
                    pureza = (dominant_count / total_n) * 100
                    print(
                        f"   Neurônio {i+1}: especializado em → {most_common}\n"
                        f"             ({dominant_count}/{total_n} amostras, pureza {pureza:.1f}%)"
                    )
                else:
                    neuron_labels[i] = "Desconhecido"
                    print(f"   Neurônio {i+1}: inativo (nenhuma amostra associada)")

        # PASSO 2: classificar dados de teste
        print("\n>> PASSO 2: Classificação do conjunto de Teste <<")
        acertos = 0
        total = len(test_data)

        for idx, p in enumerate(test_data):
            vetor = [float(p[f]) for f in self.features]
            vencedor = self.predict(vetor)
            real = p["species"]

            if train_data:
                predicao = neuron_labels[vencedor]
                correto = predicao == real
                if correto:
                    acertos += 1
                status = "✅" if correto else "❌"
                print(
                    f"   [{status}] Teste #{idx+1:02d} | Real: {real:<18} | "
                    f"N{vencedor+1} → {predicao}"
                )
            else:
                print(
                    f"   Teste #{idx+1:02d} | Real: {real:<18} | → Neurônio {vencedor+1}"
                )

        # PASSO 3: acurácia
        acuracia = None
        if train_data:
            acuracia = (acertos / total) * 100
            print("\n" + "-" * 65)
            print(f"  🏆 Acurácia: {acertos}/{total} → {acuracia:.2f}%")
            print("-" * 65)

        # PASSO 4: posições finais dos neurônios em todas as 4 dimensões
        print("\n>> PASSO 3: Posições Finais dos Neurônios (pesos) <<")
        for i, w in enumerate(self.weights):
            rotulo = neuron_labels.get(i, f"N{i+1}")
            coords = ", ".join(
                f"{self.features[j]}={w[j]:.4f}" for j in range(len(self.features))
            )
            print(f"   Neurônio {i+1} [{rotulo}]: [{coords}]")

        print("=" * 65 + "\n")
        return acuracia


# ──────────────────────────────────────────────────────────────────────────────
# Comparação WTA vs K-Means (distância euclideana entre centróides/neurônios)
# ──────────────────────────────────────────────────────────────────────────────
def euclidean(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def comparar_wta_kmeans(wta, kmeans, features):
    """
    Para cada neurônio do WTA, encontra o centróide do K-Means mais próximo
    e exibe a distância euclidiana entre eles no espaço de 4 features.
    """
    print("=" * 65)
    print("         COMPARAÇÃO: WTA × K-MEANS (espaço 4D)          ")
    print("=" * 65)

    for i, w_neuron in enumerate(wta.weights):
        melhor_dist = float("inf")
        melhor_j = -1
        for j, centroide in enumerate(kmeans.centroids):
            d = euclidean(w_neuron, centroide)
            if d < melhor_dist:
                melhor_dist = d
                melhor_j = j

        coords_wta = ", ".join(
            f"{features[k]}={w_neuron[k]:.4f}" for k in range(len(features))
        )
        coords_km = ", ".join(
            f"{features[k]}={kmeans.centroids[melhor_j][k]:.4f}"
            for k in range(len(features))
        )

        print(f"\n  Neurônio WTA {i+1}:")
        print(f"    Posição : [{coords_wta}]")
        print(f"    K-Means mais próximo (centróide {melhor_j+1}): [{coords_km}]")
        print(f"    Distância euclidiana 4D : {melhor_dist:.4f}")

    print("\n" + "=" * 65 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    # 1. Leitura do dataset Iris
    row_data = []
    with open("Iris.csv", "r") as file:
        reader = csv.reader(file)
        for row in reader:
            row_data.append(json.loads(json.dumps(row)))

    # 2. Pré-processamento via DataProcessor
    processor = DataProcessor(row_data, train_ratio=0.75)
    processor.call_functions()

    # 3. 4 features
    features_4 = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    print(f"\n>> Features utilizadas: {features_4} <<")

    # 4. Treinar WTA_4D (subclasse sem plotagem)
    wta = WTA_4D(
        processor.training_data,
        features=features_4,
        num_neurons=3,
        epochs=50,
        learning_rate=0.08,
    )
    print(f">> Treinando WTA_4D por {wta.epochs} épocas...\n")
    wta.train()

    # Plota métricas de convergência (QE, VR, DR) para o modelo 4D
    wta.plot_metricas(
        titulo_extra="4D (sepal_length, sepal_width, petal_length, petal_width)"
    )

    # Plota curvas de aprendizado
    wta.plot_curva_aprendizado(titulo_extra="4D (Learning Rate e QE)")

    # 5. Avaliar modelo e exibir logs
    acuracia = wta.test(
        test_data=processor.test_data,
        train_data=processor.training_data,
    )

    # Erro quadrático individual de cada amostra de treino (4D)
    wta.plot_erro_quadratico(
        processor.training_data,
        titulo_extra="4D (sepal_length, sepal_width, petal_length, petal_width)",
    )

    # 6. K-Means para comparação — treina sobre os vetores 4D de treino
    train_vectors_4d = [
        [float(p[f]) for f in features_4] for p in processor.training_data
    ]
    kmeans = SimpleKMeans(k=3, epochs=50)
    kmeans.fit(train_vectors_4d)

    # 7. Comparação WTA × K-Means
    comparar_wta_kmeans(wta, kmeans, features_4)


if __name__ == "__main__":
    main()
