
class WTA:
    def __init__(self, n_neurons, learning_rate=0.1, epochs=100):
        self.n_neurons = n_neurons
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None

    def _initialize_weights(self, X):
        """
        Inicializa os pesos aleatoriamente
        shape = (n_neurons, n_features)
        """
        n_features = X.shape[1]
        self.weights = np.random.rand(self.n_neurons, n_features)

    def _euclidean_distance(self, x, w):

        soma = 0

        for j in range(len(x)):
            diferenca = x[j] - w[j]
            soma += diferenca * diferenca

        return soma**0.5

    def _winner(self, x):

        menor_distancia = float("inf")
        vencedor = None

        for i in range(len(self.weights)):

            w = self.weights[i]

            distancia = self._euclidean_distance(x, w)

            if distancia < menor_distancia:
                menor_distancia = distancia
                vencedor = i

        return vencedor

    # def _winner(self, x):
    #     """
    #     Implementa a equação (3.3)
    #     Calcula a distância entre x e cada neurônio
    #     """
    #     distances = np.linalg.norm(self.weights - x, axis=1)
    #     winner = np.argmin(distances)
    #     return winner

    def _update_weights(self, winner, x):
        """
        Atualiza o neurônio vencedor
        w(t+1) = w(t) + η(x - w)
        """
        self.weights[winner] += self.learning_rate * (x - self.weights[winner])

    def fit(self, X):
        """
        Treinamento da rede WTA
        """
        self._initialize_weights(X)

        for epoch in range(self.epochs):

            for x in X:

                winner = self._winner(x)

                self._update_weights(winner, x)

    def predict(self, X):
        """
        Retorna o neurônio vencedor para cada amostra
        """
        labels = []

        for x in X:
            winner = self._winner(x)
            labels.append(winner)

        return np.array(labels)


from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X = iris.data

scaler = StandardScaler()
X = scaler.fit_transform(X)

wta = WTA(n_neurons=3, learning_rate=0.1, epochs=50)

wta.fit(X)

labels = wta.predict(X)

print(labels)


# # ==========================================
# # Rodando os dados
# # ==========================================
# dataprocessor = DataProcessor(row_data)
# dataprocessor.call_functions()

# # Inicializamos a sua rede passando apenas os dados de treino separados!
# rede_wta = Winner_take_all(
#     dataprocessor.training_data, num_neurons=3, learning_rate=0.8
# )

# # Realizamos o treinamento em si
# rede_wta.train(epochs=100)

# print("Pesos Finais da Rede Após Treinamento:")
# for i, peso in enumerate(rede_wta.weights):
#     print(f"Neurônio {i}: Sepal L: {peso[0]:.2f} | Petal W: {peso[1]:.2f}")
