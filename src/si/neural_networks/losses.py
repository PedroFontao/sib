from abc import ABC, abstractmethod

import numpy as np


# Classe base que define os métodos das funções de perda.
class LossFunction(ABC):

    # As subclasses têm de implementar o cálculo da perda.
    @abstractmethod
    def loss(self, y_true, y_pred):
        pass

    # As subclasses têm de implementar a derivada
    # em relação às previsões da rede.
    @abstractmethod
    def derivative(self, y_true, y_pred):
        pass


# Função de perda que calcula o erro quadrático médio.
class MeanSquaredError(LossFunction):

    def loss(self, y_true, y_pred):
        # Calcula a média dos quadrados das diferenças
        # entre as previsões e os valores reais.
        return np.mean((y_pred - y_true) ** 2)

    def derivative(self, y_true, y_pred):
        # Calcula a derivada do erro quadrático médio.
        # A divisão corresponde à média usada na perda.
        return 2 * (y_pred - y_true) / y_pred.size


# Função de perda para classificação binária.
class BinaryCrossEntropy(LossFunction):

    def loss(self, y_true, y_pred):
        # Limita as probabilidades para evitar logaritmos de zero.
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

        # Soma a perda associada às classes positiva e negativa.
        return -np.sum(
            y_true * np.log(y_pred)
            + (1 - y_true) * np.log(1 - y_pred)
        )

    def derivative(self, y_true, y_pred):
        # Limita as probabilidades para evitar divisões por zero.
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

        # Calcula a derivada em relação às probabilidades previstas.
        return -y_true / y_pred + (1 - y_true) / (1 - y_pred)


# Função de perda para classificação multiclasse com rótulos one-hot.
class CategoricalCrossEntropy(LossFunction):

    def loss(self, y_true, y_pred):
        # Limita as probabilidades para evitar logaritmos de zero.
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

        # Soma a perda associada às probabilidades das classes reais.
        return -np.sum(y_true * np.log(y_pred))

    def derivative(self, y_true, y_pred):
        # Limita as probabilidades para evitar divisões por zero.
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

        # Calcula a derivada em relação às probabilidades previstas.
        return -y_true / y_pred