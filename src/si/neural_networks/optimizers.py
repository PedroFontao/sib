from abc import ABC, abstractmethod

import numpy as np


# Classe base que define a estrutura dos otimizadores.
class Optimizer(ABC):

    def __init__(self, learning_rate=0.01):
        # Guarda a taxa de aprendizagem.
        self.learning_rate = learning_rate

    # As subclasses têm de implementar a atualização dos parâmetros.
    @abstractmethod
    def update(self, w, grad_loss_w):
        # w contém os parâmetros atuais: pesos ou biases.
        # grad_loss_w contém o gradiente da perda em relação a w.
        pass


# Otimizador SGD com suporte para momentum.
class SGD(Optimizer):

    def __init__(self, learning_rate=0.01, momentum=0.0):
        # Inicializa a taxa de aprendizagem através da classe base.
        super().__init__(learning_rate=learning_rate)

        # Guarda a influência dos gradientes anteriores.
        self.momentum = momentum

        # O gradiente acumulado é inicializado na primeira atualização.
        self.retained_gradient = None

    def update(self, w, grad_loss_w):
        # Inicializa o gradiente acumulado com a forma dos parâmetros.
        if self.retained_gradient is None:
            self.retained_gradient = np.zeros_like(w)

        # Combina o gradiente acumulado com o gradiente atual.
        self.retained_gradient = (
            self.momentum * self.retained_gradient
            + (1 - self.momentum) * grad_loss_w
        )

        # Devolve os parâmetros atualizados.
        return w - self.learning_rate * self.retained_gradient


# Otimizador Adam com correção do enviesamento das médias.
class Adam(Optimizer):

    def __init__(
        self,
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
    ):
        # Inicializa a taxa de aprendizagem através da classe base.
        super().__init__(learning_rate=learning_rate)

        # Guarda o fator de decaimento da média dos gradientes.
        self.beta_1 = beta_1

        # Guarda o fator de decaimento da média dos gradientes ao quadrado.
        self.beta_2 = beta_2

        # Guarda a constante usada para evitar divisões por zero.
        self.epsilon = epsilon

        # As estimativas dos momentos são inicializadas
        # na primeira atualização.
        self.m = None
        self.v = None

        # Inicializa o contador de atualizações.
        self.t = 0

    def update(self, w, grad_loss_w):
        # Inicializa as médias com a mesma forma dos parâmetros.
        if self.m is None:
            self.m = np.zeros_like(w)

        if self.v is None:
            self.v = np.zeros_like(w)

        # Incrementa o contador em cada atualização.
        self.t += 1

        # Atualiza a média móvel dos gradientes.
        self.m = (
            self.beta_1 * self.m
            + (1 - self.beta_1) * grad_loss_w
        )

        # Atualiza a média móvel dos quadrados dos gradientes.
        self.v = (
            self.beta_2 * self.v
            + (1 - self.beta_2) * grad_loss_w**2
        )

        # Corrige o enviesamento da estimativa do primeiro momento.
        m_hat = self.m / (1 - self.beta_1**self.t)

        # Corrige o enviesamento da estimativa do segundo momento.
        v_hat = self.v / (1 - self.beta_2**self.t)

        # Devolve os parâmetros atualizados.
        return (
            w
            - self.learning_rate
            * m_hat
            / (np.sqrt(v_hat) + self.epsilon)
        )