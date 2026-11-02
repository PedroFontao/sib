from abc import ABCMeta, abstractmethod

import numpy as np


class noupdate:
    """Optimizer falso: não altera os parâmetros."""

    def update(self, w, grad):
        return w


class Layer(metaclass=ABCMeta):

    @abstractmethod
    def forward_propagation(self, input, training=False):
        raise NotImplementedError

    @abstractmethod
    def backward_propagation(self, output_error):
        raise NotImplementedError

    @abstractmethod
    def output_shape(self):
        raise NotImplementedError

    @abstractmethod
    def parameters(self):
        raise NotImplementedError

    def set_input_shape(self, input_shape):
        self._input_shape = input_shape

    def input_shape(self):
        return self._input_shape

    def layer_name(self):
        return self.__class__.__name__


class DenseLayer(Layer):

    def __init__(self, n_units, input_shape=None):
        self.n_units = n_units
        self._input_shape = input_shape

        self.input = None
        self.output = None
        self.weights = None
        self.biases = None

        self.w_opt = None
        self.b_opt = None

    def initialize(self, optimizer=None):
        # Número de características de entrada
        n_features = self._input_shape[0]

        # Pesos aleatórios em [-0.5, 0.5[
        self.weights = np.random.rand(n_features, self.n_units) - 0.5

        # Um bias por neurónio
        self.biases = np.zeros((1, self.n_units))

        if optimizer is None:
            optimizer = noupdate()

        self.w_opt = optimizer
        self.b_opt = optimizer

        return self

    def parameters(self):
        return self.weights.size + self.biases.size

    def forward_propagation(self, input, training=False):
        self.input = input

        self.output = np.dot(self.input, self.weights) + self.biases

        return self.output

    def backward_propagation(self, output_error):
        # Calcular antes de atualizar os pesos
        input_error = np.dot(output_error, self.weights.T)

        weights_error = np.dot(self.input.T, output_error)

        bias_error = np.sum(output_error, axis=0, keepdims=True)

        self.weights = self.w_opt.update(self.weights, weights_error)
        self.biases = self.b_opt.update(self.biases, bias_error)

        return input_error

    def output_shape(self):
        return (self.n_units,)


class Dropout(Layer):

    def __init__(self, probability):
        if not 0 <= probability < 1:
            raise ValueError(
                "probability must be in the interval [0, 1)."
            )

        self.probability = probability
        self._input_shape = None

        self.mask = None
        self.input = None
        self.output = None
        self.scaling_factor = 1.0

    def forward_propagation(self, input, training=False):
        self.input = input

        if training:
            # Escalar os valores que não são desativados
            self.scaling_factor = 1 / (1 - self.probability)

            # Máscara binária com a mesma forma da entrada
            self.mask = np.random.binomial(
                n=1,
                p=1 - self.probability,
                size=input.shape
            )

            self.output = (
                self.input * self.mask * self.scaling_factor
            )

        else:
            # Em inferência, não alterar a entrada
            self.mask = np.ones_like(input)
            self.scaling_factor = 1.0
            self.output = input

        return self.output

    def backward_propagation(self, output_error):
        if self.mask is None:
            raise RuntimeError(
                "Call forward_propagation before backward_propagation."
            )

        # Aplicar a mesma máscara e escala usadas no forward
        return output_error * self.mask * self.scaling_factor

    def output_shape(self):
        return self.input_shape()

    def parameters(self):
        return 0


if __name__ == "__main__":
    np.random.seed(0)

    # ==========================================
    # Testes da DenseLayer
    # ==========================================

    # 3 características de entrada e 4 neurónios
    layer = DenseLayer(4, (3,)).initialize()

    # 10 amostras, cada uma com 3 características
    X = np.random.rand(10, 3)

    out = layer.forward_propagation(X, training=True)

    assert out.shape == (10, 4)
    assert layer.parameters() == 3 * 4 + 4
    assert layer.output_shape() == (4,)

    # Testar o backward com o optimizer falso
    weights_before = layer.weights.copy()
    biases_before = layer.biases.copy()

    output_error = np.ones_like(out)

    input_error = layer.backward_propagation(output_error)

    assert input_error.shape == X.shape

    np.testing.assert_allclose(
        input_error,
        output_error @ weights_before.T
    )

    np.testing.assert_array_equal(layer.weights, weights_before)
    np.testing.assert_array_equal(layer.biases, biases_before)

    print("Todos os testes da DenseLayer passaram!")
    print("Nome da camada:", layer.layer_name())
    print("Forma da entrada por amostra:", layer.input_shape())
    print("Forma dos pesos:", layer.weights.shape)
    print("Forma dos biases:", layer.biases.shape)
    print("Forma da saída completa:", out.shape)
    print("Forma da saída por amostra:", layer.output_shape())
    print("Número de parâmetros:", layer.parameters())

    # ==========================================
    # Testes do Dropout
    # ==========================================

    dropout = Dropout(probability=0.5)
    dropout.set_input_shape((3,))

    X_dropout = np.random.uniform(
        low=0.1,
        high=1.0,
        size=(6, 3)
    )

    # Forward em treino
    out_training = dropout.forward_propagation(
        X_dropout,
        training=True
    )

    training_mask = dropout.mask.copy()

    assert out_training.shape == X_dropout.shape
    assert dropout.output_shape() == (3,)
    assert dropout.parameters() == 0

    # Com probability=0.5, os valores mantidos são multiplicados por 2
    np.testing.assert_allclose(
        out_training,
        X_dropout * training_mask * 2
    )

    # Backward do mesmo forward de treino
    dropout_error = np.ones_like(out_training)

    dropout_input_error = dropout.backward_propagation(
        dropout_error
    )

    np.testing.assert_allclose(
        dropout_input_error,
        training_mask * 2
    )

    print("\nEntrada do Dropout:")
    print(X_dropout)

    print("\nMáscara do Dropout:")
    print(training_mask)

    print("\nSaída em treino:")
    print(out_training)

    print("\nGradiente propagado para trás:")
    print(dropout_input_error)

    # Forward em inferência
    out_inference = dropout.forward_propagation(
        X_dropout,
        training=False
    )

    np.testing.assert_array_equal(out_inference, X_dropout)

    print("\nSaída em inferência:")
    print(out_inference)

    # Verificar a frequência de zeros num conjunto maior
    large_input = np.ones((1000, 10))

    large_output = dropout.forward_propagation(
        large_input,
        training=True
    )

    dropped_fraction = np.mean(large_output == 0)

    assert abs(dropped_fraction - 0.5) < 0.03

    print("\nFração de entradas desativadas:", dropped_fraction)
    print("Todos os testes do Dropout passaram!")