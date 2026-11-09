import numpy as np

from .losses import MeanSquaredError
from .optimizers import SGD


# Classe que liga as camadas e executa o treino da rede neuronal.
class NeuralNetwork:

    def __init__(
        self,
        epochs=100,
        batch_size=16,
        optimizer=SGD,
        learning_rate=0.01,
        verbose=False,
        loss=MeanSquaredError,
        metric=None,
        **kwargs,
    ):
        # Guarda o número de épocas de treino.
        self.epochs = epochs

        # Guarda o número de amostras de cada mini-batch.
        self.batch_size = batch_size

        # Guarda a classe do otimizador.
        self.optimizer = optimizer

        # Guarda a taxa de aprendizagem.
        self.learning_rate = learning_rate

        # Define se os resultados são apresentados em cada época.
        self.verbose = verbose

        # Cria uma instância da função de perda.
        self.loss = loss()

        # Guarda a função usada para avaliar as previsões.
        self.metric = metric

        # Guarda os argumentos adicionais destinados ao otimizador.
        self.kwargs = kwargs

        # Inicializa a lista de camadas.
        self.layers = []

        # Inicializa o histórico dos resultados por época.
        self.history = {
            "loss": [],
            "metric": [],
        }

    def add(self, layer):
        # Usa a forma de saída da camada anterior
        # como forma de entrada da nova camada.
        if self.layers:
            layer.set_input_shape(self.layers[-1].output_shape())

        # A primeira camada tem de receber a forma de entrada.
        elif layer.input_shape() is None:
            raise ValueError(
                "The first layer must have an input shape."
            )

        # Inicializa as camadas que disponibilizam este método.
        if callable(getattr(layer, "initialize", None)):
            # Cria uma nova instância do otimizador para esta camada.
            optimizer = self.optimizer(
                learning_rate=self.learning_rate,
                **self.kwargs,
            )

            layer.initialize(optimizer)

        # Adiciona a camada à rede.
        self.layers.append(layer)

    def fit(self, dataset):
        # Obtém os dados e os rótulos do Dataset.
        X = dataset.X
        y = dataset.y

        # Obtém o número de amostras de treino.
        n_samples = X.shape[0]

        # Reinicia o histórico para esta chamada de treino.
        self.history = {
            "loss": [],
            "metric": [],
        }

        # Executa o número de épocas definido.
        for epoch in range(self.epochs):
            # Baralha os índices mantendo a correspondência entre X e y.
            indices = np.random.permutation(n_samples)

            # Percorre todos os mini-batches.
            # O último pode ter menos amostras do que batch_size.
            for start in range(0, n_samples, self.batch_size):
                batch_indices = indices[start:start + self.batch_size]

                # Seleciona os dados e os rótulos do mini-batch.
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]

                # Executa a propagação para a frente em modo de treino.
                output = X_batch

                for layer in self.layers:
                    output = layer.forward_propagation(
                        output,
                        training=True,
                    )

                # Alinha os rótulos com a forma das previsões.
                # Exemplo: transforma um vetor (n,) numa coluna (n, 1).
                y_batch = y_batch.reshape(output.shape)

                # Calcula o gradiente inicial da função de perda.
                error = self.loss.derivative(y_batch, output)

                # Propaga o gradiente da última camada para a primeira.
                # As camadas treináveis atualizam os seus parâmetros.
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error)

            # Obtém as previsões com os parâmetros do fim da época.
            predictions = self.predict(dataset)

            # Alinha os rótulos com a forma das previsões.
            y_epoch = y.reshape(predictions.shape)

            # Calcula e guarda a perda da época.
            epoch_loss = self.loss.loss(y_epoch, predictions)
            self.history["loss"].append(epoch_loss)

            # Calcula e guarda a métrica, caso tenha sido fornecida.
            if self.metric is not None:
                epoch_metric = self.metric(y_epoch, predictions)
                self.history["metric"].append(epoch_metric)

            # Apresenta os resultados se verbose estiver ativo.
            if self.verbose:
                message = (
                    f"Epoch {epoch + 1}/{self.epochs}"
                    f" - loss: {epoch_loss:.6f}"
                )

                if self.metric is not None:
                    message += f" - metric: {epoch_metric:.6f}"

                print(message)

        # Devolve a rede após o treino.
        return self

    def predict(self, dataset):
        # Obtém os dados de entrada.
        output = dataset.X

        # Executa a propagação para a frente em modo de inferência.
        for layer in self.layers:
            output = layer.forward_propagation(
                output,
                training=False,
            )

        # Numa rede com uma saída, devolve um vetor de previsões.
        if output.ndim == 2 and output.shape[1] == 1:
            return output.ravel()

        # Numa rede com várias saídas, mantém a matriz de previsões.
        return output

    def score(self, dataset):
        # Verifica se foi definida uma métrica de avaliação.
        if self.metric is None:
            raise ValueError(
                "A metric must be provided to use score()."
            )

        # Obtém as previsões da rede.
        predictions = self.predict(dataset)

        # Alinha os rótulos com a forma das previsões.
        y_true = dataset.y.reshape(predictions.shape)

        # Calcula e devolve a métrica de avaliação.
        return self.metric(y_true, predictions)