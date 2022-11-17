import numpy as np


@staticmethod
def softmax(x: np.array):
    # Thanks for softmax impl https://stackoverflow.com/a/38250088
    e_x = np.exp(x)
    return e_x / np.sum(e_x, axis=0)


class Layer:
    _neurons: int
    _biases: np.array = []
    _weights: np.array = []

    _fn_type: str

    def __init__(self, neurons: int, activation: str = 'relu') -> None:
        self._neurons = neurons   
        self._biases = np.ones(neurons)
        self._weights = np.ones(neurons)

        self._fn_type = activation

        if activation == 'relu':
            self.fn = lambda x: np.maximum(0, x)
            # Derivative function for relu
            self.dfn = lambda x: (x > 0).astype(int)
        elif activation == 'sigmoid':
            self.fn = lambda x: 1/(1+np.exp(-x))
            # Derivative function for sigmoid
            self.dfn = lambda x: self.fn(x)*(1-self.fn(x))
        elif activation == 'softmax':
            self.fn = softmax
            # Derivative function for softmax ish
            self.dfn = lambda x: self.fn(x)*(1.0 - self.fn(x))
        else:
            raise ValueError((f"`activation` must be 'relu', 'softmax', or 'sigmoid'"
                                f", but is {activation}"))

    # Shouldn't be used manually but used by the Model class
    def load(self, weights: np.array, biases: np.array) -> None:
        self._biases = biases
        self._weights = weights

    def output(self, input: np.array) -> np.array:
        # During forward propagation overwrite the non activated
        # and activated values
        self.Z = self._weights @ input + self._biases
        self.A = self.fn(self.Z)

        return self.A

    @property
    def neurons(self) -> int:
        return self._neurons

    @property
    def weights(self) -> np.array:
        return self._weights

    @property
    def biases(self) -> np.array:
        return self._biases

    @property
    def fn_type(self) -> str:
        return self._fn_type

class InputLayer(Layer):
    pass

class OutputLayer(Layer):
    pass