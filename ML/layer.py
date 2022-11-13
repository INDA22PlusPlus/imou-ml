import numpy as np

class Layer:
    _biases: np.array = []
    _weights: np.array = []

    _fn_type: str

    def __init__(self, neurons: int, activation: str = 'relu') -> None:
        self._biases = np.ones(neurons)
        self._weights = np.ones(neurons)

        self._fn_type = activation

        if activation == 'relu':
            self.fn = lambda x: np.amax(0, x)
        elif activation == 'sigmoid':
            self.fn = lambda x: 1/(1+np.exp(-x))
        else:
            raise ValueError((f"`activation` must be 'relu' or 'sigmoid'"
                                f", but is {activation}"))

    def load(self, weights: np.array, biases: np.array) -> None:
        self._biases = biases
        self._weights = weights

    def output(self, input: np.array) -> np.array:
        return self.fn(self._weights.matmul(input) + self._biases)


    @property
    def weights(self):  return self._weights

    @property
    def biases(self):   return self._biases

    @property
    def fn_type(self):  return self._fn_type

class InputLayer(Layer):
    pass

class OutputLayer(Layer):
    def output(self, input: np.array) -> int:
        out_raw = super().output(input)
        # Thanks for softmax impl https://stackoverflow.com/a/38250088
        e_x = np.exp(out_raw - np.max(out_raw))
        return np.amax(e_x / e_x.sum(axis=0))