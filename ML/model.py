import numpy as np
import h5py
import json

from .mnist import MNISTParser
from .layer import InputLayer, Layer, OutputLayer


@staticmethod
def cost(label, prediction):
    sum = np.sum(label*np.log(prediction))
    return -sum/label.shape[1]

class Model:
    _layers: list = []

    def __init__(self, layers: list, seed: int = 0) -> None:
        # Empty model, reserved for `h5load`
        if not layers:
            return
        if type(layers[0]) != InputLayer:
            raise ValueError("First layer is not an input layer")
        if type(layers[-1]) != OutputLayer:
            raise ValueError("Last layer is not an output layer")

        np.random.seed(seed)
        self.vW = list()
        self.vB = list()
        # Set right weight matrix dimension scale
        for i in range(1, len(layers)):
            current = layers[i-1]
            next = layers[i]

            # Matrix shape with current layer's neurons num as width
            # and next layer's neuron num as height
            weights = np.random.randn(next.neurons, current.neurons) \
                        * np.sqrt(1. / next.neurons)
            biases = np.random.randn(next.neurons, 1) \
                        * np.sqrt(1. / next.neurons)

            self.vW.append(np.zeros((next.neurons, current.neurons)))
            self.vB.append(np.zeros((next.neurons, 1)))
            layers[i].load(weights, biases)
                                    

        # To the output layer, just add a weight and bias matrix resulting
        # to the same number of neurons as given
        # layers[-1].load(np.random.random((layers[-1].neurons, layers[-1].neurons)),
        #                 np.random.random((layers[-1].neurons, 1)))

        self._layers = layers[1:]

    # Basically feed forward network and returns the output of the last layer
    def predict(self, input: np.array) -> int:
        out = input
        # Go through every layer and pass the output to
        # the next layer...
        for layer in self._layers:
            out = layer.output(out)

        return out


    def _backprop(self, input: np.array, label: np.array, learning_rate: float,
                    mu: float) -> None:
        # Batch size
        m = label.shape[1]

        # Forward propagation, sets the activated and non activated
        # values on each layer
        self.predict(input)
        # Delta values for each layer
        deltas = [None] * len(self._layers)
        # Gradients for the weights for each layer
        gW = [None] * len(self._layers)
        # Gradients for the biases for each layer
        gB = [None] * len(self._layers)

        for i in reversed(range(0, len(self._layers))):
            if i == len(self._layers)-1:
                deltas[i] = self._layers[i].A-label
            else:
                deltas[i] = self._layers[i+1].weights.T @ deltas[i+1] \
                            * (self._layers[i+1].dfn(self._layers[i].Z))

        # Update the weights and biases after all deltas were calculated
        for i in reversed(range(0, len(self._layers))):
            A_prev = input if i == 0 else self._layers[i-1].A
            gW[i] = (1/m) * (deltas[i] @ A_prev.T)
            gB[i] = (1/m) * np.sum(deltas[i], axis=1, keepdims=True)

            # Using momementum and the previous weight and bias
            # update iteration to get the current value
            self.vW[i] = mu * self.vW[i] - learning_rate*gW[i]
            self.vB[i] = mu * self.vB[i] - learning_rate*gB[i]

            # Updated weights and biases w.r.t momentum
            uW = self._layers[i].weights + self.vW[i]
            uB = self._layers[i].biases + self.vB[i]
            # Main update funciton
            self._layers[i].load(uW, uB)

    def train(self, learning_rate: float, mu: float, epochs: int, parser: MNISTParser,
                batch: int = 128, verbose: bool = False) -> None:


        Y = None
        for epoch in range(0, epochs):
            # Through every image in the mnist dataset
            for i in range(0, int(60000/batch)):
                # Shuffling the indexes for stochastic gradient descent
                rand_inds = np.random.permutation(np.arange(batch))
                X = np.empty((28*28, batch))
                lls = []
                for k,j in enumerate(rand_inds):
                    X[:,k] = parser.get_train_image(i*batch+j)
                    lls.append(parser.get_train_label(i*batch+j))

                # "One-hot" format
                Y = np.eye(10)[lls]
                Y = Y.T


                # The gradients of the loss functions w.r.t weights and biases
                self._backprop(X, Y, learning_rate, mu)
            

            if verbose:
                print(f"Epoch {epoch} is completed | Cost: {cost(Y,self._layers[-1].A)}")

    # Calculate the accuracy of the model by using
    # the test data
    def accuracy(self, parser: MNISTParser) -> float:
        # Go through every test image and compare the
        # prediction to the label
        occurate_predictions: int = 0
        for i in range(0, 10000):
            image = parser.get_test_image(i).reshape((-1,1))
            
            label = parser.get_test_label(i)

            # Increment if the prediction is the same as the giv
            occurate_predictions += np.argmax(self.predict(image)) == label

        return float(occurate_predictions) / 10000.0

    @property
    def layers(self):   return self._layers

    # Dumps the whole model into a file
    def dump(self, filename_json: str, filename_h5: str):
        if len(self._layers) < 2:
            raise ValueError("The model has less than 2 layers")

        model_info = dict()

        hf = h5py.File(filename_h5, 'w')
        hf.attrs['layer_num'] = len(self._layers)

        for i,layer in enumerate(self._layers):
            model_info[str(i)] = dict()
            model_info[str(i)]['fn_type'] = layer.fn_type
            model_info[str(i)]['neurons'] = layer.neurons

            w = hf.create_dataset(f"layer{i}_w", data=layer.weights)
            b = hf.create_dataset(f"layer{i}_b", data=layer.biases)
        
        hf.close()

        with open(filename_json, 'w') as jf:
            json.dump(model_info, jf)


    # Loads the whole model from a given h5 file dump
    def load(self, filename_json: str, filename_h5: str):
        hf = h5py.File(filename_h5, 'r')

        model_info = None
        layers = list()

        with open(filename_json, 'r') as jf:
            model_info = json.load(jf)

        layer_n = hf.attrs['layer_num']
        for i in range(0, layer_n):
            # Load the weights, biases
            weights = hf.get(f"layer{i}_w")[:]
            biases = hf.get(f"layer{i}_b")[:]

            layer_type = Layer

            if i == 0: layer = InputLayer
            if i+1 == layer_n: layer = OutputLayer

            layer = layer_type(model_info[str(i)]['neurons'], model_info[str(i)]['fn_type'])
            layer.load(weights, biases)

            layers.append(layer)
        
        hf.close()
        self._layers = layers