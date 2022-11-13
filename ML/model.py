import numpy as np
import h5py
from .layer import InputLayer, Layer, OutputLayer

class Model:
    _layers: list = []

    def __init__(self, layers: list) -> None:
        # Empty model, reserved for `h5load`
        if not layers:
            return
        if type(layers[0]) != InputLayer:
            raise ValueError("First layer is not an input layer")
        if type(layers[-1]) != OutputLayer:
            raise ValueError("Last layer is not an output layer")
        self._layers = layers

    def predict(self, input: np.array) -> int:
        output = input
        # Go through every layer and pass the output to
        # the next layer...
        for layer in self._layers:
            output = layer.output(output) 
        return output

    @property
    def layers(self):   return self._layers

    # Dumps the whole model into a file
    def h5dump(self, filename: str):
        if len(self._layers) < 2:
            raise ValueError("The model has less than 2 layers")

        hf = h5py.File(filename, 'w')
        hf.attrs['layer_num'] = len(self._layers)

        for i,layer in enumerate(self._layers):
            w = hf.create_dataset(f"layer{i}_w", data=layer.weights)
            w.attrs['fn'] = layer.fn_type
            b = hf.create_dataset(f"layer{i}_b", data=layer.biases)
            b.attrs['fn'] = layer.fn_type

        hf.close()

    # Loads the whole model from a given h5 file dump
    def h5load(self, filename: str):
        hf = h5py.File(filename, 'r')

        layers = list()

        layer_n = hf.attrs['layer_num']
        for i in range(0, layer_n):
            weights = hf.get(f"layer{i}_w")
            biases = hf.get(f"layer{i}_b")

            print(biases)

            layer = Layer(biases.shape[1], weights.attrs['fn'])
            if i == 0: layer = InputLayer(weights.shape[1], weights.attrs['fn'])
            if i+1 == layer_n: layer = OutputLayer(weights.shape[1], weights.attrs['fn'])

            layers.append(layer)
        
        hf.close()