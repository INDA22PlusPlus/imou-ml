import numpy as np

from ML.mnist import MNISTParser
from ML.layer import InputLayer, Layer, OutputLayer
from ML.model import Model


if __name__ == '__main__':
    # Load all the files to the parser
    parser = MNISTParser('mnist/train-labels.idx1-ubyte',
                         'mnist/train-images.idx3-ubyte',
                         'mnist/t10k-labels.idx1-ubyte',
                         'mnist/t10k-images.idx3-ubyte')

    a = InputLayer(28*28)
    b = Layer(256, activation='sigmoid')
    c = OutputLayer(10, activation='softmax')

    model = Model([a,b,c], seed=22)
    try:
        model.train(0.1, 0.7, 500, parser, batch=128, verbose=True)
    except KeyboardInterrupt:
        model.dump('m.json', 'm.h5')
    print(model.accuracy(parser))

    model2 = Model([])
    model2.load('m.json', 'm.h5')
    print(model2.accuracy(parser))