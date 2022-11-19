import numpy as np

from ML.mnist import MNISTParser
from ML.layer import InputLayer, Layer, OutputLayer
from ML.model import Model

from matplotlib import pyplot as plt


if __name__ == '__main__':
    # Load all the files to the parser
    parser = MNISTParser('mnist/train-labels.idx1-ubyte',
                         'mnist/train-images.idx3-ubyte',
                         'mnist/t10k-labels.idx1-ubyte',
                         'mnist/t10k-images.idx3-ubyte')

    # a = InputLayer(10)
    # c = Layer(512, activation='sigmoid')
    # d = OutputLayer(28*28, activation='sigmoid')

    model = Model([])
    model.load('a.json', 'a.h5')

    # model.autoencoder_train(5e-3, 0.8, 20, parser, batch=128, verbose=True)
    # model.dump('a.json', 'a.h5')
    X = np.zeros(10)
    X[9] = 1.0
    X = X.reshape((-1,1))
    plt.imshow(model.predict(X).reshape(28,28), interpolation='nearest')
    plt.show()