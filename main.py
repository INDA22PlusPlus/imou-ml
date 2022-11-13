from ML.layer import InputLayer, Layer, OutputLayer
from ML.model import Model

if __name__ == '__main__':
    a = InputLayer(10)
    b = Layer(20)
    c = OutputLayer(3)

    model = Model([a,b,c])
    model.h5dump('test.h5')

    model2 = Model(None)
    model2.h5load('test.h5')

    print(model2.layers)