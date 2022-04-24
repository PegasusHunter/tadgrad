import matplotlib.pyplot as plt
import random

from tadgrad import Network, LinLayer, Layer
from tadgrad.optim import GD
from tadgrad.losses import MSE
from tadgrad.utils import linspace


def main():
    
    def noise() -> float:
        return random.uniform(-0.05, 0.05)
    
    interval = (-1, 1)

    xs: list = linspace(*interval, 50)
    ys = [x**3 + noise() for x in xs]
    plt.plot(xs, ys, 'ro')

    # train data
    X = [[x, x**2 , x**3] for x in xs]
    labels = [[y] for y in ys]

    regression = Network(loss=MSE)
    regression.append(LinLayer(3, 1))

    optimizer = GD(regression.layers, lr=0.5)
    regression.optim = optimizer 
    
    regression.fit(X, labels, epochs=4)
   
    # test data
    X_test = [[x, x**2, x**3] for x in linspace(*interval, 30)]
    predictions = (regression.predict(x) for x in X_test)

    xs = [x[0] for x in X_test]
    ys = [v[0] for v in predictions]
    plt.plot(xs, ys, 'bo', label='predictions')    

    
    plt.legend(['train: x^3 + noise', 'predictions'])
    plt.show()


if __name__ == '__main__':
    main()

