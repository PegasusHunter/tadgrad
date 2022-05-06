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
    ys = [x**2 + noise() for x in xs]
    plt.plot(xs, ys, 'ro')

    # train data
    X = [[x, x**4] for x in xs]
    labels = [[y] for y in ys]

    regression = Network(loss=MSE)
    regression.append(LinLayer(2, 1))
    regression.optim = GD(regression.layers, lr=0.5)
    regression.fit(X, labels, epochs=4)

    # test data
    X_test = [[x, x**4] for x in linspace(*interval, 30)]
    predictions = (regression.predict(x) for x in X_test)

    xs = [x[0] for x in X_test]
    ys = [p[0] for p in predictions]
    plt.plot(xs, ys, 'bo')    
 
    plt.legend(['train: x**2 + noise', 'predictions'])
    plt.show()


if __name__ == '__main__':
    main()

