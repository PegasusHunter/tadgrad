from tadgrad import Network, Layer, LinLayer
from tadgrad.losses import MSE
from tadgrad.optim import GD
from tadgrad.activations import relu


def test_network():
    nn = Network(loss=MSE)
    
    nn.append(LinLayer(2, 3))
    nn.append(Layer(relu))
    nn.append(LinLayer(3, 4))
    nn.append(Layer(relu))
    print(nn, end='\n\n')

    nn.optim = GD(nn.layers, lr=1)

    X = [[1, 2]]
    y = [[1, 0, 0, 0]]

    nn.fit(X, y)
    print(nn)    
    return


def main():
    test_network()
    return


if __name__ == '__main__':
    main()

