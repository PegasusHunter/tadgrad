from pylinal import Matrix, Vector

from tadgrad.optim import GD
from tadgrad import Layer, LinLayer
from tadgrad.losses import MSE


def test_gd():
    lr = 1

    weights = Matrix([
        [1, 1],
        [1, 1],
        [1, 1]
    ])
    lin = LinLayer(2, 3)
    lin.weights = weights
    
    v = Vector([1, 1])
    out: Vector = lin(v)

    layers = [lin]
    optim = GD(layers=layers, lr=lr)
    
    label: list = [0, 0, 0]
    label[1] = 1

    loss_grad = MSE(label).grad(out).by_input

    optim.step([loss_grad])

    return


def main():
    test_gd()


if __name__ == '__main__':
    main()

