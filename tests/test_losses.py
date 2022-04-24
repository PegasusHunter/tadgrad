import random
from tadgrad.losses import (
    MSE,
    MAE,
)

from pylinal import Vector


def rand_vec(dim: int) -> Vector:
    return Vector(random.randint(-10, 10) for _ in range(dim))


def test_mse(tries: int = 10):

    for _ in range(tries):
        dim = random.randint(1, 15)
        v = rand_vec(dim)

        for l in range(0, dim):
            label = [0 for _ in range(dim)]
            label[l] = l

            assert MSE(label)(v) == sum((x-l)**2 for x, l in zip(v, label))
            grad = MSE(label).grad(v).by_input
            assert isinstance(grad, Vector)
            assert len(grad) == len(v)
    
    return


def test_mae(tries: int = 10):

    for _ in range(tries):
        dim = random.randint(1, 15)
        v = rand_vec(dim)

        for l in range(0, dim):
            label = [0 for _ in range(dim)]
            label[l] = l

            assert MAE(label)(v) == sum(abs(x-l) for (x, l) in zip(v, label))
            grad = MAE(label).grad(v).by_input
            assert isinstance(grad, Vector)
            assert len(grad) == len(v)
    
    return


def main():
    test_mse()
    test_mae()
    return


if __name__ == '__main__':
    main()

