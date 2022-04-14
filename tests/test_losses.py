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

        for l in range(0, 10):
            assert MSE(l)(v) == sum((x-l)**2 for x in v)
            grad = MSE(l).grad(v).by_input
            assert isinstance(grad, Vector)
            assert len(grad) == len(v)
    return


def test_mae(tries: int = 10):

    for _ in range(tries):
        dim = random.randint(1, 15)
        v = rand_vec(dim)

        for l in range(0, 10):
            assert MAE(l)(v) == sum(abs(x-l) for x in v)
            grad = MAE(l).grad(v).by_input
            assert isinstance(grad, Vector)
            assert len(grad) == len(v)
    return


def main():
    test_mse()
    test_mae()
    return


if __name__ == '__main__':
    main()

