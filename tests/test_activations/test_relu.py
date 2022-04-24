import random
from tadgrad.activations.relu import relu, _relu, _drelu
from pylinal import Vector


def rand_vec(dim: int):
    return Vector(random.randint(-10, 10) for _ in range(dim))


def test_relu():
    for _ in range(10):
        dim: int = random.randint(1, 10)
        v: Vector = rand_vec(dim)

        assert relu(v) == Vector(_relu(x) for x in v)
        assert relu.grad(v).by_input == Vector(_drelu(x) for x in v)
