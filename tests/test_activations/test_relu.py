import random
from tadgrad.activations.relu import ReLU, relu, drelu
from pylinal import Vector


def rand_vec(dim: int):
    return Vector(random.randint(-10, 10) for _ in range(dim))


def test_relu():
    for _ in range(10):
        dim = random.randint(1, 10)
        v = rand_vec(dim)

        assert ReLU(v) == Vector(relu(x) for x in v)
        assert ReLU.grad(v).by_input == Vector(drelu(x) for x in v)
