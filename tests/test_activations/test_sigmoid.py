import random
import math

from tadgrad.activations.sigmoid import sigmoid, _sigmoid
from pylinal import Vector


def rand_vec(dim: int):
    return Vector(random.randint(-10, 10) for _ in range(dim))


def test_sigmoid(tries: int = 5):
    
    for _ in range(tries):
        dim: int = random.randint(1, 10)
        v: Vector = rand_vec(dim)

        assert sigmoid(v) == Vector(_sigmoid(x) for x in v)
        assert sigmoid.grad(v).by_input == Vector(_sigmoid(x)*(1 - _sigmoid(x)) for x in v)
