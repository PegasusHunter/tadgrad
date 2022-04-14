import random
import math

from tadgrad.activations.sigmoid import Sigmoid, sigmoid
from pylinal import Vector


def rand_vec(dim: int):
    return Vector(random.randint(-10, 10) for _ in range(dim))


def test_sigmoid(tries: int = 5):
    
    for _ in range(tries):
        dim = random.randint(1, 10)
        v = rand_vec(dim)
        print(v)

        assert Sigmoid(v) == Vector(sigmoid(x) for x in v)
        assert Sigmoid.grad(v).by_input == Vector(sigmoid(x)*(1-sigmoid(x)) for x in v)
