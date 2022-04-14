import math
from pylinal import Vector
from ..typedefs import Function, Grad, T


def sigmoid(x: T) -> T: 
    div = 1 + math.exp(-x)
    return 1/div


def dsigmoid(x: T) -> T:
    s = sigmoid(x)
    grad = s*(1 - s)
    return grad


class __Sigmoid(Function):

    def __call__(self, v: Vector) -> Vector:
        return Vector([sigmoid(x) for x in v])

    def grad(self, v: Vector) -> Grad:
        gradient = Vector(dsigmoid(x) for x in v)
        return Grad(by_input=gradient)


Sigmoid = object.__new__(__Sigmoid)
