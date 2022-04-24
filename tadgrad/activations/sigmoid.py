import math
from pylinal import Vector
from ..typedefs import Function, Grad, Scalar

T = Scalar


def _sigmoid(x: T) -> T: 
    div = 1 + math.exp(-x)
    return 1/div


def _dsigmoid(x: T) -> T:
    s = _sigmoid(x)
    grad = s*(1 - s)
    return grad


class __Sigmoid(Function):

    def __new__(cls, params = None) -> '__Sigmoid':
        obj = object.__new__(cls)
        return obj

    def __call__(self, v: Vector) -> Vector:
        return Vector([_sigmoid(x) for x in v])

    def grad(self, v: Vector) -> Grad:
        gradient: Vector = Vector(_dsigmoid(x) for x in v)
        return Grad(by_input=gradient)


sigmoid: Function = __Sigmoid()
