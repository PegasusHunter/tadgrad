from typing import Callable
from pylinal import Vector, Matrix
from ..typedefs import Grad, Function


def aff(weights: Matrix, bias: Vector) -> Callable[[Vector], Vector]:

    def closure(v: Vector) -> Vector:
        return weights @ v + bias

    return closure


def daff(weights: Matrix, bias: Vector) -> Callable[[Vector], Grad]:

    def closure(v: Vector) -> Grad:
        grad = Grad(by_input=weights, by_params=v)
        return grad

    return closure


class Aff(Function):
    weights: Matrix
    bias: Vector

    def __new__(cls, weights: Matrix, bias: Vector) -> 'Aff':
        obj = object.__new__(cls)
        obj.weights = weights
        obj.bias = bias
        return obj

    def __call__(self, v: Vector) -> Vector:
        return aff(self.weights, self.bias)(v)

    def grad(self, v: Vector) -> Grad:
        return daff(self.weights, self.bias)(v)

