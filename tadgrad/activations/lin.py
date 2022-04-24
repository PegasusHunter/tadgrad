from typing import Callable
from pylinal import Vector, Matrix
from ..typedefs import Grad, Function


def lin(weights: Matrix) -> Callable[[Vector], Vector]:

    def closure(v: Vector) -> Vector:
        wv: Vector = weights @ v  # type: ignore
        return wv

    return closure


def dlin(weights: Matrix) -> Callable[[Vector], Grad]:
    
    def closure(v: Vector) -> Grad:
        grad: Grad = Grad(by_input=weights, by_params=v)
        return grad

    return closure


class Lin(Function):
    weights: Matrix

    def __new__(cls, weights: Matrix) -> 'Lin':
        obj = object.__new__(cls)
        obj.weights = weights
        return obj

    def __call__(self, v: Vector) -> Vector:
        return lin(self.weights)(v)

    def grad(self, v: Vector) -> Grad:
        return dlin(self.weights)(v)

