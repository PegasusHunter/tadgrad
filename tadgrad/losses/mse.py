from typing import Callable
from ..typedefs import Function, Grad, T
from pylinal import Vector, Matrix


def mse(label: T) -> Callable[[Vector], T]:
    
    def closure(v: Vector) -> T:
        return sum((x - label)**2 for x in v)

    return closure


def dmse(label: T):
    
    def closure(v: Vector) -> Grad:
        grad = Vector(2*(x - label) for x in v)
        return Grad(by_input=grad)

    return closure


class MSE(Function):
    label: T

    def __new__(cls, label: T) -> 'MSE':
        obj = object.__new__(cls)
        obj.label = label
        return obj

    def __call__(self, v: Vector) -> T:
        return mse(self.label)(v)

    def grad(self, v: Vector) -> Grad:
        return dmse(self.label)(v)

    def __repr__(self) -> str:
        return f'MSE(label={self.label})'

