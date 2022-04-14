from typing import Callable
from ..typedefs import Function, Grad, T
from pylinal import Vector, Matrix


def mae(label: T) -> Callable[[Vector], T]:
    
    def closure(v: Vector) -> T:
        return sum(abs(x - label) for x in v)

    return closure


def sign(x: T, label: T) -> int:
    if x > label:
        return 1
    elif x < label:
        return -1
    return 0


def dmae(label: T):

    def closure(v: Vector) -> Grad:
        grad = Vector(sign(x, label) for x in v)
        return Grad(by_input=grad)

    return closure


class MAE(Function):
    label: T

    def __new__(cls, label: T) -> 'MAE':
        obj = object.__new__(cls)
        obj.label = label
        return obj

    def __call__(self, v: Vector) -> T:
        return mae(self.label)(v)

    def grad(self, v: Vector) -> Grad:
        return dmae(self.label)(v)
    
    def __repr__(self) -> str:
        return f'MAE(label={self.label})'

