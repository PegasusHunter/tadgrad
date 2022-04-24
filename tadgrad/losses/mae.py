from typing import Callable, Iterable, Union
from ..typedefs import Function, Grad, Scalar
from pylinal import Vector, Matrix

T = Scalar


def mae(label: Iterable[T]) -> Callable[[Vector], T]:
    
    def closure(v: Vector) -> T:
        return sum(abs(x - l) for x, l in zip(v, label))

    return closure


def sign(x: T, l: T) -> int:
    if x > l:
        return 1
    elif x < l:
        return -1
    return 0


def dmae(label: Iterable[T]) -> Callable[[Vector], Grad]:

    def closure(v: Vector) -> Grad:
        grad: Vector = Vector(sign(x, l) for x, l in zip(v, label))
        return Grad(by_input=grad)

    return closure


class MAE(Function):
    label: Iterable[T]
    name: str

    def __new__(cls, label: Union[T, Iterable[T]]) -> 'MAE':
        obj = object.__new__(cls)
        obj.name = f'MSE(label={label})'
        obj.label = label if hasattr(label, '__iter__') else [label]  # type: ignore
        return obj

    def __call__(self, v: Vector) -> T:
        return mae(self.label)(v)

    def grad(self, v: Vector) -> Grad:
        return dmae(self.label)(v)
    
    def __repr__(self) -> str:
        return self.name

