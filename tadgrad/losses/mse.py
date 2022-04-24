from typing import Callable, Iterable, Union
from ..typedefs import Function, Grad, Scalar
from pylinal import Vector, Matrix

T = Scalar


def mse(label: Iterable[T]) -> Callable[[Vector], T]:
    
    def closure(v: Vector) -> T:
        return sum((x - l)**2 for x, l in zip(v, label))

    return closure


def dmse(label: Iterable[T]) -> Callable[[Vector], Grad]:
    
    def closure(v: Vector) -> Grad:
        grad: Vector = Vector(2*(x - l) for x, l in zip(v, label))
        return Grad(by_input=grad)

    return closure


class MSE(Function):
    label: Iterable[T]
    name: str

    def __new__(cls, label: Union[T, Iterable[T]]) -> 'MSE':
        obj = object.__new__(cls)
        obj.name = f'MSE(label={label})'
        obj.label = label if hasattr(label, '__iter__') else [label]  # type: ignore
        return obj

    def __call__(self, v: Vector) -> T:
        return mse(self.label)(v)

    def grad(self, v: Vector) -> Grad:
        return dmse(self.label)(v)

    def __repr__(self) -> str:
        return self.name

