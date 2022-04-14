from typing import (
    Union,
    Optional,
    Any,
    TypeVar,
)
from pylinal import Vector, Matrix

T = Union[int, float]
Tensor = Union[Vector, Matrix]


class Grad:
    by_input: Any
    by_params: Optional[Any]

    def __init__(self, *, by_input, by_params=None):
        self.by_input = by_input
        self.by_params = by_params


class Function:

    def __call__(self, inp: Tensor) -> Tensor:
        ...

    def grad(self, inp: Tensor) -> Grad:
        ...
