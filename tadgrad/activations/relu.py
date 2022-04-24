from pylinal import Vector
from ..typedefs import Function, Grad, Scalar

T = Scalar


def _relu(x: T) -> T: 
    return max(0, x)


def _drelu(x: T) -> T:
    return 1 if x > 0 else 0


class __ReLU(Function):

    def __call__(self, v: Vector) -> Vector:
        return Vector(_relu(x) for x in v)

    def grad(self, v: Vector) -> Grad:
        gradient: Vector = Vector(_drelu(x) for x in v)
        return Grad(by_input=gradient)


relu = object.__new__(__ReLU)
