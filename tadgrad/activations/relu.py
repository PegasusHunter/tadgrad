from pylinal import Vector
from ..typedefs import Function, Grad, T


def relu(x: T) -> T: 
    return max(0, x)


def drelu(x: T) -> T:
    return 1 if x > 0 else 0


class __ReLU(Function):

    def __call__(self, v: Vector) -> Vector:
        return Vector(relu(x) for x in v)

    def grad(self, v: Vector) -> Grad:
        gradient = Vector(drelu(x) for x in v)
        return Grad(by_input=gradient)


ReLU = object.__new__(__ReLU)
