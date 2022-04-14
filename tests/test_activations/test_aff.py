from tadgrad.activations import Aff
from pylinal import Vector, Matrix


def test_lin():
    v = Vector([2, 3])

    w = Matrix([
        [1, 10],
        [0, 1]
    ])
    bias = Vector([0, 0])

    aff = Aff(w, bias)
    assert aff(v) == w @ v
    assert aff.grad(v).by_input == w
    assert aff.grad(v).by_params == v
