from tadgrad.activations import Lin
from pylinal import Vector, Matrix


def test_lin():
    v = Vector([2, 3])

    w = Matrix([
        [1, 10],
        [0, 1]
    ])

    assert Lin(w)(v) == w @ v
    assert Lin(w).grad(v).by_input == w
    assert Lin(w).grad(v).by_params == v

