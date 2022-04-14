import random
from tadgrad import Layer, DenseLayer
from tadgrad.typedefs import Grad

from pylinal import Vector, Matrix


def test_dense():
    dim = random.randint(1, 10)
    dim_out = random.randint(1, 10)

    v = Vector(i for i in range(dim))
    layer = DenseLayer(len(v), dim_out)

    out = layer(v)
    inp = layer.buffer.pop()

    assert inp == v
    assert isinstance(out, Vector)
    assert len(out) == dim_out

    w = layer.weights
    assert isinstance(w, Matrix)
    assert w.shape == (dim_out, dim)

    assert isinstance(layer.grad(v), Grad)
    assert layer.grad(v).by_params == v
    assert layer.grad(v).by_input == w
