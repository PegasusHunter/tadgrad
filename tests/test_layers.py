import random
from tadgrad import Layer, LinLayer
from tadgrad.typedefs import Grad

from pylinal import Vector, Matrix


def test_linlayer():
    dim: int = random.randint(1, 10)
    dim_out: int = random.randint(1, 10)

    v: Vector = Vector(i for i in range(dim))
    layer: LinLayer = LinLayer(len(v), dim_out)

    out: Vector = layer(v)
    inp: Vector = layer.inputs.pop()
    print(inp[0])

    assert inp == v
    assert isinstance(out, Vector)
    assert len(out) == dim_out

    w = layer.weights
    assert isinstance(w, Matrix)
    assert w.shape == (dim_out, dim)

    assert isinstance(layer.grad(v), Grad)
    assert layer.grad(v).by_params == v
    assert layer.grad(v).by_input == w

    return


def main():
    test_linlayer()
    return


if __name__ == '__main__':
    main()
