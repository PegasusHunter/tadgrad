import random
from pylinal import Vector
from tadgrad.utils import random_vector, random_matrix


def test_random():
    dim = random.randint(0, 10)
    v = random_vector(dim)
    assert len(v) == dim
    assert isinstance(v, Vector)

    shape = random.randint(1, 10), random.randint(1, 10)
    m = random_matrix(shape)
    assert m.shape == shape


