import random
from typing import Tuple
from pylinal import Vector, Matrix


def random_vector(dim: int) -> Vector[float]:
    return Vector(random.uniform(-1, 1) for _ in range(dim))


def random_matrix(shape: Tuple[int, int]) -> Matrix[float]:
    rows, cols = shape
    
    weights = Matrix(
        [random.uniform(-1, 1) for col in range(cols)]
        for row in range(rows)
    )
    return weights


