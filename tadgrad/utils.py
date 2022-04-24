import random
from typing import Tuple, Iterator, Any, List
from pylinal import Vector, Matrix


def random_vector(dim: int) -> Vector[float]:
    return Vector(random.uniform(-1, 1) for _ in range(dim))


def random_matrix(shape: Tuple[int, int]) -> Matrix[float]:
    rows, cols = shape
    
    weights: Matrix = Matrix(
        [random.uniform(-1, 1) for col in range(cols)]
        for row in range(rows)
    )
    return weights


def linspace(start: float, end: float, points: int = 50) -> List[float]:
     assert start < end
     assert points > 1
 
     pts = points - 1
     delta = (end - start)/pts
 
     xs = []
     i = start
     for _ in range(pts):
         xs.append(i)
         i += delta
     xs.append(end)
     return xs

