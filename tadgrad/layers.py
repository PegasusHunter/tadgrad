from typing import (
    Tuple,
    Optional,
    Callable,
    Any,
    Tuple,
    Dict,
    List,
    Iterable,
)

from pylinal import Vector, Matrix

from .typedefs import Tensor, Function, Grad
from .utils import random_vector, random_matrix
from .activations import Aff, Lin


class Buffer:
    stack: List[Tensor]

    def __init__(self):
        self.stack = []

    def push(self, inp: Tensor) -> None:
        self.stack.append(inp)
        return

    def pop(self) -> Tensor:
        return self.stack.pop()

    def clear(self) -> None:
        self.stack = []
        return


class Layer:
    buffer: Buffer
    function: Function
    trainable: bool
    param_keys: List[str]

    def __init__(
        self,
        function: Function,
        *,
        trainable: bool = False,
        param_keys: Optional[Iterable[str]] = None,
    ) -> None:
        self.buffer = Buffer()
        self.function = function
        self.trainable = trainable
        self.param_keys = [k for k in param_keys]
        return

    def __call__(self, t: Tensor) -> Tensor:
        self.buffer.push(t)
        return self.function(t)

    @property
    def grad(self) -> Callable[[Tensor], Grad]:
        return self.function.grad

    @property
    def params(self) -> Dict:
        p = dict()
        for k in self.param_keys:
            p[k] = getattr(self.function, k)
        return p

    def update_params(self, **kwargs) -> None:
        for key, v in kwargs:
            assert hasattr(self.function, key)
            setattr(self.function, key, v)
        return


class DenseLayer(Layer):
    shape: Tuple[int, int]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = False,
        trainable: bool = True,
    ) -> None:
        self.shape = (out_features, in_features)
        self.trainable = trainable

        params = dict()
        params['weights'] = random_matrix(self.shape)
        function: Function
        if bias:
            params['bias'] = random_vector(out_features)
            function = Aff(**params)
        else:
            function = Lin(**params)

        keys = params.keys()
        super().__init__(function, trainable=trainable, param_keys=keys)
        return

    @property
    def weights(self) -> Matrix:
        return self.params['weights']
