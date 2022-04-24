from typing import (
    Tuple,
    Optional,
    Callable,
    Any,
    Tuple,
    Dict,
    List,
    Iterable,
    Generic,
    TypeVar,
)

from pylinal import Vector, Matrix

from .typedefs import Tensor, Function, Grad
from .utils import random_vector, random_matrix
from .activations import Aff, Lin

T = TypeVar('T')


class Stack(Generic[T]):
    values: List[T]

    def __init__(self, values: List[T] = []) -> None:
        self.values = values
        return
    
    def push(self, value: T) -> None:
        self.values.append(value)
        return

    def pop(self) -> Optional[T]:
        try:
            return self.values.pop()
        except IndexError:
            return None

    def clear(self) -> None:
        self.values = []
        return


class Buffer(Generic[T]):
    inputs: Stack[T]

    def __init__(self, *, inputs: List[T] = []) -> None:
        self.inputs = Stack(inputs)
        return

    def clear(self) -> None:
        self.inputs.clear()
        return


class Layer:
    buffer: Buffer[Tensor]
    function: Function
    trainable: bool
    param_keys: Iterable[str]
    shape: Optional[Tuple[int, int]]

    def __init__(
        self,
        function: Function,
        *,
        shape: Optional[Tuple[int, int]] = None,
        trainable: bool = False,
        param_keys: Iterable[str] = [],
    ) -> None:
        self.buffer = Buffer()

        self.function = function
        self.shape = shape
        self.trainable = trainable
        self.param_keys = [k for k in param_keys]
        return

    def __call__(self, t: Tensor) -> Tensor:
        self.inputs.push(t)
        return self.function(t)
    
    def clear_buffer(self) -> None:
        self.buffer.clear()
        return

    @property
    def inputs(self) -> Stack:
        return self.buffer.inputs

    @property
    def grad(self) -> Callable[[Tensor], Grad]:
        return self.function.grad

    @property
    def params(self) -> Dict:
        p = dict()
        for k in self.param_keys:
            p[k] = getattr(self.function, k)
        return p

    def set_params(self, **kwargs) -> None:
        for key, v in kwargs.items():
            assert hasattr(self.function, key)
            setattr(self.function, key, v)
        return

    def __repr__(self) -> str:
        args = f'\t{self.function},\n\t'

        if self.shape != None:
            args += f'shape={self.shape},\n\t'

        if self.params:
            args += ',\n\t'.join(f'{key}={value}' for key, value in self.params.items())
            args += ',\n\t'

        args += f'trainable={self.trainable}'

        return f'Layer(\n{args}\n)'


class LinLayer(Layer):
     
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        weights: Optional[Matrix] = None,
        trainable: bool = True,
    ) -> None:
        shape = (out_features, in_features)
        self.trainable = trainable

        _weights: Matrix = random_matrix(shape) if weights is None else weights
        assert _weights.shape == shape
        
        function: Function = Lin(weights=_weights)
        keys: set = {'weights'}
        
        super().__init__(function, shape=shape, trainable=trainable, param_keys=keys)
        return

    @property
    def weights(self) -> Matrix:
        return self.params['weights']
    
    @weights.setter
    def weights(self, w: Matrix) -> None:
        if w.shape != self.shape:
            error = f"'weights' shape must be {self.shape}, but {w.shape} provided"
            raise ValueError(error)

        self.set_params(weights=w)
        return
 

class AffLayer(Layer):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        weights: Optional[Matrix] = None,
        bias: Optional[Vector] = None,
        trainable: bool = True,
    ) -> None:
        shape: Tuple[int, int] = (out_features, in_features)
        self.trainable = trainable

        _weights: Matrix = random_matrix(shape) if weights is None else weights
        assert _weights.shape == shape
        
        _bias: Vector = random_vector(out_features)
        assert len(_bias) == out_features

        function: Function = Aff(weights=_weights, bias=_bias)
        keys: set = {'weights', 'bias'}

        super().__init__(function, shape=shape, trainable=trainable, param_keys=keys)
        return

    @property
    def weights(self) -> Matrix:
        return self.params['weights']
    
    @weights.setter
    def weights(self, w: Matrix) -> None:
        if w.shape != self.shape:
            error = f"'weights' shape must be {self.shape}, but {w.shape} provided"
            raise ValueError(error)

        self.set_params(weights=w)
        return
    
    @property
    def bias(self) -> Vector:
        return self.params['bias']
    
    @bias.setter
    def bias(self, bias: Vector) -> None:
        old_bias: Vector = self.params['bias']
        assert len(bias) == len(old_bias)

        self.set_params(bias=bias)
        return

