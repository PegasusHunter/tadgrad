from typing import List, Any, Optional, Callable, TypeVar
from pylinal import Vector

from .typedefs import Function
from .optim import Optim
from .layers import Layer

T = TypeVar('T')
Loss = Callable[[T], Function]


class Network:

    layers: List[Layer]
    loss: Loss
    optim: Optional[Optim] = None

    def __init__(self, loss: Loss, *, layers: List[Layer] = []) -> None:
        setattr(self, 'loss', loss)
        self.layers = layers
        return

    def append(self, layer: Layer) -> None:
        if self.layers == []:
            self.layers.append(layer)
            return

        if self.layers[-1].shape is not None and layer.shape is not None:
            assert layer.shape[0] == self.layers[-1].shape[1]
        
        self.layers.append(layer)
        return
    
    def pop(self) -> Layer:
        return self.layers.pop()

    def forward(self, x: Vector) -> Vector:
        out: Vector = x
        for layer in self.layers:
            out = layer(out)  # type: ignore
        return out

    def predict(self, x: List) -> List:
        return list(self.forward(Vector(x)))

    def fit(
        self,
        X: List[List[T]],
        labels: List[List[T]],
        *,
        epochs: int = 1,
        optim: Optional[Optim] = None,
    ) -> 'Network':
        optim = self.optim if optim is None else optim
        assert optim is not None
        
        _X: List[Vector] = [Vector(x) for x in X]
        _labels: List[Vector] = [Vector(label) for label in labels]

        for epoch in range(epochs):
            for x, label in zip(_X, _labels):
                out: Vector = self.forward(x)
                loss: Function = self.loss(label)  # type: ignore
                
                grad = loss.grad(out).by_input
                optim.step([grad])
        
        return self
    
    def __repr__(self) -> str:
        return f'Network(\n\tloss={self.loss},\n\tlayers={self.layers}\n)'

