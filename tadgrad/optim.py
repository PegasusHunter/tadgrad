from typing import Optional, List, Any, Iterable

from pylinal import Matrix, Vector
from .typedefs import Function, Grad
from .layers import Layer


class Optim:
    layers: List[Layer]
    lr: float

    def step(self, *args, **kwargs) -> None:
        ...


class GD(Optim):
    """Gradient Descent"""
   
    layers: List[Layer]
    lr: float

    def __init__(self, layers: List[Layer], *, lr: float = 3e-4) -> None:
        self.layers = layers
        self.lr = lr
        return

    def step(self, loss_grads: Iterable[Vector], *, lr: Optional[float] = None) -> None:  # type: ignore[override]
        lr = self.lr if lr is None else lr
        for grad in loss_grads:
            self.single_step(grad, lr=lr)
        return

    def single_step(self, loss_grad: Vector, *, lr: float) -> None:
        grad: Vector = loss_grad

        for layer in reversed(self.layers):
            inp: Vector = layer.inputs.pop()  # type: ignore
            dl: Grad = layer.grad(inp)

            if layer.trainable and hasattr(layer, 'weights'):
                dwx_dw: Vector = dl.by_params  # type: ignore

                delta: Matrix = Matrix((lr*g)*dwx_dw for g in grad)
                layer.weights -= delta  # type: ignore

                dwx_dx: Matrix = dl.by_input
                length = dwx_dx.shape[1]

                grad = Vector(
                    sum(row[k]*g for (row, g) in zip(dwx_dx, grad))
                    for k in range(length)
                )

            else:
                df_dx: Vector = dl.by_input
                grad = Vector(g*d for g, d in zip(grad, df_dx))
        
        return

