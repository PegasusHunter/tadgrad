from typing import (
    Union,
    Optional,
    Any,
    Callable,
    List,
    Iterable,
    Iterator,
)
from pylinal import Vector, Matrix

Scalar = Union[int, float]
Tensor = Union[Vector, Matrix]


class Grad:
    by_input: Any
    by_params: Optional[Any]

    def __new__(
        cls,
        *,
        by_input: Any,
        by_params: Optional[Any] = None,
    ) -> 'Grad':
        grad = object.__new__(cls)
        grad.by_input = by_input
        grad.by_params = by_params
        return grad
    
    def __repr__(self) -> str:
        return f'Grad(by_input={self.by_input}, by_params={self.by_params})'


class Function:
    
    def __new__(cls, *args, **kwargs) -> 'Function':
        ...

    def __call__(self, inp: Any) -> Any:
        ...

    def grad(self, inp: Any) -> Grad:
        ...


class Extension(Function):
    _call: Callable
    _grad: Callable

    def __new__(cls, call, *, grad) -> 'Extension':
        obj = object.__new__(cls)
        setattr(obj, '_call', call)
        setattr(obj, '_grad', grad)
        return obj

    def __call__(self, inputs: Iterable[Any]) -> List[Any]:
        return self._call(inputs)

    def grad(self, inputs: Iterable[Any]) -> Grad:
        return self._grad(inputs)
    
    def __repr__(self) -> str:
        return f'Extension({self._call}, grad={self._grad})'


def ext(constructor: Callable[[Any], Function]) -> Callable[[Any], 'Extension']:
    operator = constructor

    def closure(params: Optional[List[Any]] = None) -> 'Extension':
        
        def call(inputs: Iterable[Any]) -> List[Any]:
            results: List[Any]

            if params is None:
                results = [operator()(inp) for inp in inputs]  # type: ignore
            
            else:
                results = [
                    operator(*param)(inp)
                    for (param, inp) in zip(params, inputs)
                ]

            return results

        def grad(inputs: Iterable[Any]) -> Grad:
            grads: Iterator[Grad]

            if params is None:
                grads = (operator().grad(inp) for inp in inputs)  # type: ignore
            
            else:
                grads = (
                    operator(*param).grad(inp)
                    for (param, inp) in zip(params, inputs)
                )
            
            by_input = [d.by_input for d in grads]
            by_params = [d.by_params for d in grads]
            
            return Grad(by_input=by_input, by_params=by_params)

        return Extension(call, grad=grad)
    
    return closure

