from tadgrad.typedefs import Extension, Grad, ext


def test_extension():
    
    f = lambda x: x**2
    df = lambda x: Grad(by_input=2*x)
    f.grad = df
    
    def func():
        return f

    F: Extension = ext(func)()
    inputs = list(range(10))

    assert F(inputs) == [f(x) for x in inputs]
    assert F.grad(inputs).by_input == [f.grad(x).by_input for x in inputs]


