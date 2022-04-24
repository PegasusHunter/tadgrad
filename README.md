# tadgrad

Machine Learning from scratch


## Install

From PyPi
```sh
pip install tadgrad
```

From git
```sh
pip install git+https://github.com/cospectrum/tadgrad.git
```


## Usage

```python
from tadgrad import Network, Layer, LinLayer
from tadgrad.activations import relu
from tadgrad.losses import MSE
from tadgrad.optim import GD


nn = Network(loss=MSE)
nn.append(LinLayer(1, 4))
nn.append(Layer(relu))
nn.append(LinLayer(4, 1))
nn.optim = GD(nn.layers, lr=3e-4)

sqrt = lambda x: x**0.5
xs = [p/1000 for p in range(1800, 2200)]  # [1.8, 2.2]

X = [[x] for x in xs]
labels = [[sqrt(x)] for x in xs]

nn.fit(X, labels, epochs=25)

sqrt_2 = nn.predict([2])
print(f'sqrt(2) = {sqrt_2}')

```

