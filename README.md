# tadgrad

Machine Learning from scratch


## Install

From PyPi
```sh
pip install tadgrad
```

From GitHub
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
nn.append(LinLayer(2, 3))
nn.append(Layer(relu))
nn.append(LinLayer(3, 2))
nn.optim = GD(nn.layers, lr=3e-4)

X = [[2, 1], [3, 4], [5, 6]]
labels = [[4, 2], [-2, 3], [2, 1]]
nn.fit(X, labels)

prediction: list = nn.predict([2, 2])
```

