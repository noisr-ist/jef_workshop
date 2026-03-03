from neural_network import train_perceptron
from utils import print_gate

# X = [x1, x2, 1]
X = [[0, 0, 1],
     [0, 1, 1],
     [1, 0, 1],
     [1, 1, 1]]

y_xor = [0, 1, 1, 0]

# da input layer para hidden layer; W^T
W1T = [
    [0., 0., 0.],           # [W11, W21, b1] -> coluna 1
    [0., 0., 0.],           # [W12, W22, b2] -> coluna 2
]

# da hidden layer para output layer; W^T
W2T = [
    [0., 0., 0.]
]

W = [W1T, W2T]

W_xor = train_perceptron(X, y_xor, W)
print_gate(X, W_xor)

# TODO 6: sozinho: brinca com os parametros W, eta e max_epochs até convergires o XOR