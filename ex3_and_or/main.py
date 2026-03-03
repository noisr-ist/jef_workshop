from neural_network import train_perceptron
from utils import print_gate

# input agora é composto por 4 samples e 2 features
X = [[0, 0],
     [0, 1],
     [1, 0],
     [1, 1]]

# TODO 1: adiciona o termo de bias (1) a cada sample de X
for xi in X:
    xi.append(1)

# TODO 2: inicializa o vetor w a 0's com dimensões matriciais coerentes (já é W^T)
w = [0.] * len(X[0]) # w = [0., 0., 0.]

# TODO 3: cria as labels para o AND e OR
y_and = [0, 0, 0, 1]
y_or = [0, 1, 1, 1]

w_and = train_perceptron(X, y_and, w)
print_gate(X, w_and)

w_or = train_perceptron(X, y_or, w)
print_gate(X, w_or)

# TODO 5: define a porta y_xor, treina o perceptrão e verifica se converge
y_xor = [0, 1, 1, 0]
w_xor = train_perceptron(X, y_xor, w)
print_gate(X, w_xor)