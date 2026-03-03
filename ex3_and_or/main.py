from neural_network import train_perceptron
from utils import print_gate

# input agora é composto por 4 samples e 2 features
X = [[0, 0],
     [0, 1],
     [1, 0],
     [1, 1]]

# TODO 1: adiciona o termo de bias (1) a cada sample de X
# TODO 2: inicializa o vetor w a 0's com dimensões matriciais coerentes (já é W^T)
# TODO 3: cria as labels para o AND e OR


w_and = train_perceptron(X, y_and, w)
print_gate(X, w_and)

w_or = train_perceptron(X, y_or, w)
print_gate(X, w_or)

# TODO 5: define a porta y_xor, treina o perceptrão e verifica se converge