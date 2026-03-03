from neural_network import train_perceptron
from utils import print_gate

# TODO 1: define o resultado esperado do NOT gate
x = [0, 1]
y_not = [1, 0]
y_buf = [0, 1]

# TODO 2: declara os parâmetros de treino (peso e bias) a zero
w = 0.
b = 0.

# TODO 5: chama o percetrão e mostra a tabela de verdade correspondente
w, b = train_perceptron(x, y_not, w, b)
print_gate(x, w, b)

w, b = train_perceptron(x, y_buf, w, b)
print_gate(x, w, b)