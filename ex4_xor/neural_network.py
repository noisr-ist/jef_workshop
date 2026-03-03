import math

def dot(v1:list[float], v2:list[float]):
    assert len(v1) == len(v2)   # verifica as dimensões

    z = 0.
    for i in range(len(v1)):
        z += v1[i] * v2[i]
    return z

# TODO 1: faz o cálculo do produto do forward
def forward_dot(wT:list[list[float]], x:list[float]) -> list[float]:
    pass

# TODO 2: retorna o vetor a(z) da sigmóide aplicada ao vetor z
def sigmoid(z:list[float]) -> list[float]:
    pass

# TODO 3: calcula e retorna a DERIVADA da sigmoide (para usar no backpropagation)
def sigmoid_derivative(y:float):
    pass


def train_perceptron(
        X:list[list[int]],
        y_lab:list[float],
        W:list[list[list[float]]],
        max_epochs:int=1000,
        eta:float=.1
    ) -> list[list[list[float]]]:

    W1T, W2T = W    # faz W1T, W2T = W[0], W[1]

    # verificar dimensões de cenas
    epoch = 1
    while epoch <= max_epochs:
        print(f"\n::: Época {epoch} :::", end="\n")

        err = 0.

        for i in range(len(X)):
            xi = X[i]
            z = forward_dot(W1T, xi)    # cálculo feed forward c/ pesos
            a = sigmoid(z)
            a.append(1)

            z = forward_dot(W2T, a)     
            y_pred = sigmoid(z)[0]

            # erro total acumulado da backpropagation
            err += backpropagation(xi, a, y_lab[i], y_pred, W, eta)

        # TODO 5: faz a rede parar de aprender se o erro for inverior a 0.01 
            #print(f"A tua Rede Reural convergiu em {epoch} épocas!")

        epoch += 1

    return W

def backpropagation(
        xi: list[float],
        a: list[float],
        y_lab: float,
        y_pred: float,
        W: list[list[list[float]]],
        eta: float
    ) -> float:

    W1T, W2T = W

    # TODO 4: escreve dL/dz = dL/dy * dy/dz (deriva a função da sigmóide e da cost function)
    # output layer


    # só para guardar pesos antigos antes de atualizar
    old_W2 = W2T[0][:]

    # corrige hidden layer [2] (W2)
    for j in range(len(W2T[0])):
        W2T[0][j] = W2T[0][j] - eta * dL_dz * a[j]  # gradient descent

    # corrige hidden layer [1] (W1)
    for h in range(len(W1T)):
        h_output = a[h]                 # h1 ou h2
        weight_out = old_W2[h]          # peso que liga hidden h ao output

        delta_hidden = dL_dz * weight_out * sigmoid_derivative(h_output)

        # atualiza pesos da hidden
        for k in range(len(W1T[h])):
            W1T[h][k] = W1T[h][k] - eta * delta_hidden * xi[k]  # gradient descent

    return abs(y_pred - y_lab)