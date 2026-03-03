
def step(z:float):
    return 1 if z > 0 else 0

# TODO 3: retorna o valor do produto interno de 2 vetores v1 e v2
def dot(v1:list[float], v2:list[float]):
    assert len(v1) == len(v2)

    z = 0
    for i in range(len(v1)):
        z += v1[i] * v2[i]
    
    return z

# TODO 4: atualiza o treino do percetrão, agora para as duas features
def train_perceptron(
        X:list[list[int]],
        y_lab:list[int],
        w:list[float],
        max_epochs:float=1000,
        eta:float=.1
        ) -> list[float]:

    assert len(X) == len(y_lab)           # check dimensions

    epoch = 1
    while epoch <= max_epochs:
        print(f"\n::: Época {epoch} :::", end="\n")

        mistakes = 0

        for i in range(len(X)):                 # itera sobre cada sample
            # X = [[,,], [,,], ...]
            xi = X[i]
            z = dot(w, xi)
            y_pred = step(z)

            print(f"-> ( x = {xi[:2]}, y = {y_lab[i]} ):", end="\n\t")
            print(f"z = {z:6.3f} => step(z) = {y_pred} -> ", end="")

            if y_pred != y_lab[i]:
                for j in range(len(w)):
                    w[j] = w[j] - eta * (y_pred - y_lab[i]) * xi[j]

                mistakes += 1

                print(f"Erro! Update: w = {[round(wj,3) for wj in w]}")
            else:
                print("OK!")

        if mistakes == 0:
            print(f"\nPercetrão convergiu em {epoch} épocas :)")
            break

        epoch += 1

    else:
        print(f"\nPercetrão não convergiu nas {max_epochs} épocas... :(")
    
    return w