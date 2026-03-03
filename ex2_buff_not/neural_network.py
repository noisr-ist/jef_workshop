
# TODO 3: escreve a função de ativação step (1 se x > 0, 0 se contrário)
def step(z:float):
    return 1 if z > 0 else 0

# TODO 4: implementa o algoritmo do percetrão
def train_perceptron(
        x:list[int],
        y_lab:list[int],
        w:float,
        b:float,
        max_epochs=1000,
        eta=.1):
    
    assert len(y_lab) == len(x)

    epoch = 1
    while epoch <= max_epochs:
        print(f"\n::: Época {epoch} :::", end="\n")

        mistakes = 0
        for i in range(len(x)):
            z = w * x[i] + b    # x = [0, 1] ; x[0] = 0, x[1] = 1
            y_pred = step(z)

            print(f"z = {z:6.3f} => step(z) = {y_pred} -> ", end="")
            print(f"Erro! Update: w={w:.3f}, b={b:.3f}")

            if y_pred != y_lab[i]: # y_lab = [1, 0]
                w = w - eta * (y_pred - y_lab[i]) * x[i]
                b = b - eta * (y_pred - y_lab[i]) * 1
                
                print(f"-> ( x = {x[i]}, y = {y_lab[i]} ):", end="\n\t")
                
                mistakes += 1
            else:
                print("Ok!")
        
        if mistakes == 0:
            print(f'\nPercetrão convergiu em {epoch} épocas :)')
            break
        
        epoch += 1

    else: 
        print(f'\nPrcetrão não convergiu nas {max_epochs} épocas... :(')

    return w, b