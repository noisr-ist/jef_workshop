import math

def print_gate(X: list[list[int]], W: list[list[list[float]]]) -> None:

    W1T, W2T = W

    print("\n" + "=" * 20)
    print("GATE APRENDIDO:\n")
    print("  x1 x2 | Y ")
    print("--------+---")

    for sample in X:

        x1 = sample[0]
        x2 = sample[1]

        # hidden
        z_hidden = []
        for neuron in W1T:
            z_hidden.append(sum(neuron[i] * sample[i] for i in range(len(sample))))

        a_hidden = [1.0 / (1.0 + math.exp(-z)) for z in z_hidden]
        a_hidden.append(1)  # bias

        # output layer
        z_out = sum(W2T[0][i] * a_hidden[i] for i in range(len(a_hidden)))
        y_pred = 1.0 / (1.0 + math.exp(-z_out))

        y_class = 1 if y_pred >= 0.5 else 0

        print(f"  {x1}  {x2}  | {y_class}")

    print("=" * 20)