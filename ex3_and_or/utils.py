def print_gate(X: list[list[int]], w: list[float]) -> None:

    print("\n" + "=" * 20)
    print("GATE APRENDIDO:\n")
    print("  x1 x2 | Y ")
    print("--------+---")

    for sample in X:
        x1 = sample[0]
        x2 = sample[1]

        z = 0.0
        for i in range(len(w)):
            z += sample[i] * w[i]

        y = 1 if z > 0 else 0
        print(f"  {x1}  {x2}  | {y}")

    print("=" * 20)