def print_gate(x:list[float], w:float, b:float) -> None:

    print("\n" + "=" * 20)
    print(f"GATE APRENDIDO:\n")
    print(" X | Y ")
    print("---+---")

    for value in x:
        y = 1 if w * value + b >= 0 else 0
        print(f" {value} | {y}")

    print("=" * 20)