def square_root(a, x):
    while True:
        print(x)
        y = (x + a / x) / 2
        if abs(y-x) < 0.0000001:
            break
        x = y

square_root(9, 2)