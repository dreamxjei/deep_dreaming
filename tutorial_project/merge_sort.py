def merge_sort(x):
    if len(x) > 1:
        mid = len(x)//2
        a = x[:mid]
        b = x[mid:]

        # split until length = 1, where function will stop being called
        a = merge_sort(a)
        b = merge_sort(b)

        # reinitialize x, now that all elements are all in a or b
        x = []

        # compare values and delete. iterate
        while len(a) > 0 and len(b) > 0:
            if a[0] <= b[0]:
                x.append(a[0])
                del a[0]
            else:
                x.append(b[0])
                del b[0]

        # add any final stragglers. max 2
        for e in a:
            x.append(e)
        for e in b:
            x.append(e)

    return x


user_input = input('Specify a list of numbers to sort separated by spaces: ')
x = [int(n) for n in user_input.split()]
x = merge_sort(x)

print(x)
