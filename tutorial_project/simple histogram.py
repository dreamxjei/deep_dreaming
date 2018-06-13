'''
def histogram(s):
    d = dict()
    for c in s:
        if c not in d:
            d[c] = 1
        else:
            d[c] += 1
    return d
'''

def histogram(x):
    d = dict()
    for a in x:
        d[a] = d.get('a', 0) + 1
    return d

h = histogram('brontosaurus')
print(h)

print(h.get('a'))