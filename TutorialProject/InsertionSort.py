def InsertionSort(x):
	for index in range(1, len(x)):
		for j in range(index, 0, -1):
			i = j-1
			if x[j] < x[i]:
				temp = x[i]
				x[i] = x[j]
				x[j] = temp
	return x
	
# x = [8, 3, 6, 9, 1, 10, 4]
user_input = input('Specify a list of numbers separated by spaces: ')
x = [int(n) for n in user_input.split()]
InsertionSort(x)

print(x)
