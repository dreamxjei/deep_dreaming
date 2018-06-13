# Simple Calculator Project

while True:
    print('Two-Number Operations:')
    print('Enter "add", "subtract", "multiply", or "divide".')
    print('Enter "quit" to quit.')
    selection = input('\n:')

    try:
        if selection == 'quit':
            break
        elif selection == 'add':
            num1 = float(input('\nEnter a number: '))
            num2 = float(input('\nEnter another number: '))
            result = num1 + num2
        elif selection == 'subtract':
            num1 = float(input('\nEnter a number to subtract from: '))
            num2 = float(input('\nEnter a number to subtract: '))
            result = num1 - num2
        elif selection == 'multiply':
            num1 = float(input('\nEnter a number: '))
            num2 = float(input('\nEnter another number: '))
            result = num1 * num2
        elif selection == 'divide':
            num1 = float(input('\nEnter a number to divide from: '))
            num2 = float(input('\nEnter a number to divide: '))
            result = num1 / num2
        else:
            print('\nPlease enter a valid number.')
            continue

    except ZeroDivisionError:
        print('\nAn error occurred - can\'t divide by zero.\n')
        continue
    except ValueError:
        print('\nAn error occurred - please enter a number.\n')
        continue

    print('\nThe answer is: ' + str(result) + '\n')
