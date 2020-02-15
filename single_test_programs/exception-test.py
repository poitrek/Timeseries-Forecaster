import sys
import traceback

# res = a / b
# print('a / b =', res)

try:
    print('Type a: ')
    a = int(input())

    print('Type b: ')
    b = int(input())
    res = a / b
except Exception as e:
    # string = 'An error occurred: ' + str(e)
    # tb = sys.exc_info()[2]
    print('An error occurred:', e)
    # traceback.print_exc()
    print('Error:', traceback.format_exc().splitlines())
    # print(string)
else:
    print('a / b =', res)
finally:
    print('Program has ended.')
