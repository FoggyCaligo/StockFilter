
import random


rsult = {
    'sum': 0,
    'sub': 0,
    'mul': 0,
    'div': 0
}
npt = ''

def getRandNum():
    return int( random.randint(1, 999))

def getRandOp():
    # ops = ['+', '-', '*', '/']
    ops = ['+', '-']
    return random.choice(ops)


def calc(a,b,op):
    if op == '+':
        return a + b
    elif op == '-':
        return a - b
    elif op == '*':
        return a * b
    elif op == '/':
        if b != 0:
            return a / b
        else:
            return 'Error: Division by zero'

while npt.lower() != 'exit' : 
    nums = []
    for i in range(3):
        nums.append(getRandNum())
    ops = []
    for i in range(2):
        ops.append(getRandOp())


    formula = str(nums[0]) + ' ' + ops[0] + ' ' + str(nums[1]) + ' ' + ops[1] + ' ' + str(nums[2])
    print('\n',formula)
    npt = input()

    rsult = calc(calc(nums[0], nums[1], ops[0]), nums[2], ops[1])
    
    print(rsult)



