from __future__ import division

import random
from Stack import Stack
import re


OPERATORS = set(['+', '-', '*', '/', '(', ')'])
PRIORITY = {'+': 1, '-': 1, '*': 2, '/': 2}


def filter_equation(equation):
    equation_equals = ""

    # Clean the equation
    try:
        equation_equals = re.search(r"([a-z](\s+)?=|=(\s+)?[a-z])",
                                    equation).group(1)

        equation_equals = re.sub("=", "", equation_equals)
    except:
        pass

    equation = equation.replace(" ", "")
    equation = re.sub(r"([a-z](\s+)?=|=(\s+)?[a-z])",
                      "", equation)

    return equation, equation_equals


def infix_to_postfix(equation=""):
    filtered_equation, equation_equals = filter_equation(equation)

    stack = Stack()  # only pop when the coming op has priority
    output = ""

    for char in filtered_equation:
        if char not in OPERATORS:
            output += char
        elif char == '(':
            stack.push(char)
        elif char == ')':
            while not stack.isEmpty() and stack.peek() != '(':
                output += stack.pop()
            stack.pop()  # pop '('
        else:
            while not stack.isEmpty() and stack.peek() != '(' and PRIORITY[char] <= PRIORITY[stack.peek()]:
                output += stack.pop()

            stack.push(char)

    while not stack.isEmpty():
        output += stack.pop()

    return output


def postfix_to_infix(equation=""):
    filtered_equation, equation_equals = filter_equation(equation)

    stack = Stack()
    prev_op = None

    for char in filtered_equation:
        if not char in OPERATORS:
            stack.push(char)
        else:
            b = stack.pop()
            a = stack.pop()

            if prev_op and len(a) > 1 and PRIORITY[char] > PRIORITY[prev_op]:
                expr = '(' + a + ')' + char + b
            else:
                expr = a + char + b

            stack.push(expr)

            prev_op = char

    return stack.peek()


def infix_to_prefix(equation=""):
    filtered_equation, equation_equals = filter_equation(equation)

    op_stack = Stack()
    exp_stack = Stack()

    for char in filtered_equation:
        if not char in OPERATORS:
            exp_stack.push(char)
        elif char == '(':
            op_stack.push(char)
        elif char == ')':
            while op_stack.peek() != '(':
                op = op_stack.pop()
                a = exp_stack.pop()
                b = exp_stack.pop()

                exp_stack.push(op + b + a)

            op_stack.pop()
        else:
            while not op_stack.isEmpty() and op_stack.peek() != '(' and PRIORITY[char] <= PRIORITY[op_stack.peek()]:
                op = op_stack.pop()
                a = exp_stack.pop()
                b = exp_stack.pop()

                exp_stack.push(op + b + a)

            op_stack.push(char)

    while not op_stack.isEmpty():
        op = op_stack.pop()
        a = exp_stack.pop()
        b = exp_stack.pop()

        exp_stack.push(op + b + a)

    return exp_stack.peek()


def prefix_to_infix(equation=""):
    filtered_equation, equation_equals = filter_equation(equation)

    stack = Stack()
    prev_op = None

    for char in reversed(equation):
        if not char in OPERATORS:
            stack.push(char)
        else:
            a = stack.pop()
            b = stack.pop()

            if prev_op and PRIORITY[prev_op] < PRIORITY[char]:
                exp = '(' + a + ')' + char + b
            else:
                exp = a + char + b

            stack.push(exp)
            prev_op = char

    return stack.peek()


def convert_infix_to_prefix(equation=""):
    operator_stack = Stack()
    prefix = []

    filtered_equation, equation_equals = filter_equation(equation)

    for char in list(filtered_equation[::-1]):
        if not isOperator(char):
            prefix.append(char)
        elif char == ')':
            operator_stack.push(char)
        elif char == '(':
            topchar = operator_stack.pop()

            while topchar != ')':
                prefix.append(topchar)
                topchar = operator_stack.pop()

            operator_stack.pop()
        else:
            while not operator_stack.isEmpty() and precedence(operator_stack.peek()) >= precedence(char):
                prefix.append(operator_stack.pop())

            operator_stack.push(char)

    while not operator_stack.isEmpty():
        prefix.append(operator_stack.pop())

    expression = "".join(prefix)

    return f"{equation_equals.strip()}={expression}"


def convert_infix_to_postfix(equation=""):
    operator_stack = Stack()
    postfix = []

    filtered_equation, equation_equals = filter_equation(equation)

    for char in list(filtered_equation):
        if not isOperator(char):
            postfix.append(char)
        elif char == '(':
            operator_stack.push(char)
        elif char == ')':
            topchar = operator_stack.pop()

            while topchar != '(':
                postfix.append(topchar)
                topchar = operator_stack.pop()

            operator_stack.pop()
        else:
            while not operator_stack.isEmpty() and precedence(operator_stack.peek()) < precedence(char):
                postfix.append(operator_stack.pop())

            operator_stack.push(char)

    while not operator_stack.isEmpty():
        postfix.append(operator_stack.pop())

    expression = "".join(postfix)

    return f"{equation_equals.strip()}={expression}"


if __name__ == "__main__":
    test_equation = "x=(9 * 4)-3+12"
    test_equation2 = "12+ (9 * 4)-3=x"
    test_equation3 = "(9 * 4)-3+12  = x"
    test_equation4 = "z= ( A + B ) * C - ( D - E ) * ( F + G )"

    print("Postfix:")
    # Testing Postfix conversion
    print(test_equation)
    print(infix_to_postfix(test_equation))
    print(test_equation2)
    print(infix_to_postfix(test_equation2))
    print(test_equation3)
    print(infix_to_postfix(test_equation3))
    print(test_equation4)
    print(infix_to_postfix(test_equation4))

    print("Prefix:")
    # Testing Prefix conversion
    print(test_equation)
    print(infix_to_prefix(test_equation))
    print(test_equation2)
    print(infix_to_prefix(test_equation2))
    print(test_equation3)
    print(infix_to_prefix(test_equation3))
    print(test_equation4)
    print(infix_to_prefix(test_equation4))
