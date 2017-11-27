def multiply(a,b):
    print("Will compute", a, "times", b)
    c = 0
    for i in range(0, a):
        c = c + b
    return c

# Fibonacci numbers module

def fib(n):    # write Fibonacci series up to n
    a, b = 0, 1
    while b < n:
        print(b, end=' ')
        a, b = b, a+b
    print()
    return b

def fib2(n):   # return Fibonacci series up to n
    result = []
    a, b = 0, 1
    c = 0
    while b < n:
        result.append(b)
        a, b = b, a+b
        c = c + b
#    return result
    return c
