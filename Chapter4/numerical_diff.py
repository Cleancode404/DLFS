#not a good example

def numerical_diff(f, x):
    h = 10e-50
    return (f(x + h) - f(x)) / h
    

    

    