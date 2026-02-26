'''
function output = periodize(input,a,b);
output = a+mod(input-a,(b-a));
'''
def periodize(input, a, b):
    return a + (input - a) % (b - a)
