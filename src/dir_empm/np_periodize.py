def np_periodize(input, a, b):
    return a + np.mod(input - a, b - a)