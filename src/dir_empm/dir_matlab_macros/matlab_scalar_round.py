def matlab_scalar_round(scalar):
    if scalar % 1 == 0.5: return int(scalar)+1;
    return round(scalar)
