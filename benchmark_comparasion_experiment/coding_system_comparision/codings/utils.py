import numpy as np

def nuclear_indicator(grad, s):
    m, n  = grad.shape
    return np.sum(s)*np.sqrt(m+n)

def l1_indicator(grad):
    return np.linalg.norm(grad.reshape(-1), 1)