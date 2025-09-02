"""
Helper function for AORA.py

"""

import numpy as np

def assemble_matrix_derivative(s,K,D,M,**kwargs):
    K = np.array(K)
    D = np.array(D)
    M = np.array(M)

    omega = -1j*s
    Ap = -2*omega*M + 1j*D

    return Ap
