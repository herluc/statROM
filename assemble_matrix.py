"""
Helper function for AORA.py

"""

import numpy as np

def assemble_matrix(s,K,D,M,**kwargs):
    A = s*s*M + s*D + K

    return A
