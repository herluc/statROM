import numpy as np

def assemble_matrix(s,K,D,M,**kwargs):
    #K = np.array(K)
    #D = np.array(D)
    #M = np.array(M)

    A = s*s*M + s*D + K

    return A