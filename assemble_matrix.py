import numpy as np

def assemble_matrix(s,K,D,M,**kwargs):
    #K = np.array(K) <- löschen?
    #D = np.array(D) <- löschen?
    #M = np.array(M) <- löschen?

    A = s*s*M + s*D + K

    return A