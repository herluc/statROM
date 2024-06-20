import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform

def exponentiated_quadratic(xa, xb, lf, sigf):
    """Exponentiated quadratic kernel with σ=1"""
    # L2 distance (Squared Euclidian)
    sq_norm = -0.5 * cdist(xa, xb, 'sqeuclidean') * (1/lf**2)
   # r2 = cdist(np.atleast_2d(xa), np.atleast_2d(xb))**2
    
    return sigf**2 * np.exp(sq_norm)
    #return np.exp(2.*sigf)*np.exp(-0.5*r2*np.exp(-2.*lf))


def matern52(xa, xb, lf, sigf):
    """Exponentiated quadratic kernel with σ=1"""
    # L2 distance (Squared Euclidian)
    r = cdist(xa, xb, 'euclidean')
    sq_norm = (1+r * (np.sqrt(5)/lf) + (5/3)*r*r* (1/lf**2))*np.exp(-np.sqrt(5) * r * (1/lf))

    return sigf**2 * sq_norm


def matern52_log(xa, xb, lf, sigf):
    """Exponentiated quadratic kernel with σ=1"""
    lf = np.exp(lf)
    sigf = np.exp(sigf)
    # L2 distance (Squared Euclidian)
    r = cdist(xa, xb, 'euclidean')
    sq_norm = (1+r * (np.sqrt(5)/lf) + (5/3)*r*r* (1/lf**2))*np.exp(-np.sqrt(5) * r * (1/lf))

    return sigf**2 * sq_norm


def exponentiated_quadratic_log(xa, xb, lf, sigf):
    """Exponentiated quadratic kernel with σ=1"""
    lf = np.exp(lf)
    sigf = np.exp(sigf)
    # L2 distance (Squared Euclidian)
    sq_norm = -0.5 * cdist(xa, xb, 'sqeuclidean') * (1/lf**2)
    r2 = cdist(np.atleast_2d(xa), np.atleast_2d(xb))**2
    
    return sigf**2 * np.exp(sq_norm)
    #return np.exp(2.*sigf)*np.exp(-0.5*r2*np.exp(-2.*lf))


def exponentiated_quadratic1D(xa, xb, lf, sigf):
    """Exponentiated quadratic kernel with σ=1"""
    # L2 distance (Squared Euclidian)
    sq_norm = -0.5 * cdist(xa, xb, 'sqeuclidean') * (1/lf**2)
    
    return sigf**2 * np.exp(sq_norm)




def exponentiated_quadratic_log_old(xa, xb, l, sig):
    """ Exponentiated quadratic kernel, expects input parameters as log(par). """
    #return np.exp(2.*sig) * np.exp(-0.5 * scipy.spatial.distance.cdist(xa, xb)**2 * np.exp(-2.*l))
    #return np.exp(sig)**2 * np.exp(-0.5 * cdist(xa, xb)**2 * 1/(np.exp(l)**2	))
    x = np.expand_dims(xa[:,0], 1)
    K = np.exp(-0.5 * pdist(x / np.exp(l), metric="sqeuclidean"))
    K = squareform(K)
    np.fill_diagonal(K, 1)
    K = np.exp(sig)**2 * K
    return K


def matern_log(xa, xb, l, sig):
    """ Matern Kernel, expects input parameters as log(par). """
    r = cdist(xa, xb, 'euclidean')
    #return np.exp(2*sig) * np.exp(-1 * cdist(xa, xb, 'euclidean') * np.exp(-1*l)) #nu = 1/2
    #return np.exp(2*sig) * ((1 + np.sqrt(3) * r *np.exp(-1*l)) *  np.exp(-1 * np.sqrt(3) * r * np.exp(-1*l))   ) # nu = 3/2
    return np.exp(2*sig) * ((1 + np.sqrt(5) * r *np.exp(-1*l) +  5*r*r/3*np.exp(-1*l)*np.exp(-1*l)  ) *  np.exp(-1 * np.sqrt(5) * r * np.exp(-1*l))   ) # nu = 5/2


def periodic(xa,xb,lf,sigf,p):
    r = cdist(xa, xb, 'euclidean')
    sq_norm_matern= (1+r * (np.sqrt(5)/lf) + (5/3)*r*r* (1/lf**2))*np.exp(-np.sqrt(5) * r * (1/lf))
    sq_norm = -2*np.sin(np.pi * cdist(xa, xb, 'euclidean') /p)**2* (1/lf**2)
    return sigf**2 * (np.exp(sq_norm)* np.exp(sq_norm_matern))