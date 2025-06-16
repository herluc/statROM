"""
statROM Helmholtz 1D example
"""

__author__ = "Lucas Hermann"
__version__ = "0.1.1"
__license__ = "MIT"


import numpy as np
from exampleHelmholtz import StatROM_1D


class UserParameters:
    def __init__(self):
        pass
    f = 460#Hz      # frequency
    s = 100#Hz      # AORA expansion frequency
    m = 5          # number of matched moments in AORA

    n = 100         # number of degrees of freedom
    n_fine = 1000   # number of degrees of freedom in the data generating process
    n_qmc = 256     # number of QMC sample points. Only powers of two are permitted

    ns = 11         # number of sensors
    no = 200        # number of observations per sensor
    sig_o = 1e-1#Pa # observation noise

    n_est = 12      # number of error estimator training points



if __name__ == '__main__':
    up = UserParameters
    funcs = StatROM_1D(up)

    ### Generate data from fine mesh:
    funcs.switchMesh_self("ground_truth")
    funcs.generateParameterSamples(up.n_qmc) 
    funcs.getFullOrderPrior() # solve for "ground truth"
    funcs.get_noisy_data_from_solution(0,np.real(funcs.u_mean_std)) # extract noisy data at sensors
    ####

    funcs.switchMesh_self("coarse")
    funcs.generateParameterSamples(up.n_qmc) 
    ### Compute ROM basis:
    for i,sample in enumerate(funcs.f_samples):
        print("computing basis for sample no. "+str(i))
        funcs.computeROMbasisSample(sample,i)
    ####

    # Full order reference:
    funcs.getFullOrderPrior() 

    # ROM prior and error estiamte:
    funcs.calcROMprior()
    
    # FOM reference posterior, classical ROM posterior and proposed ROM posterior:
    funcs.getFullOrderPosterior()
    funcs.getEasyROMPosterior()
    funcs.getCorrectedROMPosterior()

    # plotting etc.
    funcs.saveData()
    funcs.plotPriorPosteriorComparison(prior=True,posterior=True) # choose if you want to see the prior, the posterior or both
    funcs.plotRomError()
