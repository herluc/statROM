"""
statROM Helmholtz 2D scattering example
"""

__author__ = "Lucas Hermann"
__version__ = "0.1.1"
__license__ = "MIT"


import numpy as np
from exampleHelmholtz_scatter import StatROM_2D


class UserParameters:
    def __init__(self):
        pass
    f = 400#Hz      # frequency
    s = 250#Hz      # AORA expansion frequency
    m = 12#8#12          # number of matched moments in AORA

    #               # the mesh for the model and a finer one for data are hardcoded with gmsh.
    #               #
    n_qmc = 256     # number of QMC sample points. Only powers of two are permitted

    ns = 5          # number of sensors
    no = 20         # number of observations per sensor
    sig_o = 5e-4#Pa # observation noise

    n_est = 200     # number of error estimator training points


if __name__ == '__main__':
    up = UserParameters
    funcs = StatROM_2D(up)
    
    ### Generate data from fine mesh:
    funcs.switchMesh_self("ground_truth")
    funcs.generateParameterSamples(up.n_qmc,save=False,load=True)
    #funcs.generateParameterSamples(up.n_qmc,save=True,load=False)
    funcs.getFullOrderPrior() # solve for "ground truth"
    funcs.get_noisy_data_from_solution(0,np.abs(funcs.u_mean_std)) # extract noisy data at sensors
    ####

    funcs.switchMesh_self("coarse")

    funcs.generateParameterSamples(up.n_qmc,load = False) # get new parameter sample

    # Full order reference:
    funcs.getFullOrderPrior()

    ### Compute ROM basis:
    _, funcs.V_adj_list = funcs.getAORAbasis(Nr=funcs.L,rhs_sp=np.zeros(np.shape(funcs.RBmodel.coordinates_coarse)[0]),adj=True)[0:2] # adjoint solves
    for i,sample in enumerate(funcs.f_samples):
        print("primal sample number: "+str(i))
        funcs.computeROMbasisSample(sample,i)
    ####

    # ROM prior and error estiamte:
    funcs.calcROMprior()

    funcs.plotPriorVtk()

    # FOM reference posterior, classical ROM posterior and proposed ROM posterior:
    funcs.getFullOrderPosterior()
    funcs.getEasyROMPosterior()
    funcs.getCorrectedROMPosterior()

    funcs.plotPosteriorVtk()
