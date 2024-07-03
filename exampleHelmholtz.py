"""
statROM Helmholtz 1D example
"""

__author__ = "Lucas Hermann"
__version__ = "0.1.0"
__license__ = "MIT"


import json
import os
import numpy as np
import scipy
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import qmcpy as qp

import ufl
from ufl import grad, inner, dot
from dolfinx.fem import Function, FunctionSpace, assemble_scalar, form

from mpi4py import MPI

from RB_Solver import RBClass
from AORA import AORA
import kernels

usr_home = os.getenv("HOME")
plt.rcParams.update({
   "pgf.texsystem": "pdflatex",
   "text.usetex": True,
   "font.family": "serif",
   "font.sans-serif": ["Helvetica"]})
#mpl.use('pgf')
plt.rc('text', usetex=True)
plt.rc('text.latex',  preamble=r'\usepackage{amsmath}\usepackage[utf8]{inputenc}\usepackage{bm}')



class StatROM_1D:
    """ 
    Methods to call FEM/ROM solvers and statFEM routines.
    """
    def __init__(self,up):
        self.up = up
        ne = up.n
        self.RBmodel = RBClass(self.up)
        self.RBmodel.problem = "Helmholtz"
        self.reset(ne)


    def reset(self,ne):
        """doc"""
        self.RBmodel.reset(ne=ne)
        self.L = self.up.m  #5   	                    # size of basis
        self.par1 = self.up.f  #460                 # frequency
        self.par2 = 1.15#1.1011236


    def wrapper_AORA(self,rhs_sp=None,mat_par=None):
        self.V_AORA,self.V_adj_list = self.getAORAbasis(Nr=self.L,rhs_sp=rhs_sp,matCoef=mat_par)[0:2]
        self.V_AORA_mean,self.V_adj_list = self.getAORAbasis(Nr=self.L,rhs_sp=np.pi**2/50,matCoef=np.array([0,0,0]))[0:2]


    def getAORAbasis(self,Nr,freq_exp=100,matCoef=None,rhs_sp=None): #freq_exp = 100
        """ Calls the adaptive order rational Arnoldi ROM
            code to compute and store ROM projection matrices V.
            This is basically the majority of the offline part of the method.
        """
        freq_exp = self.up.s
        s0 = 2*np.pi*freq_exp / 343
        if rhs_sp==None:
            rhs_sp=np.pi**2/50
        self.RBmodel.doFEMHelmholtz(freq = self.par1,rhsPar=rhs_sp,mat_par=matCoef)
        V_AORA,_,LinSysFac,_,_= AORA(self.RBmodel.Msp.todense(),self.RBmodel.Dsp.todense(),self.RBmodel.KKsp.todense(),self.RBmodel.FFnp,self.RBmodel.C,[s0],Nr)
        P_error_est = self.RBmodel.getP(self.y_points_error_est)
        V_adj_list = [AORA(self.RBmodel.Msp.todense().T,self.RBmodel.Dsp.todense().T,self.RBmodel.KKsp.todense().T,P_error_est[i],self.RBmodel.C,[s0],Nr,LinSysFac=LinSysFac)[0] for i in range(np.shape(P_error_est)[0])]

        self.nAora = Nr
        return V_AORA,V_adj_list,Nr


    def generateParameterSamples(self,n_samp):   
        """ sample the parameter space using QMC """ 
        diag_all = [1.0 for _ in range(3)] # material parameter, standard normal, goes into KLE later
        diag_all.insert(0,0.02**2)  # right hand side, Neumann
        g_all = qp.true_measure.gaussian.Gaussian(qp.discrete_distribution.digital_net_b2.digital_net_b2.DigitalNetB2(dimension=4),mean=[np.pi**2/50,0.0,0.0,0.0],covariance=np.diag(diag_all))
        all_par_samples = g_all.gen_samples(n_samp)
        self.f_samples = all_par_samples[:,0]
        self.par_samples = all_par_samples[:,1:]

        return 0


    def saveBasis(self,V,V_adj,i):
        with open('Results/bases/basisAORA1D_sample'+str(i)+'.npy', 'wb') as fileArray:
            np.save(fileArray,V)
        with open('Results/bases/basisAdj1D_sample'+str(i)+'.npy', 'wb') as fileArray:
            np.save(fileArray,np.array(V_adj))
        with open('Results/bases/rom_mean_basis.npy', 'wb') as fileArray:
            np.save(fileArray,self.V_AORA_mean)


    def computeROMbasisSample(self,sample,i):
        self.wrapper_AORA(rhs_sp=sample,mat_par=self.par_samples[i])
        self.saveBasis(self.V_AORA,self.V_adj_list,i)


    def loadBasis(self,i):
        with open('Results/bases/basisAORA1D_sample'+str(i)+'.npy', 'rb') as fileArray:
               self.V_AORA = np.load(fileArray)
        with open('Results/bases/basisAdj1D_sample'+str(i)+'.npy', 'rb') as fileArray:
               self.V_adj_list = np.load(fileArray)
        with open('Results/bases/rom_mean_basis.npy', 'rb') as fileArray:
               self.V_AORA_mean = np.load(fileArray)


    def calcROMprior(self):
        """ Computes the prior mean and covariance for the ROM""" 
        u_rom = []
        d_rom = []
        for i,sample in enumerate(self.f_samples):
            self.loadBasis(i)
            u = self.RBmodel.getPriorAORA(self.V_AORA,[self.par1,sample,self.par_samples[i]])
            u_rom.append(u)
            dr = self.romErrorEst([self.par1,sample,self.par_samples[i]],ur=u,multiple_bases=True)
            d_rom.append(dr)
    
        C_u = np.cov(np.array(u_rom), rowvar=0, ddof=0)
        self.u_mean = np.mean(np.array(u_rom),axis=0)
        self.dr_mean, self.dr_cov = self.errorGPnoisy(d_rom,self.y_points_error_est)
        ident = np.identity(np.shape(C_u)[0])
        self.C_u = C_u + 9e-11*ident #-4
        
        self.C_uDiag = np.sqrt(np.diagonal(self.C_u))

        return 0



    def get_noisy_data_from_solution(self,n_sens,solution):
        """ Computes a data generating solution and samples noisy data from it at sensor points.
            Also handles the error estimator training points positions.    
        """
        n_sens = self.up.ns
        self.n_sens = n_sens

        size_fine = np.shape(self.RBmodel.coordinates)[0]-1
        idx = np.round(np.linspace(0, size_fine, n_sens)).astype(int)
        n_error_est = self.up.n_est 
        idx_error_est = np.round(np.linspace(1, self.RBmodel.ne, n_error_est)).astype(int)

        self.y_points = [self.RBmodel.coordinates.tolist()[i] for i in idx] # sensor locations

        self.y_points_error_est = [self.RBmodel.coordinates_coarse.tolist()[i] for i in idx_error_est] # training points for the error estimator

        values_at_indices = [solution[x]+0.0 for x in idx]
        n_obs = self.up.no
        self.n_obs = n_obs
        self.RBmodel.no = n_obs
        y_values_list = []
        self.RBmodel.get_C_f()

        for i in range(n_obs):
            y_values_list.append([x+np.random.normal(0,self.up.sig_o) for j,x in enumerate(values_at_indices)]) #1e-1
        a = np.array(y_values_list)
        self.y_values_list = a.tolist()
        self.true_process = solution


    def romErrorEst(self,par,ur = None,multiple_bases = False):
        """ Provides a cheap estimate for the ROM error
            using evaluations of the adoint solution at
            given points throughout the domain and a
            GP regression.
        """

        P = self.RBmodel.getP(self.y_points_error_est)
        V_state = self.V_AORA
        
        _, A, _ = self.RBmodel.doFEMHelmholtz(freq=par[0],rhsPar=par[1],mat_par=par[2],assemble_only=True)
        
        f_rhs = self.RBmodel.FFnp.copy()
        ai, aj, av = A.getValuesCSR()
        Asp = csr_matrix((av, aj, ai))
        A = Asp.todense()
        Ar = self.RBmodel.getAr(V_state,A)
        if isinstance(ur,type(None)):
            ur = self.u_mean
        else: 
            ur=ur
        f = f_rhs

        residual = f - A@ur
        dr=[]
        dr_exact = []
        for i in range(np.shape(P)[0]):
            V = self.V_adj_list[i]
            A_r_T = np.transpose(V)@np.transpose(A)@V
            zr = np.linalg.solve(A_r_T,(V.T)@P[i])
            z_est = V@zr
            z_exact = np.linalg.solve(A.T,P[i])
            dr_i = z_est.T@np.array(residual)[0]
            dr_i_exact = z_exact.T@np.array(residual)[0]
            dr.append(dr_i)
            dr_exact.append(dr_i_exact)

        self.dr_est = dr
        self.dr_ex  = dr_exact
        reference = np.copy(self.dr_ex)
        solution = self.dr_est
        print(np.linalg.norm(reference-solution)/np.linalg.norm(reference))
        if multiple_bases == False:
            self.dr_mean, self.dr_cov = self.errorGP(dr,self.y_points_error_est)
        else:
            return dr


    def errorGP(self,dr,y_points):
        """ Computes the GP regression for the ROM error
            given the approximative adjoint solution for
            the error at given points.
        """
        with open('./Results/dr.npy', 'wb') as fileArray:
            np.save(fileArray,dr)
        ident_coord = np.identity(len(self.RBmodel.coordinates))
        y_points = np.array(y_points)

        l = 343/self.par1/4
        sig = np.max(np.abs(dr))*2

        prior_cov = kernels.matern52(self.RBmodel.coordinates,self.RBmodel.coordinates,lf=l,sigf=sig) +1e-15*ident_coord
        noise_std = np.max(np.abs(dr))/10

        noise = noise_std**2 * np.identity(np.shape(y_points)[0])
        data_kernel = kernels.matern52(y_points,y_points,lf=l,sigf=sig)+noise#+1e-15*ident_y
        mixed_kernel = kernels.matern52(y_points,self.RBmodel.coordinates,lf=l,sigf=sig)

        solved = scipy.linalg.solve(data_kernel, mixed_kernel, assume_a='pos').T
        post_mean = solved @ dr
        post_cov = prior_cov - (solved@mixed_kernel)

        ys_post = np.random.multivariate_normal(
        mean=np.real(post_mean), cov=np.real(post_cov), 
        size=5)
        fig = plt.figure(figsize=(6, 4))
        for i in range(5):
            plt.plot(self.RBmodel.coordinates, ys_post[i], linestyle='--')
        plt.plot(self.RBmodel.coordinates, post_mean, linestyle='-')
        plt.scatter(y_points,dr)
        plt.xlabel('$x$', fontsize=13)
        plt.ylabel('$y = f(x)$', fontsize=13)
        plt.title((
            'estimated ROM error'))
        fig.savefig("Results/error_GP.pdf", bbox_inches='tight')

        with open('Results/dr_cov.npy', 'wb') as fileArray:
            np.save(fileArray,post_cov)

        return post_mean,post_cov
    

    def errorGPnoisy(self,dr,y_points):
        """ Computes the GP regression for the ROM error
            given the approximative adjoint solution for
            the error at given points, 
            variant for with multiple samples of d_r
        """
        n_obs = np.shape(dr)[0]
        dr_mean = np.mean(np.real(dr),axis=0)
        dr_sum = np.sum(np.real(dr),axis=0)
        with open('./Results/dr.npy', 'wb') as fileArray:
            np.save(fileArray,dr_mean)
        dr_cov = np.cov(np.real(dr), rowvar=0, ddof=0)
        ident_coord = np.identity(len(self.RBmodel.coordinates))
        y_points = np.array(y_points)

        # simple way to choose hyperparameters
        l = 343/self.par1/1
        sig = np.max(np.abs(dr))*1.0

        prior_cov = kernels.periodic(self.RBmodel.coordinates,self.RBmodel.coordinates,lf=l,sigf=sig,p=2.5) +1e-15*ident_coord

        data_kernel = kernels.periodic(y_points,y_points,lf=l,sigf=sig,p=2.5)*n_obs+dr_cov#+1e-8*ident_y
        mixed_kernel = kernels.periodic(y_points,self.RBmodel.coordinates,lf=l,sigf=sig,p=2.5) 

        solved = scipy.linalg.solve(data_kernel, mixed_kernel).T
        post_mean = solved @ dr_sum
        post_cov = prior_cov - n_obs*(solved@mixed_kernel)

        with open('Results/dr_cov.npy', 'wb') as fileArray:
            np.save(fileArray,post_cov)

        return post_mean,post_cov


    def getEasyROMPosterior(self):
        """ compute the statFEM posterior in the classical way.
            The ROM prior is used but no correction terms.
        """
        print("Classical ROM posterior START")
        (u_mean_y_easy, C_u_y_easy, postGP) = self.RBmodel.computePosteriorMultipleY(self.y_points,self.y_values_list,self.u_mean,self.C_u)
        self.C_u_y_easy_Diag = np.sqrt(np.diagonal(C_u_y_easy))
        self.d_ROM = np.copy(np.diag(self.RBmodel.C_d_total))
        self.sigd_easy = np.copy(self.RBmodel.sigd)
        print("Classical ROM posterior FINISH")
        self.u_mean_y_easy, self.C_u_y_easy = u_mean_y_easy, C_u_y_easy
        return u_mean_y_easy, C_u_y_easy


    def getCorrectedROMPosterior(self):
        """ compute the statFEM posterior in the new way, as proposed in our paper.
            The ROM prior is used with correction terms.
        """
        print("Corrected ROM posterior START")
        (u_mean_y, C_u_y, u_mean_y_pred_rom, postGP) = self.RBmodel.computePosteriorROM(self.y_points,self.y_values_list,self.u_mean,self.C_u,self.dr_mean,self.dr_cov)
        self.C_u_y_Diag = np.sqrt(np.diagonal(C_u_y))
        print("Corrected ROM posterior FINISH")
        self.u_mean_y, self.C_u_y = u_mean_y, C_u_y
        self.u_mean_y_pred_rom = u_mean_y_pred_rom
        self.sigd_adv = np.copy(self.RBmodel.sigdROM)
        return u_mean_y, C_u_y


    def getFullOrderPrior(self):
        # prior calculation Full Order Model
        u = []
        for i,samp in enumerate(self.f_samples):
            _, _, ui  = self.RBmodel.doFEMHelmholtz(freq=self.par1,rhsPar=samp,mat_par=self.par_samples[i])
            u.append(ui)
        
        u_mean_std = np.mean(np.array(u),axis=0)
        C_u_std = np.cov(np.array(u), rowvar=0, ddof=0)
        ident = np.identity(np.shape(C_u_std)[0])
        C_u_std = C_u_std + 9e-11*ident  #-4
        self.C_u_stdDiag = np.sqrt(np.diagonal(C_u_std))

        self.u_mean_std,self.C_u_std = u_mean_std,C_u_std
        return u_mean_std,C_u_std


    def getFullOrderPosterior(self):
        """ Solve the full order posterior counterpart for reference
        """
        print("Full Order START")
        (u_mean_y_std, C_u_y_std, postGP) = self.RBmodel.computePosteriorMultipleY(self.y_points,self.y_values_list,self.u_mean_std,self.C_u_std)
        self.C_u_y_std_Diag = np.sqrt(np.diagonal(C_u_y_std))
        self.d_FEM = np.copy(np.diag(self.RBmodel.C_d_total))
        self.sigd_std = np.copy(self.RBmodel.sigd)
        print("Full Order FINISH")
        self.u_mean_y_std, self.C_u_y_std = u_mean_y_std, C_u_y_std
        return u_mean_y_std, C_u_y_std


    def computeErrorNorm(self,solution,reference):
        bar_p_sq = 0
        for p in solution:
            val = np.sqrt(np.abs(p*p))
            bar_p_sq+=val
        bar_p_sq = bar_p_sq/np.shape(solution)[0]
        mean_p = bar_p_sq

        degree_raise = 1
        uh = Function(self.RBmodel.V)
        uh.x.array[:] = np.real(solution)
        u_ex = Function(self.RBmodel.V_ground_truth)
        u_ex.x.array[:] = np.real(reference)
        # Create higher order function space
        degree = uh.function_space.ufl_element().degree()
        family = uh.function_space.ufl_element().family()
        mesh = uh.function_space.mesh
        W = FunctionSpace(mesh, (family, degree+degree_raise))
        # Interpolate approximate solution
        u_W = Function(W)
        u_W.interpolate(uh)

        # Interpolate exact solution, special handling if exact solution
        # is a ufl expression or a python lambda function
        u_ex_W = Function(W)
        u_ex_W.interpolate(u_ex)
        
        # Compute the error in the higher order function space
        e_W = Function(W)
        e_W.x.array[:] = u_W.x.array - u_ex_W.x.array
        
        # Integrate the error
        error = form(inner(e_W, e_W) * ufl.dx)
        error_H10 = form(dot(grad(e_W), grad(e_W)) * ufl.dx)
        error_local = assemble_scalar(error)
        error_local_H10 = assemble_scalar(error_H10)
        error_global = mesh.comm.allreduce(error_local, op=MPI.SUM)
        error_global_H10 = mesh.comm.allreduce(error_local_H10, op=MPI.SUM)

        k = 2 * np.pi * self.par1 / 343
        error_sobolev = np.sqrt(error_global) + k**(-2)*np.sqrt(error_global_H10)

        u = form(inner(u_ex_W, u_ex_W) * ufl.dx)
        u_H10 = form(dot(grad(u_ex_W), grad(u_ex_W)) * ufl.dx)
        u_local = assemble_scalar(u)
        u_local_H10 = assemble_scalar(u_H10)
        u_global = mesh.comm.allreduce(u_local, op=MPI.SUM)
        u_global_H10 = mesh.comm.allreduce(u_local_H10, op=MPI.SUM)

        u_sobolev = np.sqrt(u_global) + k**(-2)*np.sqrt(u_global_H10)

        return error_sobolev, mean_p



    def switchMesh(self,grade):
        if grade == "ground_truth":
            funcs.RBmodel.msh = funcs.RBmodel.msh_ground_truth
            funcs.RBmodel.V = funcs.RBmodel.V_ground_truth
            funcs.RBmodel.coordinates = funcs.RBmodel.coordinates_ground_truth

        if grade == "coarse":
            funcs.RBmodel.msh = funcs.RBmodel.msh_coarse
            funcs.RBmodel.V = funcs.RBmodel.V_coarse
            funcs.RBmodel.coordinates = funcs.RBmodel.coordinates_coarse


    def switchMesh_self(self,grade):
        if grade == "ground_truth":
            self.RBmodel.msh = self.RBmodel.msh_ground_truth
            self.RBmodel.V = self.RBmodel.V_ground_truth
            self.RBmodel.coordinates = self.RBmodel.coordinates_ground_truth

        if grade == "coarse":
            self.RBmodel.msh = self.RBmodel.msh_coarse
            self.RBmodel.V = self.RBmodel.V_coarse
            self.RBmodel.coordinates = self.RBmodel.coordinates_coarse



    def saveData(self):
        data = {'number of elements': self.RBmodel.ne,
                'number of sensors': self.n_sens,
                'number of observations': self.RBmodel.no,
                'freq': self.par1,
                'rho': self.par2,
                'size of basis': self.L
        }

        with open('./Results/parameters.txt', 'w') as filePar:
            json.dump(data, filePar, sort_keys = True, indent = 4,
            ensure_ascii = False)

        with open('./Results/solutions.npy', 'wb') as fileArray:
            np.save(fileArray, self.RBmodel.coordinates)

            np.save(fileArray, self.y_points)
            np.save(fileArray, self.y_values_list)

            np.save(fileArray, self.u_mean)
            np.save(fileArray, self.C_uDiag)

            np.save(fileArray, self.u_mean_std)
            np.save(fileArray, self.C_u_stdDiag)

            np.save(fileArray, self.u_mean_y)
            np.save(fileArray, self.C_u_y_Diag)

            np.save(fileArray, self.u_mean_y_std)
            np.save(fileArray, self.C_u_y_std_Diag)

            np.save(fileArray, self.u_mean_y_easy)
            np.save(fileArray, self.C_u_y_easy_Diag)
            
            np.save(fileArray, self.u_mean_y_pred_rom)
            np.save(fileArray, self.dr_mean)
        np.savez('./Results/solutions.npz', coordinates = self.RBmodel.coordinates, y_points = self.y_points, u_mean = self.u_mean, C_uDiag = self.C_uDiag, \
            u_mean_std = self.u_mean_std, C_u_stdDiag = self.C_u_stdDiag, u_mean_y = self.u_mean_y, C_u_y_Diag = self.C_u_y_Diag, u_mean_y_std = self.u_mean_y_std, \
             C_u_y_std_Diag = self.C_u_y_std_Diag, u_mean_y_easy = self.u_mean_y_easy,  C_u_y_easy_Diag = self.C_u_y_easy_Diag, u_mean_y_pred_rom = self.u_mean_y_pred_rom, \
               dr_mean = self.dr_mean )
        with open('./Results/y_points_rom_est.npy', 'wb') as fileArray:
            np.save(fileArray,self.y_points_error_est)



    def plotPriorPosteriorComparison(self,prior=True,posterior=True):
        fig = plt.figure(figsize=(6,3), dpi=300)
        
        if prior == True:
            plt.plot(self.RBmodel.coordinates, self.u_mean_std, linestyle='-', color = 'red',lw = 1.0, label='Prior FEM')
            plt.fill_between(np.transpose(self.RBmodel.coordinates)[0], np.transpose(self.u_mean_std)+1.96*self.C_u_stdDiag, np.transpose(self.u_mean_std)-1.96*self.C_u_stdDiag,color = 'red',alpha=0.1)

            plt.plot(self.RBmodel.coordinates, self.u_mean, linestyle='-', color = 'green',lw = 1.0, label='Prior ROM')
            plt.fill_between(np.transpose(self.RBmodel.coordinates)[0], np.transpose(self.u_mean)+1.96*self.C_uDiag, np.transpose(self.u_mean)-1.96*self.C_uDiag,color = 'green',alpha=0.1)
        if posterior == True:
            plt.plot(self.RBmodel.coordinates, np.transpose(self.u_mean_y_std), linestyle='-', color = 'blue',lw = 1.5,label='Posterior mean FEM')
            plt.fill_between(np.transpose(self.RBmodel.coordinates)[0], np.transpose(self.u_mean_y_std)+1.96*self.C_u_y_std_Diag, np.transpose(self.u_mean_y_std)-1.96*self.C_u_y_std_Diag,color = 'blue',alpha=0.1)

            plt.fill_between(np.transpose(self.RBmodel.coordinates)[0], np.transpose(self.u_mean_y_easy)+1.96*self.C_u_y_easy_Diag, np.transpose(self.u_mean_y_easy)-1.96*self.C_u_y_easy_Diag,color = 'goldenrod',alpha=0.1,label='$2\sigma$ Posterior w/o ROM error')
            plt.plot(self.RBmodel.coordinates, np.transpose(self.u_mean_y_easy), linestyle='-', color = 'goldenrod',lw = 1.5,label='Posterior mean ROM w/o rom error')
            plt.fill_between(np.transpose(self.RBmodel.coordinates)[0], np.transpose(self.u_mean_y_pred_rom)+1.96*self.C_u_y_Diag, np.transpose(self.u_mean_y_pred_rom)-1.96*self.C_u_y_Diag,color = 'purple',alpha=0.1,label='$2\sigma$ Posterior with ROM error')
            plt.plot(self.RBmodel.coordinates, np.transpose(self.u_mean_y_pred_rom), linestyle='--', color = 'purple',lw = 1.5,label='Posterior mean ROM with rom error')
            
        print("proposed statROM on ROM prior posterior error:")
        norm,_ = self.computeErrorNorm(self.u_mean_y_pred_rom,self.true_process)
        print(norm)
        print("classical statFEM on ROM prior posterior error:")
        norm,_ = self.computeErrorNorm(self.u_mean_y_easy,self.true_process)
        print(norm)
        print("classical statFEM on FEM prior posterior error (reference):")
        norm,_ = self.computeErrorNorm(self.u_mean_y_std,self.true_process)
        print(norm)

        plt.plot(self.RBmodel.coordinates_ground_truth, np.transpose(self.true_process), linestyle='--', color = 'blue',lw = 1.0,label=r'ground truth $\bar{\bm{u}}_\mathrm{ref}$')

        for obs in self.y_values_list:
            plt.scatter(self.y_points, obs,s=5, color = 'red',alpha=0.6)
        plt.ylabel("$pressure [Pa]$")
        plt.xlabel("$x$")

        plt.grid()
        plt.legend()
        fig.savefig("./Results/End_result_statFEM.pdf", bbox_inches='tight')
        plt.close(fig)		



    def plotRomError(self):
        fig = plt.figure(figsize=(8,4), dpi=100)
        plt.plot(self.RBmodel.coordinates, np.transpose(self.u_mean_std - self.u_mean), linestyle='-', color = 'black',lw = 1.5,label='exact')
        plt.plot(self.RBmodel.coordinates, np.transpose(self.dr_mean), linestyle='-', color = 'purple',lw = 1.5,label='estimate')
        plt.fill_between(np.transpose(self.RBmodel.coordinates)[0], np.transpose(self.dr_mean)+1.96*np.sqrt(np.diagonal(self.dr_cov)), np.transpose(self.dr_mean)-1.96*np.sqrt(np.diagonal(self.dr_cov)),color = 'purple',alpha=0.1,label='$2\sigma$ estimate')
        plt.grid()
        plt.legend()
        fig.savefig("./Results/ROMerror.pdf", bbox_inches='tight')
 


    

#end of file