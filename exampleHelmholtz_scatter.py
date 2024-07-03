"""
statROM Helmholtz 2D scattering example
"""

__author__ = "Lucas Hermann"
__version__ = "0.1.0"
__license__ = "MIT"

import os
import numpy as np
import scipy
import qmcpy as qp
import matplotlib.pyplot as plt
from RB_Solver_scatter import RBClass

from AORA import AORA
import kernels

import ufl
from dolfinx.fem import Function, FunctionSpace, assemble_scalar, form
from mpi4py import MPI
from ufl import grad, inner, dot
usr_home = os.getenv("HOME")
plt.rcParams.update({
   "pgf.texsystem": "pdflatex",
   "text.usetex": True,
   "font.family": "serif",
   "font.sans-serif": ["Helvetica"]})
plt.rc('text', usetex=True)
plt.rc('text.latex',  preamble=r'\usepackage{amsmath}\usepackage[utf8]{inputenc}')



class StatROM_2D:
    """ 
    Methods to call FEM/ROM solvers and statFEM routines.
    """
    def __init__(self,up):
        self.up = up
        self.RBmodel = RBClass(self.up)
        self.RBmodel.problem = "Helmholtz"
        self.reset()


    def reset(self):
        """Enables simple re-initialisation during convergence studies"""
        self.RBmodel.reset()
        self.RBmodel.lowrank = False # use lowrank statfem
        self.L=self.up.m#12#16		         # size of basis
        self.par1 = self.up.f#360#366          # frequency
        self.par2 = 1.15#1.1011236  



    def snapshotsHandling(self,rhs_sp=None):
        self.V_AORA,_ = self.getAORAbasis(Nr=self.L,rhs_sp=rhs_sp)[0:2]
        self.V_AORA_mean = self.getAORAbasis(Nr=self.L,rhs_sp=np.zeros(np.shape(self.RBmodel.coordinates)[0]))[0]


    def getAORAbasis(self,Nr,freq_exp=250,matCoef=None,rhs_sp=None,adj=False):
        """ Calls the adaptive order rational Arnoldi ROM
            code to compute and store ROM projection matrices V.
            This is basically the majority of the offline part of the method.
        """
        freq_exp = self.up.s
        s0 = 2*np.pi*freq_exp / 340 # expansion frequency expressed as wave number
        self.RBmodel.doFEMHelmholtz(freq = self.par1,rhsPar=rhs_sp)
        V_AORA,_,LinSysFac,_,_= AORA(self.RBmodel.Msp,self.RBmodel.Dsp,self.RBmodel.KKsp,self.RBmodel.FFnp,self.RBmodel.C,[s0],Nr)
        V_adj_list = []

        # compute the projection matrices for the error estimator. Previous LU decomp. is reused in LinSysFac
        if adj == True:
            self.RBmodel.doFEMHelmholtz(freq = self.par1,rhsPar=np.zeros(np.shape(self.RBmodel.coordinates)[0]))
            P_error_est = self.RBmodel.getP(self.y_points_error_est)
            V_adj_list = []
            for i in range(np.shape(P_error_est)[0]):
                V_adj_list.append(AORA(self.RBmodel.Msp.conj().T,self.RBmodel.Dsp.conj().T,self.RBmodel.KKsp.conj().T,P_error_est[i],self.RBmodel.C,[s0],Nr,LinSysFac=LinSysFac)[0])
                print("adjoint sample nr. "+str(i))
            #V_adj_list = [AORA(self.RBmodel.Msp.conj().T,self.RBmodel.Dsp.conj().T,self.RBmodel.KKsp.conj().T,P_error_est[i],self.RBmodel.C,[s0],Nr,LinSysFac=LinSysFac)[0] for i in range(np.shape(P_error_est)[0])]
        self.nAora = Nr

        return V_AORA,V_adj_list,Nr


    def generateParameterSamples(self,n_samp,load=False,save=False):
        """ sample the parameter space using QMC """
        cov = kernels.matern52
        dim = np.shape(self.RBmodel.coordinates)[0]
        cov_re = cov(self.RBmodel.coordinates,self.RBmodel.coordinates,lf=0.6,sigf=0.8)+np.identity(dim)*1e-6
        cov_im = cov(self.RBmodel.coordinates,self.RBmodel.coordinates,lf=0.6,sigf=0.8)+np.identity(dim)*1e-6
        cov_block = np.block([
        [cov_re,               np.zeros((dim, dim))],
        [np.zeros((dim, dim)), cov_im               ]
        ])
        g_all = qp.true_measure.gaussian.Gaussian(qp.discrete_distribution.digital_net_b2.digital_net_b2.DigitalNetB2(dimension=2*dim),mean=np.zeros(2*dim),covariance=cov_block)
        all_par_samples = g_all.gen_samples(n_samp)
        self.f_samples = all_par_samples[:,0:dim]
        self.f_samples_im = all_par_samples[:,dim:]

        self.het_samples = np.random.normal(loc=0.0, scale=1e-16,size=n_samp)

        if load == True:
            with open('Results/f_samples.npy', 'rb') as fileArray:
                self.f_samples = np.load(fileArray)
            with open('Results/f_samples_im.npy', 'rb') as fileArray:
                self.f_samples_im = np.load(fileArray)

        if save == True:
            with open('Results/f_samples.npy', 'wb') as fileArray:
                np.save(fileArray,self.f_samples)
            with open('Results/f_samples_im.npy', 'wb') as fileArray:
                np.save(fileArray,self.f_samples_im)



    def saveBasis(self,V,i):
        with open('Results/bases/basisAORA1D_sample'+str(i)+'.npy', 'wb') as fileArray:
            np.save(fileArray,V)
        try:
            self.V_adj_list
            with open('Results/bases/basisAdj1D_sample'+str(i)+'.npy', 'wb') as fileArray:
                np.save(fileArray,np.array(self.V_adj_list))
        except:
            print("Adjoint ROM basis wasn't computed! Has to be loaded.")

        with open('Results/bases/rom_mean_basis.npy', 'wb') as fileArray:
            np.save(fileArray,self.V_AORA_mean)



    def computeROMbasisSample(self,sample,i):
        self.snapshotsHandling(rhs_sp=sample+1j*self.f_samples_im[i])
        self.saveBasis(self.V_AORA,i)


    def loadBasis(self,i):
        with open('Results/bases/basisAORA1D_sample'+str(i)+'.npy', 'rb') as fileArray:
               self.V_AORA = np.load(fileArray)
        with open('Results/bases/basisAdj1D_sample'+str(i)+'.npy', 'rb') as fileArray:
               self.V_adj_list = np.load(fileArray)
        with open('Results/bases/rom_mean_basis.npy', 'rb') as fileArray:
               self.V_AORA_mean = np.load(fileArray)


    def calcROMprior(self,multiple_bases = False):
        """ Computes the prior mean and covariance for the ROM""" 
        if multiple_bases == False:
            (self.u_mean, self.C_u) = self.RBmodel.getPriorAORA(self.V_AORA,[self.par1,1])
            C_u = self.RBmodel.get_C_u_ROM_MC(self.par1,self.V_AORA)
            self.u_mean = np.real(self.u_mean)
            C_u = np.real(C_u)
        else:
            u_rom = []
            u_rom_unified = []
            u_rom_real = []
            u_rom_imag = []
            self.loadBasis(0)
            self.rom_mean = self.RBmodel.getPriorAORA(self.V_AORA_mean,[self.par1,np.zeros(np.shape(self.RBmodel.coordinates)[0]),self.het_samples[0]])[0]
            for i,sample in enumerate(self.f_samples):
                self.loadBasis(i)
                u_sc,_ = self.RBmodel.getPriorAORA(self.V_AORA,[self.par1,sample+1j*self.f_samples_im[i],self.het_samples[i]])
                u_rom.append(u_sc)
                u_rom_real.append(np.real(u_sc))
                u_rom_imag.append(np.imag(u_sc))
                u_rom_unified.append(np.concatenate([np.real(u_sc),np.imag(u_sc)]))

            self.u_mean_real = np.mean(np.array(u_rom_real),axis=0)
            self.u_mean_imag = np.mean(np.array(u_rom_imag),axis=0)
            self.u_mean = np.mean(np.array(u_rom),axis=0)
            C_u = np.cov(np.array(u_rom_unified), rowvar=0, ddof=0)
            C_u_approx = np.cov(np.array(u_rom), rowvar=0, ddof=0)
            C_u_real = np.cov(np.array(u_rom_real), rowvar=0, ddof=0)
            C_u_imag = np.cov(np.array(u_rom_imag), rowvar=0, ddof=0)
          
            self.romErrorEst([self.par1,np.zeros(np.shape(self.RBmodel.coordinates)[0])],ur=self.u_mean,Cu=C_u_approx,Cu_real=C_u_real,Cu_imag=C_u_imag,multiple_bases=False)
            
          
        ident = np.identity(np.shape(C_u)[0])
        self.C_u = C_u + 9e-8*ident
        ident_small = np.identity(np.shape(C_u_real)[0])
        self.C_u_real = C_u_real + 9e-8*ident_small
        self.C_u_imag = C_u_imag + 9e-8*ident_small
        
        self.C_uDiag = np.sqrt(np.diagonal(self.C_u))
        self.C_uDiag_real = np.sqrt(np.diagonal(self.C_u_real))
        self.C_uDiag_imag = np.sqrt(np.diagonal(self.C_u_imag))
        self.C_uDiag = self.C_uDiag_real + 1j*self.C_uDiag_imag

        return



    def get_noisy_data_from_solution(self,n_sens,solution):
        """ Computes a data generating solution and samples noisy data from it at sensor points.
            Also handles the error estimator training points positions.    
        """
        n_sens = self.up.ns#5
        solution =solution*0.999
        solution = self.RBmodel.solveFEMData(sample=self.f_samples[0]+1j*self.f_samples_im[0])[2]
  #      with open('./Results/data_vector.npy', 'rb') as fileArray:
  #          solution = np.load(fileArray)
        import dolfinx.io
        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "./Results/data.xdmf", "w") as xdmf:
            xdmf.write_mesh(self.RBmodel.msh)
            ut = dolfinx.fem.Function(self.RBmodel.V)
            ut.x.array[:] = solution
            xdmf.write_function(ut)
        
        with open('./Results/data_vector.npy', 'wb') as fileArray:
            np.save(fileArray, solution)
            
        self.data_solution = solution
        size_fine = np.shape(self.RBmodel.coordinates_ground_truth)[0]-1
        size_coarse = np.shape(self.RBmodel.coordinates_coarse)[0]-4

        idx = np.round(np.linspace(0, size_fine, n_sens)).astype(int) #854
        n_error_est = self.up.n_est#200
        idx_error_est = np.round(np.linspace(0, size_coarse, n_error_est)).astype(int)
        idx_boundary = self.RBmodel.dofs_dirichl_coarse
        self.num_boundary = np.shape(idx_boundary)[0]

        self.y_points = [self.RBmodel.coordinates.tolist()[i] for i in idx]#*2
       
        idx_total = np.unique(np.concatenate([idx_error_est,idx_boundary]))
        self.y_points_error_est = [self.RBmodel.coordinates_coarse.tolist()[i] for i in idx_total]
      
        #funcs.y_points = self.y_points

        values_at_indices = [solution[x]+0.0 for x in idx]

        n_obs = self.up.no#20
        self.RBmodel.no = n_obs
        y_values_list = []
        y_values_list_real = []
        y_values_list_imag = []

 
        for i in range(n_obs):
            y_values_list.append([np.abs(x)+np.random.normal(0,self.up.sig_o) for j,x in enumerate(values_at_indices)]) #5e-4
            y_values_list_real.append([np.real(x)+np.random.normal(0,self.up.sig_o) for j,x in enumerate(values_at_indices)])
            y_values_list_imag.append([np.imag(x)+np.random.normal(0,self.up.sig_o) for j,x in enumerate(values_at_indices)])
        y_values_list_unified = np.block([np.array(y_values_list_real),np.array(y_values_list_imag)])
        a = np.array(y_values_list)
        self.y_values_list = a.tolist()
        #funcs.y_values_list = self.y_values_list

        self.y_values_list_real = np.array(y_values_list_real).tolist()
        #funcs.y_values_list_real = self.y_values_list_real

        self.y_values_list_imag = np.array(y_values_list_imag).tolist()
        #funcs.y_values_list_imag = self.y_values_list_imag

        self.y_values_list_unified = y_values_list_unified.tolist()
        #funcs.y_values_list_unified = self.y_values_list_unified

        self.true_process = solution



    def romErrorEst(self,par,ur = None,Cu=None,Cu_real=None,Cu_imag=None,multiple_bases = False):
        """ Provides a cheap estimate for the ROM error
            using evaluations of the adoint solution at
            given points throughout the domain and a
            GP regression.
        """

        P = self.RBmodel.getP(self.y_points_error_est)
        V_mean = self.V_AORA_mean
        
        _, A, _ = self.RBmodel.doFEMHelmholtz(par[0],par[1],assemble_only=True)
        A_r = V_mean.conj().T@A@V_mean
        ident = np.identity(np.shape(A_r)[0])
        VAr_inv = V_mean @ np.linalg.solve(A_r,ident)
        f = self.RBmodel.FFnp.copy()
        A = A.todense()

        if isinstance(ur,type(None)):
            ur = self.u_mean
        else: 
            ur=ur

        residual = f - A@(ur)
        dr=[]
        dr_var = []
        Cf = kernels.matern52(self.RBmodel.coordinates,self.RBmodel.coordinates,lf=0.6,sigf=0.8)
        Cf = self.RBmodel.get_C_f() +np.identity(np.shape(A)[0])*1e-10
        Cf = Cf+ 1j*Cf

        #Cf = np.cov(self.f_samples.T+1j*self.f_samples_im.T)


        for i in range(np.shape(P)[0]):
            V = self.V_adj_list[i]
            A_r_T = V.conj().T@A.conj().T@V
            zr = np.linalg.solve(A_r_T,(V.conj().T)@P[i])
            z_est = V@zr
            dr_i = z_est.conj().T@np.array(residual)[0]
            dr.append(dr_i)
            #VAr_inv = V @ np.linalg.solve(A_r,ident)  ##neu
            zAVA = z_est.conj().T@A@VAr_inv

       #     Cdr = z_est.conj().T@Cf@z_est + np.array(zAVA)@(V_mean.conj().T@Cf@V_mean)@np.array(zAVA).conj().T
       #     Cdr_real = np.real(z_est).T@Cf@np.real(z_est) + np.real(np.array(zAVA)@V_mean.conj().T)@Cf@np.real(V_mean@np.array(zAVA).conj().T)
       #     Cdr_imag = np.imag(z_est).T@np.imag(Cf)@np.imag(z_est) + np.imag(np.array(zAVA)@V_mean.conj().T)@np.imag(Cf)@np.imag(V_mean@np.array(zAVA).conj().T)
            Cf_approx = A@Cu@A.conj().T
            Cdr = z_est.conj().T@Cf@z_est + z_est.conj().T@Cf_approx@z_est
            Cdr_real = np.real(z_est).conj().T@Cf@np.real(z_est) + np.real(z_est).conj().T@Cu_real@np.real(z_est)
            Cdr_imag = np.imag(z_est).conj().T@Cf@np.imag(z_est) + np.imag(z_est).conj().T@Cu_imag@np.imag(z_est)
            print("Cdr")
            print(Cdr)
            print("Cdr_real")
            print(Cdr_real)
            print(np.real(z_est).T@Cf@np.real(z_est) + np.real(np.array(zAVA)@V_mean.conj().T)@Cf@np.real(V_mean@np.array(zAVA).conj().T))
            
            print("Cdr_imag")
            print(Cdr_imag)
            print(np.imag(z_est).T@np.imag(Cf)@np.imag(z_est) + np.imag(np.array(zAVA)@V_mean.conj().T)@np.imag(Cf)@np.imag(V_mean@np.array(zAVA).conj().T))

            #Cdr = z_est.conj().T@Cf@z_est + np.array(zAVA)@(V.conj().T@Cf@V)@np.array(zAVA).conj().T
          #  Cdr_real = np.real(z_est).T@np.real(Cf)@np.real(z_est) + np.real(np.array(zAVA)@V.conj().T)@np.real(Cf)@np.real(V@np.array(zAVA).conj().T)
          #  Cdr_imag = np.imag(z_est).T@np.imag(Cf)@np.imag(z_est) + np.imag(np.array(zAVA)@V.conj().T)@np.imag(Cf)@np.imag(V@np.array(zAVA).conj().T)
           
            #Cdr_real = np.real(Cdr)
            #Cdr_imag = np.imag(Cdr)
          
           # dr_var.append(Cdr_real + 1j*Cdr_imag)
            dr_var.append(Cdr)

        if multiple_bases == False:
            self.dr_var = np.nan_to_num(np.array(dr_var))
            print(self.dr_var)
          #  self.dr_mean_real, self.dr_cov_real = self.errorGP(np.nan_to_num(np.real(dr)),self.y_points_error_est,dr_vari=np.real(self.dr_var[:,0]))
          #  self.dr_mean_imag, self.dr_cov_imag = self.errorGP(np.nan_to_num(np.imag(dr)),self.y_points_error_est,dr_vari=np.imag(self.dr_var[:,0]))

            self.dr_mean_real, self.dr_cov_real = self.errorGP(np.nan_to_num(np.real(dr)),self.y_points_error_est,dr_vari=np.real(self.dr_var[:,0]))
            self.dr_mean_imag, self.dr_cov_imag = self.errorGP(np.nan_to_num(np.imag(dr)),self.y_points_error_est,dr_vari=np.imag(self.dr_var[:,0]))
        
        else:
            return dr


    def errorGP(self,dr,y_points,dr_vari):
        """ Computes the GP regression for the ROM error
            given the approximative adjoint solution for
            the error at given points.
        """
        dr=[0.0 if np.abs(dri) < 1e-14 else dri for dri in dr]

        ident_coord = np.identity(len(self.RBmodel.coordinates_coarse))
        y_points = np.array(y_points)

        # simple way to find suitable hyperparameters
        l = 340/self.par1/3
        sig = np.max(np.abs(dr))*3#4 #3
        noise = np.diag(dr_vari[:,0])
      #  noise = np.diag(dr_vari)
        # training noise comes pre-computed from the adjoint estimator
        prior_cov = kernels.matern52(self.RBmodel.coordinates_coarse,self.RBmodel.coordinates_coarse,lf=l,sigf=sig) +1e-10*ident_coord #l=0.012
        data_kernel = kernels.matern52(y_points,y_points,lf=l,sigf=sig)+noise#+1e-9*ident_y
        #data_kernel = noise
        data_kernel = 0.5*(data_kernel+data_kernel.T)
        mixed_kernel = kernels.matern52(y_points,self.RBmodel.coordinates_coarse,lf=l,sigf=sig)

        solved = scipy.linalg.solve(data_kernel, mixed_kernel).T
        post_mean = solved @ dr
        post_cov = prior_cov - (solved@mixed_kernel)

        return post_mean,post_cov
    

    def errorGPnoisy(self,dr,y_points):
        dr = np.nan_to_num(dr)
        dr_mean = np.mean(dr,axis=0)
        dr_mean = np.nan_to_num(dr_mean)
        #dr_mean=[0.0 if np.abs(dri) < 1e-12 else dri for dri in dr_mean]

        dr_cov = np.cov(dr, rowvar=0, ddof=0)
        dr_cov = np.nan_to_num(dr_cov)
        ident_coord = np.identity(len(self.RBmodel.coordinates_coarse))
        y_points = np.array(y_points)

        l = 340/self.par1/6
        sig = np.max(np.abs(dr_mean))*0.05

        prior_cov = kernels.matern52(self.RBmodel.coordinates_coarse,self.RBmodel.coordinates_coarse,lf=l,sigf=sig) +1e-15*ident_coord

        data_kernel = kernels.matern52(y_points,y_points,lf=l,sigf=sig)+dr_cov
        mixed_kernel = kernels.matern52(y_points,self.RBmodel.coordinates_coarse,lf=l,sigf=sig)

        solved = scipy.linalg.solve(data_kernel, mixed_kernel).conj().T
        post_mean = solved @ dr_mean
        post_cov = prior_cov - (solved@mixed_kernel)

        return post_mean,post_cov




    def getFullOrderPosterior(self):
        """ Solve the full order posterior counterpart for reference
        """
        print("Full Order START")
        (u_mean_y_std_real, C_u_y_std_real, postGP) = self.RBmodel.computePosteriorMultipleY(self.y_points,self.y_values_list_real,np.real(self.u_mean_std),self.C_u_std_real)
        (u_mean_y_std_imag, C_u_y_std_imag, postGP) = self.RBmodel.computePosteriorMultipleY(self.y_points,self.y_values_list_imag,np.imag(self.u_mean_std),self.C_u_std_imag)
        self.C_u_y_std_Diag_real = np.sqrt(np.diagonal(C_u_y_std_real))
        self.C_u_y_std_Diag_imag = np.sqrt(np.diagonal(C_u_y_std_imag))
        self.C_u_y_std_Diag = self.C_u_y_std_Diag_real + 1j*self.C_u_y_std_Diag_imag
        self.d_FEM = np.copy(np.diag(self.RBmodel.C_d_total))
        print("Full Order FINISH")
        self.u_mean_y_std, self.C_u_y_std = u_mean_y_std_real+1j*u_mean_y_std_imag, C_u_y_std_real+1j*C_u_y_std_imag
        return self.u_mean_y_std, self.C_u_y_std



    def getEasyROMPosterior(self):
        """ compute the statFEM posterior in the classical way.
            The ROM prior is used but no correction terms.
        """
        print("Classical ROM posterior START")
        (u_mean_y_easy_real, C_u_y_easy_real, postGP) = self.RBmodel.computePosteriorMultipleY(self.y_points,self.y_values_list_real,np.real(self.u_mean),self.C_u_real)
        (u_mean_y_easy_imag, C_u_y_easy_imag, postGP) = self.RBmodel.computePosteriorMultipleY(self.y_points,self.y_values_list_imag,np.imag(self.u_mean),self.C_u_imag)
        self.C_u_y_easy_Diag_real = np.sqrt(np.diagonal(C_u_y_easy_real))
        self.C_u_y_easy_Diag_imag = np.sqrt(np.diagonal(C_u_y_easy_imag))
        self.C_u_y_easy_Diag = self.C_u_y_easy_Diag_real + 1j*self.C_u_y_easy_Diag_imag
        self.d_ROM = np.copy(np.diag(self.RBmodel.C_d_total))
        print("Classical ROM posterior FINISH")
        self.u_mean_y_easy, self.C_u_y_easy = u_mean_y_easy_real+1j*u_mean_y_easy_imag, C_u_y_easy_real+1j*C_u_y_easy_imag
        return self.u_mean_y_easy, self.C_u_y_easy

    def getCorrectedROMPosterior(self):
        """ compute the statFEM posterior in the new way, as proposed in our paper.
            The ROM prior is used with correction terms.
        """
        print("Corrected ROM posterior START")
        (u_mean_y_real, C_u_y_real, u_mean_y_pred_rom_real, postGP) = self.RBmodel.computePosteriorROM(self.y_points,self.y_values_list_real,np.real(self.u_mean),self.C_u_real,self.dr_mean_real,self.dr_cov_real)
        (u_mean_y_imag, C_u_y_imag, u_mean_y_pred_rom_imag, postGP) = self.RBmodel.computePosteriorROM(self.y_points,self.y_values_list_imag,np.imag(self.u_mean),self.C_u_imag,self.dr_mean_imag,self.dr_cov_imag)
        self.C_u_y_Diag_real = np.sqrt(np.diagonal(C_u_y_real))
        self.C_u_y_Diag_imag = np.sqrt(np.diagonal(C_u_y_imag))
        print("Corrected ROM posterior FINISH")
        self.u_mean_y_pred_rom = u_mean_y_pred_rom_real + 1j*u_mean_y_pred_rom_imag
        self.C_u_y_Diag = self.C_u_y_Diag_real + 1j*self.C_u_y_Diag_imag
        return 


    def getFullOrderPrior(self,multiple_bases = False):
        # prior calculation Full Order Model
        if multiple_bases == False:
            u_mean_std = np.abs(self.RBmodel.get_U_mean_standard([0,0]) - self.RBmodel.ui.x.array)#+2
            self.u_mean_test = u_mean_std
            self.u_mean_data = self.RBmodel.get_U_mean_standard([0,0]) - self.RBmodel.ui.x.array
            C_u_std = self.RBmodel.get_C_u_MC()
            self.C_u_stdDiag = np.sqrt(np.diagonal(C_u_std))
        else:
            _, A, _ = self.RBmodel.doFEMHelmholtz(self.par1,np.zeros(np.shape(self.RBmodel.coordinates)[0]),assemble_only=True)
            f_rhs = self.RBmodel.FFnp.copy()
            A = A.todense()
            f = f_rhs
            self.sol_mean=np.linalg.solve(A,f)
            _, _, self.sol_mean  = self.RBmodel.solveFEM(rhsPar=np.zeros(np.shape(self.RBmodel.coordinates)[0]))
            u = []
            u_unified = []
            u_data = []
            for j,samp in enumerate(self.f_samples):
                self.RBmodel.doFEMHelmholtz(freq=self.par1,rhsPar=samp+1j*self.f_samples_im[j])
                _, _, uj  = self.RBmodel.solveFEM(rhsPar=samp+1j*self.f_samples_im[j])
                uD = uj - self.RBmodel.ui.x.array
                u_data.append(uD)
                u.append(uj)
                u_unified.append(np.concatenate([np.real(uj),np.imag(uj)]))
            u_mean_std = np.mean(np.array(u),axis=0)  
            self.u_mean_data = np.mean(np.array(u_data),axis=0)
            C_u_std = np.cov(u, rowvar=0, ddof=0)
            self.C_u_std_real = np.cov(np.real(u), rowvar=0, ddof=0)
            self.C_u_std_imag = np.cov(np.imag(u), rowvar=0, ddof=0)
            ident_long = np.identity(np.shape(C_u_std)[0])
            ident = np.identity(np.shape(self.C_u_std_real)[0])
            C_u_std = C_u_std + 9e-8*ident_long
            self.C_u_std_real = self.C_u_std_real + 9e-8*ident
            self.C_u_std_imag = self.C_u_std_imag + 9e-8*ident
            self.C_u_stdDiag = np.sqrt(np.diagonal(C_u_std))
            self.C_u_stdDiag_real = np.sqrt(np.diagonal(self.C_u_std_real))
            self.C_u_stdDiag_imag = np.sqrt(np.diagonal(self.C_u_std_imag))
            self.C_u_stdDiag = self.C_u_stdDiag_real + 1j*self.C_u_stdDiag_imag

        self.u_mean_std,self.C_u_std = u_mean_std,C_u_std


        return u_mean_std,C_u_std




    def computeErrorNorm(self,solution,reference):

        bar_p_sq = 0
        for p in solution:
            val = np.sqrt(np.abs(p*p))
            bar_p_sq+=val
        bar_p_sq = bar_p_sq/np.shape(solution)[0]

        degree_raise = 4
        uh = Function(self.RBmodel.V_coarse)
        uh.x.array[:] = solution
        u_ex = Function(self.RBmodel.V_ground_truth)
        u_ex.x.array[:] = reference
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

        k = 2 * np.pi * self.par1 / 340
        e_W.x.array[:] = np.real(u_W.x.array - u_ex_W.x.array)
        error = form(inner(e_W, e_W) * ufl.dx)
        error_H10 = form(dot(grad(e_W), grad(e_W)) * ufl.dx)
        error_local = assemble_scalar(error)
        error_local_H10 = assemble_scalar(error_H10)
        error_global = mesh.comm.allreduce(error_local, op=MPI.SUM)
        error_global_H10 = mesh.comm.allreduce(error_local_H10, op=MPI.SUM)
        error_sobolev_real = error_global + k**(-2)*error_global_H10


        e_W.x.array[:] = np.imag(u_W.x.array - u_ex_W.x.array)
        error = form(inner(e_W, e_W) * ufl.dx)
        error_H10 = form(dot(grad(e_W), grad(e_W)) * ufl.dx)
        error_local = assemble_scalar(error)
        error_local_H10 = assemble_scalar(error_H10)
        error_global = mesh.comm.allreduce(error_local, op=MPI.SUM)
        error_global_H10 = mesh.comm.allreduce(error_local_H10, op=MPI.SUM)
        error_sobolev_imag = error_global + k**(-2)*error_global_H10

        u_ex_W_real = Function(W)
        u_ex_W_real.x.array[:] = np.real(u_ex_W.x.array)
        u_ex_W_imag = Function(W)
        u_ex_W_imag.x.array[:] = np.imag(u_ex_W.x.array)
        u = form(inner(u_ex_W_real, u_ex_W_real) * ufl.dx)
        u_H10 = form(dot(grad(u_ex_W_real), grad(u_ex_W_real)) * ufl.dx)
        u_local = assemble_scalar(u)
        u_local_H10 = assemble_scalar(u_H10)
        u_global = mesh.comm.allreduce(u_local, op=MPI.SUM)
        u_global_H10 = mesh.comm.allreduce(u_local_H10, op=MPI.SUM)

        u_sobolev_real = np.copy(u_global + k**(-2)*u_global_H10)

        u = form(inner(u_ex_W_imag, u_ex_W_imag) * ufl.dx)
        u_H10 = form(dot(grad(u_ex_W_imag), grad(u_ex_W_imag)) * ufl.dx)
        u_local = assemble_scalar(u)
        u_local_H10 = assemble_scalar(u_H10)
        u_global = mesh.comm.allreduce(u_local, op=MPI.SUM)
        u_global_H10 = mesh.comm.allreduce(u_local_H10, op=MPI.SUM)

        u_sobolev_imag = np.copy(u_global + k**(-2)*u_global_H10)


        return error_sobolev_real/u_sobolev_real, error_sobolev_imag/u_sobolev_imag
 


    def switchMesh(self,grade):
        if grade == "ground_truth":
            funcs.RBmodel.msh = funcs.RBmodel.msh_ground_truth
            funcs.RBmodel.V = funcs.RBmodel.V_ground_truth
            funcs.RBmodel.coordinates = funcs.RBmodel.coordinates_ground_truth
            funcs.RBmodel.facet_markers = funcs.RBmodel.facet_markers_ground_truth
            funcs.RBmodel.cell_markers = funcs.RBmodel.cell_markers_ground_truth

        if grade == "coarse":
            funcs.RBmodel.msh = funcs.RBmodel.msh_coarse
            funcs.RBmodel.V = funcs.RBmodel.V_coarse
            funcs.RBmodel.coordinates = funcs.RBmodel.coordinates_coarse
            funcs.RBmodel.facet_markers = funcs.RBmodel.facet_markers_coarse
            funcs.RBmodel.cell_markers = funcs.RBmodel.cell_markers_coarse


    def switchMesh_self(self,grade):
        if grade == "ground_truth":
            self.RBmodel.msh = self.RBmodel.msh_ground_truth
            self.RBmodel.V = self.RBmodel.V_ground_truth
            self.RBmodel.coordinates = self.RBmodel.coordinates_ground_truth
            self.RBmodel.facet_markers = self.RBmodel.facet_markers_ground_truth
            self.RBmodel.cell_markers = self.RBmodel.cell_markers_ground_truth

        if grade == "coarse":
            self.RBmodel.msh = self.RBmodel.msh_coarse
            self.RBmodel.V = self.RBmodel.V_coarse
            self.RBmodel.coordinates = self.RBmodel.coordinates_coarse
            self.RBmodel.facet_markers = self.RBmodel.facet_markers_coarse
            self.RBmodel.cell_markers = self.RBmodel.cell_markers_coarse
  

    def plotPriorVtk(self):
        import dolfinx.io
        u_fem = self.u_mean_std
        u_rom = self.u_mean
        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "./Results/priorFEM_total.xdmf", "w") as xdmf:
            xdmf.write_mesh(self.RBmodel.msh)
            ut = dolfinx.fem.Function(self.RBmodel.V)
            ut.x.array[:] = u_fem
            xdmf.write_function(ut)

        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "./Results/priorFEM_total_field.xdmf", "w") as xdmf:
            xdmf.write_mesh(self.RBmodel.msh)
            ut = dolfinx.fem.Function(self.RBmodel.V)
            ut.x.array[:] = np.abs(u_fem)
            xdmf.write_function(ut)

        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "./Results/priorFEM_cov.xdmf", "w") as xdmf:
            xdmf.write_mesh(self.RBmodel.msh)
            ut = dolfinx.fem.Function(self.RBmodel.V)
            ut.x.array[:] = self.C_u_stdDiag
            xdmf.write_function(ut)

        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "./Results/priorFEM_cov_total_field.xdmf", "w") as xdmf:
            xdmf.write_mesh(self.RBmodel.msh)
            ut = dolfinx.fem.Function(self.RBmodel.V)
            ut.x.array[:] = np.abs(self.C_u_stdDiag)
            xdmf.write_function(ut)

        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "./Results/incident_wave.xdmf", "w") as xdmf:
            xdmf.write_mesh(self.RBmodel.msh)
            xdmf.write_function(self.RBmodel.ui)

        figure = plt.figure()
        axes = figure.add_subplot(111)
        caxes = axes.matshow(self.C_u_std_real, interpolation ='nearest')
        figure.colorbar(caxes)
        figure.savefig("./Results/cov_std.pdf", bbox_inches='tight')
        plt.close(figure)

        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "./Results/priorROM_total.xdmf", "w") as xdmf:
            xdmf.write_mesh(self.RBmodel.msh)
            ut = dolfinx.fem.Function(self.RBmodel.V)
            ut.x.array[:] = u_rom
            xdmf.write_function(ut)

        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "./Results/priorROM_cov.xdmf", "w") as xdmf:
            xdmf.write_mesh(self.RBmodel.msh)
            ut = dolfinx.fem.Function(self.RBmodel.V)
            ut.x.array[:] = self.C_uDiag
            xdmf.write_function(ut)
        
        figure = plt.figure()
        axes = figure.add_subplot(111)
        caxes = axes.matshow(self.C_u_real, interpolation ='nearest')
        figure.colorbar(caxes)
        figure.savefig("./Results/cov_rom.pdf", bbox_inches='tight')
        plt.close(figure)
            
        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "./Results/ROM_error_cov_est.xdmf", "w") as xdmf:
            xdmf.write_mesh(self.RBmodel.msh)
            ut = dolfinx.fem.Function(self.RBmodel.V)
            ut.x.array[:] = np.sqrt(np.diagonal(self.dr_cov_real + 1j*self.dr_cov_imag))
            xdmf.write_function(ut)

        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "./Results/ROM_error_mean_est.xdmf", "w") as xdmf:
            xdmf.write_mesh(self.RBmodel.msh)
            ut = dolfinx.fem.Function(self.RBmodel.V)
            ut.x.array[:] = self.dr_mean_real+1j*self.dr_mean_imag
            xdmf.write_function(ut)

        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "./Results/ROM_error_mean_exact.xdmf", "w") as xdmf:
            xdmf.write_mesh(self.RBmodel.msh)
            ut = dolfinx.fem.Function(self.RBmodel.V)
            ut.x.array[:] = u_fem-u_rom
            xdmf.write_function(ut)
        
        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "./Results/ROM_error_mean_exact_total.xdmf", "w") as xdmf:
            xdmf.write_mesh(self.RBmodel.msh)
            ut = dolfinx.fem.Function(self.RBmodel.V)
            ut.x.array[:] = np.abs(u_fem-u_rom)
            xdmf.write_function(ut)

        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "./Results/ROM_error_var_exact_total.xdmf", "w") as xdmf:
            xdmf.write_mesh(self.RBmodel.msh)
            ut = dolfinx.fem.Function(self.RBmodel.V)
            ut.x.array[:] = np.abs(self.C_u_stdDiag-self.C_uDiag)
            xdmf.write_function(ut)

        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "./Results/ROM_error_mean_est_total.xdmf", "w") as xdmf:
            xdmf.write_mesh(self.RBmodel.msh)
            ut = dolfinx.fem.Function(self.RBmodel.V)
            ut.x.array[:] = np.abs(self.dr_mean_real+1j*self.dr_mean_imag)
            xdmf.write_function(ut)

        return

    def plotPosteriorVtk(self):
        import dolfinx.io
        u_fem = self.u_mean_y_std
        u_rom_easy = self.u_mean_y_easy
        u_rom_adv = self.u_mean_y_pred_rom

        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "./Results/postFEM_total.xdmf", "w") as xdmf:
            xdmf.write_mesh(self.RBmodel.msh)
            ut = dolfinx.fem.Function(self.RBmodel.V)
            ut.x.array[:] = u_fem
            xdmf.write_function(ut)
        
        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "./Results/postFEM_cov.xdmf", "w") as xdmf:
            xdmf.write_mesh(self.RBmodel.msh)
            ut = dolfinx.fem.Function(self.RBmodel.V)
            ut.x.array[:] = self.C_u_y_std_Diag
            xdmf.write_function(ut)
        
        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "./Results/postROM_total.xdmf", "w") as xdmf:
            xdmf.write_mesh(self.RBmodel.msh)
            ut = dolfinx.fem.Function(self.RBmodel.V)
            ut.x.array[:] = u_rom_easy
            xdmf.write_function(ut)

        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "./Results/postROM_cov.xdmf", "w") as xdmf:
            xdmf.write_mesh(self.RBmodel.msh)
            ut = dolfinx.fem.Function(self.RBmodel.V)
            ut.x.array[:] = self.C_u_y_easy_Diag
            xdmf.write_function(ut)

        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "./Results/postAdvROM_total.xdmf", "w") as xdmf:
            xdmf.write_mesh(self.RBmodel.msh)
            ut = dolfinx.fem.Function(self.RBmodel.V)
            ut.x.array[:] = u_rom_adv
            xdmf.write_function(ut)

        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "./Results/postAdvROM_cov.xdmf", "w") as xdmf:
            xdmf.write_mesh(self.RBmodel.msh)
            ut = dolfinx.fem.Function(self.RBmodel.V)
            ut.x.array[:] = self.C_u_y_Diag
            xdmf.write_function(ut)
    
        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "./Results/postErrorAdv.xdmf", "w") as xdmf:
            print("adv norm:")
            print(self.computeErrorNorm(u_rom_adv,self.data_solution))

        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "./Results/postErrorEasy.xdmf", "w") as xdmf:
            print("easy norm:")
            print(self.computeErrorNorm(u_rom_easy,self.data_solution))

        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "./Results/postErrorFem.xdmf", "w") as xdmf:
            print("fem norm:")
            print(self.computeErrorNorm(u_fem,self.data_solution))




if __name__ == '__main__':
    funcs = StatROM_2D()
    
    funcs.switchMesh("ground_truth")
    funcs.RBmodel.doFEMHelmholtz(freq=funcs.par1,rhsPar=np.zeros(np.shape(funcs.RBmodel.coordinates_ground_truth)[0]),assemble_only=True)
    funcs.generateParameterSamples(256,save=False,load=True)
    funcs.getFullOrderPrior(multiple_bases = True)
    print("full order")
    funcs.get_noisy_data_from_solution(0,np.abs(funcs.u_mean_std))
    funcs.switchMesh("coarse")

    funcs.generateParameterSamples(256,load = False)
    funcs.getFullOrderPrior(multiple_bases = True)
    _, funcs.V_adj_list = funcs.getAORAbasis(Nr=funcs.L,rhs_sp=np.zeros(np.shape(funcs.RBmodel.coordinates_coarse)[0]),adj=True)[0:2] # adjoint solves
    for i,sample in enumerate(funcs.f_samples):
        print("primal sample number: "+str(i))
        funcs.computeROMbasisSample(sample,i)


    funcs.calcROMprior(multiple_bases = True)
    funcs.plotPriorVtk()

    funcs.getFullOrderPosterior()
    funcs.getEasyROMPosterior()
    funcs.getCorrectedROMPosterior()

    funcs.plotPosteriorVtk()
   
    

#end of file