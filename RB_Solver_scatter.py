"""
statROM Helmholtz 1D example

Here, the FEniCSx FEM solver is implemented along with futher low-level methods for data assimilation. 
The file also wraps AORA.py to generate the ROM basis.

The most important methods are 
doFEMHelmholtz() ->           Implements the FEniCSx FEM solver. This is used to compute the ROM basis.
getPriorAORA() ->             Uses a given AORA basis to compute a ROM prior
computePosteriorROM() ->      Implements the low level data assimilation routines

"""

__author__ = "Lucas Hermann"
__version__ = "0.2.0"
__license__ = "MIT"



import ufl
from dolfinx.fem import Function, FunctionSpace, form, Constant, locate_dofs_topological, dirichletbc
from dolfinx.fem.petsc import LinearProblem, assemble_matrix, assemble_vector, apply_lifting, set_bc
from dolfinx.io import XDMFFile
from dolfinx.mesh import locate_entities, meshtags
from dolfinx.geometry import BoundingBoxTree, compute_collisions, compute_colliding_cells

from ufl import dx, grad, inner, Measure
from mpi4py import MPI
from petsc4py import PETSc


import numpy as np
from numpy.linalg import multi_dot
import scipy
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

import kernels
from meshesHelmholtz import generate_mesh_with_obstacle

import os
usr_home = os.getenv("HOME")

import time




class RBClass:
    """solver class for a reduced order statFEM approach"""

    def __init__(self,up):
        '''Input arguments: user parameters up'''
        self.up = up
        self.reset()


    def reset(self):
        """ Generates meshes using gmsh and computes the corresponding function spaces
            with FEniCSx.
        """
        # approximation space polynomial degree
        deg = 1
        L = 1
        H = 1
        lc =.9

        self.msh_coarse, self.cell_markers_coarse, self.facet_markers_coarse  = generate_mesh_with_obstacle(
                 Lx=L,
                 Ly=H,
                 lc=lc,
                 refine = 0.04)
        
        self.msh_ground_truth, self.cell_markers_ground_truth, self.facet_markers_ground_truth  = generate_mesh_with_obstacle(
                 Lx=L,
                 Ly=H,
                 lc=lc,
                 refine = 0.035)
        self.msh = self.msh_coarse
        self.cell_markers = self.cell_markers_coarse
        self.facet_markers = self.facet_markers_coarse

        self.V_coarse = FunctionSpace(self.msh_coarse, ("CG", deg)) # Function space
        self.coordinates_coarse = self.V_coarse.tabulate_dof_coordinates()[:,0:2]

        self.V_ground_truth = FunctionSpace(self.msh_ground_truth, ("CG", deg)) # Function space
        self.coordinates_ground_truth = self.V_ground_truth.tabulate_dof_coordinates()[:,0:2]
        self.times_reducedorder = []
        



    def incident(self,x):
        '''Input arguments: coordinate x'''
        # Plane wave travelling in positive x-direction
        return 10*np.exp(1.0j * self.k *x[0])
        

    def doFEMHelmholtz(self,freq,rhsPar=0,het_par=0,assemble_only=False):
        """basic FEM solver for the Helmholtz equation 
        Returns the mean solution for the prior and expects the frequency parameters.
        Input arguments: frequency, RHS parameter, material parameter, assemble only flag
        """
        c = 340
        k = 2 * np.pi * freq / c # wave number
        self.k = k

        # material coefficient:
        self.hetCoef = Function(self.V)

        # functions for the RHS:
        self.s = Function(self.V)
        self.s.vector.array[:] = np.real(rhsPar)
        self.si = Function(self.V)
        self.si.vector.array[:] = np.imag(rhsPar)

        # Define variational problem
        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)
        self.v = v
        dx = ufl.Measure("dx")
        ds = ufl.Measure("ds", domain=self.msh, subdomain_data=self.facet_markers)

        # incident wave interpolation:
        ui = Function(self.V)
        ui.interpolate(self.incident)
        self.ui=ui 

        # the bilinear form:
        b= 1j*self.k
        a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx - self.k**2 * ufl.inner(u, v) * dx  - b * ufl.inner(u , v) * ds(40)  # usually: g*ufl.inner(...)
        L = -ufl.inner(ui+1j*self.si+self.s, v) *dx

        # set dirichlet BCs:
        dofs_dirichl_coarse = locate_dofs_topological(self.V_coarse, self.msh_coarse.topology.dim-1, self.facet_markers_coarse.find(50))
        self.dofs_dirichl_coarse = dofs_dirichl_coarse
        dofs_dirichl = locate_dofs_topological(self.V, self.msh.topology.dim-1, self.facet_markers.find(50))
        self.dofs_dirichl = dofs_dirichl
        uD = Function(self.V)
        uD.interpolate(lambda x: 0.0*x[0]+0.0*x[1])
        bc = dirichletbc(uD, dofs_dirichl)

        # complete system matrix A assembly:
        A = assemble_matrix(form(a))
        A.assemble()
        self.rhs = L
        LL = assemble_vector(form(L))
        LL.assemble()
        self.LLnp = LL.getArray()

        # A split into M, D, K for AORA:
        m = -inner(u, v) * dx
        M = assemble_matrix(form(m), bcs=[bc])
        M.assemble()
        mi, mj, mv = M.getValuesCSR()
        self.Msp = csr_matrix((mv, mj, mi))
        
        d = - 1j * inner(u,v) * ds(40) 
        D = assemble_matrix(form(d), bcs=[bc])
        D.assemble()
        di, dj, dv = D.getValuesCSR()
        self.Dsp = csr_matrix((dv, dj, di))

        kk = inner(grad(u), grad(v)) * dx
        KK = assemble_matrix(form(kk), bcs=[bc])
        KK.assemble()
        kki, kkj, kkv = KK.getValuesCSR()
        self.KKsp = csr_matrix((kkv, kkj, kki))

        # RHS assembly:
        ff = -inner(ui+1j*self.si+self.s, v) * dx
        FF = assemble_vector(form(ff))
        apply_lifting(FF, [form(a)], bcs=[[bc]])
        FF.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(FF, [bc])
        FF.assemble()
        FFnp = FF.getArray()
        self.FFnp = FF.getArray()

        C = np.zeros((np.shape(FFnp)))
        C[0] = 1
        self.C = C

        Asp = k*k*self.Msp + k*self.Dsp+ self.KKsp
        self.A = Asp
        
        # Compute solution:
        uh = Function(self.V)
        uh.name = "u"

        self.problem = LinearProblem(a, L,bcs=[bc],u=uh)

        # sometimes only assembly and no solve is necessary:
        if assemble_only == False:
            self.problem.solve()
            uh_np = uh.vector.getArray()
            uh_np = np.copy(uh_np)
            U = uh.vector
            p_full = spsolve(Asp,self.FFnp)
            uh_np = p_full
            return U, self.A, uh_np
        elif assemble_only == True:
            U,uh_np = None,None
            return U, self.A, uh_np

        
    
    def solveFEM(self, rhsPar):
        ''' separate solve function to minimize allocations'''
        '''Input arguments: RHS parameter'''
        p_full = spsolve(self.A,self.FFnp)
        uh_np = p_full
        return None, None, uh_np
    

    def solveFEMData(self,sample):
        ''' solve FEM on a fine mesh to get the ground truth
            Input arguments: RHS sample
        '''
        self.s.vector.array[:] = np.real(sample) * (np.cos(self.coordinates[:,1]* 4.5 *np.pi)*0.8+1) + 1j*np.imag(sample) * (np.sin(self.coordinates[:,1]* 4.5 *np.pi)*0.8+1)
        with XDMFFile(MPI.COMM_WORLD, "./Results/data_f.xdmf", "w") as xdmf:
            xdmf.write_mesh(self.msh)
            ut = Function(self.V)
            ut.x.array[:] = np.real(sample) * (np.cos(self.coordinates[:,1]* 4.5 *np.pi)*0.8+1) + 1j*np.imag(sample) * (np.sin(self.coordinates[:,1]* 4.5 *np.pi)*0.8+1)
            xdmf.write_function(ut)

        with XDMFFile(MPI.COMM_WORLD, "./Results/sample_f.xdmf", "w") as xdmf:
            xdmf.write_mesh(self.msh)
            ut = Function(self.V)
            ut.x.array[:] = sample
            xdmf.write_function(ut)
        
        uh = self.problem.solve()

        uh_np = uh.vector.getArray()
        uh_np = np.copy(uh_np)
        U = uh.vector
        return uh, U, uh_np



    def getAr(self,V,A):
        """projects a matrix A into reduced space.
           Input arguments: ROM projection matrix V, system matrix A
        """
        A_r = V.conj().T@A@V
        return A_r


    def get_C_f(self):
        """compute covariance matrix for the random parameter
        
        it can be chosen whether a Sq.Exp. or Matern covariance matrix
        shall be used.
        """
        integratedTestF = assemble_vector(form(inner(1.0 , self.v) * dx))
        integratedTestF.assemble()
        self.integratedTestF = integratedTestF
        c_f = kernels.matern52(self.coordinates,
                self.coordinates, lf=0.6, sigf=0.8)

        self.c_f = c_f
        dim = np.shape(self.coordinates)[0]
        C_f = np.zeros((dim,dim))

        # Acc. to Girolami et al 2021, lumped mass approximation of C_f
        for i in range(dim):
            for j in range(dim):
                C_f[i,j] = self.integratedTestF[i] * c_f[i,j] * self.integratedTestF[j]
        return csr_matrix(C_f)
    



    def getPriorAORA(self,V,par):
        """ With a given projection matrix V and parameter sample, compute the ROM result
            Input arguments: projection matrix V, FEM parameters
        """
        
        _ = self.doFEMHelmholtz(par[0],par[1],par[2],assemble_only=True)[1] # assemble A
        start = time.time() 
        c = 340
        k = 2 * np.pi * par[0] / c # wave number

        # project matrices with V to get the ROM:
        V_H = V.conj().T
        Mr = V_H@self.Msp@V
        Dr = V_H@self.Dsp@V
        Kr = V_H@self.KKsp@V
        Fr=V_H@self.FFnp
        Asp_rom = k*k*Mr + k*Dr+ Kr
        
        p_rom = spsolve(Asp_rom,Fr) # solve ROM
        p_rom_re = V@p_rom # project solution back to FEM dofs
        end = time.time()
        duration = end - start
        self.times_reducedorder.append(duration)
        self.u_mean_r = p_rom
        u_mean = p_rom_re

        return (u_mean,0) 
    

    def getLogPostMultiple(self,params,y_points,y_values,C_u,P,Pu):
        """compute the negative log likelihood for multiple observations
        
        returns the neg log likelihood
        expects a set of parameters, the sensor locations and the prior covariance matrix.
        Input arguments: set of hyperparameters, sensor locations, observation data, prior covariance matrix, sensor projection matrix.
        """
        # parameters for the model error GP:
        rho = params[0]
        sigd = params[1]
        ld = params[2]

        # data:
        y_valuesList=y_values
        y_points = np.transpose(np.atleast_2d(np.array(y_points)))

        logpost = 0
        rho = np.exp(rho) # enforce positivity

        # project the covariance to sensor space:
        C_u_trans = multi_dot([P,C_u,P.T])
        C_u_trans = 0.5*(C_u_trans+C_u_trans.T)

        # add all covariances:
        K_y = self.get_C_d(y_points = y_points,ld=ld,sigd=sigd)+ self.C_e + rho**2 * C_u_trans

        # faster solve with cholesky
        L = cho_factor(K_y)
        Log_K_y_det = 2 * np.sum(np.log(np.diag(L[0])))
        i=0

        # iterate through all observations to get total log marginal likelihood:
        for obs in y_valuesList:
            y_values = np.array(obs)
            ny = len(y_values)  
            y = y_values - rho * Pu
        
            K_y_inv_y = cho_solve(L, y)
        
            Log_K_y_det = 2 * np.sum(np.log(np.diag(L[0])))
            logpost2 = 0.5 * (-np.dot(np.transpose(y), K_y_inv_y )  -Log_K_y_det - ny * np.log(2* np.pi))
            logpost = logpost + logpost2
            i=i+1


        return logpost*(-1)



    def getLogLikelihoodROM(self,params,y_points,y_values,C_u,C_d_r,mean_d_r,P,Pu):
        """compute the negative log likelihood for multiple observations for the ROM case
        
        returns the neg log likelihood
        Input arguments: set of hyperparameters, sensor locations, observation data, prior covariance matrix, ROM error covariance, ROM error mean, sensor projection matrix.
        """

        rho = params[0]
        sigd = params[1]
        ld = params[2]

        y_valuesList=y_values
        y_points = np.transpose(np.atleast_2d(np.array(y_points)))
        logpost = 0
        rho = np.exp(rho)
        C_u_trans = multi_dot([P,C_u,P.T])
        C_u_trans = 0.5*(C_u_trans+C_u_trans.T)

        # add all covariances:
        K_y = self.get_C_d(y_points = y_points,ld=ld,sigd=sigd)+ rho**2*C_d_r + self.C_e + rho**2 * C_u_trans
        L = cho_factor(K_y)
        Log_K_y_det = 2 * np.sum(np.log(np.diag(L[0])))
        i=0

        # iterate through all observations to get total log marginal likelihood:
        for obs in y_valuesList:
            if isinstance(obs, np.float64):
                ny = 1
            else:
                y_values = np.array(obs)
                ny = len(y_values)

            y = y_values - rho * Pu - rho*mean_d_r  
            K_y_inv_y = cho_solve(L, y)

            Log_K_y_det = 2 * np.sum(np.log(np.diag(L[0])))
            logpost2 = 0.5 * (-np.dot(np.transpose(y), K_y_inv_y )  -Log_K_y_det - ny * np.log(2* np.pi))
            logpost = logpost + logpost2
            i=i+1


        return logpost*(-1)


    def estimateHyperpar(self,y_points,y_values, C_u,u_mean, dROM=None,CROM=None,ROM=True):
        """find an optimal set of hyperparameters given data.

        expects sensor locations, observation data and the prior covariance matrix.
        returns a set of hyperparameters.
        Input arguments: sensor locations, observation data, prior covariance matrix, prior mean, ROM error estimate mean, ROM error estimate cov, ROM/FEM flag.
        """

        # project the dof vector:
        P = self.getP(y_points) # projection matrix
        Pu = np.dot(P,u_mean)

        y_points = np.transpose(np.atleast_2d(np.array(y_points)))
        y_values = np.array(y_values)

        # provide initial values for the optimizer:
        rho_est = np.log(1.001)
        sigd_est = np.log(0.01)
        ld_est = np.log(0.5)

        # run an optimizer to find the optimal hyperparameters given the data:
        if ROM == False:
            result = scipy.optimize.minimize(fun=self.getLogPostMultiple,method='L-BFGS-B',bounds=((-2,1),(-15,2),(-20,5)),x0=np.array([rho_est,sigd_est,ld_est]),args=(y_points, y_values,C_u,P,Pu),tol=1e-8)
        elif ROM == True:
            result = scipy.optimize.minimize(fun=self.getLogLikelihoodROM,method='L-BFGS-B',bounds=((-2,1),(-15,2),(-20,5)),x0=np.array([rho_est,sigd_est,ld_est]),args=(y_points, y_values,C_u,CROM,dROM,P,Pu),tol=1e-8)
        print("Results optimizer:")
        print(result.x)
        print(result.success)
        
        res = result.x
        print(np.exp(res))

        return res[0], res[1], res[2]


    def get_C_e(self,size):
        """create the measurement error covariance matrix (diagonal)"""
        sige_square = (5e-4)**2
        C_e = sige_square * np.identity(size)
        self.C_e = C_e

        return C_e


    def get_C_d(self,y_points,ld, sigd):
        """create the model mismatch covariance matrix
        
        expects the sensor locations and kernel parameters
        returns the resulting covariance matrix.
        It can be chosen between a Sq. Exp. and Matern type kernel function.
        """
       
        C_d = kernels.matern52_log(y_points, y_points, lf=ld, sigf=sigd)
        self.C_d = C_d

        return C_d



    def getP(self,y_points):
        """create the statFEM projection matrix
        
        expects the sensor locations and the prior mean
        returns P
        Input arguments: sensor locations
        """
        y_points = y_points
        ny = len(y_points) # number of sensors
        ne = np.shape(self.Msp.todense())[0] # number of dofs
        P = np.zeros((ny, ne), dtype = float) # pre-allocate matrix

        for j,point in enumerate(y_points):
            # check in which cell the sensor lies:
            point = np.atleast_2d(point)
            bb = BoundingBoxTree(self.msh, self.msh.topology.dim)
            bbox_collisions = compute_collisions(bb, point)
            cells = compute_colliding_cells(self.msh, bbox_collisions, point)
            cell = cells.links(0)[0]
            # check reference coordinates in the cell:
            geom_dofs = self.msh.geometry.dofmap.links(cell)
            x_ref = self.msh.geometry.cmap.pull_back(point, self.msh.geometry.x[geom_dofs])
            el = self.V.element.basix_element
            ref_coord = el.tabulate(0,x_ref)
            # assemble P:
            for i in range(self.V.element.space_dimension):
               P[j,geom_dofs[i]] = ref_coord[0][0][i]

            P = np.copy(P)

        self.P = P
        self.P_T = np.transpose(P)
        return P



    def computePosteriorMultipleY(self,y_points,y_values,u_mean,C_u):
        """computes the statFEM posterior for multiple observations
        
        expects the sensor locations, the observation data vector and the prior GP.
        returns the posterior GP
        here, y_values is a vector of different measurement sets. 
        """
        C_e = self.get_C_e(len(y_points))
        P = self.getP(y_points)
        self.C_e = C_e
        pars = self.estimateHyperpar(y_points,y_values,C_u,u_mean=u_mean,ROM=False)
        rho=pars[0]
        sigd = pars[1]
        ld = pars[2]
        self.sigd=np.exp(sigd)
    
        y_points = np.transpose(np.atleast_2d(np.array(y_points)))
        y_values = np.array(y_values)
        self.no = np.shape(y_values)[0]

        sum_y = np.sum(y_values,axis=0)
        P_T = np.transpose(P)
        rho = np.exp(rho)
        C_d = self.get_C_d(y_points=y_points.T,ld=ld,sigd=sigd)
        self.C_d_total = self.get_C_d(self.coordinates,ld=ld,sigd=sigd)
        
        c, low = cho_factor((1/(self.no*rho*rho))*(C_d+C_e)+P@C_u@P_T)
        C_u_y = C_u - C_u@P_T@cho_solve((c, low), P@C_u)
        B = multi_dot([P,C_u,P_T])+1/(rho*rho*self.no)*(C_d+C_e)

        u_mean_y = np.array(u_mean) + (1/(rho*self.no))*multi_dot([C_u,P_T,np.linalg.solve(B,sum_y-rho*self.no*np.dot(P,u_mean))])
        u_mean_y = rho*u_mean_y #eq. 57 statFEM paper
        C_u_y = rho**2 * C_u_y + self.C_d_total + self.get_C_e(np.shape(self.C_d_total)[0]) #eq. 57 statFEM paper

        posteriorGP = 0
        return u_mean_y,C_u_y,posteriorGP


    def computePosteriorROM(self,y_points,y_values,u_mean,C_u,dROMfull,CROMfull):
        """computes the statFEM posterior for multiple observations
        
        This is the version which incorporates a ROM error into the data model.
        expects the sensor locations, the observation data vector and the prior GP.
        returns the posterior GP
        here, y_values is a vector of different measurement sets. 
        Input arguments: sensor locations, observation data, prior mean, prior covariance matrix, ROM error mean, ROM error covariance.
        """
        C_e = self.get_C_e(len(y_points)) # get noise covariance
        P = self.getP(y_points) # get sensor projection matrix
      
        self.C_e = C_e
        dROM,CROM = P@dROMfull,P@CROMfull@P.T  # project ROM error estimates to sensor space

        # find optimal hyperparameters:
        pars = self.estimateHyperpar(y_points,y_values,C_u,dROM=dROM,CROM=CROM,u_mean=u_mean,ROM=True)
        rho=pars[0]
        sigd = pars[1]
        self.sigdROM=np.exp(sigd)
        ld = pars[2]

        y_points = np.transpose(np.atleast_2d(np.array(y_points)))
        y_values = np.array(y_values)

        sum_y = np.sum(y_values,axis=0)

        # given the hyperparametes, construct model error covariance:
        C_d = self.get_C_d(y_points.T,ld=ld,sigd=sigd)
        self.C_d_total = self.get_C_d(self.coordinates,ld=ld,sigd=sigd)

        P_T = np.transpose(P)
        rho = np.exp(rho)

        no = np.shape(y_values)[0]

        # Bayesian update:
        c, low = cho_factor((1/(no*rho*rho))*(C_d+rho**2*CROM+C_e)+P@C_u@P_T)
        C_u_y = C_u - C_u@P_T@cho_solve((c, low), P@C_u)

        B = multi_dot([P,C_u,P_T])+1/(rho*rho*self.no)*(C_d+rho**2*CROM+C_e)
        u_mean_y = np.array(u_mean) + (1/(rho*self.no))*multi_dot([C_u,P_T,np.linalg.solve(B,sum_y-rho*self.no*np.dot(P,u_mean)-self.no*rho*dROM)])
        u_mean_y = rho*u_mean_y#
        u_mean_y_pred_rom= u_mean_y + rho*dROMfull #eq. 57 statFEM paper
        C_u_y = rho**2 * C_u_y + self.C_d_total + CROMfull + self.get_C_e(np.shape(self.C_d_total)[0]) #eq. 57 statFEM paper
        

        posteriorGP = np.random.multivariate_normal(
            mean = np.real(u_mean_y), cov=np.real(C_u_y),
            size=1)

        return u_mean_y,C_u_y,u_mean_y_pred_rom,posteriorGP






