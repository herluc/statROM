import ufl
from dolfinx.fem import Function, FunctionSpace, assemble_scalar, form, Constant, locate_dofs_topological, dirichletbc
from dolfinx.fem.petsc import LinearProblem, assemble_matrix, assemble_vector
from dolfinx.la import matrix_csr
from dolfinx.io import XDMFFile
from dolfinx.mesh import create_unit_square, create_interval, create_rectangle, locate_entities, meshtags, exterior_facet_indices
from dolfinx.geometry import BoundingBoxTree, compute_collisions, compute_colliding_cells

from ufl import dx, grad, nabla_grad, inner, Measure, lhs, rhs
from mpi4py import MPI
from petsc4py import PETSc
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import seaborn as sns
import numpy as np
from numpy.linalg import multi_dot, norm#, eigh
import scipy
from scipy.linalg import cho_factor, cho_solve, cholesky, eigh
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.interpolate import interp1d

import kernels

import os
usr_home = os.getenv("HOME")

#np.random.seed(76)

class KLE_expression:
    def __init__(self,inputVector,coords):
        self.inputVector = inputVector
        self.coords = coords
    def eval(self, x):
        xGrid = np.linspace(-0.000001,1.0,100)
        yGrid = xGrid
        [X_grid,Y_grid] = np.meshgrid(xGrid,yGrid)

        sigma = np.sqrt(5e-2)
        correlationLength = 0.3
        truncOrder = int(np.sqrt(len(self.inputVector)))

        CovarianceMatrix = sigma*np.exp(-np.abs(X_grid-Y_grid)/correlationLength)

        w,v = np.linalg.eig(CovarianceMatrix)
        eigenValues = w[0:truncOrder]
        eigenFunctions = v[:,0:truncOrder]

        matCoef = 0
        for i in range(0,truncOrder):
            eigenVectorX = interp1d(xGrid,eigenFunctions[:,i])
            for j in range(0,truncOrder):
                eigenVectorY = interp1d(yGrid,eigenFunctions[:,j])
                globalInd = np.ravel_multi_index(np.array([i,j]),(truncOrder,truncOrder))
                matCoef = matCoef + np.sqrt(eigenValues[i]*eigenValues[j])*self.inputVector[globalInd]*np.multiply(eigenVectorX(x[0]),eigenVectorY(x[1]))

        matCoef = np.exp(matCoef)

        return matCoef



class RBClass:
    """solver class for a reduced basis statFEM approach"""

    def __init__(self,up,ne=100):
        self.problem = None
        self.up = up
        self.reset(up.n)


    def reset(self,ne):
        """doc"""
        self.ne	= ne #number of elements
        # approximation space polynomial degree
        deg = 1
        self.msh_coarse = create_interval(MPI.COMM_WORLD,self.ne,[0.0,1.0]) #define mesh
        self.msh_ground_truth = create_interval(MPI.COMM_WORLD,self.up.n_fine,[0.0,1.0]) #define fine mesh for ground truth
        self.msh = self.msh_coarse

        self.V_coarse = FunctionSpace(self.msh_coarse, ("CG", deg)) # Function space
        self.coordinates_coarse = self.V_coarse.tabulate_dof_coordinates()[:,0:1]

        self.V_ground_truth = FunctionSpace(self.msh_ground_truth, ("CG", deg)) # Function space
        self.coordinates_ground_truth = self.V_ground_truth.tabulate_dof_coordinates()[:,0:1]



    def doFEMHelmholtz(self,freq,rhsPar=1,mat_par=np.array([0]),assemble_only=False):
        """basic FEM solver for the Helmholtz equation 
        Returns the mean solution for the prior and expects the frequency and parameters.
        """
        Amp = rhsPar
        c = 343
        k = 2 * np.pi * freq / c
        self.k = k

        # approximation space polynomial degree
        deg = 1
        # Test and trial function space
        V = FunctionSpace(self.msh, ("CG", deg))

        # Define variational problem
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        self.v = v
        f = Constant(self.msh, PETSc.ScalarType(Amp * k**2))
        self.f = f

        beta = k  

        boundaries = [(1, lambda x: np.isclose(x[0], 0)),
                    (2, lambda x: np.isclose(x[0], 1.0))]

        facet_indices, facet_markers = [], []
        fdim = self.msh.topology.dim - 1
        for (marker, locator) in boundaries:
            facets = locate_entities(self.msh, fdim, locator)
            facet_indices.append(facets)
            facet_markers.append(np.full_like(facets, marker))
        facet_indices = np.hstack(facet_indices).astype(np.int32)
        facet_markers = np.hstack(facet_markers).astype(np.int32)
        sorted_facets = np.argsort(facet_indices)
        facet_tag = meshtags(self.msh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])


        ds = Measure("ds", domain=self.msh, subdomain_data=facet_tag)
        self.ds = ds

        
        bcs = []

        # Setup KLE material coef as function of input Vector
        self.hetCoef = Function(self.V)
        hetCoef_KLE = KLE_expression(mat_par.tolist(),self.coordinates)
        self.hetCoef.interpolate(hetCoef_KLE.eval)
        
        a = inner(grad(u), grad(v)) * dx - self.hetCoef * k**2 * inner(u, v) * dx# - (0.1j*k*beta) * inner(u,v) * ds(2)
        L = inner(f, v) * ds(1)
        A = assemble_matrix(form(a))
        A.assemble()
        self.rhs = L
        LL = assemble_vector(form(L))
        LL.assemble()
        self.LLnp = LL.getArray()

        m = - self.hetCoef * inner(u, v) * dx
        M = assemble_matrix(form(m))
        M.assemble()
        mi, mj, mv = M.getValuesCSR()
        self.Msp = csr_matrix((mv, mj, mi))

        
        d = - (0.1j*beta) * inner(u,v) * ds(2) 
        D = assemble_matrix(form(d))
        D.assemble()
        di, dj, dv = D.getValuesCSR()
        self.Dsp = csr_matrix((dv, dj, di))*0

        kk = inner(grad(u), grad(v)) * dx
        KK = assemble_matrix(form(kk))
        KK.assemble()
        kki, kkj, kkv = KK.getValuesCSR()
        self.KKsp = csr_matrix((kkv, kkj, kki))

        ff = inner(f, v) * ds(1)
        FF = assemble_vector(form(ff))
        FF.assemble()
        FFnp = FF.getArray()
        self.FFnp = FF.getArray()

        C = np.zeros((np.shape(FFnp)))
        C[0] = 1
        self.C = C

        Asp = k*k*self.Msp + k*self.Dsp+ self.KKsp
        

        # Compute solution
        uh = Function(V)
        uh.name = "u"
        problem = LinearProblem(a, L, u=uh, bcs=bcs)

        if assemble_only == False:
            problem.solve()
            uh_np = uh.vector.getArray()
            uh_np = np.copy(uh_np)
            U = uh.vector

            p_full = spsolve(Asp,self.FFnp)
            uh_np = p_full
            return U, A, uh_np
        elif assemble_only == True:
            # no solve, only assembling
            U,uh_np = None,None
            return U, A, uh_np





    def getAr(self,V,A):
        """projects a matrix A into reduced space."""
        A_r = np.transpose(V)@A@V
 
        return A_r


    def get_C_f(self):
        """compute covariance matrix for the random parameter
        
        it can be chosen whether a Sq.Exp. or Matern covariance matrix
        shall be used.
        """
        integratedTestF = assemble_vector(form(inner(1.0 , self.v) * dx))
        integratedTestF.assemble()
        self.integratedTestF = np.real(integratedTestF.getArray())
        c_f = kernels.matern52(self.coordinates,
                self.coordinates, lf=0.25, sigf=0.3)

        self.c_f = c_f
        C_f = np.zeros((self.ne+1,self.ne+1))

        # Acc. to Girolami et al 2021, lumped mass approximation of C_f
        for i in range(self.ne+1):
            for j in range(self.ne+1):
                C_f[i,j] = self.integratedTestF[i] * c_f[i,j] * self.integratedTestF[j]
        return C_f


    def getPriorAORA(self,V,par):
        """ With a given projection matrix V and parameter sample, compute the ROM result
        """
        A = self.doFEMHelmholtz(par[0],par[1],par[2],assemble_only=True)[1]
        c = 343
        k = 2 * np.pi * par[0] / c

        Mr = V.T@self.Msp@V
        Dr = V.T@self.Dsp@V
        Kr = V.T@self.KKsp@V
        Fr=V.T@self.FFnp
        Asp_rom = k*k*Mr + k*Dr+ Kr
        p_rom = spsolve(Asp_rom,Fr)
        p_rom_re = V@p_rom
        self.u_mean_r = p_rom
        u_mean = p_rom_re

        return u_mean 
    

    def getLogPostMultiple(self,params,y_points,y_values,C_u,P,Pu):
        """compute the negative log likelihood for multiple observations
        
        returns the neg log likelihood
        expects a set of parameters, the sensor locations and the prior covariance matrix.
        """
        rho = params[0]
        sigd = params[1]
        ld = params[2]
        y_valuesList=y_values
        y_points = np.transpose(np.atleast_2d(np.array(y_points)))
        logpost = 0
        rho = np.exp(rho)

        C_u_trans = multi_dot([P,C_u,P.T])
        C_u_trans = P@C_u@P.T

        K_y = self.get_C_d(y_points = y_points,ld=ld,sigd=sigd)+ self.C_e + rho**2 * C_u_trans

        L = cho_factor(K_y)
        Log_K_y_det = 2 * np.sum(np.log(np.diag(L[0])))
        i=0

        for obs in y_valuesList:
            if isinstance(obs, np.float64):
                ny = 1
            else:
                y_values = np.array(obs)
                ny = len(y_values)
            y = y_values - rho * Pu
            K_y_inv_y = cho_solve(L, y)
            logpost2 = 0.5 * (-np.dot(np.transpose(y), K_y_inv_y )  -Log_K_y_det - ny * np.log(2* np.pi))#Version Paper
            logpost = logpost + logpost2
            i=i+1

        return logpost*(-1)



    def getLogLikelihoodROM(self,params,y_points,y_values,C_u,C_d_r,mean_d_r,P,Pu):
        """compute the negative log likelihood for multiple observations for the ROM case
        
        returns the neg log likelihood
        expects a set of parameters, the sensor locations and the prior covariance matrix.
        """

        rho = params[0]
        sigd = params[1]
        ld = params[2]
     
        y_valuesList=y_values
        y_points = np.transpose(np.atleast_2d(np.array(y_points)))
        logpost = 0
        rho = np.exp(rho)
        C_u_trans = multi_dot([P,C_u,P.T])

        K_y = self.get_C_d(y_points = y_points,ld=ld,sigd=sigd)+ rho**2*C_d_r + self.C_e + rho**2 * C_u_trans
        L = cho_factor(K_y)
        Log_K_y_det = 2 * np.sum(np.log(np.diag(L[0])))
        i=0

        for obs in y_valuesList:
            if isinstance(obs, np.float64):
                ny = 1
            else:
                y_values = np.array(obs)
                ny = len(y_values)
            y = y_values - rho * Pu - rho*mean_d_r      
            K_y_inv_y = cho_solve(L, y)
            logpost2 = 0.5 * (-np.dot(np.transpose(y), K_y_inv_y )  -Log_K_y_det - ny * np.log(2* np.pi))#Version Paper
            logpost = logpost + logpost2
            i=i+1

        return logpost*(-1)


    def estimateHyperpar(self,y_points,y_values, C_u,u_mean, dROM=None,CROM=None,ROM=True):
        """find an optimal set of hyperparameters based on given data.

        expects sensor locations, observation data and the prior covariance matrix.
        returns a set of hyperparameters.
        """
 
        P = self.getP(y_points)
        y_points = np.transpose(np.atleast_2d(np.array(y_points)))
        y_values = np.array(y_values)
        

        Pu = np.dot(P,u_mean)
        rho_est = np.log(1.001)
        sigd_est = np.log(0.01)
        ld_est = np.log(0.5)

        #suppress warnings about complex values. In this 1D setting without damping, we only look at the real part of the solution.
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            if ROM == False:
                result = scipy.optimize.minimize(fun=self.getLogPostMultiple,method='L-BFGS-B',bounds=((-2,1),(-15,2),(-20,5)),x0=np.array([rho_est,sigd_est,ld_est]),args=(y_points, y_values,C_u,P,Pu),tol=1e-8)
            elif ROM == True:
                result = scipy.optimize.minimize(fun=self.getLogLikelihoodROM,method='L-BFGS-B',bounds=((-2,1),(-15,2),(-20,5)),x0=np.array([rho_est,sigd_est,ld_est]),args=(y_points, y_values,C_u,CROM,dROM,P,Pu),tol=1e-8)
        print("Results optimizer:")
        print(result.x)
        
        res = result.x
        print(np.exp(res))

        return res[0], res[1], res[2]


    def get_C_e(self,size):
        """create the measurement error covariance matrix (diagonal)"""
        sige_square = (1e-1)**2
        C_e = sige_square * np.identity(size)
        self.C_e = C_e

        return C_e


    def get_C_d(self,y_points,ld, sigd):
        """create the model mismatch covariance matrix
        
        expects the sensor locations and kernel parameters
        returns the resulting covariance matrix.
        It can be chosen between a Sq. Exp. and Matern type kernel function.
        """
        #C_d = kernels.exponentiated_quadratic(y_points, y_points, lf=ld, sigf=sigd)
        C_d = kernels.matern52_log(y_points, y_points, lf=ld, sigf=sigd)

        self.C_d = C_d

        return C_d


        
    def getP(self,y_points):
        """create the statFEM projection matrix
        
        expects the sensor locations and the prior mean
        returns P
        """
        ny = len(y_points)
        P = np.zeros((ny, self.ne+1), dtype = float)

        for j,point in enumerate(y_points):
            point = np.atleast_2d(point)
            bb = BoundingBoxTree(self.msh, self.msh.topology.dim)
            bbox_collisions = compute_collisions(bb, point)
            cells = compute_colliding_cells(self.msh, bbox_collisions, point)
            cell = cells.links(0)[0]
            geom_dofs = self.msh.geometry.dofmap.links(cell)
            x_ref = self.msh.geometry.cmap.pull_back(point, self.msh.geometry.x[geom_dofs])
            el = self.V.element.basix_element
            ref_coord = el.tabulate(0,x_ref)
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

        C_u_y = (C_u - C_u@P_T@np.linalg.solve( (1/(self.no*rho*rho))*(C_d+C_e)+P@C_u@P_T,P@C_u))

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
        """
        C_e = self.get_C_e(len(y_points))
        P = self.getP(y_points)
        self.C_e = C_e
        dROM,CROM = P@dROMfull,P@CROMfull@P.T
        pars = self.estimateHyperpar(y_points,y_values,C_u,dROM=dROM,CROM=CROM,u_mean=u_mean,ROM=True)
        rho=pars[0]
        sigd = pars[1]
        self.sigdROM=np.exp(sigd)
        ld = pars[2]
        print("ROM pars:")
        print(np.exp(pars))
        y_points = np.transpose(np.atleast_2d(np.array(y_points)))
        y_values = np.array(y_values)
        sum_y = np.sum(y_values,axis=0)
        C_d = self.get_C_d(y_points.T,ld=ld,sigd=sigd)
        self.C_d_total = self.get_C_d(self.coordinates,ld=ld,sigd=sigd)
        P_T = np.transpose(P)
        rho = np.exp(rho)
        no = np.shape(y_values)[0]

        c, low = cho_factor((1/(no*rho*rho))*(C_d+CROM+C_e)+P@C_u@P_T)
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


    


    


