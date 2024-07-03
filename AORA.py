import numpy as np
import scipy as sp
import scipy.linalg
from scipy.sparse import csc_matrix, linalg as sla
from scipy.sparse.linalg import spsolve
from assemble_matrix import assemble_matrix
from assemble_matrix_derivative import assemble_matrix_derivative
###
# python adaptation of the MATLAB code by M. Bollhöfer, TU Braunschweig
###
# auskommentiere Zeilen löschen?
def AORA(M, D, K, B, C, s, Nr, LinSysFac = None):

    SOAR = 1
    if SOAR == False:
        FIRST_ORDER = 1
    else:
        FIRST_ORDER = 0
    
    info = []
   # info.IdxExpPts = [] ?!?!?!?!

    # Normalization coefficient hPi(s_i) = \prod_{j} || r^{(j-1)}(s_i) ||
    try:
        len_s = np.shape(s)[0]
    except:
        if isinstance(s,float) or isinstance(s,int):
            len_s = 1
        else:
            print("check len of s!")
    hPi = np.ones((len_s,1))
    # Number of given expansion points
    NumExpPts = len_s
    # Residual vector for every expansion point in each iteration step
    R = np.zeros((np.shape(M)[0],NumExpPts),dtype=np.complex_)
    # Projection matrix for reduced order model
    V = np.zeros((np.shape(M)[0],Nr),dtype=np.complex_)

    # Initialization of Krylov subspaces for each expansion point 
    if (LinSysFac == None):
        LinSysFac = []
    for i in range(NumExpPts):
        #print("len LU:")
        #print(len(LinSysFac))
        if (len(LinSysFac) != 0):
        #if (LinSysFac != None):
            if (i > len(LinSysFac)):
                A = assemble_matrix(s[i],K,D,M)
                (Ps,Ls,Us) = sp.linalg.lu(A) # here: dense matrices. cchange for future work on larger systems
                LinSysFac.append([Ps,Ls,Us])
            else:
                (Ps,Ls,Us,Qs) = (LinSysFac[i][0], LinSysFac[i][1], LinSysFac[i][2], LinSysFac[i][3])
               # print("using old LU!")
        else:
          #  print("using new LU!")
            A = assemble_matrix(s[i],K,D,M)
            na = np.shape(A)[0]
            #(Ps,Ls,Us) = sp.linalg.lu(A)
            
            A = csc_matrix(A)
            lu = sla.splu(A)
            #print("LU done")
            Ls = lu.L.A
            Us = lu.U.A
            Ps = csc_matrix((np.ones(na), (lu.perm_r, np.arange(na))))
            Qs = csc_matrix((np.ones(na), (np.arange(na), lu.perm_c)))
           # print("sparse conv done")
            LinSysFac.append([Ps,Ls,Us,Qs])
        #R[:,i] = np.linalg.solve(Us, np.linalg.solve(np.transpose(Ps)@Ls,B))
        R[:,i] = spsolve((Us@np.transpose(Qs)), spsolve(np.transpose(Ps)@Ls,B))
        #print("first solve done")
    T = np.zeros((Nr,Nr),dtype=np.complex_)
    if SOAR:
        f = np.zeros((np.shape(M)[0],1))
    elif FIRST_ORDER:
        R2 = np.zeros((np.shape(M)[0],NumExpPts))

    # Main iteration loop of AORA method
    for j in range(Nr):
        # Choice of expansion frequency with maximum output moment error
        maxMomErr = np.abs(hPi[0,0]*C@R[:,0])
        idxMomErr = 0
        #(print("abs done"))
        for i in range(1,NumExpPts):
            if (np.abs(hPi[i]@C@R[:,i]) > maxMomErr):
                maxMomErr = np.abs(hPi[i]@C@R[:,i])
                idxMomErr = i

        # Orthonormal vector for s[idxMomErr]
        if FIRST_ORDER:
            normRes = np.linalg.norm(np.array([R[:,idxMomErr],R2[:,idxMomErr]]).T)
        else:
            normRes = np.linalg.norm(np.array([R[:,idxMomErr]]).T)
        #print("norm Res done")
        V[:,j] = R[:,idxMomErr] / normRes
        hPi[idxMomErr] = hPi[idxMomErr] * normRes
        if SOAR and j>0:
           # print(i)
           # print(j)
            T[j,j-1] = normRes
            b = np.zeros((j,1),dtype=np.complex_)
            b[0,:] = 1
            #vec = np.linalg.lstsq(T[1:j+1,0:j],np.array([1,np.zeros((j-1,1),dtype=np.complex_)]))
            vec = spsolve(T[1:j+1,0:j],b).reshape(j,1)
            f = V[:,0:j] @ vec
        elif FIRST_ORDER:
            V2[:,j] = R2[:,idxMomErr] /normRes

        # Update residual for next iteration
        for i in range(NumExpPts):
            #print("in the big loop")
            if (i==idxMomErr):
                (Ps,Ls,Us,Qs) = (LinSysFac[i][0], LinSysFac[i][1], LinSysFac[i][2], LinSysFac[i][3])
                #print("assemble deriv start")
                Ap = assemble_matrix_derivative(s[i],K,D,M)
                #print("assemble deriv done")
                if SOAR:
                   # print("loop solve start")
                    R[:,i] = spsolve(  -1*(Us@np.transpose(Qs)), spsolve(  np.transpose(Ps)@Ls, (np.atleast_2d(Ap@V[:,j]).T+1j*(M@f))  )   ).flatten()
                   # print("loop solve done")
                elif FIRST_ORDER:
                    R[:,i] = spsolve(  -1*(Us@np.transpose(Qs)), spsolve(  np.transpose(Ps)@Ls, (Ap@V[:,j]+1j*(M@V2[:,j]))  )   )
                    R2[:,i] = V[:,j]
                else:
                    R[:,i] = spsolve(  -1*(Us@np.transpose(Qs)), spsolve(  np.transpose(Ps)@Ls, (Ap@V[:,j])  )   )

                # Complete modified Gram-Schmidt procedure for the new moment
                for t in range(j+1):
                    if SOAR:
                        # Orthogonal projection of R(:,i) onto V(:,t)
                        T[t,j] = np.transpose(V[:,t]).conj()  @ R[:,i]
                        R[:,i] = R[:,i] - T[t,j] * V[:,t]
                    elif FIRST_ORDER:
                        # Orthogonal projection of [R(:,i);R2(:,i)] onto [V(:,t);V2(:,t)]
                        alpha =  np.transpose(V[:,t]).conj()  @ R[:,i] + np.transpose(V2[:,t]).conj()  @ R2[:,i]
                        R[:,i] = R[:,i] - alpha @ V[:,t]
                        R2[:,i] = R2[:,i] - alpha @ V2[:,t]
                    else:
                        # Orthogonal projection of R(:,i) onto V(:,t)
                        alpha = np.transpose(V[:,t]).conj()  @ R[:,i]
                        R[:,i] = R[:,i] - alpha @ V[:,t]
            else:
                # one further step modified Gram-Schmidt procedure for the old moment
                t=j
                if FIRST_ORDER:
                    # Orthogonal projection of [R(:,i);R2(:,i)] onto [V(:,t);V2(:,t)]
                    alpha =  np.transpose(V[:,t]) @ R[:,i] + np.transpose(V2[:,t]) @ R2[:,i]
                    R[:,i] = R[:,i] - alpha @ V[:,t]
                    R2[:,i] = R2[:,i] - alpha @ V2[:,t]
                else:
                    # Orthogonal projection of R(:,i) onto V(:,t)
                    alpha = np.transpose(V[:,t]) @ R[:,i]
                    R[:,i] = R[:,i] - alpha @ V[:,t]
    if FIRST_ORDER:
        (V, RR) = sp.linalg.qr(V)
        (URR,SRR,VRR) = sp.linalg.svd(RR)
        rnk = np.linalg.matrix_rank(SRR)
        V = V@URR[:,0:rnk]



    return (V, info, LinSysFac, R,T)