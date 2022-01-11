import numpy as np
import scipy.sparse as sparse
import dos.tools as tools
from scipy.linalg import block_diag
import osqp     # OSQP solver
import logging
logging.basicConfig()
import os.path
#import yaml

class SHAcO_qp:
    def __init__(self,D,W2,W3,K,wfsMask,umin,umax,verbose=logging.INFO,**kwargs):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(verbose)
        self.logger.info(' - - - Initializing AcO QP-based algorithm! - - - ')

        self.mount_included = False
        if not ((D.shape[1]+2) % 7):
            n_bm = ((D.shape[1]+2)//7) - 12
        elif not ((D.shape[1]+2 -2) % 7):
            n_bm = ((D.shape[1])//7) - 12
            self.mount_included = True
        else:
            self.logger.error('Unable to get the correct number of bending modes. Check D!')

        # W1 can be used to remove mean slopes
        self.W1 = np.eye(D.shape[0])
        if 'Cs' in kwargs.keys():
            self.W1 = kwargs['Cs']
        if kwargs['rm_mean_slopes']:
            # - - - Probe mean slope removal matrix Rs (but retains the overall contribution)
            O = np.kron(np.eye(6),np.ones((48*48,1)))
            V_pr = np.zeros((D.shape[0],6))
            for iv in range(6):
                V_pr[:,iv] = np.hstack([*[O[MaskSeg.ravel(),iv] for MaskSeg in wfsMask]])
            Lambda_pr = np.diag(1/np.sum(V_pr,axis=0))
            
            R_g = np.kron(np.eye(2),np.array([[1,1,1]]).T)
            Lambda_g = np.diag(1/np.sum(V_pr@R_g,axis=0))
            Rs = np.eye(D.shape[0]) - (V_pr @ (Lambda_pr - R_g@Lambda_g@R_g.T) @ V_pr.T)
            self.W1 = Rs.T.dot(self.W1).dot(Rs)
            self.logger.info('Mean slope removal feature incorporated')

        self.DT_W1_D = D.T.dot(self.W1).dot(D)
        self.W1_D = self.W1.dot(D)

        # It is assumed that W3 incorporates the Tu transformation effect
        self.W2, self.W3, self.k_I = W2, W3, K
        self.wfsMask = wfsMask
        # Constraints
        self.umin = umin
        self.umax = umax

        try:
            self._Tu = kwargs['_Tu']
        except:
            self._Tu = np.eye(D.shape[1])

        if 'rho3' in kwargs.keys():
            self.rho3 = kwargs['rho3']
        else:
            self.rho3 = 1.0e-1

        if 'J1_J3_ratio' in kwargs.keys():
            self.J1_J3_ratio = kwargs['J1_J3_ratio']
        else:
            self.J1_J3_ratio = 10

        self.logger.info('AcO: k_I=%.3g(integral gain) and rho3(0)=%.3g'%(self.k_I,self.rho3))

        # Indices to insert (or remove) S7Rz columns
        self.iM1S7Rz = ((12+n_bm)*6) + 5
        self.iM2S7Rz = ((12+n_bm)*6) + 10   # Add 1 to delete
        if 'end2end_ordering' in kwargs.keys():
            if kwargs['end2end_ordering']:
                self.iM1S7Rz, self.iM2S7Rz = 41, 82
        # WFS interaction matrix with M1/2-S7Rz
        self.DwS7Rz = np.insert(D,[self.iM1S7Rz,self.iM2S7Rz],0,axis=1)

        # Reconstructor dimensions
        self.nc = D.shape[1]

        # QP reconstructor matrices
        P = sparse.csc_matrix(self.DT_W1_D+ self.W2+ self.rho3*(self.k_I**2)*self.W3)
        #Ptriu = sparse.triu(P)
        #self.logger.debug('Ptriu: %s',Ptriu.toarray()[:4,:4])
        
        # Inequality constraint matrix: lb <= Ain*u <= ub
        self.Ain = sparse.csc_matrix(   # Remove S7Rz from _Tu
            np.delete(self._Tu,[self.iM1S7Rz,self.iM2S7Rz+1], axis=1)*self.k_I)
             
        # Create an OSQP object as global
        self.qpp = osqp.OSQP()
        # Setup QP problem
        self.qpp.setup(P=P, q=np.zeros(self.nc), A=self.Ain,
            l=-np.inf*np.ones(self.Ain.shape[0]),u=np.inf*np.ones(self.Ain.shape[0]),
            eps_abs = 1.0e-8, eps_rel = 1.0e-6, max_iter = 500*self.nc,
            verbose=True, warm_start=True)
        self.qpp.update(Px=sparse.triu(P).data)    
        
        # Integral controller initial state
        self.__u = np.zeros(0)


    def init(self):
        self.__u = 1e-6 + np.zeros(self.nc+2) # ! Insert zeros for M1/2S7-Rz
        self.invTu = np.linalg.pinv(self._Tu)


    def update(self,y_sh):
        # AcO state reconstructor
        y_sh = y_sh.ravel()
        y_valid = np.hstack([*[y_sh[MaskSeg.ravel()] for MaskSeg in self.wfsMask]])
        
        # Remove S7-Rz
        u_ant = np.delete(self.__u,[self.iM1S7Rz,self.iM2S7Rz+1])
        # Update linear QP term
        q = -y_valid.T.dot(self.W1_D) - self.rho3*u_ant.T.dot(self.W3)*self.k_I
        
        # Update bounds to inequality constraints
        _Tu_u_ant = self._Tu.dot(self.__u)
        lb = self.umin -_Tu_u_ant
        ub = self.umax -_Tu_u_ant
        # Update QP object and solve problem - 1st step
        self.qpp.update(q=q, l=lb, u=ub)
        
        self.logger.debug('q type: %s',q.dtype)
        X = self.qpp.solve()
        # Check solver status
        if X.info.status == 'solved':
            # Insert zeros for M1/2S7-Rz
            c_hat = np.insert(X.x[:self.nc],[self.iM1S7Rz,self.iM2S7Rz],0)
        else:
            self.logger.info('QP info: %s', X.info.status)
            self.logger.warning('Infeasible QP problem!!!')
            c_hat = np.zeros_like(self.__u)

        epsilon = y_valid - self.DwS7Rz.dot(c_hat)
        J1 = epsilon.T.dot(self.W1).dot(epsilon)
        delta = np.delete(self.k_I*c_hat - self.__u,[self.iM1S7Rz,self.iM2S7Rz+1])
        J3 = delta.T.dot(self.W3).dot(delta)

        # J3 is zero if delta is also zero -> no need for the 2nd step
        if (J3):#False:#
            if(self.rho3):                
                norm_s = np.linalg.norm(y_valid)
                self.logger.info('1st-> J1:%0.3g, J3:%0.3g, ratio:%0.3g, ||s||:%0.3g' %(J1,J3,J1/(self.rho3*J3),norm_s**2))
            else:
                self.logger.info('1st-> J1:%0.3g, J3:%0.3g, ratio:%0.3g, rho3:-X-' %(J1,J3,J1/J3))

            # Update J3 weight
            self.rho3 = max((J1/(self.J1_J3_ratio*J3)),1.0e-6)
            # Update QP object and solve problem - 2nd step
            P = sparse.csc_matrix(self.DT_W1_D+ self.W2+ self.rho3*(self.k_I**2)*self.W3)
            q = -y_valid.T.dot(self.W1_D) - self.rho3*u_ant.T.dot(self.W3)*self.k_I
            self.qpp.update(q=q)
            self.qpp.update(Px=sparse.triu(P).data)
            # Solve QP - 2nd Step
            X = self.qpp.solve()
            # Check solver status            
            if X.info.status != 'solved':
                self.logger.warning('QP info: %s', X.info.status)
                self.logger.warning('Infeasible QP problem!!!')
        
            # Insert zeros for M1/2S7-Rz
            c_hat = np.insert(X.x[:self.nc],[self.iM1S7Rz,self.iM2S7Rz],0)

            epsilon = y_valid - self.DwS7Rz.dot(c_hat)
            J1 = epsilon.T.dot(self.W1).dot(epsilon)
            delta = np.delete(self.k_I*c_hat - self.__u,[self.iM1S7Rz,self.iM2S7Rz+1])
            J3 = delta.T.dot(self.W3).dot(delta)
            
            self.logger.info('2nd> J1:%0.3g, J3:%0.3g, ratio:%0.3g, rho3:%0.3g' %(J1,J3,J1/(self.rho3*J3),self.rho3))
            
        self.logger.info('c: %s',c_hat[:7].T)
        # Integral controller
        self.__u = self.__u -self.k_I*c_hat

        # Clip the control signal to the saturation limits [umin,umax] - Should not be necessary if using QP
        if not (empty(self.umin) and empty(self.umax)):
            clip_iter, clip_tol = 0, 1.1
            while (clip_iter<0) and (
                        any(self._Tu.dot(self.__u) > clip_tol*self.umax) or 
                        any(self._Tu.dot(self.__u) < clip_tol*self.umin)):
                clip_iter = clip_iter + 1        
                self.__u = self.invTu.dot(np.clip(self._Tu.dot(self.__u), self.umin, self.umax))
            # Warn clipping iterations required    
            if(clip_iter):
                self.logger.warning('Number of clipping iterations: %d',clip_iter)

        self.logger.debug('u: %s',self.__u)

    def output(self):
        if not self.mount_included:
            return np.atleast_2d(self.__u)
        else:
            self.logger.info('u_mount: %s',self.__u[-2:])
            return np.atleast_2d(self.__u[:-2])

# Function used to test empty constraint vectors
def empty(value):
    try:
        value = np.array(value)
    except ValueError:
        pass
    if value.size:
        return False
    else:
        return True
