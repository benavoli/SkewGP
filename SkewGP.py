import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.stats import multivariate_normal
from joblib import Parallel, delayed
import random
from scipy.optimize import minimize, dual_annealing, differential_evolution
from commoncode import *

'''
Skew Gaussian Process
author: Alessio Benavoli
12/2020
'''

class SkewGP:
    """
    Class SkewGP
    """
    def __init__(self,X, Kernel, Delta, params, W=[],Z=[], Y=[],C=[], latent_dim = 0, type_y='affine', jitter=1e-8,batchsize_dim=70,num_cores=4):
        """
        Initialize class SkewGP

        :Kernel is an object containing the kernel Omega
        :Delta is the skewness function
        :params is a dictionary including the hyperparameters of the model
        :latent_dim is the latent dimension of the skewGP process, default to zero, i.e. a GP
        :type_y is a string containing the type of output data, 'affine' for classification and preference, 
        'regression' for regression and 'mixed' for mixed problems.
        Note that for a mixed classification example you can use the 'preference' type and pass the appropriate matrix W
        :W,Z,C,Y are the data dependent matrices
        :batchsize_dim is the batch dimension for the approximation of the normal CDF computation in the marginal likelihood
        :num_cores we parallelize the computation of the normal CDF  
        
        """
        self._Kernel = Kernel #function
        self._Delta = Delta #function
        self.params=params
        self.γ = []
        self.Γ = []
        self.γp = []#posterior params
        self.Γp = []#posterior params
        self.latent_dim = latent_dim
        #data
        self.X=X
        self.W=W
        self.Y=Y
        self.Z=Z
        self.C=C
        self.type_y=type_y #'class'/'regression'/'mixed'
        self.jitter=jitter
        self.batchsize_dim=batchsize_dim
        self.num_cores=num_cores



    
    def _compute_gammas(self,params,X,W,Z,Y,C):
        '''
        Computes the small gamma and capital Gamma 
        '''
        if self.type_y=='affine':
            self._compute_gammas_affine(params,X,W,Z)
        elif self.type_y=='regression':
            self._compute_gammas_regression(params,X,Y,C)
        elif self.type_y=='mixed':
            self._compute_gammas_mixed(params,X,W,Z,Y,C)
    
    def compute_gammas_regression(self,params,X,Y,C):
        '''
        Computes the small gamma and capital Gamma for the normal likelihood case
        '''
        noise_variance=self.params['noise_variance']['value']
        γ = params['gamma']['value']
        u=params['u']['value']
        U = self._Kernel(u,u,params) 
        Du=np.diag(1/np.sqrt(np.diag(U)))
        Γ = np.diag(params['phase_u']['value'])@Du@U@Du@np.diag(params['phase_u']['value']) 
        Δ = self._Delta(X,params)
        Ω = self._Kernel(X,X,params) + self.jitter*np.eye(X.shape[0])
        #PROCESS Omega
        ω = np.diag(np.sqrt(np.diag(Ω)))
        #print(Ω.shape)
        L = cholesky(Ω,lower=True)
        L_inv = solve_triangular(L.T,np.eye(L.shape[0]))
        IΩ = L_inv@L_inv.T #solve(Ω+self.jitter*np.eye(Ω.shape[0]), np.eye(Ω.shape[0]))
        # process C        
        xi = np.zeros((Ω.shape[0],1))+0.0
        Kxx = C@Ω@C.T + noise_variance * np.eye(Y.shape[0])
        Kx = C@Ω
        L = cholesky(Kxx,lower=True)
        L_inv = solve_triangular(L.T,np.eye(L.shape[0]))
        IKxx = L_inv@L_inv.T
        #computed posterior parameters
        xip = xi+Kx.T@IKxx@(Y-C@xi)
        Ωp = Ω -Kx.T@IKxx@Kx
        ωp = np.diag(np.sqrt(np.diag(Ωp)))
        Δp = np.diag(1/np.diag(ωp))@Ωp@IΩ@ω@Δ
        γp = γ+Δ.T@ω@IΩ@(xip-xi)
        Γp = Γ+ self.jitter*np.eye(u.shape[0])-Δ.T@ω@IΩ@ω@Δ+ Δp.T@ωp@(IΩ+C.T@C/noise_variance)@ωp@Δp
        return γp, Γp, γ, Γ
    
    def compute_gammas_affine(self,params,X,W0,Z0): 
        '''
        Computes the small gamma and capital Gamma for the affine probit likelihood case
        '''
        noise_variance=self.params['noise_variance']['value']
        W=W0/np.sqrt(noise_variance)#AAAAAAAAA
        Z=Z0/np.sqrt(noise_variance)#AAAAAAAAA
        if self.latent_dim>0:
            γ = params['gamma']['value']
            u=params['u']['value']
            U = self._Kernel(u,u,params) 
            Du=np.diag(1/np.sqrt(np.diag(U)))
            Γ = np.diag(params['phase_u']['value'])@Du@U@Du@np.diag(params['phase_u']['value']) 
            Δ = self._Delta(X,params)
        else:
            γ=[]
            Γ=[]
            Δ=[]
        Ω = self._Kernel(X,X,params) + self.jitter*np.eye(X.shape[0])
        #PROCESS Omega
        ω = np.diag(np.sqrt(np.diag(Ω)))
        #print(Ω.shape)
        xi = np.zeros((Ω.shape[0],1))+0.0
        #computed posterior parameters
        xip = xi
        Ωp = Ω

        if self.latent_dim>0:
            γp = np.vstack([γ, Z+W@xi])
            #print(Γ.shape,Δ.shape)
            Γp = np.vstack([np.hstack([Γ,Δ.T@ω@W.T]),np.hstack([W@ω@Δ,W@Ω@W.T+np.eye(W.shape[0])])])
        else:
            γp = Z+W@xi
            Γp = W@Ω@W.T+np.eye(W.shape[0])
        return γp, Γp, γ, Γ
    
    def compute_gammas_mixed(self,params,X,Y,C,W0,Z0):
        γp0, Γp0, γ, Γ = self.compute_gammas_affine(params,X,W0,Z0)
        noise_variance=self.params['noise_variance']['value']
        W=W0/np.sqrt(noise_variance)
        Ω = self._Kernel(X,X,params) + self.jitter*np.eye(X.shape[0])
        #PROCESS Omega
        ω = np.diag(np.sqrt(np.diag(Ω)))
        if self.latent_dim>0:
            Δ = self._Delta(X,params)
            M = np.hstack([Δ, np.diag(1/np.diag(ω))@Ω@W.T])
        else:
            M = np.diag(1/np.diag(ω))@Ω@W.T
        L = cholesky(Ω,lower=True)
        L_inv = solve_triangular(L.T,np.eye(L.shape[0]))
        IΩ = L_inv@L_inv.T #solve(Ω+self.jitter*np.eye(Ω.shape[0]), np.eye(Ω.shape[0]))
        # process C        
        xi = np.zeros((Ω.shape[0],1))+0.0
        Kxx = C@Ω@C.T + noise_variance * np.eye(Y.shape[0])
        Kx = C@Ω
        L = cholesky(Kxx,lower=True)
        L_inv = solve_triangular(L.T,np.eye(L.shape[0]))
        IKxx = L_inv@L_inv.T
        #computed posterior parameters
        xip = xi+Kx.T@IKxx@(Y-C@xi)
        Ωp = Ω -Kx.T@IKxx@Kx
        γp =γp0 +M.T@ω@IΩ@(xip-xi)
        ωp = np.diag(np.sqrt(np.diag(Ωp)))
        Δi= np.diag(1/np.diag(ωp))@Ωp@IΩ@ω@M
        Γp = Γp0 -M.T@ω@IΩ@ω@M+Δi.T@ωp@(IΩ+C.T@C/noise_variance)@ωp@Δi
        return γp, Γp, γ, Γ, M
       
        

            
    def log_marginal_likelihood_normal_pdf(self):
        """Computes the log marginal likelihood of the Normal"""
        noise_variance=self.params['noise_variance']['value']
        Kxx = self.C@self._Kernel(self.X, self.X, self.params)@self.C.T + (noise_variance+self.jitter) * np.eye(self.Y.shape[0])
        try:
            mu = np.linalg.solve(Kxx, self.Y)
            (sign, logdet) = np.linalg.slogdet(2 * np.pi * Kxx)        
            logp1 =   -0.5*np.asscalar(self.Y.T@mu)-0.5*logdet        
        except:
            logp1=-10.0**300
        return logp1

    def log_marginal_likelihood_normal_cdf(self):
        """Computes the log marginal likelihood of the CDF"""
        #we define the loop for the batchsize
        num_batches = int(np.ceil(self.W.shape[0] /  self.batchsize_dim))
        slices=np.array_split(np.arange(0,self.W.shape[0]),num_batches)
        def batch_indices(iter):
            idx = iter 
            return slice(slices[idx][0],slices[idx][-1]+1)
        
        batch_slices=[batch_indices(iter) for iter in range(num_batches)]
        #print(batch_slices,num_batches,self.batchsize_dim)
        def innerloop(slices):
            if type(slices)!=list:
                slices=[slices]
            #print(slices)
            ml=[]
            for idx in slices:
                if self.type_y=='affine':
                    γp, Γp, _, _ = self.compute_gammas_affine(self.params,self.X,self.W[idx,:],self.Z[idx,:])
                elif self.type_y=='mixed':
                    γp, Γp, _, _, _ = self.compute_gammas_mixed(self.params,self.X,self.Y,self.C,self.W[idx,:],self.Z[idx,:])
                #print(y1.shape)
                res = gaussianCDF(Γp,-np.ones((γp.shape[0],1))*np.inf,γp)
                ml.append(res)
            return ml
        
        if self.type_y=='affine':
    
            results = Parallel(n_jobs=self.num_cores )(delayed(innerloop)(b) for b in batch_slices)
            #print(results)
            res1=np.sum(results)
                     
            _, _, γ, Γ = self.compute_gammas_affine(self.params,self.X,self.W[[0],:],self.Z[[0],:])#we only need γ, Γ
            #print()
            if self.latent_dim>0:
                res2 = gaussianCDF(Γ+self.jitter*np.eye(Γ.shape[0]),-np.ones((γ.shape[0],1))*np.inf,γ)
                logres2 = np.log(res2+1e-200)
            else:
                logres2 = 0.0
            #print( np.log(res1+1e-300),logres2)
            res= np.log(res1+1e-300)-logres2 
        elif self.type_y=='regression':
            if self.latent_dim>0:
                γp, Γp, γ, Γ = self.compute_gammas_regression(self.params,self.X,self.Y,self.C)
                res2 = gaussianCDF(Γ+self.jitter*np.eye(Γ.shape[0]),-np.ones((γ.shape[0],1))*np.inf,γ)
                #from scipy.stats import multivariate_normal
                try:
                    res1 = gaussianCDF(Γp,-np.ones((γp.shape[0],1))*np.inf,γp)
                    res= np.log(res1+1e-300)-np.log(res2+1e-300)
                except:
                    #print(self.params, Γp)
                    res=-10.0**300
            else:
                return 0.0
        elif self.type_y=='mixed':
            results = Parallel(n_jobs=self.num_cores )(delayed(innerloop)(b) for b in batch_slices)
            res1=np.sum(results)
            _, _, γ, Γ = self.compute_gammas_affine(self.params,self.X,self.W[[0],:],self.Z[[0],:])#we only need γ, Γ
            if self.latent_dim>0:
                res2 = gaussianCDF(Γ+self.jitter*np.eye(Γ.shape[0]),-np.ones((γ.shape[0],1))*np.inf,γ)
                logres2 = np.log(res2+1e-200)
            else:
                logres2 = 0.0
            res= np.log(res1+1e-300)-logres2
        if np.isnan(res):
            return -10.0**300            
        else:
            return  res

        
    def log_marginal_likelihood_regression(self):
        if self.latent_dim>0:
            return self.log_marginal_likelihood_normal_cdf()+self.log_marginal_likelihood_normal_pdf()
        else:
            return self.log_marginal_likelihood_normal_pdf()
    
    def log_marginal_likelihood_affine(self):
        return self.log_marginal_likelihood_normal_cdf()
    
    def log_marginal_likelihood_mixed(self):
        return self.log_marginal_likelihood_normal_cdf()+self.log_marginal_likelihood_normal_pdf()
    
    def log_marginal_likelihood(self):
        if self.type_y=='affine':
            return self.log_marginal_likelihood_affine()
        elif self.type_y=='regression':
            return self.log_marginal_likelihood_regression()
        elif self.type_y=='mixed':
            return self.log_marginal_likelihood_mixed()
    
    def predict_regression(self, Xpred,nsamples=2000, tune=100, progress=True ,points2=[]):
        noise_variance=self.params['noise_variance']['value']
        Ω = self._Kernel(self.X,self.X,self.params)+self.jitter * np.eye(self.X.shape[0])
        ω = np.diag(np.sqrt(np.diag(Ω)))        
        xi = np.zeros((Ω.shape[0],1))+0.0
        Kxx = self.C@Ω@self.C.T + (noise_variance) * np.eye(self.Y.shape[0])
        Kxz = self.C@self._Kernel(self.X,Xpred,self.params)
        Kzz = self._Kernel(Xpred,Xpred,self.params)
        
        L = cholesky(Kxx,lower=True)
        L_inv = solve_triangular(L.T,np.eye(L.shape[0]))
        IKxx = L_inv@L_inv.T

        
        xixx = Kxz.T@IKxx@self.Y
        Ωxx = Kzz -Kxz.T@IKxx@Kxz
        iωxx = np.diag(np.sqrt(1/np.diag(Ωxx)))
        if self.latent_dim>0:
            γp, Γp, γ, Γ = self.compute_gammas_regression(self.params,self.X,self.Y,self.C)
            Δ =  self._Delta(self.X,self.params)
            Δx = self._Delta(Xpred,self.params)
            Δxx = iωxx@(np.diag(np.sqrt(np.diag(Kzz)))@Δx-Kxz.T@IKxx@self.C@ω@Δ)
        else:
            #the skewness is zero in this case
            Δxx = np.array([0.0])
            #these are just dummy values
            γp = np.array([1.0])
            Γp = np.array([[1.0]])
        return self.sample(xixx, Ωxx, Δxx, γp, Γp, nsamples=nsamples, tune=tune, progress=progress,points2=points2)

    def predict_affine(self, Xpred,nsamples=2000, tune=100,  progress=True, points2=[]):
        
        xix = np.zeros((Xpred.shape[0],1))+0.0
        γp, Γp, γ, Γ = self.compute_gammas_affine(self.params,self.X,self.W,self.Z)
        ΩxX = self._Kernel(Xpred,self.X,self.params)
        Ωxx = self._Kernel(Xpred,Xpred,self.params)
        iωxx = np.diag(np.sqrt(1/np.diag(Ωxx)))
        
        noise_variance=self.params['noise_variance']['value']
        W=self.W/np.sqrt(noise_variance)
        if self.latent_dim>0:
            Δx = self._Delta(Xpred,self.params)
            Δp = np.hstack([Δx, iωxx@ΩxX@W.T])
        else:
            Δp = iωxx@ΩxX@W.T
        return self.sample(xix, Ωxx, Δp, γp, Γp, nsamples=nsamples, tune=tune,progress=progress, points2=points2)
    
    def predict_mixed(self, Xpred,nsamples=2000, tune=100, cores=None, progress=True, points2=[]):
        noise_variance=self.params['noise_variance']['value']
        W=self.W/np.sqrt(noise_variance)
        Ω = self._Kernel(self.X,self.X,self.params)+self.jitter * np.eye(self.X.shape[0])
        ω = np.diag(np.sqrt(np.diag(Ω))) 
        ΩxX = self._Kernel(Xpred,self.X,self.params)
        Ωxx = self._Kernel(Xpred,Xpred,self.params)
        iωxx = np.diag(np.sqrt(1/np.diag(Ωxx)))
        
        xi = np.zeros((Ω.shape[0],1))+0.0
        Kxx = self.C@Ω@self.C.T + (noise_variance) * np.eye(self.Y.shape[0])
        Kxz = self.C@ΩxX.T
        Kzz = Ωxx
        
        L = cholesky(Kxx,lower=True)
        L_inv = solve_triangular(L.T,np.eye(L.shape[0]))
        IKxx = L_inv@L_inv.T

        
        xip = Kxz.T@IKxx@self.Y
        Ωp = Kzz -Kxz.T@IKxx@Kxz
        if self.latent_dim>0:
            Δx = self._Delta(Xpred,self.params)
            Mx = np.hstack([Δx, iωxx@ΩxX@W.T])
        else:
            Mx = iωxx@ΩxX@W.T
        γp, Γp, γ, Γ, M = self.compute_gammas_mixed(self.params,self.X,self.Y,self.C,self.W,self.Z)
        

        Δp = np.diag(1/np.sqrt(np.diag(Ωp)))@( np.diag(np.sqrt(np.diag(Kzz)))@Mx -Kxz.T@IKxx@self.C@ω@M)
        
        return self.sample(xip, Ωp, Δp, γp, Γp, nsamples=nsamples, tune=tune, progress=progress,points2=points2)
      
    def predict(self,Xpred, nsamples=2000, tune=100,  progress=True,  points2=[]):
        """
        Computes nsamples from the posterior predictive distribution at Xpred

        Xpred test points locations
        nsamples number of posterior samples
        tune number of burn-in samples used in sampling methods
        points2 the posterior samples at the training points, if they are passed this sampling is very fast

        """
        if self.type_y=='affine':
            return self.predict_affine(Xpred, nsamples, tune, progress,  points2)
        elif self.type_y=='regression':
            return self.predict_regression(Xpred, nsamples, tune,  progress,  points2)
        elif self.type_y=='mixed':
            return self.predict_mixed(Xpred, nsamples, tune,  progress,  points2)
            
        
    def sample(self, xi, Ω, Δ, γ, Γ, nsamples=2000, tune=100, progress=True, points2=[]):
        """
        Computes nsamples from the posterior distribution at the training points1

        Note that this is the expensive step in the predictive posterior sampling and it can be done once.

        xi the xi parameter of the posterior SUN
        Ω the Ω parameter of the posterior SUN
        Δ the Δ parameter of the posterior SUN
        γ the γ parameter of the posterior SUN
        Γ the Γ parameter of the posterior SUN
        nsamples number of posterior samples
        tune number of burn-in samples used in sampling methods
        points2: these are samples from the truncated normal, it can be precomputed. This is useful in active learning
        """

        iω = np.diag(1/np.sqrt(np.diag(Ω)))
        Ω_c = np.linalg.multi_dot([iω  , Ω , iω])  #correlation matrix
        L = cholesky(Γ+self.jitter*np.eye(Γ.shape[0]),lower=True)
        M = cho_solve(( L,True),Δ.T).T
        M1 = Ω_c-np.linalg.multi_dot([M,Δ.T])

        del Ω_c
        M1=0.5*(M1+M1.T)+self.jitter*np.identity(M1.shape[0])

        L=cholesky(M1,lower=True)
        trunc = -γ
        rv1=multivariate_normal(np.zeros(M1.shape[0]),np.identity(M1.shape[0]))
        del M1
        points1 = np.dot(L,rv1.rvs(nsamples).T)
        if len(points2)>0:
            self.points2=points2
        else:
            print("Start Lin-Ess sampler")
            self.points2 = sample_truncated(trunc,np.zeros(Γ.shape[0]), Γ, nsamples, tune=tune,  sign=1,progress=progress)
        return xi+ np.dot(np.diag(np.sqrt(np.diag(Ω))), points1 + M@self.points2.T)
        
    def optimize(self,  num_restarts=1, max_iters=100, max_f_eval=300.0,  method='Anneal'):
        """
        Computes the optimal hyperparameters by maximising the marginal likelihood
        
        If method=='Anneal' useas dual_annealing, otherwise calls directly method in scipy.minimize.
        
        num_restarts: number of restarts of the optimisation
        max_iters: maximum number of iterations
        max_f_eval: max number of function evaluations
        """
        dic = DictVectorizer()
        # flatten the parameters
        init_params,bounds=dic.fit_transform(self.params)
        #we minimise minus the marginal likelihood
        def objective(params_flatten):
            self.params=dic.inverse_transform(params_flatten,bounds)
            val = -self.log_marginal_likelihood()
            return val# we want to maximize it
        
       
        #run ptimisation with multiple restarts
        optml=np.inf
        for i in range(num_restarts):
            #minimise function
            if method=='Anneal':
                res=dual_annealing(objective,bounds, maxiter=max_iters, maxfun=max_f_eval, x0=init_params)
            else:
        
                res = minimize(objective, init_params, 
                              bounds=bounds, method=method,options={'maxiter': max_iters, 'disp': False})
            #print("Iteration "+str(i)+" ",-res.fun)
            if res.fun<optml:
                params_best=res.x #init_params 
                optml=res.fun
            init_params=bounds[:,0]+(bounds[:,1]-bounds[:,0])*np.random.rand(len(bounds[:,0]))
            print("Iteration "+str(i)+" ",-res.fun)
        #params_best=res.x
        #optml=res.fun
        self.params=dic.inverse_transform(params_best,bounds)
        return -optml
        
