from scipy.optimize import minimize

# import time
import SkewGP as SkewGP
import numpy as np
import GPy as GPy
from scipy.stats import multivariate_normal, bernoulli
# from sklearn.preprocessing import StandardScaler

from scipy.linalg import cholesky, cho_solve, solve_triangular
import pymc3 as pm

lik = GPy.likelihoods.Bernoulli()


def acquisition(Xpred, SGP, Xref, samples, nsamples, X,
                ell, variancef, fMAP, LambdaMAP, SigmaMAP,
                kernel, surrogateM='SGP', typea='IGAIN', iter_s=42, kappa=0.1):
    """
    Compute the acquisition function at Xpred with reference point Xref on surrogate SGP

    Xpred point where to compute the acquisition function
    SGP surrogate Gaussian Process (either Laplace approximation or SkewGP)
    Xref the best point to compare to
    samples the posterior samples at the training points, used to reduce computational cost
    nsamples number of posterior predictive samples
    X training inputs, used for GPL predictions
    ell kernel lengthscales parameters to be passed to GPL predict
    variancef kernel variance parameter to be passed to GPL predict
    fMAP MAP for GPL
    LambdaMAP Lambda MAP for GPL
    SigmaMAP Sigma MAP for GPL
    kernel GPy kernel object for Omega
    surrogateM choice of either 'SGP' (skewGP) or 'GPL' (Laplace approximation)
    typea string selecting the acquisition function:
    'IGAIN' (infogain), 'EPI' (expected probability of improvement),
    'Thompson' (Thompson sampling), 'UCB' (UCB), 'EPI_IGAIN' (EPI IGAIN as defined in the paper)
    iter_s seed random random seed
    kappa parameter for EPI_IGAIN
    """
    Xpred = np.atleast_2d(Xpred)
    XX = np.vstack([Xpred, Xref])
    np.random.seed(iter_s)
    if surrogateM == 'SGP':
        predict = SGP.predict(XX, nsamples=nsamples, tune=0, points2=samples)
        # print(predict.shape)
    else:
        # print(samples,nsamples)
        predict = predict_GPL(XX, X, np.exp(ell), np.exp(variancef), fMAP, LambdaMAP, SigmaMAP, kernel,
                              nsamples=nsamples)
        # print(predict.shape)
    if typea == 'IGAIN':  # infogain
        # Xpred=np.atleast_2d(Xpred) #.reshape(-1,1)
        # XX = np.vstack([Xpred,Xref])
        # print(XX)
        # np.random.seed(42)
        # predict = SGP.predict(XX,nsamples=samples.shape[1], tune=0, method="LinEss", points=samples)
        PI = lik.gp_link.transf(predict[0:1, :] - predict[1:2, :])
        meanP = np.mean(PI, axis=1)
        # print(meanP)
        H1 = he(meanP)  # H[y|x,D]
        H2 = np.mean(he(PI), axis=1)
        return -(H1 - H2)
    elif typea == 'EPI':
        # predict = SGP.predict(XX,nsamples=2, tune=0, method="LinEss", points=samples[:,-3:-1])[:,0:1]
        PI = lik.gp_link.transf(predict[0:1, :] - predict[1:2, :])
        # PI =  predict[0:1,:]-predict[1:2,:]
        meanP = np.mean(PI, axis=1)
        return -meanP
    elif typea == 'Thompson':
        # np.random.seed(time.time())
        cols = -1  # np.random.randint(predict.shape[1])
        return -(predict[0:1, cols] - predict[1:2, cols])
    # elif typea=='Thompson':
    #    return PI =  predict[0:1,-1]-predict[1:2,-1]
    elif typea == 'UCB':
        # Xpred=np.atleast_2d(Xpred) #.reshape(-1,1)
        # XX = np.vstack([Xpred,Xref])
        # print(XX)
        # np.random.seed(42)
        # predict = SGP.predict(XX,nsamples=samples.shape[1], tune=0, method="LinEss", points=samples)
        # predict = SGP.predict(XX,nsamples=2, tune=0, method="LinEss", points=samples[:,-3:-1])[:,0:1]
        # PI =  lik.gp_link.transf(predict[0:1,:]-predict[1:2,:])
        PI = predict[0:1, :] - predict[1:2, :]
        meanP = np.mean(PI, axis=1)
        # meanP = np.sum(lik.gp_link.transf(predict-Fref)>0.5,axis=1)/nsamples
        credib_int = pm.stats.hpd(PI.T, credible_interval=0.95)  # pm.stats.hpd(PI.T,alpha=0.05)
        ##print(Xpred,np.mean(PI,axis=1))
        # penalty = np.maximum(0,7-10*classp)**2
        return -credib_int[:, 1]  # -meanP-(credib_int[:,1]-credib_int[:,0])/2
    elif typea == 'EPI_IGAIN':
        # Xpred=np.atleast_2d(Xpred) #.reshape(-1,1)
        # XX = np.vstack([Xpred,Xref])
        # print(XX)
        # np.random.seed(42)
        # predict = SGP.predict(XX,nsamples=samples.shape[1], tune=0, method="LinEss", points=samples)
        # predict = SGP.predict(XX,nsamples=2, tune=0, method="LinEss", points=samples[:,-3:-1])[:,0:1]
        PI = lik.gp_link.transf(predict[0:1, :] - predict[1:2, :])
        # PI =  predict[0:1,:]-predict[1:2,:]
        meanP = np.mean(PI, axis=1)
        H1 = he(meanP)  # H[y|x,D]
        H2 = np.mean(he(PI), axis=1)
        eps = 1e-15
        p1 = np.maximum(eps, np.minimum(1 - eps, meanP))
        return -kappa * np.log(p1) - (H1 - H2)  # -0.1*np.log(p1)-(H1-H2)


def he(p):
    """
    Compute the entropy of binary variable p
    """
    eps = 1e-15
    p1 = np.maximum(eps, np.minimum(1 - eps, p))
    return -p1 * np.log(p1) - (1 - p1) * np.log(1 - p1)


def queryf(x_new, x_old, i, j, f, valid):
    """
    Query the function

    x_new new x input
    x_old old input
    i position of comparison
    j position of comparison
    f query function
    valid a function that checks if the new input is valid, used in case of mixed inputs
    """
    if valid(x_new, f) == 1.0:
        v1 = f(x_new)
        v2 = f(x_old)
        if v1 >= v2:
            return 1.0, [i, j]
        elif v1 < v2:
            return 1.0, [j, i]
        else:
            print("error")
    else:  # valid(x_new,f)==0.0:
        return -1.0, []


def compute_W(X, Class, Pairs):
    """
    Compute the matrix W

    Attention: Class is not used here

    X input points
    Class vector containing the class output
    Pairs vector containing pairs indicating the preference output
    """
    W = []
    # W=np.zeros((len(Pairs)+len(Class),X.shape[0]))
    # for i in range(len(Class)):
    #    W[i,i]=Class[i]
    # j = len(Class)
    j = 0
    W = np.zeros((len(Pairs), X.shape[0]))
    for i in range(len(Pairs)):
        W[j + i, Pairs[i, 0]] = 1
        W[j + i, Pairs[i, 1]] = -1

    return W


def predict_GPL(Xpred, X, ell, variancef, fMAP, LambdaMAP, Sigma, kernel, nsamples=200):
    """
    Predict GPL posterior

    Xpred point where to compute the prediction
    X training points
    ell kernel lengthscales
    variancef kernel variance
    fMAP MAP for GPL
    LambdaMAP Lambda MAP for GPL
    Sigma Sigma MAP for GPL
    kernel GPy kernel object for Omega
    nsamples number of samples
    """
    M = np.linalg.inv(LambdaMAP + 1e-6 * np.eye(LambdaMAP.shape[0]))
    #Kkernel = kernel(Xpred.shape[1], ARD=True, lengthscale=ell, variance=variancef)  # AAAAAAAAAAAAAA
    pars = {'lengthscale': {'value': ell},
            'variance': {'value': variancef}
            }
    K_s = kernel(Xpred, X, params=pars)
    K_ss = kernel(Xpred, Xpred, params=pars)
    SS = Sigma + 1e-6 * np.eye(LambdaMAP.shape[0])
    L = cholesky(SS, lower=True)
    L1 = cholesky(SS + M, lower=True)
    v = cho_solve((L1, True), K_s.T)
    alpha = cho_solve((L, True), fMAP)
    mup = K_s.dot(alpha)
    covp = K_ss - K_s.dot(v) + 1e-6 * np.eye(len(Xpred))
    # print(np.linalg.eig(covp))
    L = cholesky(covp, lower=True)
    rv1 = multivariate_normal(np.zeros(covp.shape[0]), np.identity(covp.shape[0]))
    points1 = np.dot(L, rv1.rvs(nsamples).T)
    return mup + points1


class BO():
    """
    Class BO: Bayesian optimization
    """

    def __init__(self, X, Preference, Class, bounds, kernel, params, oracle, valid, Xref, nsamples=300,
                 surrogateM='SGP', acquisition='EPI_IGAIN', alternate_optim=30, kappa=0.1):
        """
        Initialize BO class

        X training points
        Preference preference data
        Class class data
        bounds bounds of the input space
        kernel GPy kernel object for Omega
        params kernel hyper-parameters
        oracle oracle function
        valid function that returns whether the location returns an output
        Xref best input point
        nsamples number of samples used in sampling functions
        surrogateM type of surrogate for BO: 'SGP' for SkewGP, 'GPL' for Laplace approximation
        acquisition string selecting the acquisition function:
        'IGAIN' (infogain), 'EPI' (expected probability of improvement),
        'Thompson' (Thompson sampling), 'UCB' (UCB), 'EPI_IGAIN' (EPI IGAIN as defined in the paper)
        alternate_optim
        kappa parameter for EPI_IGAIN
        """
        self.kernel = kernel
        self.params = params
        # Here we convert back the parameters in a simpler form to pass them to MATLAB
        self.log_params = {'log_lengthscale': (np.log(self.params['lengthscale']['value']),
                                               np.log(self.params['lengthscale']['range'])),
                           'log_variance': (np.log(self.params['variance']['value']),
                                            np.log(self.params['variance']['range']))}

        self.nsamples = nsamples
        self.samples = []
        self.surrogateM = surrogateM
        self.bounds = bounds
        self.iref = 0
        self.Xref = Xref
        self.X = X.copy()
        self.Preference = Preference.copy()
        self.Class = Class.copy()
        self.oracle = oracle
        self.valid = valid
        self.acquisition = acquisition
        self.SGP = []
        self.fMAP = []
        self.SigmaMAP = []
        self.LambdaMAP = []
        self.kernelName = 'stationary'
        self.alternate_optim = alternate_optim
        self.kappa = kappa

    def update_surrogate(self, X, W, iter_s, update_model=True):
        """
        Update the surrogate model

        X training points
        W matrix containing the preference
        iter_s number of the iteration
        update_model a flag to choose whether to update the hyper-parameters
        """
        if self.surrogateM == 'SGP':
            # define skewness function
            def Delta(X, params):
                # empty
                return []

            # Z is only because SkewGP works with affine likelihoods, here Z is zeros
            Z = np.zeros((W.shape[0], 1), float)
            SGP = SkewGP.SkewGP(X, self.kernel, Delta, self.params,
                                W=W, Z=Z, latent_dim=0, type_y='affine', batchsize_dim=np.minimum(W.shape[0], 45))
            # SGP.compute_posterior(X,W,self.params)
            # SGP = SkewGP.SkewGP(self.kernel, self.kernelName, latent_dim = 0, type_y='preference')
            # SGP.compute_posterior(X,W,self.params)

            # SGP.optimize(params, optimizer='Anneal', batchsize_dim=30, max_iters=130, max_f_eval=130)
            print(iter_s)
            opt = False
            if update_model == True:
                if self.X.shape[0] > self.alternate_optim:
                    if np.mod(iter_s, 10) == 0:
                        opt = True
                else:
                    opt = True
            if opt == True:
                resSGP = SGP.optimize(method='Anneal', max_iters=150, max_f_eval=150)
                # resSGP=SGP.optimize(self.params, niter=30,  bounds=[[-4.0,4.3]]*X.shape[1]+[[-4,5]], verbose=False ,batchsize_dim=np.minimum(W.shape[0],45), u_fixed=False)
                self.params = SGP.params
            else:
                print(self.params)
            nsamples = self.nsamples

            γp, Γp, γ, Γ = SGP.compute_gammas_affine(SGP.params, X, W, Z)
            Ω = SGP._Kernel(X, X, self.params) + SGP.jitter * np.eye(X.shape[0])
            iω = np.diag(np.sqrt(1 / np.diag(Ω)))
            # # ω = np.diag(np.sqrt(np.diag(Ω)))
            xi = np.zeros((Ω.shape[0], 1)) + 0.0
            if SGP.latent_dim > 0:
                Δx = SGP._Delta(X, self.params)
                Δp = np.hstack([Δx, iω @ Ω @ W.T])
            else:
                Δp = iω@Ω@W.T

            samples = SGP.sample(xi, Ω, Δp, γp, Γp, nsamples=nsamples, tune=500)
            #samples = SGP.predict(X, nsamples=nsamples, tune=500)
            self.samples = SGP.points2
            self.SGP = SGP
        else:
            import matlab.engine
            matlabeng = matlab.engine.start_matlab()
            matlabeng.addpath('matlab_GP_preference/')
            Xbtrain = np.vstack(self.Preference) + 1.0
            opt = False
            # print(update_model)
            if update_model == True:
                if self.X.shape[0] > self.alternate_optim:
                    if np.mod(iter_s, 10) == 0:
                        opt = True
                else:
                    opt = True
            # print(opt)
            if opt == True:
                print(X.shape, Xbtrain.shape, self.log_params)
                try:
                    Res_matlab = np.array(
                        matlabeng.main_BO_alessio(matlab.double(X.tolist()), matlab.double(Xbtrain.tolist()),
                                                  matlab.double(np.exp(self.log_params['log_lengthscale'][0]).tolist()),
                                                  matlab.double(np.exp(self.log_params['log_variance'][0]).tolist()), 1))
                except:

                    # matlabeng = matlab.engine.start_matlab()
                    # matlabeng.addpath('matlab_GP_preference/')
                    Res_matlab = np.array(
                        matlabeng.main_BO_alessio(matlab.double(X.tolist()), matlab.double(Xbtrain.tolist()),
                                                  matlab.double(np.exp(self.log_params['log_lengthscale'][0]).tolist()),
                                                  matlab.double(np.exp(self.log_params['log_variance'][0]).tolist()), 0))
            else:
                Res_matlab = np.array(
                    matlabeng.main_BO_alessio(matlab.double(X.tolist()), matlab.double(Xbtrain.tolist()),
                                              matlab.double(np.exp(self.log_params['log_lengthscale'][0]).tolist()),
                                              matlab.double(np.exp(self.log_params['log_variance'][0]).tolist()), 0))
            # opts.SE, opts.sigmae2, D.fMAP, opts.Sigma, opts.LambdaMAP
            print(self.log_params)
            ell = np.atleast_2d(np.sqrt(Res_matlab[0]['l2']))
            variancef = np.atleast_1d(Res_matlab[0]['sigmaf2'])
            variancee = np.atleast_1d(Res_matlab[1])  # ['sigmaf2']
            print("Parameters: ", ell, variancef, variancee)
            fMAP = np.array(Res_matlab[2])
            Sigma = np.array(Res_matlab[3])
            LambdaMAP = np.array(Res_matlab[4])
            # self.params['log_lengthscale'][0] =np.log(ell)
            # self.params['log_variance'][0] =np.log(np.array([variancee])) #AAAAAAAAAA
            self.log_params = {'log_lengthscale': (np.log(np.ones((1, X.shape[1])) * ell), [[-5.0, 5.0]] * X.shape[1]),
                           'log_variance': (np.log(np.array([variancef])), [[-5.0, 5.0]])}
            self.fMAP = fMAP
            self.SigmaMAP = Sigma
            self.LambdaMAP = LambdaMAP
            matlabeng.exit()

    def find_next(self, oracle, iter_s, ninit=1000, niter=50, update_model=True):

        """
        Find the next input

        oracle oracle function
        iter_s iteration number
        ninit number of initial points for the acquistion function optimization
        niter number of optimization iterations 
        update_model a flag to choose whether to update the hyper-parameters
        """

        W = compute_W(self.X, np.vstack(self.Class), np.vstack(self.Preference))
        self.update_surrogate(self.X, W, iter_s, update_model)
        np.random.seed(iter_s)
        # Fref = self.SGP.predict(self.Xref,nsamples=self.nsamples, tune=30, method="LinEss", points=self.samples)
        # self.Fref = np.mean(Fref,axis=1)

        bb = np.vstack(self.bounds)
        # print(bb)
        xx = bb[:, 0:1] + (bb[:, 1:2] - bb[:, 0:1]) * np.random.rand(self.X.shape[1], ninit)
        # xx = np.linspace(bounds[0][0],bounds[0][1],100)
        # print(xx[:,0])
        # (Xpred,SGP,Xref,samples,X, ell,fMAP,LambdaMAP,SigmaMAP, kernel ,surrogateM='SGP',typea='IGAIN')
        FX = [acquisition(xx[:, i], self.SGP, self.Xref, self.samples, self.nsamples, self.X,
                          self.log_params['log_lengthscale'][0], self.log_params['log_variance'][0], self.fMAP, self.LambdaMAP,
                          self.SigmaMAP, self.kernel, self.surrogateM, self.acquisition, iter_s, self.kappa) for i in
              range(ninit)]
        # print(FX)
        indmin = np.argmin(FX)
        x0 = xx[:, indmin]
        print(x0, FX[indmin])
        args = (self.SGP, self.Xref, self.samples, self.nsamples, self.X, self.log_params['log_lengthscale'][0],
                self.log_params['log_variance'][0], self.fMAP, self.LambdaMAP, self.SigmaMAP, self.kernel, self.surrogateM,
                self.acquisition, iter_s, self.kappa)
        res = minimize(acquisition, bounds=self.bounds, x0=x0, method='l-bfgs-b',
                       options={'disp': True, 'maxiter': 50, 'maxfun': 50}, args=args)
        Xnew = np.atleast_2d(res.x)
        self.X = np.vstack([self.X, Xnew])
        inew = self.X.shape[0] - 1
        # print(Xnew,self.Xref,inew,self.iref,self.oracle(Xnew),self.valid(self.oracle,Xnew))
        try:
            cl, pref = queryf(Xnew, self.Xref, inew, self.iref, self.oracle, self.valid)
        except:
            cl, pref = queryf(Xnew, self.Xref, inew, self.iref, self.oracle, self.valid)
        print(Xnew, self.Xref, self.oracle(Xnew), self.oracle(self.Xref), cl, pref)
        if cl == 1.0:
            # print("AAA",cl,pref)
            if pref[0] == inew:
                self.iref = inew  # we have found a better point
                self.Xref = Xnew  # X[-1:,:]
                # print(Xref)
        self.Class.append(cl)
        if len(pref) > 0:
            self.Preference.append(pref)
        print("Xref=", self.Xref)
