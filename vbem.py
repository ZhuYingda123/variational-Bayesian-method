import numpy as np
import numpy.linalg as LA
from scipy import special
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import scipy.stats as st


def KL(alpha1,beta1,mu1,nu1,W1, alpha2,beta2,mu2,nu2,W2, E_pi, E_lnph0, D):
    '''
    This function is used for calculating the KL divergence between the joint variational distribution of model parameters Q and the ground truth posterior P
    '''
    K = len(alpha1)
    #Sort the hyperparameters
    index=np.argsort(mu1[:,1])
    a = np.zeros(K)
    b = np.zeros(K)
    n = np.zeros(K)
    m = np.zeros((K,D))
    W = []
    for k in range(K):
        a[k]=alpha1[index[k]]
        b[k]=beta1[index[k]]
        n[k]=nu1[index[k]]
        m[k,:]=mu1[index[k],:]
        W.append(W1[index[k]])

    alpha1 = a
    beta1 = b
    nu1 = n
    mu1 = m
    W1 = W
    
    G1 = 0.0
    G2 = 0.0
    A1 = np.zeros(K)
    A2 = np.zeros(K)
    first_term = np.zeros(K)
    second_term = np.zeros(K)
    third_term = np.zeros(K)
    forth_term = np.zeros(K)
    for i in range(K):
        G1 = G1+special.gammaln(alpha1[i])
        G2 = G2+special.gammaln(alpha2[i])
        phi1 = 0
        phi2 = 0
        for j in range(D):
            phi1 += special.gammaln((nu1[i]+1-j)/2)
            phi2 += special.gammaln((nu2[i]+1-j)/2)
        A1[i] = -D/2.0*np.log(abs(beta1[i]))+nu1[i]/2.0*np.log(LA.det(W1[i]))+D*nu1[i]/2.0*np.log(2)+phi1
        A2[i] = -D/2.0*np.log(abs(beta2[i]))+nu2[i]/2.0*np.log(LA.det(W2[i]))+D*nu2[i]/2.0*np.log(2)+phi2
        first_term[i] = (nu1[i]-nu2[i])/2.0*E_lnph0[i]
        second_term[i] = nu1[i]/2.0*(np.trace(LA.inv(W2[i]).dot(W1[i]))-D) + nu1[i]/2.0*(beta2[i]*mu2[i].dot(W1[i].dot(mu2[i].T))-beta1[i]*mu1[i].dot(W1[i].dot(mu1[i].T)))
        third_term[i] = nu1[i]*(beta1[i]*mu1[i].dot(W1[i].dot(mu1[i].T))-beta2[i]*mu2[i].dot(W1[i].dot(mu1[i].T)))
        forth_term[i] = D/2.0*beta2[i]/beta1[i]-D/2.0+nu1[i]/2.0*(beta2[i]-beta1[i])*(mu1[i].dot(W1[i].dot(mu1[i].T)))
    lnB1 = G1-special.gammaln(alpha1.dot(np.ones(K)))
    lnB2 = G2-special.gammaln(alpha2.dot(np.ones(K)))
    
    KL1 = (alpha1-alpha2).dot(E_pi)-lnB1+lnB2
    KL2 = (first_term + second_term + third_term + forth_term-A1+A2).dot(np.ones(K))

    return KL1+KL2


class VBEM:
    def __init__(self):
        self._data = None
        self._K = None
        self._alpha0 = None
        self._beta0 = None
        self._nu0 = None
        self._W0 = None
        self._mu0 = None

        self._alpha = None
        self._beta = None
        self._nu = None
        self._W = []
        self._mu = None


    def train(self, data, alpha0, beta0, nu0, W0, mu0, K):
	self._data = data
        self._K = K
        self._alpha0 = alpha0
        self._beta0 = beta0
        self._nu0 = nu0
        self._W0 = W0
        self._mu0 = mu0
	for k in range(self._K):
	    self._W.append(W0)
        numsamples, dim = self._data.shape
        Num_Iter = 0
	Nk_old = np.zeros(self._K)
        while Num_Iter <2000:
	    '''
            VB-M step
            '''
	    if Num_Iter==0:
	    	r_ = KMeans(n_clusters=self._K).fit_predict(self._data)
                r = np.zeros((numsamples, self._K))
                for i in range(numsamples):
		    r[i, r_[i]] = 1
	    	Nk = np.zeros(self._K)
	    	for k in range(self._K):
		    Nk[k] = len(np.nonzero(r_[:]==k)[0])
            else:
            	Nk = r.T.dot(np.ones(numsamples))
	    Nk[np.nonzero(Nk[:]<10)] = 10
	    if LA.norm(Nk-Nk_old) < 0.000001:
 		print "stop condition reached!"
		break
            self._alpha = self._alpha0*np.ones(K)+Nk
            self._beta = self._beta0*np.ones(K)+Nk
            self._mu = ((r.T.dot(self._data)).T/self._beta).T
            self._nu = self._nu0*np.ones(K)+Nk
            for k in range(self._K):
                self._W[k] = self._W0 - self._beta[k]*self._mu[k:(k+1)].T.dot(self._mu[k:(k+1)]) + (self._data.T*r[:,k]).dot(self._data)

	    '''
            VB-E step
            '''
            ro,kl = self.posterior()
            KLf[Num_Iter]=kl
            ro_sum = ro.dot(np.ones(self._K))
            r = (ro.T/ro_sum).T #r described the distribution of category label Z
            Nk_old = Nk
            Num_Iter += 1
            print "%d-th iteration, the KL is %f" %(Num_Iter,kl)
        cluster = np.argmax(r, axis = 1)
        return cluster


    def posterior(self):
        W_inv = []
        numsamples, dim = self._data.shape
        Lamda_ba = np.zeros(self._K)
        pi_ba = np.zeros(self._K)
        ro = np.zeros((numsamples, self._K))
        for k in range(self._K):
            W_inv.append(LA.inv(self._W[k]))
            phi0 = 0
            for i in range(dim):
                phi0 += special.psi((self._nu[k]+1-i)/2)

            Lamda_ba[k] = np.exp(phi0 + dim*np.log(2)+np.log(LA.det(W_inv[k])))
            pi_ba[k] = np.exp(special.psi(self._alpha[k])-special.psi(self._alpha.dot(np.ones(self._K))))
        for k in range(self._K):
            ro[:,k:k+1] =pi_ba[k]*np.sqrt(Lamda_ba[k])*np.exp(-dim/(2*self._beta[k]))*np.exp(-self._nu[k]/2*(((self._data-np.kron(np.ones((numsamples,1)),self._mu[k,:])).dot(W_inv[k]))*(self._data-np.kron(np.ones((numsamples,1)),self._mu[k,:]))).dot(np.ones((dim,1))))
        kl = KL(self._alpha,self._beta,self._mu,self._nu, W_inv, alpha_g,beta_g,mu_g,nu_g,W_g, np.log(pi_ba), np.log(Lamda_ba), dim)
        return ro,kl


if __name__ == "__main__":
    file = open('data/data.txt','r')
    data=np.loadtxt(file)
    file.close()
    
    alpha0 = 0
    beta0 = 0.5
    nu0 = 2
    W0 = np.array([[1,0],[0,1]])
    mu0 = np.zeros((1,2))
    K = 4
    alpha_g = np.array([1038.30019283, 1017.90672199, 1284.7863048, 1659.00678037])
    beta_g = alpha_g + 0.5*np.ones(4)
    nu_g = alpha_g + 2.0*np.ones(4)
    mu_g = np.array([[-4.0372274,-5.51469245],[2.48384244,-1.01063354],[-3.5278735, 3.0214866],[-0.01768184, 6.50601058]])
    W_g = [LA.inv(np.array([[2233.91380892,78.3113725],[78.3113725,2324.31902559]])),LA.inv(np.array([[639.17974906,-3.09314915],[-3.09314915,648.07961503]])),LA.inv(np.array([[2160.1125502,-1578.7632118],[-1578.7632118,2173.68223529]])),LA.inv(np.array([[1894.4440879,1273.27141417],[1273.27141417,1795.73556566]]))]

    KLf = np.zeros(2000)#record the convergence curve of the KL divergence

    vbem = VBEM()
    cluster = vbem.train(data, alpha0, beta0, nu0, W0, mu0, K)


