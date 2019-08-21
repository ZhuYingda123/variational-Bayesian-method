import numpy as np
import numpy.linalg as LA
from scipy import special
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import scipy.stats as st
import random
from vbem import KL


class SVB:
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
	self._minbatch = 100

        
    def train(self, data, alpha0, beta0, nu0, W0, mu0, K, tau, d0):
	self._data = data
        self._K = K
        self._alpha0 = alpha0
        self._beta0 = beta0
        self._nu0 = nu0
        self._W0 = W0
        self._mu0 = mu0
	self._alpha = alpha0
        self._beta = beta0
        self._nu = nu0
        self._mu = np.zeros((self._K,2))
	for k in range(self._K):
	    self._W.append(W0)
        numsamples, dim = self._data.shape
        Num_Iter = 0
	Nk_old = np.zeros(self._K)
        
        while Num_Iter <10000:
	    '''
            VB-M step
            '''
	    if Num_Iter==0:
	   	index = random.sample(range(0,numsamples),self._minbatch)
		r_ = KMeans(n_clusters=self._K).fit_predict(self._data)
                r = np.zeros((numsamples, self._K))
                for i in range(numsamples):
		    r[i, r_[i]] = 1
	    	r = r[index,:]
	    data2 = np.tile(self._data[index,:], (numsamples/self._minbatch,1))
	    r = np.tile(r,(numsamples/self._minbatch,1))

	    Nk = r.T.dot(np.ones(numsamples))
	    if LA.norm(Nk-Nk_old) == 0.001:
 		print "stop condition reached!"
		break
	    if Num_Iter == 0:
                self._alpha = self._alpha0*np.ones(K)+Nk
                self._beta = self._beta0*np.ones(K)+Nk
                self._mu = ((r.T.dot(data2)).T/self._beta).T
                self._nu = self._nu0*np.ones(K)+Nk
                for k in range(self._K):
                    self._W[k] = self._W0 - self._beta[k]*self._mu[k:(k+1)].T.dot(self._mu[k:(k+1)]) + (data2.T*r[:,k]).dot(data2) #self ._mu[k:(k+1)]
            else:
		alphax = self._alpha0*np.ones(K)+Nk
		self._alpha += ((d0+Num_Iter*tau)**(-1))*(alphax - self._alpha)
		betax = self._beta0*np.ones(K) + Nk
		self._beta += ((d0+Num_Iter*tau)**(-1))*(betax - self._beta)
		mux = ((r.T.dot(data2)).T/betax).T
  		self._mu += ((d0+Num_Iter*tau)**(-1))*(mux - self._mu)
		nux = self._nu0*np.ones(K)+Nk
		self._nu += ((d0+Num_Iter*tau)**(-1))*(nux-self._nu)
		for k in range(self._K):
		    Wx = self._W0 - betax[k]*mux[k:k+1].T.dot(mux[k:k+1]) + (data2.T*r[:,k]).dot(data2)
		    self._W[k] = self._W[k]+((d0+Num_Iter*tau)**(-1))*(Wx-self._W[k])
               
	    '''
            VB-E step
            '''
            index = random.sample(range(0,numsamples),self._minbatch) #generate a batch with batch size equals self._minbatch
            ro,kl = self.posterior(index)
            KLf[Num_Iter]=kl
            ro_sum = ro.dot(np.ones(self._K))
            r = (ro.T/ro_sum).T
            '''
            r described the distribution of category label Z
            '''
            Nk_old = Nk
            Num_Iter += 1
            print "%dth iteration, KL:%f" %(Num_Iter,kl)
            
	r = self.posterior(np.arange(0,numsamples))[0]
	cluster = np.argmax(r, axis = 1)
                
        min_x = np.min(self._data[:,0])
        max_x = np.max(self._data[:,0])
        min_y = np.min(self._data[:,1])
        max_y = np.max(self._data[:,1])
        x = np.arange(min_x-2, max_x+2, 0.2)
        y = np.arange(min_y-2, max_y+2, 0.2)
                
        X, Y = np.meshgrid(x,y)
        xy = np.stack((X, Y), axis=2)
        plt.figure(1, figsize=(15, 15))
        for k in range(self._K):
	    plt.scatter(self._data[np.nonzero(cluster[:]==k),0],self._data[np.nonzero(cluster[:]==k),1],s=200,color=color[k],marker='.')
        for k in range(self._K):
            norm_2d = st.multivariate_normal(self._mu[k], self._W[k]/self._nu[k])
            Z = norm_2d.pdf(xy)
            contour=plt.contour(X, Y, Z, [0.004], colors='darkseagreen', linewidth = 10)
        plt.xlabel('x',fontsize = 30)
        plt.ylabel('y',fontsize = 30)
        plt.savefig('stochastic_vb')
        return cluster


    def posterior(self,index):
        W_inv = []
        dim = self._data.shape[1]
	minbatch = len(index)
        Lamda_ba = np.zeros(self._K)
        pi_ba = np.zeros(self._K)
        ro = np.zeros((minbatch, self._K))
        for k in range(self._K):
            W_inv.append(LA.inv(self._W[k]))
            phi0 = 0
            for i in range(dim):
                phi0 += special.psi((self._nu[k]+1-i)/2)
            Lamda_ba[k] = np.exp(phi0 + dim*np.log(2)+np.log(LA.det(W_inv[k])))
            pi_ba[k] = np.exp(special.psi(self._alpha[k])-special.psi(self._alpha.dot(np.ones(self._K))))

            
        for k in range(self._K):
	    ro[:,k:k+1] =pi_ba[k]*np.sqrt(Lamda_ba[k])*np.exp(-dim/(2*self._beta[k]))*np.exp(-self._nu[k]/2*(((self._data[index,:]-np.kron(np.ones((minbatch,1)),self._mu[k,:])).dot(W_inv[k]))*(self._data[index,:]-np.kron(np.ones((minbatch,1)),self._mu[k,:]))).dot(np.ones((dim,1))))

        kl = KL(self._alpha,self._beta,self._mu,self._nu, W_inv, alpha_g,beta_g,mu_g,nu_g,W_g, np.log(pi_ba), np.log(Lamda_ba), dim)
        return ro,kl


if __name__ == "__main__":
    color = ['green','red','darkslateblue','darkorange','brown','black','blue','darkgreen','yellow','cyan','magenta' 'darkkhaki','darkgray','darkred','chocolate','darkorchid','chartreuse','blanchedalmond','darkmagenta','coral','darkgoldenrod','cornflowerblue','cornsilk','crimson','darkblue','darkcyan',
                 'darkolivegreen','darksalmon','blueviolet',
                 'darkseagreen','aliceblue','antiquewhite','aqua','aquamarine','azure','burlywood','cadetblue','beige','bisque']
    alpha_g = np.array([1038.30019283, 1017.90672199, 1284.7863048, 1659.00678037])
    beta_g = alpha_g + 0.5*np.ones(4)
    nu_g = alpha_g + 2.0*np.ones(4)
    mu_g = np.array([[-4.0372274,-5.51469245],[2.48384244,-1.01063354],[-3.5278735, 3.0214866],[-0.01768184, 6.50601058]])
    W_g = [LA.inv(np.array([[2233.91380892,78.3113725],[78.3113725,2324.31902559]])),LA.inv(np.array([[639.17974906,-3.09314915],[-3.09314915,648.07961503]])),LA.inv(np.array([[2160.1125502,-1578.7632118],[-1578.7632118,2173.68223529]])),LA.inv(np.array([[1894.4440879,1273.27141417],[1273.27141417,1795.73556566]]))]

    KLf = np.zeros(10000)

    file = open('data/data.txt','r')
    data=np.loadtxt(file)
    file.close()
    
    '''
    hyperparameters define
    '''
    alpha0 = 0
    beta0 = 0.5
    nu0 = 2
    W0 = np.array([[1,0],[0,1]])
    mu0 = np.zeros((1,2))
    K = 4
    tau = 1
    d0 = 1
    
    svb = SVB()
    cluster = svb.train(data, alpha0, beta0, nu0, W0, mu0, K, tau, d0)
