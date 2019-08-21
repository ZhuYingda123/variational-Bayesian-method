import numpy as np
import numpy.linalg as LA
from scipy import special
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import scipy.stats as st
import multiprocessing
import time
from vbem import KL


class DVBEM(multiprocessing.Process):
    def __init__(self, data, alpha0, beta0, nu0, W0, mu0, K, A, nodeid, mu_p, sd, sv, sem, N):
        multiprocessing.Process.__init__(self)
        self.daemon = True
        self._data = data
        self._K = K
        self._id = id
        self._alpha0 = alpha0
        self._beta0 = beta0
        self._nu0 = nu0
        self._W0 = W0
        self._mu0 = mu0

        self._alpha = None
        self._beta = None
        self._nu = None
        self._W = []
        self._mu = None

        self._A = A
        self._id = nodeid
        self._mu_p = mu_p
        self._sd = sd
        self._sv = sv
        self._sem = sem
        self._N = N

	self.KLf = np.zeros(2000)

    def run(self):
        start = time.time()
        cluster = self.train()
	end = time.time()
	print 'process %d complete! running time: %f' %(self._id, end-start)

	#np.savetxt("Dkl/KL_d_4_"+str(self._id)+".txt",self.KLf)
	#np.savetxt("Dkl/cent_4_"+str(self._id)+".txt",self._mu)

        
        
    def train(self):
	for k in range(self._K):
	    self._W.append(self._W0)
        numsamples, dim = self._data.shape
        numnodes = self._A.shape[0]
        B = np.sum(self._A, axis=1)
        Num_Iter = 0
	Nk_old = np.zeros(self._K)
        self._beta = np.zeros(self._K)
        self._mu = np.zeros((self._K, 2))
        self._nu = np.zeros(self._K)
	self._alpha = np.zeros(self._K)
        
        while Num_Iter <500:
            '''
            Broadcast step
            '''
            self._sem.acquire()
            for j in range(self._A.shape[0]):
                if self._A[self._id, j]!=0:
                    self._sd[self._id][0][j] = self._beta
                    self._sd[self._id][1][j] = self._mu
                    self._sd[self._id][2][j] = self._nu
                    self._sd[self._id][3][j] = self._W
		    self._sd[self._id][4][j] = self._alpha
                    
            self._sv.value += 1
            self._sem.release()
 	    while (self._sv.value//self._A.shape[0])==Num_Iter:
         	pass
            
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
		    Nk[k] = self._N/numsamples*len(np.nonzero(r_[:]==k)[0])
            else:
            	Nk = self._N/numsamples*r.T.dot(np.ones(numsamples))
	    Nk[np.nonzero(Nk[:]<10)] = 10
            '''
            calculate B_ik, m_1ki, etc.
            '''
            B_ik = np.zeros(self._K)
	    N_ik = np.zeros(self._K)
            m_1kiT = np.zeros((2, self._K))
            nu_ki = np.zeros(self._K)
            W_ki = []
            for k in range(self._K):
	        W_ki.append([[0,0],[0,0]])
            for j in range(self._A.shape[0]):
                if self._A[self._id, j]!=0:
                    '''
                    0---beta_jk
                    1---mu_jk
                    2---nu_jk
                    3---W_jk
		    4---alpha_jk
                    '''
		    N_ik += self._sd[j][4][self._id]
                    B_ik += self._sd[j][0][self._id]
                    m_1kiT += self._sd[j][0][self._id]*self._sd[j][1][self._id].T
                    nu_ki += self._sd[j][2][self._id]
                    for k in range(self._K):
                        W_ki[k] += self._sd[j][3][self._id][k] + self._sd[j][0][self._id][k]*self._sd[j][1][self._id][k:(k+1)].T.dot(self._sd[j][1][self._id][k:(k+1)])
	    self._alpha = (self._alpha0*np.ones(K)+Nk+self._mu_p*N_ik)/(1+self._mu_p*B[self._id])
            self._beta = (self._beta0*np.ones(K)+Nk+self._mu_p*B_ik)/(1+self._mu_p*B[self._id])
            self._mu = (((self._N/numsamples*r.T.dot(self._data)).T+self._mu_p*m_1kiT)/(self._beta0*np.ones(K)+Nk+self._mu_p*B_ik)).T
            self._nu = (self._nu0*np.ones(K)+Nk+self._mu_p*nu_ki)/(1+self._mu_p*B[self._id])
            for k in range(self._K):
                self._W[k] = (self._W0 + self._mu_p*W_ki[k] + (self._N/numsamples)*(self._data.T*r[:,k]).dot(self._data))/(1+self._mu_p*B[self._id]) - self._beta[k]*self._mu[k:(k+1)].T.dot(self._mu[k:(k+1)]) 
	    '''
            VB-E step
            '''
            ro,kl = self.posterior()
            self.KLf[Num_Iter]=kl
            ro_sum = ro.dot(np.ones(self._K))
            r = (ro.T/ro_sum).T
            '''
            r described the distribution of category label Z
            '''
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
            #print "pi", pi_ba[k]

        for k in range(self._K):
            ro[:,k:k+1] =pi_ba[k]*np.sqrt(Lamda_ba[k])*np.exp(-dim/(2*self._beta[k]))*np.exp(-self._nu[k]/2*(((self._data-np.kron(np.ones((numsamples,1)),self._mu[k,:])).dot(W_inv[k]))*(self._data-np.kron(np.ones((numsamples,1)),self._mu[k,:]))).dot(np.ones((dim,1))))
        kl = KL(self._alpha,self._beta,self._mu,self._nu, W_inv, alpha_g,beta_g,mu_g,nu_g,W_g, np.log(pi_ba), np.log(Lamda_ba), dim)
        return ro,kl


if __name__ == "__main__":
    alpha_g = np.array([1038.30019283, 1017.90672199, 1284.7863048, 1659.00678037])
    beta_g = alpha_g + 0.5*np.ones(4)
    nu_g = alpha_g + 2.0*np.ones(4)
    mu_g = np.array([[-4.0372274,-5.51469245],[2.48384244,-1.01063354],[-3.5278735, 3.0214866],[-0.01768184, 6.50601058]])
    W_g = [LA.inv(np.array([[2233.91380892,78.3113725],[78.3113725,2324.31902559]])),LA.inv(np.array([[639.17974906,-3.09314915],[-3.09314915,648.07961503]])),LA.inv(np.array([[2160.1125502,-1578.7632118],[-1578.7632118,2173.68223529]])),LA.inv(np.array([[1894.4440879,1273.27141417],[1273.27141417,1795.73556566]]))]

    numprocess = 50
    data = []
    for i in range(numprocess):
        file = open('data/data'+str(i)+'.txt','r')
        data.append(np.loadtxt(file))
        file.close()
        
    N=0
    for i in range(numprocess):
        N += data[i].shape[0]
        
        
    sd = [[] for i in range(numprocess)]
    for i in range(numprocess):
        for j in range(5):
            sd[i].append(multiprocessing.Manager().dict()) 
    sv = multiprocessing.Manager().Value('d',0)
    sem=multiprocessing.Semaphore(1)

    '''
    Read adjacency matrix A from txt file!
    '''
    file1 = open("A.txt","r")
    A = np.loadtxt(file1)


    '''
    hyperparameters define
    '''
    alpha0 = 0
    beta0 = 0.5
    nu0 = 2
    W0 = np.array([[1,0],[0,1]])
    mu0 = np.zeros((1,2))
    K = 4
    mu_p = 4
    
    ps = []
    
    for i in range(numprocess):
        p = DVBEM(data[i], alpha0, beta0, nu0, W0, mu0, K, A, i, mu_p, sd, sv, sem, N)
        p.start()
        ps.append(p)
        
    for p in ps:
        p.join()

