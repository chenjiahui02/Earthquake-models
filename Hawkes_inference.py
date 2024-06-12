import numpy as np
import matplotlib.pyplot as plt


## MLE
def Hawkes_MLE(T, tl, iters = 10, plot=False):
    """
    Computing MLE of a Hawkes process.
    """
    import scipy.optimize as sc
    
    
    def minus_log_likelihood(theta, T, tl):
        """
        Expression of log likelihood by the recursive structure A.
        """
        import numpy as np

        lamb, a, b = theta
        N = len(T)
        A = np.zeros(N)
        A[0] = 0

        for i in range(1,N):
            A[i] = np.exp(-b*(T[i] - T[i-1])) * (1 + A[i-1])

        l = -lamb*tl
        for i, t in enumerate(T):
            l += np.log(lamb + a * A[i]) - a/b * (1 - np.exp(-b *(tl - t)))

        return -l
    
    parameters = []
    for i in range(iters):
        x0 = np.random.rand(3)
        res = sc.minimize(minus_log_likelihood, x0, args=(T, tl), bounds=((1e-5, np.inf),(1e-5, np.inf),(1e-5, np.inf)))
        parameters.append(res.x.tolist())
        
    x = np.array(parameters)
    if plot:
        plt.figure()
        plt.hist(x[:,0], alpha=0.6, bins=1, label="estimated mu")
        plt.hist(x[:,1], alpha=0.6, label="estimated alpha")
        plt.hist(x[:,1], alpha=0.6, label="estimated beta")
        plt.legend()
        plt.show()
        
    return x, np.mean(x, axis=0)



## EM


def Hawkes_EM(T, tl, eta=0.001):

    import numpy as np
    import scipy.optimize as sc
    
    def minus_complete_log_likelihood(theta, Z, T, tl):
        """
        Calculate the complete data log likelihood.
        input:
        theta: a list of parameters, lamb, a, b
        Z: a matrix of N x N where N is the number events in T
        T: a list of time of observed events
        tl: the time length of observation
        output:
        l: a scalar value of log likelihood
        """
        import numpy as np

        lamb, a, b = theta
        T = np.array(T)
        l = - lamb * tl + np.log(lamb) * np.sum(Z[0]) - a/b * np.sum(1 - np.exp(-b*(tl - T)))

        for i in range(1, len(T)):
            l += np.sum(Z[1:i+1, i] * (np.log(a)-b*(T[i] - T[0:i])))

        return -l

    def update_Z(theta, T):
        """
        Update Z matrix by new parameters.
        Input:
        theta: a list of parameters, lamb, a, b
        T: a list of time of observed events
        output:
        Z: the updated Z matrix 
        """
        import numpy as np

        lamb, a, b = theta
        T = np.array(T)
        N = len(T)
        Z = np.zeros((N, N))
        Z[0,0] = 1

        for i in range(1, N):
            terms = a*np.exp(-b*(T[i] - T[0:i]))
            numerator = np.append([lamb], terms)
            demoninator = lamb + np.sum(terms)
            Z[0:i+1, i] = numerator / demoninator

        return Z
    
    # initialisation
    theta0 = np.random.rand(3)
    diff = 1
    while diff > eta:
        Z = update_Z(theta0,T) 
        x0 = np.random.rand(3)
        res = sc.minimize(minus_complete_log_likelihood, x0, args=(Z, T, tl), bounds=((1e-5, np.inf),(1e-5, np.inf),(1e-5, np.inf)))
        theta1 = res.x
        diff = -(minus_complete_log_likelihood(theta1, Z, T, tl)-minus_complete_log_likelihood(theta0, Z, T, tl))
        theta0 = theta1
    
    return theta0


## MCMC


def Hawkes_MCMC(T, tl, max_iters=2000, burnin=500, plot=False):

    def log_likelihood(lamb, a, b, T, tl):
        """
        Expression of log likelihood by the recursive structure A.
        """
        import numpy as np

        N = len(T)
        A = np.zeros(N)
        A[0] = 0

        for i in range(1,N):
            A[i] = np.exp(-b*(T[i] - T[i-1])) * (1 + A[i-1])

        l = -lamb*tl
        for i, t in enumerate(T):
            l += np.log(lamb + a * A[i]) - a/b * (1 - np.exp(-b *(tl - t)))

        return l

    # assume lamb, a, b follow a uniform(0,5) iid
    # assume the proposals of lamb'|lamb, a'|a, b'|b are N(lamb,1), N(a,1), N(b,1) respectively
    # thus the log acceptance rate is min(logA, 0) where logA = loglikelihood(parameter') - loglikelihood(parameter)

    # initialise parameters
    lamb, a, b = np.random.rand(3)
    parameter_list = [0] * max_iters
    k = 0

    while k < max_iters:
        # sample lamb, a, b in order 
        # sample lamb
        lamb1 = np.random.normal(lamb, 1)
        A = np.exp(log_likelihood(lamb1,a,b,T,tl) - log_likelihood(lamb,a,b,T,tl))
        u = np.random.uniform()
        if u < min(A, 1):
            lamb = lamb1
        # sample a
        a1 = np.random.normal(a, 1)
        A = np.exp(log_likelihood(lamb,a1,b,T,tl) - log_likelihood(lamb,a,b,T,tl))
        u = np.random.uniform()
        if u < min(A, 1):
            a = a1
        # sample b
        b1 = np.random.normal(b, 1)
        A = np.exp(log_likelihood(lamb,a,b1,T,tl) - log_likelihood(lamb,a,b,T,tl))
        u = np.random.uniform()
        if u < min(A, 1):
            b = b1

        parameter_list[k] = [lamb, a, b]
        k += 1
    
    parameter_list = np.array(parameter_list)
    
    if plot:
        plt.figure()
        plt.hist(parameter_list[burnin:,0], bins=50, alpha=0.6, label="samples of lamb")
        plt.hist(parameter_list[burnin:,1], bins=50, alpha=0.6, label="samples of a")
        plt.hist(parameter_list[burnin:,2], bins=50, alpha=0.6, label="samples of b")
        plt.plot(np.mean(parameter_list[burnin:,0]),10, "x", label="lamb")
        plt.plot(np.mean(parameter_list[burnin:,1]),10, "x", label="a")
        plt.plot(np.mean(parameter_list[burnin:,2]),10, "x", label="b")
        plt.legend()
        plt.show()
        
    return np.mean(parameter_list[burnin:,:], axis=0)