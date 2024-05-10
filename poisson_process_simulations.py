# this file contains simulations of homogeneous Poisson process by two methods and the simulation of inhomogeneous Poisson process

def homo_poi_simu(lamb,tl,ini=0,method="uni",plot=False):
    """
    Simulate a single homogeneous poisson process with intensity "lamb".
    Input:
    lamb: the intensity of Poisson proess
    tl: the time length of observation interval 
    ini: the initial time of observation
    method:"uni" or "exp"
        "uni" represent the method of estimating the number of events first, then use uniform distribution to sample the event time.
        "exp" represent the method of sampling inter-arrival time by exponential distribution.
    plot: output a plot or not, default = False
    Output:
    T: a list of event time 
    [lamb]: the intensity of the Poisson process
    [tl]: the time length of observation 
    [ini]: the initial time of observation
    A plot of poisson process
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # if the intensity is non-positive, return none
    if lamb <= 0:
        return [], [lamb], [tl], [ini]
    # if the time length is non-positive, return none
    if tl <= 0:
        return [], [lamb], [tl], [ini]
    
    if method == "uni":
        # the number of events in a t-length interval follows a Poisson distribution with parameter lambda*t
        N = np.random.poisson(lamb*tl)
        # each event time can be sample by uniform distribution in the t-length interval
        T = np.random.uniform(ini,ini+tl,N)
        T = sorted(T)
    
    
    if method == "exp":
        T = [] # list to contain inter-arrival times
        while sum(T) < tl:
            # each inter-arrival time follows an exponential distribution 
            inter_arr = np.random.exponential(1/lamb)
            T.append(inter_arr)
        T.pop() # drop the last one since it is out of time length
        # compute the time of each event by summing up the inter-arrival time 
        T = np.cumsum(T)
        T = T + ini

    if plot: 
        l = len(T)
        x = range(0, l+1)
        T_p = T.copy()
        T_p = np.insert(T_p, 0, ini)
        plt.figure(figsize=(8,4))
        plt.step(T_p, x, where="post")
        plt.plot(T_p, x, "x")
        plt.grid()
        plt.title("counting plot")
        plt.show()
    
    return list(T), [lamb], [tl], [ini]




def inhomo_poi_simu(lamb,tl,ini=0,plot=False):
    """
    Simulate a single inhomogeneous Poisson process with intensity function "lamb".
    Input
    lamb: the intensity function
    tl: the time length of the observation interval
    ini: the initial time of observation
    plot: output a plot or not, default=Flase
    Output
    T: a list of event time
    [lamb]: True or Flase which shows if the intensity function is positive or not
    [tl]: time length of observation 
    [ini]: starting time of observation 
    A plot of poisson process
    """
    import scipy.optimize as opt
    import numpy as np
    import matplotlib.pyplot as plt
    
    # the minimum and maximum of the intensity in the observation interval 
    lamb_min = lamb(opt.fminbound(lamb,ini,ini+tl))
    lamb_max = lamb(opt.fminbound(lambda x: -lamb(x),ini,ini+tl))
    
    # if the intensity function is non-positive, return none
    if lamb_min < float(0):
        return [], [-1], [tl], [ini]
    # if the intensity function is all zero, return none
    if lamb == 0:
        return [], [0], [tl], [ini]
    # if the time length is non-positive, return none
    if tl <= 0:
        return [], [1], [tl], [ini]
    
    
    T1 = [0] # store all events we sampled
    T2 = [] # store events we didn't discard
    
    while T1[-1] < tl:
        # generate inter-arrival time following an exponential distribution
        inter_arr = np.random.exponential(1/lamb_max)
        T1.append(inter_arr+T1[-1])
        u = np.random.uniform()
        if u <= lamb(T1[-1])/lamb_max:
            T2.append(T1[-1])
    
    T = np.array(T2)+ini
    
    if plot: 
        l = len(T)
        x = range(0, l+1)
        T_p = T.copy()
        T_p = np.insert(T_p, 0, ini)
        c = np.linspace(ini, ini+tl,50)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,4))
        ax1.step(T_p, x, where="post")
        ax1.plot(T_p, x, "x")
        ax1.grid()
        ax2.plot(c,lamb(c))
        ax1.set_title("counting plot")
        ax2.grid()
        ax2.set_title("intensity function")
        plt.show()
    
    return list(T), [1], [tl], [ini]

# example of intensity function def f(x): return 2*np.exp(-0.2*x)