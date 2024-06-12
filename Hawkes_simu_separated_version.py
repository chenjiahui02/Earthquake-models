def hawkes_simu_sup(lamb,a,b,tl,ini=0,plot=False):
    """
    Simulate a single Hawkes process by the superposition of Poisson processes
    Input
    lamb: the background intensity (a constant)
    a,b: the parameter in the excitation function, u(t)=a*exp(-b*t)
    tl: the time length of the observation interval
    ini: the initial time of observation
    plot: output a plot or not, default=Flase
    Output
    T: a list of event time
    [lamb,a,b]: lamb,a,b parameters
    [tl]: time length of observation 
    [ini]: starting time of observation 
    A plot of poisson process
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    
    # if the intensity is non-positive, return none
    if lamb <= 0:
        return [], [0,a,b], [tl], [ini]
    # if the time length is non-positive, return none
    if tl <= 0:
        return [], [lamb,a,b], [tl], [ini]
    
    # the number of parent events triggered from background intensity, which follows Poi(lambda*t)
    N_parent = np.random.poisson(lamb*tl)
    # the arrival time of each parent event can be generated by Uniform(ini,ini+tl)
    T_parent = np.random.uniform(ini, ini+tl, N_parent)
    
    
    # initialise a list to contain the times of offspring events of all generations
    T_offspring = []    
    
    
    
    
    
    T0 = T_parent.copy().tolist()
    while T0:
        t = T0.pop(0)
        D = np.random.poisson(a/b)
        if D == 0:
            continue
        # the arrival time of offspring event of one background event is T_bg + E_i where E_i follows Exp(b)
        T_arrival = np.random.exponential(1/b,D)
        # the time of offsprings
        T1 = t + T_arrival
        # drop the event out of the observation window
        T1 = T1[T1 < ini+tl]
        # append the new event time to T0 and T_offspring
        T0 = T0 + T1.tolist()
        T_offspring = T_offspring + T1.tolist()
    
    T = np.array(sorted(list(T_parent) + T_offspring))    
    
    if plot: 
        plt.figure(figsize=(8,4))
        x = np.linspace(ini, ini+tl, tl*20)
        y = [lamb + a * np.sum(np.exp(-b * (i - T[T<=i]))) for i in x]
        plt.plot(x, y, "red", linewidth=1.0, alpha=0.7)
        plt.plot(T_parent, np.ones_like(T_parent)*lamb, "x", color="k", label="parent events")
        plt.plot(T_offspring, np.ones_like(T_offspring)*lamb + 1, "o", color="k", label="offspring events")
        plt.legend()
        plt.grid()
        plt.xlabel("time")
        plt.ylabel("intensity")
        plt.title("Simulation of Hawkes process")
        plt.show()

    return list(T), [lamb,a,b], [tl], [ini]


#example hawkes_simu_sup(0.5,1,2,30,plot=True)


def hawkes_simu_inhomo(lamb,a,b,tl,ini=0,plot=False):
    """
    Simulate a single Hawkes process by the thinning method from simulating inhomogeneous Poisson processes
    Input
    lamb: the background intensity (a constant)
    a,b: the parameter in the excitation function, u(t)=a*exp(-b*t)
    tl: the time length of the observation interval
    ini: the initial time of observation
    plot: output a plot or not, default=Flase
    Output
    T: a list of event time
    [lamb,a,b]: lamb,a,b parameters
    [tl]: time length of observation 
    [ini]: starting time of observation 
    A plot of poisson process
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    
    # if the intensity is non-positive, return none
    if lamb <= 0:
        return [], [0,a,b], [tl], [ini]
    # if the time length is non-positive, return none
    if tl <= 0:
        return [], [lamb,a,b], [tl], [ini]
    
    
    T1 = [0] # store all events we sampled
    T2 = [] # store events we didn't discard
    intensity_maxs = [lamb]

    # sample the first event time by distribution exp(lamb)  (homo Poisson process)
    inter_arr = np.random.exponential(1/lamb)
    T1.append(inter_arr)
    T2.append(inter_arr)
    int_max = lamb+a
    intensity_maxs.append(int_max)

    # we define the max intensity function that will be used in the iterations below to compute the max intensity
    def intensity(t):
        return lamb + np.sum(a*np.exp(-b*(t-np.array(T2))))

    while T1[-1] < tl:
        # generate rest inter-arrival times by thinning inhomo Poisson process      
        inter_arr = np.random.exponential(1/int_max)
        T1.append(inter_arr+T1[-1])
        u = np.random.uniform()
        if u <= intensity(T1[-1])/int_max:
            T2.append(T1[-1])
            int_max = intensity(T2[-1])
            intensity_maxs.append(int_max)
   

    T = np.array(T2)+ini
    T = T[T <= ini+tl]
    
    if plot:
        plt.figure(figsize=(8,4))
        
        # plot intensity 
        x = np.linspace(ini, ini+tl, tl*20)
        y = [lamb + a * np.sum(np.exp(-b * (i - T[T<=i]))) for i in x]
        plt.plot(x, y, "red", linewidth=1.0, alpha=0.7)
        plt.plot(T, np.ones_like(T)*lamb, "kx")
        plt.xlabel("time")
        plt.ylabel("intensity")
        plt.title("Simulation of Hawkes process")
        plt.grid()
        plt.show()
    
    return list(T), [lamb,a,b], [tl], [ini]


#example hawkes_simu_inhomo(0.5,1,2,30,plot=True)