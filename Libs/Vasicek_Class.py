
import numpy as np 
import matplotlib.pyplot as plt 

def vasicek_r(n,alpha, b, sigma, r0, L, T, N): 
    delta_t = T/N
    r_t = np.zeros(N)
    sims = np.zeros((n,N)) #An nxN matrix containing all paths
    pf = 0.25
    pers = int(T/pf)

    for i in range(0,n):
        #np.random.seed(i) (Not neccesary. Just for in case)
        r_t = np.zeros(N)
        r_t[0] = r0
        
        for t in range(0,N-1): 
            r_t[t+1] = r_t[t] + alpha*(b - r_t[t])*delta_t + sigma*np.sqrt(delta_t)*np.random.standard_normal()
            #r_t[t+1] = r_t[t]*np.exp(-alpha*delta_t) + b*(1-np.exp(-alpha*delta_t)) + sigma*np.sqrt((1-np.exp(-2*delta_t)/2*alpha))*np.random.standard_normal()
        sims[i,:] = r_t

    mean_path = np.zeros(N)
    for j in range(0,N):
        mean_path[j] = np.mean(sims[:,j]) 
        
    #YIELD CURVE 
    t = np.array([pf*t for t in range(1,pers+1)])
    y = np.zeros(len(t))

    alpha = 0.15  #speed of mean reversion
    b = 0.065     #long-term mean level
    sigma = 0.05    #Volatility
    r0 = 0.07     #Current repo rate

    for i in range(0,len(t)):
        B = (1/alpha)*(1-np.exp(-alpha*(t[i])))
        A = np.exp((b-0.5*sigma**2/alpha**2)*(B-t[i]) - (B**2)*sigma**2/(4*alpha))
        P = A*np.exp(-B*mean_path[i])
        y[i] = -(np.log(P))/t[i]

    return(y)

def swap_w_curve(L,T,y):
    pf = 0.25
    pers = int(T/pf)
    r0 = 0.07
    t = np.array([pf*i for i in range(0,pers+1)])

    #DISCOUNT FACTORS
    DF = np.zeros(pers+1)
    DF[0] = 1
    for i in range(1,len(t)): 
        DF[i] = np.exp(-t[i]*y[i-1]) #(assuming continuous compounding)

    #FORWARD RATE
    fwd_rate = np.zeros(pers)
    for i in range(0,pers):
        fwd_rate[i] = (1/pf)*(DF[i]/DF[i+1] - 1)

    #SWAP VALUATION
    m = int(T/pf) # = 40
    r_fi = 0.060452
    tau = 0.25

    V = np.sum(DF[1:m+1]*L*tau*(fwd_rate[0:m] - r_fi))

    return(V)

def exposure(n,alpha,b, sigma, r0, L, T, N):
    delta_t = T/N
    r_t = np.zeros(N)
    sims = np.zeros((n,N)) #An nxN matrix containing all paths
    pf = 0.25
    pers = int(T/pf)
    r_fi = 0.060452

    for i in range(0,n):
        #np.random.seed(i) (Not neccesary. Just for in case)
        r_t = np.zeros(N)
        r_t[0] = r0
        
        for t in range(0,N-1): 
            r_t[t+1] = r_t[t] + alpha*(b - r_t[t])*delta_t + sigma*np.sqrt(delta_t)*np.random.standard_normal()
            #r_t[t+1] = r_t[t]*np.exp(-alpha*delta_t) + b*(1-np.exp(-alpha*delta_t)) + sigma*np.sqrt((1-np.exp(-2*delta_t)/2*alpha))*np.random.standard_normal()
        sims[i,:] = r_t

    #DISCOUNT FACTORS
    V = 0
    t = np.arange(T/N,T + T/N ,T/N)
    r = sims[1,:]
    B = (1/alpha)*(1-np.exp(-alpha*(t)))
    A = np.exp((b-0.5*sigma**2/alpha**2)*(B-t) - (B**2)*sigma**2/(4*alpha))
    P = A*np.exp(-B*r)

    V = L*(P[len(t)-1] - P[0]) + L*sum(pf*r_fi*P)

    return(V)

def V_Rates(n,alpha, b, sigma, r0, T, N):
    delta_t = T/N
    r_t = np.zeros(N)
    sims = np.zeros((n,N)) #An nxN matrix containing all paths
    pf = 0.25
    pers = int(T/pf)

    for i in range(0,n):
        #np.random.seed(i) (Not neccesary. Just for in case)
        r_t = np.zeros(N)
        r_t[0] = r0
        
        for t in range(0,N-1): 
            r_t[t+1] = r_t[t] + alpha*(b - r_t[t])*delta_t + sigma*np.sqrt(delta_t)*np.random.standard_normal()
            #r_t[t+1] = r_t[t]*np.exp(-alpha*delta_t) + b*(1-np.exp(-alpha*delta_t)) + sigma*np.sqrt((1-np.exp(-2*delta_t)/2*alpha))*np.random.standard_normal()
        sims[i,:] = r_t
    
    return(sims)
