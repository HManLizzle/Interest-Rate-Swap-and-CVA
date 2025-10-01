import numpy as np 
import matplotlib.pyplot as plt

"""
Vasicek model parameters

alpha = 0.15    #speed of mean reversion
b = 0.065       #long-term mean level
sigma = 0.05    #Volatility
r0 = 0.07       #Current repo rate
W_t             #Wiener process (to be discretised)
"""

def vasicek_EM(alpha, b, sigma, r0, T, N): 
    delta_t = T/N
    r_t = np.zeros(N)
    r_t[0] = r0
    for t in range(0,N-1): 
        # Using Euler Maruyama for speed
        r_t[t+1] = r_t[t] + alpha*(b - r_t[t])*delta_t + sigma*np.sqrt(delta_t)*np.random.standard_normal()

    return(r_t)


alpha = 0.15  #speed of mean reversion
b = 0.065     #long-term mean level
sigma = 0.05    #Volatility
r0 = 0.07     #Current repo rate
theta = alpha*b
T = 10
N = 10000
pf = 0.25
#print(vasicek_EM(alpha,b,sigma,r0,T,N))

def plot_Vasicek(n, alpha, b, sigma, r0, T, N): 
    delta_t = T/N
    time = np.arange(0,T,delta_t)
    f = plt.figure()
    f.set_figwidth(10)
    f.set_figheight(4)
    for i in range(0,n):
        vi = vasicek_EM(alpha,b,sigma,r0,T,N)
        plt.plot(time,vi, linewidth = 0.7)
        
    plt.title("Simulated Paths of the Vasicek Model")
    plt.xlabel("Time (t years)")
    plt.ylabel("Rate (r)")
    plt.grid(True)
    #plt.show()

n = 50000
plot_Vasicek(n,alpha,b,sigma,r0,T,N)
# %%
#Computing floating rates 
n = 50000
paths = np.zeros((n,N))
for i in range(0,n):
    vi = vasicek_EM(alpha,b,sigma, r0,T,N)
    paths[i,:] = vi
    
# %%
curve = np.zeros(N)
for j in range(0,N): 
    mean = np.mean(paths[:,j])
    curve[j] = mean

np.savetxt("C:/Users/Mean_Path_Alt.txt", curve)


T = [0.25*i for i in range(1,int(T/pf))]

for j in T: 
    paths = np.zeros((n,N))
    for i in range(0,n): 
        vi = vasicek_EM(alpha,b,sigma, r0,T,N)
        paths[i,:] = vi
    

# Plotting Some Paths and b of the  Vasicek Model
n= 10
times = np.arange(0,T,(T/N))
plot_Vasicek(n,alpha,b,sigma,r0,T,N)
b_ = [b for i in times]
plt.plot(times,b_, color = "black", linewidth = 2, label = "Long-Term Mean (b)")
plt.legend()
plt.show()

# Constructing the Yield Curve
r_ts = np.loadtxt("C:/CVA_Code/Mean_Path_Alt.txt")
pers = int(T/pf)
t = [pf*i for i in range(1,pers +1)]
y = np.zeros(pers)

for i in range(0,pers):
    B = (1/alpha)*(1-np.exp(-alpha*(t[i])))
    A = np.exp((b-0.5*sigma**2/alpha**2)*(B-t[i]) - (B**2)*sigma**2/(4*alpha))
    P = A*np.exp(-B*r_ts[i])
    y[i] = -(np.log(P))/t[i]

np.savetxt("C:/Users/Human/Desktop/Honours S2/WTW 762 Financial Engineering/CVA_Project/CVA_Code/Yield_Curve_Alt.txt", y)

plt.plot(t,y, color = "blue", marker = 'o', mfc = 'orange',mec = 'orange', ms = 5)
plt.title("Yield Curve for Simulated Short Rates (Vasicek Model)")
plt.xlabel("Time to Maturity")
plt.ylabel("Interest Rate (in decimal)")
plt.grid()
plt.show()


