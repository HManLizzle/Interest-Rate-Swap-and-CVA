#%%
import numpy as np 
import matplotlib.pyplot as plt

"""
PARAMETERS 

r_fi = 0.060452      #Fixed Rate 
pf = 0.25           #Payment Frequancy (quarterly)
L = 1000000         #Notional Amount (R1 000 000)
T = 10              #Maturity

r_fl = 'read from data set' ()
curve = np.loadtxt("C:/Users/Human/Desktop/Honours S2/WTW 762 Financial Engineering/CVA_Project/CVA_Code/Term_Structure.txt")
Note that N = 10 000 for curve so, 1 year will be at curve[999] (approximately)

DF = 'calculated from data set'     #Discount Factors 

CASHFLOWS 
 -> Fixed Leg

"""

#Parameters
r0 = 0.07           #Current Repo Rate
r_fi = 0.060452     #Fixed Rate 
pf = 0.25           #Payment Frequancy (quarterly)
L = 1000000         #Notional Amount (R1 000 000)
T = 10              #Maturity

r_fi *= pf
#Yield Curve
curve = np.loadtxt("C:/Users/Human/Desktop/Honours S2/WTW 762 Financial Engineering/CVA_Project/CVA_Code/Yield_Curve_Alt.txt")
N = np.size(curve)

# %%
#Calculating Floating Rates r_fl

pers = int(T/pf) #=40
r_fl = np.zeros(int(pers)+1)
r_fl[0] = r0
for i in range(1,int(pers)+1): 
    r_fl[i] = curve[int(N/pers)*(i-1)]

# %%
#Calculating the dicount factors [DF] (assuming continuous compounding)
t = [pf*i for i in range(0,int(T/pf)+1)]
DF = np.zeros(int(T/pf)+1)

for i in range(0,len(t)): 
    DF[i] = np.exp(-t[i]*r_fl[i])

K = (DF[0]-DF[-1])/np.sum(0.25*DF)
print(K)
# %%
#Forward rate calculations
fwd_rate = np.zeros(pers)
#fwd_rate[0] = r0
for i in range(0,pers):
    fwd_rate[i] = (1/pf)*(DF[i]/DF[i+1] - 1)

# %%
#Alternative formula 
m = 40
r_fi = 0.060452
tau = 0.25
V = L*(DF[m]- DF[0])+L*np.sum(tau*r_fi*DF[1:m+1])

print("V = ", V)
# %%

