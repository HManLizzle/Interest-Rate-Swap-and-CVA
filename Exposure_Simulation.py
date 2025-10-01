#%%
import numpy as np
from Libs.Vasicek_Class import V_Rates

alpha = 0.15    #speed of mean reversion
b = 0.065       #long-term mean level
sigma = 0.05    #Volatility
r0 = 0.07       #Current repo rate
T = 10
n = 50000
N = 10000

Exposure_Paths = V_Rates(n,alpha, b, sigma, r0, T, N) 
np.savetxt("C:/Users/Human/Desktop/Honours S2/WTW 762 Financial Engineering/CVA_Project/CVA_Code/Exposure_Paths.txt", Exposure_Paths)
# %%
paths = np.loadtxt("C:/Users/Human/Desktop/Honours S2/WTW 762 Financial Engineering/CVA_Project/CVA_Code/Exposure_Paths.txt")
# %%
