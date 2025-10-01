#%%
import numpy as np
import matplotlib.pyplot as plt

# Parameters
alpha = 0.15  # Speed of mean reversion
b = 0.065     # Long-term mean
sigma = 0.05  # Volatility
r0 = 0.07     # Initial short rate
T = 10.0      # Maturity in years
delta = 0.25  # Quarterly payments
n_steps = int(T / delta)  # Number of steps
t = np.arange(0, T + delta, delta)  # Time grid
n_paths = 50000  # Number of simulation paths (increase to 10000 for accuracy)
K = 0.060452  # Fixed rate
notional = 1000000  # Notional amount (scale to 1000000 if needed)
h = 0.083333  # Hazard rate
R = 0.5       # Recovery rate

# Simulate short rate paths using the exact Vasicek solution
dt = delta
exp_alpha_dt = np.exp(-alpha * dt)
var_term = sigma ** 2 * (1 - np.exp(-2 * alpha * dt)) / (2 * alpha)
std = np.sqrt(var_term)
r = np.zeros((n_paths, n_steps + 1))
r[:, 0] = r0
for i in range(1, n_steps + 1):
    r[:, i] = r[:, i-1] * exp_alpha_dt + b * (1 - exp_alpha_dt) + std * np.random.normal(0, 1, n_paths)

# Function to compute B(tau)
def B(alpha, tau):
    return (1 - np.exp(-alpha * tau)) / alpha

# Function to compute A(tau)
def A(alpha, b, sigma, tau):
    B_val = B(alpha, tau)
    return (b - sigma**2 / (2 * alpha**2)) * (B_val - tau) - (sigma**2 * B_val**2) / (4 * alpha)

# Compute expected exposure EE_t at each t_i
EE = np.zeros(n_steps + 1)
for i in range(n_steps + 1):
    V_swap = np.zeros(n_paths)
    for j in range(n_paths):
        rt = r[j, i]
        fixed_pv = 0.0
        float_pv = 0.0
        remaining_times = t[i+1:] if i < n_steps else np.array([])
        if len(remaining_times) > 0:
            tau = remaining_times - t[i]
            B_vals = B(alpha, tau)
            A_vals = A(alpha, b, sigma, tau)
            P = np.exp(A_vals - B_vals * rt)
            # Fixed leg PV
            fixed_pv = K * delta * np.sum(P) * notional
            # Floating leg PV using telescoping (risk-neutral valuation)
            float_pv = (P[0] - P[-1]) * notional  # Assumes no accrued interest for simplicity
        V_swap[j] = float_pv - fixed_pv
    # Exposure from bank's perspective (pays fixed, so max(-V_swap, 0))
    exposure = np.maximum(V_swap, 0)
    EE[i] = np.mean(exposure)

# Compute survival probabilities S(t)
S = np.exp(-h * t)

# Compute interval default probabilities q(T_{i-1}, T_i)
q = np.zeros(n_steps)
for i in range(1, n_steps + 1):
    q[i-1] = S[i-1] - S[i]

# Compute CVA (ignore EE[0] as at inception exposure is typically 0)
CVA = (1 - R) * np.sum(EE[1:] * q[0:]) 


print("Calculated CVA:", CVA)
# %%
plt.figure(figsize=(10, 6))
plt.plot(t, EE)
plt.xlabel('Time (years)')
plt.ylabel('Expected Exposure')
plt.title('Expected Exposure Over 10-Year Period')
plt.grid(True)
plt.show()
# %%
#Plotting some exposure paths
exposure_paths = np.zeros((n_paths, n_steps + 1))
for j in range(n_paths):
    for i in range(n_steps + 1):
        rt = r[j, i]
        fixed_pv = 0.0
        float_pv = 0.0
        remaining_times = t[i+1:] if i < n_steps else np.array([])
        if len(remaining_times) > 0:
            tau = remaining_times - t[i]
            B_vals = B(alpha, tau)
            A_vals = A(alpha, b, sigma, tau)
            P = np.exp(A_vals - B_vals * rt)
            # Fixed leg PV
            fixed_pv = K * delta * np.sum(P) * notional
            # Floating leg PV using telescoping (risk-neutral valuation)
            float_pv = (P[0] - P[-1]) * notional  # Assumes no accrued interest for simplicity
        V_swap = float_pv - fixed_pv
        # Exposure from bank's perspective (pays fixed, so max(-V_swap, 0))
        exposure_paths[j, i] = max(V_swap, 0)

plt.figure(figsize=(10, 6))
for j in range(10):
    plt.plot(t, exposure_paths[j])  # Label first 5 paths for clarity
plt.xlabel('Time (years)')
plt.ylabel('Exposure')
plt.title('Some Simulated Exposure Paths Over 10-Year Period')
plt.grid(True)
plt.legend()
plt.show()
# %%

