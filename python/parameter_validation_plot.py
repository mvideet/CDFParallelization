
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from scipy.stats import t as tdist, chi2
#hard coded values for the distribution matching based on the results from the kernel and the ground truth
dims = np.arange(1, 11)

mu_true = np.array([
    -2.5092,  9.0143,  4.6399,  1.9732, -6.8796,
    -6.8801, -8.8383,  7.3235,  2.0223,  4.1615
])
sigma_true = np.array([
    
    1.1546, 1.7142, 2.1043, 1.9481, 2.0321,
    1.8613, 1.9014, 2.0788, 1.2742, 1.6439
])
mu_hat = np.array([
    -2.5236,  9.0350,  4.6362,  1.9637, -6.8792,
    -6.8667, -8.8229,  7.3302,  2.0364,  4.1553
])
s_hat = np.array([
    1.1532, 1.7322, 2.0835, 1.9535, 2.0268,
    1.8708, 1.9003, 2.0871, 1.2646, 1.6394
])
n = 10000
alpha = 0.05
df = n -1
chi2_lo = chi2.ppf(alpha/2, df)
chi2_hi = chi2.ppf(1 - alpha/2, df)
sigma_lo = s_hat * np.sqrt(df / chi2_hi)
sigma_hi = s_hat * np.sqrt(df / chi2_lo)
plt.figure(figsize=(10, 6))
x_pos = np.arange(len(dims))
bar_width = 0.35
plt.bar(x_pos - bar_width/2, sigma_true, bar_width, 
        label='True σ', color='lightblue', alpha=0.7)
plt.bar(x_pos + bar_width/2, s_hat, bar_width, 
        label='Estimated σ', color='orange', alpha=0.7)

plt.errorbar(x_pos + bar_width/2, s_hat, 
             yerr=[s_hat - sigma_lo, sigma_hi - s_hat], 
             fmt='none', color='black', capsize=3)
plt.xlabel('Dimension')
plt.ylabel('Standard Deviation')
plt.title('Standard Deviation Comparison')
plt.legend()
plt.xticks(x_pos, [f'Dim {d}' for d in dims])
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
