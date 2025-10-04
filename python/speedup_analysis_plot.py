import numpy as np
import matplotlib.pyplot as plt
#plotting the speedup comparison between the python and the cuda implementation for 
#creating a joint cdf/pdf for a given dataset
labels = [
    "Method 0\n(Naive Python)",
    "Method 1\n(Naive CUDA)",
    "Method 2\n(Shared Memory)",
    "Method 3\n(Sum of Squares)",
    "Method 4\n(Welford)",
]
times_ms = [22.173166275024414, 4.7136, 0.16896, 0.088288, 0.087296]

# Runtime comparison
plt.figure(figsize=(10, 6))
plt.bar(range(len(labels)), times_ms)
plt.yscale('log')
plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
plt.ylabel('Time (ms)')
plt.title('Runtime Comparison')
plt.tight_layout()
plt.savefig('runtime_comparison.png')
plt.show()

# Speedup vs Method 0
base0 = times_ms[0]
speedup_m0 = [base0 / t for t in times_ms]

plt.figure(figsize=(10, 6))
plt.bar(range(len(labels)), speedup_m0)
plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
plt.ylabel('Speedup x')
plt.title('Speedup vs Method 0')
plt.axhline(1.0, color='red', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('speedup_vs_m0.png')
plt.show()

# Speedup vs Method 1
base1 = times_ms[1]
speedup_m1 = [base1 / t for t in times_ms]

plt.figure(figsize=(10, 6))
plt.bar(range(len(labels)), speedup_m1)
plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
plt.ylabel('Speedup x ')
plt.title('Speedup vs Method 1')
plt.axhline(1.0, color='red', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('speedup_vs_m1.png')
plt.show()

# Combined plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Runtime
axes[0].bar(range(len(labels)), times_ms)
axes[0].set_yscale('log')
axes[0].set_xticks(range(len(labels)))
axes[0].set_xticklabels(labels, rotation=45, ha='right')
axes[0].set_ylabel('Time (ms)')
axes[0].set_title('Runtime')

# Speedup vs M0
axes[1].bar(range(len(labels)), speedup_m0)
axes[1].set_xticks(range(len(labels)))
axes[1].set_xticklabels(labels, rotation=45, ha='right')
axes[1].set_ylabel('Speedup x')
axes[1].set_title('Speedup vs Method 0')
axes[1].axhline(1.0, color='red', linestyle='--', alpha=0.7)

# Speedup vs M1
axes[2].bar(range(len(labels)), speedup_m1)
axes[2].set_xticks(range(len(labels)))
axes[2].set_xticklabels(labels, rotation=45, ha='right')
axes[2].set_ylabel('Speedup x')
axes[2].set_title('Speedup vs Method 1')
axes[2].axhline(1.0, color='red', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('combined_plots.png')
plt.show()
