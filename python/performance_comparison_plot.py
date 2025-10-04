import matplotlib.pyplot as plt
#plotting the performance comparison between the python and the cuda implementation for detecting
#non-gaussian distributions
methods = ['Python', 'CUDA']
times_ms = [480.0, 8.76]
plt.figure(figsize=(8, 5))
plt.bar(methods, times_ms, color=['lightblue', 'orange'])
plt.yscale('log')
plt.ylabel('Time (ms)')
plt.title('Performance Comparison')
plt.grid(True, alpha=0.3)

for i, time in enumerate(times_ms):
    plt.text(i, time * 1.1, f'{time:.1f} ms', ha='center', va='bottom')

speedup = times_ms[0] / times_ms[1]
plt.text(0.5, max(times_ms) * 0.3, f'{speedup:.0f}x faster', 
         ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()
