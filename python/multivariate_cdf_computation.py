import pandas as pd
import numpy as np
import time
from tqdm import tqdm  
# computing empirical multivariate cumulative distribution function for each column

data_path = "mixed_distributions_150k_20f.parquet"
df = pd.read_parquet(data_path)
data = df.values.astype(np.float64)
n_rows, n_cols = data.shape
multivariate_cdf = np.zeros(n_rows, dtype=np.float32)

start_time = time.time()
for i in tqdm(range(n_rows), desc="Computing CDF", unit="row"):
    current_row = data[i]
    row_comparisons = data< current_row[None, :]
    all_less = np.all(row_comparisons, axis=1)
    multivariate_cdf[i] = np.sum(all_less) / n_rows

total_time = time.time() - start_time
print("Total Time", total_time)
result_df = df.copy()
result_df['multivariate_empirical_cdf'] = multivariate_cdf
result_df.to_parquet("mixed_distributions_with_cdf_150k_20f.parquet", index=False)
print("Mean CDF", multivariate_cdf.mean())