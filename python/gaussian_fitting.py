import math
import numpy as np
import pandas as pd
from dataclasses import dataclass

def gaussian_cdf(x, mean, std):
    std = max(float(std), 1e-12) #clamps tiny std to avoid div-by-zero
    z = (x - mean) / (std * math.sqrt(2.0))
    from numpy import special as nps
    return 0.5 * (1.0 + nps.erf(z))
@dataclass
class GaussianParams:
    mean: float
    std: float
import time 
def fit_gaussians_and_export_cdfs(csv_path, output_csv="gaussian_cdf_values_10k_1kf.csv", print_sample=True):
    df = pd.read_csv(csv_path, header=None)
    X = df.values.astype(np.float64)
    rows, cols = X.shape
    params = []
    for j in range(cols):
        col = X[:, j]
        mu = float(np.mean(col))
        var = float(np.mean(col * col) - mu * mu)
        std = math.sqrt(max(var, 1e-12))
        params.append(GaussianParams(mu, std))
fit_gaussians_and_export_cdfs("/data/TTS/videet/gaussian_data_final_10k_1000f.csv")