import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import IntEnum

class DistributionType(IntEnum):
    GAUSSIAN = 0
    STUDENT_T = 1
    EXPONENTIAL = 2
    GUMBEL = 3
    LAPLACE = 4
    UNKNOWN = 5

@dataclass
class DistributionParams:
    type: DistributionType
    param1: float = 0.0
    param2: float = 0.0
    param3: float = 0.0
    logLikelihood: float = -np.inf

PI = math.pi
EULER_MASCHERONI = 0.5772156649015329
EPS = 1e-10
def gaussian_pdf(x, mean, std):
    std = max(std, 1e-6)
    z = (x - mean) / std
    return np.exp(-0.5 * z * z) / (std * math.sqrt(2.0 * PI))

def student_t_pdf(x, loc, scale, df):
    scale = max(scale, 1e-6)
    t = (x - loc) / scale
    num = math.gamma((df + 1.0) / 2.0)
    den = math.sqrt(df * PI) * math.gamma(df / 2.0) * scale
    base = 1.0 + (t * t) / df
    return (num / den) * (base ** (-(df + 1.0) / 2.0))

def exponential_pdf(x, rate):
    out = np.zeros_like(x, dtype=np.float64)
    mask = (x >= 0)
    out[mask] = rate * np.exp(-rate * x[mask])
    return out

def gumbel_pdf(x, loc, scale):
    scale = max(scale, 1e-6)
    z = (x - loc) / scale
    return (1.0 / scale) * np.exp(-(z + np.exp(-z)))

def laplace_pdf(x, loc, scale):
    scale = max(scale, 1e-6)
    return (1.0 / (2.0 * scale)) * np.exp(-np.abs(x - loc) / scale)


def estimate_gaussian_params(col):
    mean = float(np.mean(col))
    var = float(np.mean(col * col) - mean * mean)
    std = math.sqrt(max(var, 1e-6))
    return mean, std

def estimate_exponential_params(col):
    safe = np.maximum(col, 1e-6)
    rate = len(col) / float(np.sum(safe))
    return rate

def estimate_laplace_params(col):
    loc = float(np.mean(col))
    mad = float(np.mean(np.abs(col - loc)))
    scale = max(mad, 1e-6)
    return loc, scale

def estimate_gumbel_params(col):
    mean = float(np.mean(col))
    var = float(np.mean(col * col) - mean * mean)
    std = math.sqrt(max(var, 1e-6))
    scale = std * math.sqrt(6.0) / PI
    loc = mean - scale * EULER_MASCHERONI
    return loc, max(scale, 1e-6)

def estimate_student_t_params(col, df=3.0):
    loc = float(np.mean(col))
    var = float(np.mean(col * col) - loc * loc)
    scale = math.sqrt(max(var * (df - 2.0) / df, 1e-6))
    return loc, scale, df


def column_loglik(col, params: DistributionParams):
    x = col
    if params.type == DistributionType.GAUSSIAN:
        pdf = gaussian_pdf(x, params.param1, params.param2)
    elif params.type == DistributionType.STUDENT_T:
        pdf = student_t_pdf(x, params.param1, params.param2, params.param3)
    elif params.type == DistributionType.EXPONENTIAL:
        if np.any(x < 0):
            return -np.inf
        pdf = exponential_pdf(x, params.param1)
    elif params.type == DistributionType.GUMBEL:
        pdf = gumbel_pdf(x, params.param1, params.param2)
    elif params.type == DistributionType.LAPLACE:
        pdf = laplace_pdf(x, params.param1, params.param2)
    else:
        return -np.inf

    return float(np.sum(np.log(np.maximum(pdf, EPS))))

def aic(loglik, k):
    return -2.0 * loglik + 2.0 * k
def fit_best_distribution_for_column(col):
    candidates = []
    mu, sd = estimate_gaussian_params(col)
    p = DistributionParams(DistributionType.GAUSSIAN, mu, sd, 0.0, 0.0)
    log_likelihood = column_loglik(col, p)
    candidates.append( (aic(log_likelihood, 2), p._replace(logLikelihood=log_likelihood) if hasattr(p,'_replace') else DistributionParams(p.type, p.param1, p.param2, p.param3, log_likelihood)) )

    loc, sc, df = estimate_student_t_params(col, 3.0)
    p = DistributionParams(DistributionType.STUDENT_T, loc, sc, df, 0.0)
    log_likelihood = column_loglik(col, p)
    candidates.append( (aic(log_likelihood, 2), DistributionParams(p.type, p.param1, p.param2, p.param3, log_likelihood)) )
    if np.all(col >= 0):
        rate = estimate_exponential_params(col)
        p = DistributionParams(DistributionType.EXPONENTIAL, rate, 0.0, 0.0, 0.0)
        log_likelihood = column_loglik(col, p)
        candidates.append( (aic(log_likelihood, 1), DistributionParams(p.type, p.param1, p.param2, p.param3, log_likelihood)) )
    else:
        candidates.append( (np.inf, DistributionParams(DistributionType.EXPONENTIAL, 0.0,0.0,0.0, -np.inf)) )

    loc, sc = estimate_gumbel_params(col)
    p = DistributionParams(DistributionType.GUMBEL, loc, sc, 0.0, 0.0)
    log_likelihood = column_loglik(col, p)
    candidates.append( (aic(log_likelihood, 2), DistributionParams(p.type, p.param1, p.param2, p.param3, log_likelihood)) )

    loc, sc = estimate_laplace_params(col)
    p = DistributionParams(DistributionType.LAPLACE, loc, sc, 0.0, 0.0)
    log_likelihood = column_loglik(col, p)
    candidates.append( (aic(log_likelihood, 2), DistributionParams(p.type, p.param1, p.param2, p.param3, log_likelihood)) 
    best_aic, best_params = min(candidates, key=lambda t: t[0])
    return best_params, best_aic

import time 
def fit_distributions_and_export_cdfs(csv_path, output_csv="cdf_values.csv", print_sample=True):
    start = time.time()
    df = pd.read_csv(csv_path, header=None)
    X = df.values.astype(np.float64)
    rows, cols = X.shape
    best_params = []
    best_aics = []
    for j in range(cols):
        col = X[:, j]
        p, a = fit_best_distribution_for_column(col)
        best_params.append(p)
        best_aics.append(a)

    end = time.time()
    print(f"Processed {rows} rows × {cols} columns in {end - start:.2f}s")
    print(f"Best distributions: {[best_params[i].type.name for i in range(min(5, cols))]}...")

fit_distributions_and_export_cdfs("mixed_distributions_150k_200f_cuda.csv")