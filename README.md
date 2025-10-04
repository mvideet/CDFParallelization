# HackMIT 2025 Technical Challenge Winner & Voloridge $5K prize Winning Code: Statistical Computing & GPU Acceleration 

Videet Mehta and Advay Goel

## What This Project Does

This project implements high-performance statistical computing tools with GPU acceleration:

1. **Data Generation**: Creates datasets with 5 different statistical distributions (Gaussian, t-distribution, Exponential, Laplace, Gumbel) and saves them in both Python-friendly (Parquet) and CUDA-friendly (CSV) formats.

2. **Multivariate CDF Computation**: Calculates empirical cumulative distribution functions for high-dimensional data with multi-core parallelization.

3. **GPU-Accelerated CUDA Kernels**: Implements various statistical algorithms on GPU including sum computation, variance calculation using Welford's algorithm, and covariance matrix computation.

## Quick Start

```bash
# Clone and setup
git clone https://github.com/mvideet/hackmit.git
cd hackmit
python3 -m venv venv
source venv/bin/activate
pip install numpy pandas scipy pyarrow fastparquet ml_dtypes

# Generate data
python mixed_distribution_generator.py data.parquet 15000

# Compute CDF
python cdf_labels.py

# Compile and run CUDA kernels
nvcc -o gaussians gaussians.cu -lcurand
./gaussians
```

## Performance Results

We tested 5 different statistical computation methods on GPU:

| Method | Runtime | Speedup vs Python |
|--------|---------|-------------------|
| **Method 0** (Naive Python) | 20.0 ms | 1.0× (baseline) |
| **Method 1** (Naive CUDA) | 4.7 ms | **4.2×** |
| **Method 2** (Shared Memory) | 0.17 ms | **118×** |
| **Method 3** (Sum of Squares) | 0.088 ms | **227×** |
| **Method 4** (Welford's Algorithm) | 0.087 ms | **230×** |

**Key Findings:**
- GPU acceleration provides **4-230× speedup** over Python
- Shared memory and optimized algorithms are crucial for performance
- Welford's algorithm (Method 4) gives the best performance at **0.087 ms**

## Project Structure

### CUDA Implementation (`cuda/`)
- `distribution_detector.cu` - Main CUDA distribution detection
- `gaussians.cu` - CUDA kernels for statistical computation
- `Makefile` - Build configuration

### Python Analysis (`python/`)
- `distribution_detection.py` - Multi-distribution detection algorithm
- `gaussian_fitting.py` - Gaussian distribution fitting
- `multivariate_cdf_computation.py` - Multivariate CDF computation
- `xgboost_regression.py` - XGBoost regression model

### Data Generation
- `gaussian_data_generator.py` - Gaussian data generator
- `mixed_distribution_generator.py` - Multi-distribution data generator

### Visualization
- `speedup_analysis_plot.py` - Performance speedup analysis
- `performance_comparison_plot.py` - Python vs CUDA comparison
- `parameter_validation_plot.py` - Parameter validation plots
