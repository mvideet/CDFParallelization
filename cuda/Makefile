# Makefile for CUDA Gaussian Statistics Calculator

# CUDA compiler
NVCC = nvcc

# Compiler flags
NVCCFLAGS = -O3 -arch=sm_70 -std=c++11
# Note: Change -arch=sm_70 to match your GPU architecture:
# - sm_30: Kepler
# - sm_35: Kepler (K40, K80)
# - sm_50: Maxwell
# - sm_60: Pascal (P100)
# - sm_70: Volta (V100)
# - sm_75: Turing (T4, RTX 20xx)
# - sm_80: Ampere (A100, RTX 30xx)
# - sm_86: Ampere (RTX 30xx)
# - sm_89: Ada Lovelace (RTX 40xx)
# - sm_90: Hopper (H100)

# Target executables
TARGET = gaussians
DIST_TARGET = distribution_detector

# Source files
SOURCES = gaussians.cu
DIST_SOURCES = distribution_detector.cu

# Default target
all: $(TARGET) $(DIST_TARGET) generate_data

# Compile CUDA programs
$(TARGET): $(SOURCES)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $(SOURCES)

$(DIST_TARGET): $(DIST_SOURCES)
	$(NVCC) $(NVCCFLAGS) -o $(DIST_TARGET) $(DIST_SOURCES)

# Generate sample data
generate_data:
	@echo "Generating sample Gaussian data..."
	@python3 gaussian_data_generator.py gaussian_data.csv 1000 10

# Run the Gaussian statistics program with sample data
run: $(TARGET) generate_data
	./$(TARGET) gaussian_data.csv

# Generate mixed distribution data
mixed_data:
	@echo "Generating mixed distribution data..."
	@python3 mixed_distribution_generator.py mixed_distributions.csv 1000

# Run distribution detection
run_dist: $(DIST_TARGET) mixed_data
	./$(DIST_TARGET) mixed_distributions.csv

# Generate large dataset for performance testing
large_data:
	@echo "Generating large Gaussian dataset (10000 rows x 100 columns)..."
	@python3 gaussian_data_generator.py large_gaussian_data.csv 10000 100

# Run with large dataset
run_large: $(TARGET) large_data
	./$(TARGET) large_gaussian_data.csv

# Clean build artifacts
clean:
	rm -f $(TARGET) $(DIST_TARGET)
	rm -f *.o

# Clean everything including generated data
clean_all: clean
	rm -f gaussian_data.csv
	rm -f large_gaussian_data.csv
	rm -f mixed_distributions.csv

# Check CUDA installation
check_cuda:
	@echo "Checking CUDA installation..."
	@nvcc --version
	@echo ""
	@echo "Checking GPU devices..."
	@nvidia-smi --query-gpu=name,compute_cap --format=csv

# Help
help:
	@echo "Available targets:"
	@echo "  make            - Build all CUDA programs and generate sample data"
	@echo "  make run        - Run Gaussian statistics with sample data"
	@echo "  make run_dist   - Run distribution detection with mixed data"
	@echo "  make run_large  - Run Gaussian statistics with large dataset"
	@echo "  make mixed_data - Generate mixed distribution test data"
	@echo "  make clean      - Remove compiled binaries"
	@echo "  make clean_all  - Remove binaries and generated data"
	@echo "  make check_cuda - Check CUDA installation and GPU info"
	@echo "  make help       - Show this help message"

.PHONY: all generate_data run large_data run_large clean clean_all check_cuda help
