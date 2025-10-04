#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <iostream>

#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define TILE_SIZE 16

// sums each column using a single thread per column
__global__ void calculateSumNaive_Implementation(float* d_data,float* d_sums,int rows, int cols) {
    int col_num = blockIdx.x *blockDim.x + threadIdx.x;
    if (col_num < cols) {
        float sum_of_col = 0.0f;
        for (int row = 0; row < rows; row++) {
            sum_of_col +=d_data[row *cols +col_num];
        }
        d_sums[col_num] = sum_of_col;
    }
}

// computes column sums using many threads w/ atomic adds
__global__ void calculateSum_Atomic_Implementation(float* d_data, float* d_sums, int rows, int cols) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * gridDim.x;
    for (int idx =thread_id;idx < rows *cols; idx += total_threads) {
        int row = idx / cols;
        int col_num = idx % cols;
        //using atomic add here while parallelized
        atomicAdd(&d_sums[col_num], d_data[idx]);
    }
}

// computes column sums using block-level shared memory reduction
__global__ void calculateSum_SharedReduction_Implementation(float* d_data,float* d_sums,int rows, int cols) {
    extern __shared__ float shared_data[];
    int col_num = blockIdx.x;
    int thread_id = threadIdx.x;
    int stride_of_threads = blockDim.x;
    if (col_num >= cols) return;
    float sum = 0.0f;
    for (int row= thread_id; row < rows; row += stride_of_threads) {
        sum += d_data[row * cols + col_num];
    }
    shared_data[thread_id] =sum; //storing the sum in shared memory
    __syncthreads();
    for(int s =blockDim.x /2; s >0; s>>= 1) { //iterating through shared data to get the sum
        if (thread_id < s) {
            shared_data[thread_id]+= shared_data[thread_id + s];
        }
        __syncthreads();
    }
    if (thread_id == 0) {
        d_sums[col_num] = shared_data[0];
    }
}

struct WelfordState { //welford state to compute mean and variance
    float avg;
    float M2; //mean of sqares or like seocond moment
    int count; //num
};

// updates a welford state with one new value
__device__ WelfordState welfordUpdate(WelfordState curr_state, float value) {
    curr_state.count++;
    float delta = value - curr_state.avg;
    curr_state.avg += delta / curr_state.count;
    float delta2 = value - curr_state.avg;
    curr_state.M2 += delta * delta2;
    return curr_state;
}

// combines two welford states for parallel reduction
__device__ WelfordState welfordCombine(WelfordState a, WelfordState b) {
    if (a.count == 0) return b;
    if (b.count == 0) return a;
    WelfordState combined;
    combined.count =a.count + b.count;
    float delta = b.avg- a.avg;
    combined.avg = (a.count* a.avg + b.count *b.avg) /combined.count;
    combined.M2 = a.M2 +b.M2+delta * delta * a.count *b.count /combined.count;
    return combined;
}

// computes per-column mean and variance using welford's algorithm with shared reduction
__global__ void calculateMeanVarianceWelford(float* d_data,float* d_means,float* d_variances, int num_rows, int num_cols) {
    extern __shared__ WelfordState shared_states[]; //shared memory for welford states so that we can retrieve them later
    
    int col = blockIdx.x;
    int thread_id = threadIdx.x;
    int stride_of_threads = blockDim.x;    
    WelfordState state = {0.0f, 0.0f, 0}; //initizlie with 0's
    for (int row =thread_id; row < num_rows; row +=stride_of_threads) {
        float value =d_data[row *num_cols + col];
        state = welfordUpdate(state, value);
    }
    shared_states[thread_id] = state;
    __syncthreads();
    int blockDim_half = blockDim.x /2;
    for (int s = blockDim_half; s > 0; s >>= 1) { //iterating through shared data to get the sum
        if (thread_id < s) { 
            shared_states[thread_id] =welfordCombine(shared_states[thread_id], shared_states[thread_id+ s]);
        }
        __syncthreads(); //sync all threads tgt for a column
    }
    if (thread_id == 0) {
        d_means[col] = shared_states[0].avg;
        d_variances[col] = shared_states[0].M2 / (shared_states[0].count - 1); //move everything to thread 0 for averaging everything 
        //after they have been combined
    }
}


// computes per-column mean and variance via sums and sums of squares in one pass
__global__ void calculateMeanVarianceFaster(float* d_data, float* d_means, float* d_variances, int rows, int cols) {
    extern __shared__ float shared_sum[]; //shared memory for sum and sum of squares
    float* shared_var = &shared_sum[blockDim.x];
    int col =blockIdx.x;
    int num_warps =blockDim.x/WARP_SIZE;

    int thread_id =threadIdx.x;
    int lane = thread_id%WARP_SIZE;
    int warp_id= thread_id/WARP_SIZE;
    
    if (col >= cols) return;
    
    float sum =0.0f;
    float sum_sq = 0.0f;
    //storing E[X] and E[X^2] over  threads
    for (int row =thread_id; row< rows; row += blockDim.x) { 
    // for (int row = 0; row < rows; row++) {
        float val = d_data[row * cols + col];
        sum += val;
        sum_sq += val * val;
    }
    // __syncthreads();
    // #pragma unroll
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) { //log n reduction for sum and sum of squares
        sum+= __shfl_down_sync(0xffffffff, sum, offset); 
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }
    // if (lane == num_warps - 1) {
    if (lane == 0) {
        shared_sum[warp_id]=sum;
        shared_var[warp_id]=sum_sq;
    }
    __syncthreads(); //sync all threads have all their partial saved to the warp
    
    // only threads in first warp do final reduction
    if (thread_id< num_warps) {
        sum= shared_sum[thread_id];
        sum_sq = shared_var[thread_id];
        #pragma unroll
        for (int offset = num_warps/ 2; offset > 0; offset /= 2) {
            // shuffle down for reduction
            sum += __shfl_down_sync(0xffffffff, sum, offset);
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
        }
        if (thread_id == 0) {
            // thread 0 does final calc
            float mean = sum / rows;
            d_means[col] =mean;
            d_variances[col] =(sum_sq- sum * mean) / (rows - 1);//sample variance
            //d_variances[col] = (sum_sq/rows) - (mean*mean);s
        }
    }
}
// computes per-column variance given precomputed means
__global__ void calculateVariance(float* d_data, float* d_means, float* d_variances, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cols) {
        float mean = d_means[col];
        float variance = 0.0f;
        for (int row =0;row < rows; row++) {
            float diff= d_data[row* cols+ col] - mean;
            variance += diff *diff;
        }
        d_variances[col] = variance / (rows - 1); //we used n-1 for sample variance but n also works. converges to 0
    }
}

// computes per-column mean and variance using a two-pass shared-memory reduction
__global__ void calculateMeanVarianceShared(float* d_data, float* d_means, float* d_variances, int rows, int cols) {
    extern __shared__ float shared_data[];
    
    int col = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;
    
    if (col >= cols) return;
    float sum = 0.0f;
    for (int row = tid; row < rows; row += stride) {
        sum += d_data[row * cols + col];
    }
    shared_data[tid] = sum;
    __syncthreads();
    // #pragma unroll
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        d_means[col]= shared_data[0] / rows; //thread 0 does final calc
    }
    __syncthreads();
    // grab the mean we calculated earlier
    float mean = d_means[col];
    float variance_sum = 0.0f;
    // each thread handles multiple rows with stride pattern
    for (int row = tid; row < rows; row += stride) {
        // classic variance calculation - subtract mean and square it
        float diff = d_data[row * cols + col] - mean;
        variance_sum += diff * diff;
    }
    shared_data[tid] = variance_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }
    // thread 0 writes the final result
    if (tid == 0) {
        d_variances[col] = shared_data[0] / (rows - 1);
    }
}
//TODO: cache these values somehow to intermediately calculate the coviarance matrix faster.
// computes the full covariance matrix in parallel with shared-memory reduction
__global__ void computeCovarianceMatrix(float* d_data, float* d_means, float* d_covariance, 
                                       int rows, int cols) {
    int x_block= blockIdx.x; 
    int y_block= blockIdx.y;
    
    if (x_block >= cols || y_block >= cols) return;
    extern __shared__ float shared_data[];
    int thread_id = threadIdx.x;
    float mean_i = d_means[x_block]; //get means for the blocks
    float mean_j = d_means[y_block];
    
    float sum = 0.0f;
    for (int row = thread_id; row < rows; row += blockDim.x) { //iterate through rows
        float val_i = d_data[row* cols +x_block]- mean_i;
        float val_j = d_data[row*cols +y_block]- mean_j;
        sum += val_i * val_j; //variance = sum of (x-mean_x)(y-mean_y)
    }
    shared_data[thread_id] =sum; //store sum in shared memory
    __syncthreads();
    for (int s = blockDim.x /2; s >0;s>>= 1) { //>>= does right shif
        if (thread_id< s) {
            // shared_data[thread_id+s] += shared_data[thread_id];
            shared_data[thread_id] += shared_data[thread_id + s];
        }
        __syncthreads();
    }
    if (thread_id == 0) {
        float cov = shared_data[0] /(rows - 1);
        d_covariance[x_block * cols + y_block] = cov;
        if (x_block != y_block) {
            d_covariance[y_block * cols + x_block] = cov;
        }
    }
}

// computes the covariance matrix using 2d tiling for locality and reuse
//TODO: deprecated/it doesn't work rn
__global__ void computeCovarianceTiled(float* d_data, float* d_means, float* d_covariance,
                                      int rows, int cols) {
    __shared__ float tile_i[TILE_SIZE][BLOCK_SIZE];
    __shared__ float tile_j[TILE_SIZE][BLOCK_SIZE];
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int col_i = bx * TILE_SIZE + tx;
    int col_j = by * TILE_SIZE + ty;
    if (col_i >= cols || col_j >= cols) return;
    float mean_i = d_means[col_i];
    float mean_j = d_means[col_j];
    float sum = 0.0f;
    for (int tile = 0; tile < (rows + BLOCK_SIZE - 1) / BLOCK_SIZE; tile++) {
        int row = tile * BLOCK_SIZE + ty;
        if (row < rows && tx < TILE_SIZE) {
            tile_i[tx][ty] = d_data[row * cols + col_i] - mean_i;
            tile_j[tx][ty] = d_data[row * cols + col_j] - mean_j;
        } else {
            tile_i[tx][ty] = 0.0f;
            tile_j[tx][ty] = 0.0f;
        }
        __syncthreads();
        
        if (tx < TILE_SIZE && ty < TILE_SIZE) {
            for (int k = 0; k < BLOCK_SIZE && (tile * BLOCK_SIZE + k) < rows; k++) {
                sum += tile_i[tx][k] * tile_j[ty][k];
            }
        }
        __syncthreads();
    }
    if (col_i < cols && col_j < cols && tx < TILE_SIZE && ty < TILE_SIZE) {
        d_covariance[col_i * cols + col_j] = sum / (rows - 1);
    }
}


// TODO: parallelize this kernel
//traditional cholesky decomp for sampling from a multivariate gaussian
__global__ void choleskyDecomposition(float* d_covariance, float* d_cholesky, int n) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = 0; i < n * n; i++) {
            d_cholesky[i] = 0.0f;
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                float sum = 0.0f;
                if (i == j) {
                    for (int k = 0; k < j; k++) {
                        sum += d_cholesky[i * n + k] * d_cholesky[i * n + k];
                    }
                    float diag = d_covariance[i * n + i] - sum;
                    d_cholesky[i * n + j] = sqrtf(fmaxf(diag, 1e-6f));
                } else {
                    for (int k = 0; k < j; k++) {
                        sum += d_cholesky[i * n + k] * d_cholesky[j * n + k];
                    }
                    d_cholesky[i * n + j] = (d_covariance[i * n + j] - sum) / d_cholesky[j * n + j];
                }
            }
        }
    }
}

// initializes per-thread curand states for random number generation
__global__ void initRNG(curandState* states, unsigned long seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

// generates multivariate gaussian samples using a cholesky factor
__global__ void generateMultivariateGaussianSamples(curandState* states, float* d_samples,
                                                   float* d_means, float* cholesky_data,
                                                   int num_samples, int num_dims) {
    int thread_id= blockIdx.x* blockDim.x + threadIdx.x;
    if (thread_id >= num_samples) return;
    
    extern __shared__ float shared_mem[];
    float* z_scores = shared_mem;
    float* result = &shared_mem[num_dims]; 
    curandState localState = states[thread_id];
    for (int i = 0; i <num_dims; i++) {
        z_scores[i] = curand_normal(&localState); //generate normal distribution
        result[i] =0.0f;
    }
    
    for (int i = 0; i < num_dims; i++) {
        float sum = d_means[i];
        for (int j = 0; j <= i;j++) {
            sum +=cholesky_data[i* num_dims + j] * z_scores[j];
        }
        result[i] = sum;
    }
    for (int i = 0; i < num_dims; i++) {
        d_samples[thread_id * num_dims + i] = result[i];
    }
    states[thread_id] = localState;
}


// reads a csv of floats into a 2d vector
bool csv_reader(const char* filename, std::vector<std::vector<float>>& data) {
    std::ifstream file(filename);
    if (!file.is_open()) return false;
    std::string line;
    while (std::getline(file, line)) {
        std::vector<float> row;
        std::stringstream ss(line);
        std::string cell;
        
        while (std::getline(ss, cell, ',')) {
                row.push_back(std::stof(cell));
        }
            data.push_back(row);
    }
    return !data.empty();
}

// flattens 2d data into a flattned arr  and returns dimensions
float* transposer(const std::vector<std::vector<float>>& data, int& rows, int& cols) {
    rows = data.size();
    cols = data[0].size();
    for (const auto& row :data) {
        if (row.size() != cols) {
            cols = std::min(cols, (int)row.size());
        }
    }
    float* flattned =new float[rows * cols]; //similar to flattened matmul
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            // flattned[j * cols + i] =data[i][j];
            flattned[i * cols + j] =data[i][j];
        }
    }
    return flattned;
}


// loads data, runs kernels, reports timings, and writes outputs
int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <csv_file>" << std::endl;
        return 1;
    }
    std::vector<std::vector<float>> data;
    if (!csv_reader(argv[1], data)) {
        return 1;
    }
    int rows, cols;
    float* h_data = transposer(data, rows, cols);
    std::cout << "Data: " << rows << " rows, " << cols << " columns" << std::endl;
    float* h_means = new float[cols];
    float* h_variances = new float[cols];
    float *d_data, *d_means, *d_variances, *d_sums;
    size_t dataSize = rows * cols * sizeof(float);
    size_t statsSize = cols * sizeof(float);
     //allocate device memory
    cudaMalloc(&d_data, dataSize);
    cudaMalloc(&d_means, statsSize);
    cudaMalloc(&d_variances, statsSize);
    cudaMalloc(&d_sums, statsSize);
    
    cudaMemcpy(d_data, h_data, dataSize, cudaMemcpyHostToDevice);
    
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (cols + threadsPerBlock - 1) / threadsPerBlock;
    
    //create event objects
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Method 1: Naive
    cudaEventRecord(start);
    
    calculateSumNaive_Implementation<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_sums, rows, cols);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_means, d_sums, statsSize, cudaMemcpyDeviceToHost);
    for (int i = 0; i < cols; i++) {
        h_means[i] /= rows;
    }
    cudaMemcpy(d_means, h_means, statsSize, cudaMemcpyHostToDevice);
    
    calculateVariance<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_means, d_variances, rows, cols);
    cudaDeviceSynchronize();
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds1 = 0;
    cudaEventElapsedTime(&milliseconds1, start, stop);
    cudaMemcpy(h_variances, d_variances, statsSize, cudaMemcpyDeviceToHost);
    



    // Method 2: Shared Memory
    cudaMemset(d_means, 0, statsSize);
    cudaMemset(d_variances, 0, statsSize);
    
    cudaEventRecord(start);
    
    int sharedMemSize = BLOCK_SIZE * sizeof(float);
    calculateMeanVarianceShared<<<cols, BLOCK_SIZE, sharedMemSize>>>(d_data, d_means, d_variances, rows, cols);
    cudaDeviceSynchronize();
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds2 = 0;
    cudaEventElapsedTime(&milliseconds2, start, stop);
    
    cudaMemcpy(h_means, d_means, statsSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_variances, d_variances, statsSize, cudaMemcpyDeviceToHost);
    
    // Method 3: Sum of Squares
    cudaMemset(d_means, 0, statsSize);
    cudaMemset(d_variances, 0, statsSize);
    
    cudaEventRecord(start);
    
    int sharedMemSize2 = 2 * BLOCK_SIZE * sizeof(float);  // Space for both sum and variance at the same time
    calculateMeanVarianceFaster<<<cols, BLOCK_SIZE, sharedMemSize2>>>(d_data, d_means, d_variances, rows, cols);
    cudaDeviceSynchronize();
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds3 = 0;
    cudaEventElapsedTime(&milliseconds3, start, stop);
    
    cudaMemcpy(h_means, d_means, statsSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_variances, d_variances, statsSize, cudaMemcpyDeviceToHost);
    
    // Method 4: Welford's Algorithm
    cudaMemset(d_means, 0, statsSize);
    cudaMemset(d_variances, 0, statsSize);
    
    cudaEventRecord(start);
    
    int sharedMemSize3 = BLOCK_SIZE * sizeof(WelfordState);
    calculateMeanVarianceWelford<<<cols, BLOCK_SIZE, sharedMemSize3>>>(d_data, d_means, d_variances, rows, cols);
    cudaDeviceSynchronize();
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds4 = 0;
    cudaEventElapsedTime(&milliseconds4, start, stop);
    
    cudaMemcpy(h_means, d_means, statsSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_variances, d_variances, statsSize, cudaMemcpyDeviceToHost);
    
    
    for (int i = 0; i < cols; i++) {
        float stddev = sqrt(h_variances[i]);
        printf("Column %3d: %.4f ± %.4f\n", i + 1, h_means[i], stddev);
    }
    printf("Algorithm Times: Naive=%.2f, Shared=%.2f, SumSq=%.2f, Welford=%.2f ms\n", 
           milliseconds1, milliseconds2, milliseconds3, milliseconds4);
    printf("Speedup vs Naive: Shared=%.1fx, SumSq=%.1fx, Welford=%.1fx\n", 
           milliseconds1/milliseconds2, milliseconds1/milliseconds3, 
           milliseconds1/milliseconds4);
    
    // Covariance Matrix
    
    float* d_covariance;
    float* d_cholesky;
    size_t matrixSize = cols * cols * sizeof(float);
    
    cudaMalloc(&d_covariance, matrixSize);
    cudaMalloc(&d_cholesky, matrixSize);
    
    cudaMemset(d_covariance, 0, matrixSize);
    
    cudaEventRecord(start);
    
    dim3 covGrid(cols, cols);
    dim3 covBlock(BLOCK_SIZE);
    int sharedMemCov = BLOCK_SIZE * sizeof(float);
    
    computeCovarianceMatrix<<<covGrid, covBlock,sharedMemCov>>>(d_data, d_means, d_covariance, rows, cols);
    cudaDeviceSynchronize();
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float covTime = 0;
    cudaEventElapsedTime(&covTime, start, stop);
    printf("Covariance time: %.2f ms\n", covTime);
    
    float* h_covariance = new float[cols * cols];
    cudaMemcpy(h_covariance, d_covariance, matrixSize, cudaMemcpyDeviceToHost);
    
    int display_size = std::min(5, cols);
    printf("Covariance Matrix (%dx%d):\n", display_size, display_size);
    for (int i = 0; i < display_size; i++) {
        for (int j = 0; j < display_size; j++) {
            printf("%8.4f ", h_covariance[i * cols + j]);
        }
        printf("\n");
    }
    
    // Cholesky decomposition
    cudaEventRecord(start);
    choleskyDecomposition<<<1, 1>>>(d_covariance, d_cholesky, cols);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float choleskyTime = 0;
    cudaEventElapsedTime(&choleskyTime, start, stop);
    printf("Cholesky time: %.2f ms\n", choleskyTime);
    
    // Gaussian Sampling
    
    int num_samples = 1000;
    printf("Generating %d samples...\n", num_samples);
    
    float* d_samples;
    curandState* d_states;
    size_t samplesSize = num_samples * cols * sizeof(float);
    size_t statesSize = num_samples * sizeof(curandState);
    
    cudaMalloc(&d_samples, samplesSize);
    cudaMalloc(&d_states, statesSize);
    
    int rngBlocks = (num_samples + BLOCK_SIZE - 1) / BLOCK_SIZE;
    initRNG<<<rngBlocks, BLOCK_SIZE>>>(d_states, time(NULL), num_samples);
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    
    int sampleBlocks = (num_samples + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int sharedMemSample = 2 * cols * sizeof(float);
    
    generateMultivariateGaussianSamples<<<sampleBlocks, BLOCK_SIZE, sharedMemSample>>>(
        d_states, d_samples, d_means, d_cholesky, num_samples, cols);
    cudaDeviceSynchronize();
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float sampleTime = 0;
    cudaEventElapsedTime(&sampleTime, start, stop);
    printf("Sample time: %.2f ms\n", sampleTime);
    
    float* h_samples = new float[num_samples * cols];
    cudaMemcpy(h_samples, d_samples, samplesSize, cudaMemcpyDeviceToHost);
    
    printf("Sample Verification:\n");
    printf("Col | Orig Mean | Sample Mean | Orig Std | Sample Std\n");
    for (int col = 0; col <std::min(10, cols); col++) {
        float sample_mean =0.0f;
        float sample_var = 0.0f;
        for (int s = 0; s < num_samples; s++) {
            sample_mean += h_samples[s * cols + col];
        }
        sample_mean /= num_samples;
        for (int s = 0; s < num_samples; s++) {
            float diff = h_samples[s * cols + col] - sample_mean;
            sample_var += diff * diff;
        }
        sample_var /= (num_samples - 1);
        printf("%6d | %13.4f | %11.4f | %12.4f | %9.4f\n",
               col + 1, h_means[col], sample_mean, 
               sqrt(h_variances[col]), sqrt(sample_var));
    }
    
    std::ofstream sampleFile("multivariate_samples.csv");
    if (sampleFile.is_open()) {
        for (int s = 0; s < num_samples; s++) {
            for (int c = 0; c < cols; c++) {
                sampleFile << h_samples[s * cols + c];
                if (c < cols - 1) sampleFile << ",";
            }
            sampleFile << "\n";
        }
        sampleFile.close();
    }
    
    // Cleanup
    delete[] h_covariance;
    delete[] h_samples;
    delete[] h_data;
    delete[] h_means;
    delete[] h_variances;
    
    cudaFree(d_data);
    cudaFree(d_means);
    cudaFree(d_variances);
    cudaFree(d_sums);
    cudaFree(d_covariance);
    cudaFree(d_cholesky);
    cudaFree(d_samples);
    cudaFree(d_states);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}