#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <algorithm>

#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define MAX_ITERATIONS 100
#define CONVERGENCE_THRESHOLD 1e-6

enum DistributionType {
    GAUSSIAN = 0,
    STUDENT_T = 1,
    EXPONENTIAL = 2,
    GUMBEL = 3,
    LAPLACE = 4,
    UNKNOWN = 5
};

struct DistributionParams {
    DistributionType type;
    float param1;
    float param2;
    float param3;
    float logLikelihood;
};

__constant__ float PI = 3.14159265358979323846f;
__constant__ float E = 2.71828182845904523536f;
__constant__ float EULER_MASCHERONI = 0.5772156649015329f;

// computes gaussian probability density function
__device__ float gaussianPDF(float x, float mean, float stddev){
    float diff = x-mean;
    return expf(-0.5f * diff * diff / (stddev * stddev)) / (stddev * sqrtf(2.0f * PI));
}

//student-t pdf
__device__ float studentTPDF(float x, float location, float scale, float df) {
    float t =(x -location) /scale;
    float numerator = tgammaf((df + 1.0f) /2.0f);
    float denominator = sqrtf(df * PI) * tgammaf(df / 2.0f) *scale;
    float base =1.0f + (t *t) / df;
    return (numerator /denominator) *powf(base, -(df + 1.0f) / 2.0f);
}

//exponential pdf
__device__ float exponentialPDF(float x, float rate) {
    if (x < 0) return 0.0f;
    return rate * expf(-rate * x);
}

//gumbel pdf
__device__ float gumbelPDF(float x, float location, float scale) {
    float z = (x -location) / scale;
    return (1.0f / scale) * expf(-(z + expf(-z)));
}

//laplace pdf
__device__ float laplacePDF(float x, float location, float scale) {
    return (1.0f /(2.0f *scale)) * expf(-fabsf(x -location) / scale);
}

//gaussian cdf
__device__ float gaussianCDF(float x, float mean, float stddev) {
    float z =(x -mean) / (stddev *sqrtf(2.0f));
    return 0.5f * (1.0f + erff(z));
}

//student-t cdf
__device__ float studentTCDF(float x, float location, float scale, float df) {
    float t = (x -location) / scale;
    float x_val = df /(df +t * t);
    if (t >= 0) {
        return 1.0f - 0.5f *powf(x_val, df / 2.0f);
    } else {
        return 0.5f *powf(x_val, df / 2.0f);
    }
}

//exponential cdf
__device__ float exponentialCDF(float x, float rate) {
    if (x < 0) return 0.0f;
    return 1.0f -expf(-rate * x);
}

//gumbel cdf
__device__ float gumbelCDF(float x,float location, float scale) {
    float z = (x -location) / scale;
    return expf(-expf(-z));
}

//laplace cdf
//TODO: figure out why laplace is very innaccurate. mayybe its because of the way we are estimating the parameters.s
__device__ float laplaceCDF(float x, float location, float scale) {
    if (x < location) {
        return 0.5f * expf((x - location) / scale);
    } else {
        return 1.0f - 0.5f * expf(-(x - location) / scale);
    }
}

//gaussian params
__device__ void estimateGaussianParams(float* data, int n, int col, int cols, float& mean, float& stddev) {
    float sum = 0.0f;
    float sum_sq = 0.0f;
    for (int i = 0; i < n; i++) {
        float val = data[i * cols + col];
        sum += val;
        sum_sq += val *val;
    }
    mean = sum / n;
    float variance = (sum_sq / n) - (mean * mean);
    stddev = sqrtf(fmaxf(variance, 1e-6f));
}

//exponential params
__device__ void estimateExponentialParams(float* data, int n, int col, int cols, float& rate) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += fmaxf(data[i*cols + col], 1e-6f);
    }
    rate = n /sum;
}

//laplace params estimation
__device__ void estimateLaplaceParams(float* data, int n, int col, int cols, float& location, float& scale) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += data[i *cols + col];
    }
    location = sum /n;
    float mad = 0.0f;
    for (int i = 0; i < n; i++) {
        mad +=fabsf(data[i * cols + col] - location);
    }
    scale = fmaxf(mad /(n*0.6745f), 1e-6f);
}

//gumbel params estimation
__device__ void estimateGumbelParams(float* data, int n, int col, int cols, float& location, float& scale) {
    float sum = 0.0f;
    float sum_sq = 0.0f;
    for (int i = 0; i < n; i++) {
        float val = data[i * cols + col];
        sum +=val;
        sum_sq +=val* val;
    }
    float mean = sum / n;
    float variance = (sum_sq / n) - (mean * mean);
    float stddev =sqrtf(fmaxf(variance, 1e-6f));
    scale = stddev *sqrtf(6.0f) / PI;
    location = mean - scale *EULER_MASCHERONI;
}

//student-t params estimation
__device__ void estimateStudentTParams(float* data, int n, int col, int cols, float& location, float& scale, float df) {
    float sum = 0.0f;
    float sum_sq = 0.0f;
    for (int i = 0; i < n; i++) {
        float val = data[i * cols + col];
        sum += val;
        sum_sq += val * val;
    }
    location = sum/ n;
    float variance = (sum_sq / n) - (location * location);
    scale = sqrtf(fmaxf(variance * (df -2.0f) / df, 1e-6f));
}

//log likelihood ethod
//TODO: parallelize this
__device__ float calculateLogLikelihood(float* data, int n, int col, int cols, DistributionParams params) {
    float logLik = 0.0f;
    for (int i = 0; i < n; i++) {
        float x = data[i * cols + col];
        // Enforce distribution support where applicable
        if (params.type == EXPONENTIAL && x < 0.0f) {
            return -INFINITY;
        }
        float pdf = 0.0f;
        switch (params.type) {
            case GAUSSIAN:
                pdf = gaussianPDF(x, params.param1, params.param2);
                break;
            case STUDENT_T:
                pdf = studentTPDF(x, params.param1, params.param2, params.param3);
                break;
            case EXPONENTIAL:
                pdf = exponentialPDF(x, params.param1);
                break;
            case GUMBEL:
                pdf = gumbelPDF(x, params.param1, params.param2);
                break;
            case LAPLACE:
                pdf = laplacePDF(x, params.param1, params.param2);
                break;
        }
        logLik += logf(fmaxf(pdf, 1e-30f));
    }
    return logLik;
}

//fit distribution
__global__ void fitDistribution(float* d_data, DistributionParams* d_all_params, 
                                int rows, int cols, int num_distributions) {
    int col = blockIdx.x;
    int dist_type = blockIdx.y;
    if (col >= cols ||dist_type >= num_distributions) return;
    int param_idx = col *num_distributions +dist_type;
    DistributionParams params;
    switch (dist_type) {
        case GAUSSIAN:
            params.type = GAUSSIAN;
            estimateGaussianParams(d_data, rows, col, cols, params.param1, params.param2);
            params.param3 = 0.0f;
            break;
            
        case STUDENT_T:
            params.type = STUDENT_T;
            params.param3 = 3.0f;
            estimateStudentTParams(d_data, rows, col, cols, params.param1, params.param2, params.param3);
            break;
            
        case EXPONENTIAL:
            params.type = EXPONENTIAL;
            estimateExponentialParams(d_data, rows, col, cols, params.param1);
            params.param2 = 0.0f;
            params.param3 = 0.0f;
            break;
            
        case GUMBEL:
            params.type = GUMBEL;
            estimateGumbelParams(d_data, rows, col, cols, params.param1, params.param2);
            params.param3 = 0.0f;
            break;
            
        case LAPLACE:
            params.type = LAPLACE;
            estimateLaplaceParams(d_data, rows, col, cols, params.param1, params.param2);
            params.param3 = 0.0f;
            break;
            
        default:
            return;
    }
    
    params.logLikelihood = calculateLogLikelihood(d_data, rows, col, cols, params);
    
    d_all_params[param_idx] = params;
}

// selects best distribution for each column using aic
__global__ void selectBestDistribution(DistributionParams* d_all_params, DistributionParams* d_best_params,
                                      int cols, int num_distributions) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= cols) return;
    int base_idx = col * num_distributions;
    float bestAIC = INFINITY;
    int bestIdx = 0;
    DistributionParams best = d_all_params[base_idx];
    for (int dist = 0; dist < num_distributions; dist++) {
        DistributionParams params = d_all_params[base_idx + dist];
        if (params.logLikelihood == -INFINITY) continue;
        int numParams = (params.type == EXPONENTIAL) ? 1 : ((params.type == STUDENT_T) ? 2 : 2);
        float aic = -2.0f * params.logLikelihood + 2.0f * (float)numParams;
        if (aic < bestAIC) {
            bestAIC = aic;
            bestIdx = dist;
            best = params;
        }
    }
    d_best_params[col] = best;
}

//compute cdf
__global__ void computeCDF(float* d_x_values, float* d_cdf_values, DistributionParams* d_params, 
                           int num_x, int num_cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y;
    if (idx >= num_x || col >= num_cols) return;
    float x = d_x_values[idx];
    DistributionParams params = d_params[col];
    float cdf = 0.0f;
    //TODO: parallelize this
    if (params.type == GAUSSIAN) {
        cdf = gaussianCDF(x, params.param1, params.param2);
    } else if (params.type == STUDENT_T) {
        cdf = studentTCDF(x, params.param1, params.param2, params.param3);
    } else if (params.type == EXPONENTIAL) {
        cdf = exponentialCDF(x, params.param1);
    } else if (params.type == GUMBEL) {
        cdf = gumbelCDF(x, params.param1, params.param2);
    } else if (params.type == LAPLACE) {
        cdf = laplaceCDF(x, params.param1, params.param2);
    } else {
        cdf = 0.5f;
    }
    d_cdf_values[col * num_x + idx] = cdf;
}
//compute cdf for all data points
__global__ void computeCDFForAllData(float* d_data, float* d_cdf_values, DistributionParams* d_params, 
                                     int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y;
    if (row >= rows || col >= cols) return;
    
    int idx = row * cols + col;
    float x = d_data[idx];
    DistributionParams params = d_params[col];
    float cdf = 0.0f;
    switch (params.type) {
        case GAUSSIAN:
            cdf = gaussianCDF(x, params.param1, params.param2);
            break;
        case STUDENT_T:
            cdf = studentTCDF(x, params.param1, params.param2, params.param3);
            break;
        case EXPONENTIAL:
            cdf = exponentialCDF(x, params.param1);
            break;
        case GUMBEL:
            cdf = gumbelCDF(x, params.param1, params.param2);
            break;
        case LAPLACE:
            cdf = laplaceCDF(x, params.param1, params.param2);
            break;
        default:
            cdf = 0.5f;
    }
    d_cdf_values[idx] = cdf;
}

//distribution name
const char* getDistributionName(DistributionType type) {
    switch (type) {
        case GAUSSIAN: return "Gaussian";
        case STUDENT_T: return "Student-t (df=3)";
        case EXPONENTIAL: return "Exponential";
        case GUMBEL: return "Gumbel";
        case LAPLACE: return "Laplace";
        default: return "Unknown";
    }
}

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
float* transposer(const std::vector<std::vector<float>>& data, int& rows, int& cols) {
    rows = data.size();
    cols = data[0].size();
    for (const auto& row : data) {
        if (row.size() != cols) {
            cols = std::min(cols, (int)row.size());
        }
    }
    float* flattned = new float[rows * cols];
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            flattned[i * cols + j] = data[i][j];
        }
    }
    return flattned;
}

// minimal cuda error check
inline void check(cudaError_t error) {
    if (error != cudaSuccess) exit(1);
}

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
    
    std::cout << rows << "x" << cols << std::endl;
    
    float* d_data;
    DistributionParams* d_params;
    size_t dataSize = rows * cols * sizeof(float);
    size_t paramsSize = cols * sizeof(DistributionParams);
    
    check(cudaMalloc(&d_data, dataSize));
    check(cudaMalloc(&d_params, paramsSize));
    
    check(cudaMemcpy(d_data, h_data, dataSize, cudaMemcpyHostToDevice));
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    
    
    int num_distributions = 5;
    DistributionParams* d_all_params;
    size_t allParamsSize = cols * num_distributions * sizeof(DistributionParams);
    check(cudaMalloc(&d_all_params, allParamsSize));
    
    cudaEventRecord(start);
    
    dim3 gridDim(cols, num_distributions);
    dim3 blockDim(1);  // One thread per block for this approach
    
    fitDistribution<<<gridDim, blockDim>>>(d_data, d_all_params, rows, cols, num_distributions);
    cudaDeviceSynchronize();
    
    int selectThreads = 256;
    int selectBlocks = (cols + selectThreads - 1) / selectThreads;
    selectBestDistribution<<<selectBlocks, selectThreads>>>(d_all_params, d_params, cols, num_distributions);
    cudaDeviceSynchronize();
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << milliseconds << " ms\n";
    
    cudaFree(d_all_params);
    
    DistributionParams* h_params = new DistributionParams[cols];
    check(cudaMemcpy(h_params, d_params, paramsSize, cudaMemcpyDeviceToHost));
    
    for (int i = 0; i < cols; i++) {
        std::cout << i + 1 << "," << getDistributionName(h_params[i].type) << "," << h_params[i].param1 << "," << h_params[i].param2 << "," << h_params[i].param3 << "," << h_params[i].logLikelihood << "\n";
    }
    
    
    
    
    
    float* d_cdf_values;
    check(cudaMalloc(&d_cdf_values, dataSize));
    
   
    dim3 cdfBlocks((rows + BLOCK_SIZE - 1) / BLOCK_SIZE, cols);
    dim3 cdfThreads(BLOCK_SIZE);
    
    computeCDFForAllData<<<cdfBlocks, cdfThreads>>>(d_data, d_cdf_values, d_params, rows, cols);
    cudaDeviceSynchronize();
    
    float* h_cdf_values = new float[rows * cols];
    check(cudaMemcpy(h_cdf_values, d_cdf_values, dataSize, cudaMemcpyDeviceToHost));
    
    std::string outputFilename = "cdf_values.csv";
    std::ofstream outFile(outputFilename);
    for (int j = 0; j < cols; j++) {
        outFile << "Column" << (j + 1) << "_CDF";
        if (j < cols - 1) outFile << ",";
    }
    outFile << "\n";
    outFile.setf(std::ios::fixed);
    outFile << std::setprecision(6);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            outFile << h_cdf_values[i * cols + j];
            if (j < cols - 1) outFile << ",";
        }
        outFile << "\n";
    }
    outFile.close();
    
    
    delete[] h_data;
    delete[] h_params;
    delete[] h_cdf_values;
    
    cudaFree(d_data);
    cudaFree(d_params);
    cudaFree(d_cdf_values);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}