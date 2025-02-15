#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include "firefly.h"

__device__ double rosenbrockFunctionCUDA(const double *x, int dim) {
    double sum = 0.0;
    for (int i = 0; i < dim - 1; i++) {
        sum += 100 * pow(x[i + 1] - x[i] * x[i], 2) + pow(1 - x[i], 2);
    }
    return sum;
}

// Initialisation des fireflies sur GPU
__global__ void initializeFireflies(double *positions, int pop_size, int dim, double lb, double ub, curandState *states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < pop_size) {
        curandState localState = states[idx];
        for (int i = 0; i < dim; i++) {
            positions[idx * dim + i] = lb + (curand_uniform(&localState) * (ub - lb));
        }
        states[idx] = localState;
    }
}

__global__ void computeFitness(double *positions, double *intensities, int pop_size, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < pop_size) {
        intensities[i] = 1.0 / (1.0 + rosenbrockFunctionCUDA(&positions[i * dim], dim));
    }
}

__global__ void reduceMax(double *d_intensities, double *d_max, int pop_size) {
    extern __shared__ double sharedData[];
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    sharedData[tid] = (index < pop_size) ? d_intensities[index] : -1.0;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedData[tid] = max(sharedData[tid], sharedData[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_max[blockIdx.x] = sharedData[0];
    }
}

void runFireflyCUDA(int pop_size, int dim, int epochs, double lb, double ub, double *h_best_per_iteration) {
    double *d_positions, *d_intensities, *d_max;
    cudaMalloc(&d_positions, pop_size * dim * sizeof(double));
    cudaMalloc(&d_intensities, pop_size * sizeof(double));
    cudaMalloc(&d_max, pop_size * sizeof(double));

    int threadsPerBlock = 256;
    int blocksPerGrid = (pop_size + threadsPerBlock - 1) / threadsPerBlock;

    double *h_max_intermediate = (double*) malloc(blocksPerGrid * sizeof(double));

    for (int t = 0; t < epochs; t++) {
        if (t % 100 == 0) {  // ✅ Affiche toutes les 100 itérations pour éviter trop de logs
            std::cout << "🔥 CUDA - Epoch: " << t << std::endl;
        }
        computeFitness<<<blocksPerGrid, threadsPerBlock>>>(d_positions, d_intensities, pop_size, dim);
        cudaDeviceSynchronize();

        if (t % 10 == 0) {
            reduceMax<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(double)>>>(d_intensities, d_max, pop_size);
            cudaDeviceSynchronize();

            cudaMemcpy(h_max_intermediate, d_max, blocksPerGrid * sizeof(double), cudaMemcpyDeviceToHost);

            double best_intensity = -1.0;
            for (int i = 0; i < blocksPerGrid; i++) {
                best_intensity = max(best_intensity, h_max_intermediate[i]);
            }

            std::cout << "🔥 CUDA - Meilleur score trouvé à Epoch " << t << " : " << best_intensity << std::endl;
            h_best_per_iteration[t / 10] = best_intensity;
        }
    }

    cudaFree(d_positions);
    cudaFree(d_intensities);
    cudaFree(d_max);
    free(h_max_intermediate);
}
