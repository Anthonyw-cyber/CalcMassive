#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <sstream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_E
#define M_E 2.71828182845904523536
#endif

using namespace std;

struct Agent {
    double *x;
    double B;
    double I;
    double objectivFunc;
};

__global__ void initializeCurand(curandState *states, int pop_size, unsigned long long seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < pop_size) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

__device__ double sphereFunction(const double *x, int dim) {
    double sum = 0.0;
    for (int i = 0; i < dim; i++) {
        sum += x[i] * x[i];
    }
    return sum;
}

__global__ void evaluateFitness(Agent *fireflies, int pop_size, int dim) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < pop_size) {
        fireflies[idx].objectivFunc = sphereFunction(fireflies[idx].x, dim);
        fireflies[idx].I = 1.0 / fireflies[idx].objectivFunc;
    }
}

__global__ void updateFireflies(Agent *fireflies, int pop_size, int dim, double beta_base, double gamma, double alpha, double lb, double ub, curandState *states) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < pop_size) {
        curandState state = states[i];

        for (int j = 0; j < pop_size; j++) {
            double r = 0.0;
            for (int k = 0; k < dim; k++) {
                r += (fireflies[i].x[k] - fireflies[j].x[k]) * (fireflies[i].x[k] - fireflies[j].x[k]);
            }
            r = sqrt(r);

            if (fireflies[j].I > fireflies[i].I) {
                fireflies[i].B = beta_base * exp(-gamma * r * r);
                for (int k = 0; k < dim; k++) {
                    double u = alpha * (curand_uniform(&state) - 0.5);
                    fireflies[i].x[k] += fireflies[i].B * (fireflies[j].x[k] - fireflies[i].x[k]) + u;
                    fireflies[i].x[k] = max(lb, min(ub, fireflies[i].x[k]));
                }
            }
        }
        states[i] = state;
        __syncthreads();
    }
}

void generate_population(int pop_size, int dim, Agent *fireflies, double lb, double ub) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dis(lb, ub);

    for (int i = 0; i < pop_size; i++) {
        fireflies[i].B = 1.0;
        cudaMallocManaged(&fireflies[i].x, dim * sizeof(double));
        for (int j = 0; j < dim; j++) {
            fireflies[i].x[j] = dis(gen);
        }
    }
}

int main() {
    vector<int> dimensions = {10, 30, 50};
    vector<int> popSizes = {30, 50, 70};

    for (int dim : dimensions) {
        for (int popSize : popSizes) {
            cout << "dim: " << dim << " popSize: " << popSize << endl;
            stringstream ss;
            ss << "results_dim_" << dim << "_pop_" << popSize << ".txt";
            string fileName = ss.str();
            ofstream outFile(fileName);
            if (!outFile) {
                cerr << "Cannot open output file: " << fileName << endl;
                return 1;
            }

            int pop_size = popSize;
            int n_dim = dim;
            double lb = -10.0, ub = 10.0;
            double alpha = 0.2, beta_base = 2.0, gamma = 0.001;
            unsigned long long seed = 12345678;

            Agent *fireflies;
            cudaMallocManaged(&fireflies, pop_size * sizeof(Agent));
            generate_population(pop_size, n_dim, fireflies, lb, ub);

            curandState *d_states;
            cudaMalloc(&d_states, pop_size * sizeof(curandState));
            initializeCurand<<<(pop_size + 255) / 256, 256>>>(d_states, pop_size, seed);
            cudaDeviceSynchronize();

            int blockSize = 256;
            int numBlocks = (pop_size + blockSize - 1) / blockSize;

            // Mesure du temps avec CUDA Events
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);

            for (int run = 0; run < 10; run++) {
                for (int t = 0; t < 5000; t++) {
                    updateFireflies<<<numBlocks, blockSize>>>(fireflies, pop_size, n_dim, beta_base, gamma, alpha, lb, ub, d_states);
                    evaluateFitness<<<numBlocks, blockSize>>>(fireflies, pop_size, n_dim);
                    cudaDeviceSynchronize();
                }

                double bestFitness = fireflies[0].objectivFunc;
                for (int i = 1; i < pop_size; i++) {
                    if (fireflies[i].objectivFunc < bestFitness) {
                        bestFitness = fireflies[i].objectivFunc;
                    }
                }
                outFile << bestFitness << endl;
                printf("Run %d: Best Fitness = %lf\n", run, bestFitness);
            }

            // Fin de la mesure du temps
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            printf("Execution Time for dim=%d, popSize=%d: %f ms\n", dim, popSize, milliseconds);

            outFile.close();

            for (int i = 0; i < pop_size; i++) {
                cudaFree(fireflies[i].x);
            }
            cudaFree(fireflies);
            cudaFree(d_states);

            // Libération des CUDA Events
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }
    }

    return 0;
}
