#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <random>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <sstream>  // Ajouter cet include

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_E
#define M_E 2.71828182845904523536
#endif

using namespace std;

__device__ double sphereFunction(const double *x, int dim)
{
    double sum = 0.0;
    for (int i = 0; i < dim; i++)
    {
        sum += x[i] * x[i]; // Somme des carrés des éléments
    }
    return sum;
}


// Fonction de Rosenbrock
__device__ double rosenbrockFunction(const double *x, int dim)
{
    double sum = 0.0;
    for (int i = 0; i < dim - 1; i++)
    {
        sum += 100 * pow(x[i + 1] - x[i] * x[i], 2) + pow(1 - x[i], 2);
    }
    return sum;
}
// Fonction d'Ackley
__device__ double ackleyFunction(const double *x, int dim)
{
    double sum1 = 0.0, sum2 = 0.0;
    for (int i = 0; i < dim; i++)
    {
        sum1 += x[i] * x[i];
        sum2 += cos(2.0 * M_PI * x[i]);
    }
    return -20.0 * exp(-0.2 * sqrt(sum1 / dim)) - exp(sum2 / dim) + 20.0 + M_E;
}
// Fonction de Rastrigin
__device__ double rastriginFunction(const double *x, int dim)
{
    double sum = 10.0 * dim;
    for (int i = 0; i < dim; i++)
    {
        sum += x[i] * x[i] - 10.0 * cos(2.0 * M_PI * x[i]);
    }
    return sum;
}

__global__ void computeFitness(double *positions, double *intensities, int popSize, int dim)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < popSize)
    {
       double  fitness = rosenbrockFunction(&positions[i * dim], dim);//changer la fonction ici
        intensities[i] = fitness ;
    }
}
__global__ void enforceBoundaries(double *positions, int popSize, int dim, double lb, double ub)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < popSize)
    {
        for (int k = 0; k < dim; k++)
        {
            if (positions[i * dim + k] < lb) positions[i * dim + k] = lb;
            if (positions[i * dim + k] > ub) positions[i * dim + k] = ub;
        }
    }
}

__global__ void updateFireflies(double *positions, double *intensities, double *betas,double beta_base, int popSize, int dim, double gamma, double alpha, double lb, double ub, curandState *states)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < popSize)
    {
        curandState localState = states[i];
        for (int j = 0; j < popSize; j++)
        {

                double r = 0.0;
                for (int k = 0; k < dim; k++)
                {
                    r += (positions[i * dim + k] - positions[j * dim + k]) * (positions[i * dim + k] - positions[j * dim + k]);
                }
                r = sqrt(r);
                if (intensities[j] > intensities[i]) {
                    betas[i] = beta_base * exp(-gamma * pow(r, 2));
                    for (int k = 0; k < dim; k++) {
                        double E = curand_uniform(&localState); // Génération d'un nombre dans [0,1]
                        double u = alpha * (E - 0.5);
                        positions[i * dim + k] += betas[i] * (positions[j * dim + k] - positions[i * dim + k]) + u;
                    }
                }
        }
        states[i] = localState;
    }

}
__global__ void initPopulation(double *positions, int popSize, int dim, double lb, double ub, curandState *states) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < popSize * dim) {
        curandState localState = states[i];
        positions[i] = lb + (ub - lb) * curand_uniform(&localState); // Génération uniforme entre lb et ub
        states[i] = localState; // Sauvegarde de l'état
    }
}

__global__ void initCurand(curandState *states, unsigned long seed, int popSize)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < popSize)
    {
        curand_init(seed + i,0, 0, &states[i]);
    }
}

int main()
{
    int dimensions[] = {10, 30, 50}; // Dimensions
    int popSizes[] = {30, 50, 70}; // Tailles de la population
    double lb = -10.0, ub = 10.0;
    double gamma = 0.001, alpha = 0.2;
    int epochs = 5000;
    double beta_base = 2;

    for (int dim : dimensions)
    {
        for (int popSize : popSizes) {
            for (int run = 0; run < 10; run++) {
                // Nom du fichier en fonction de la dimension et de la population
                stringstream ss;
                ss << "results_dim_" << dim << "_pop_" << popSize << ".txt";
                string fileName = ss.str();
                ofstream outFile(fileName, ios::app);

                if (!outFile) {
                    cerr << "Impossible d'ouvrir le fichier de sortie : " << fileName << endl;
                    return 1;
                }

                // Initialisation des variables
                double bestFitness = DBL_MAX;
                vector<double> positions(popSize * dim);
                vector<double> intensities(popSize);
                vector<double> betas(popSize, 1.0);

                random_device rd;
                mt19937 gen(rd());
                uniform_real_distribution<double> dist(lb, ub);

 // Début Initialisation des positions aléatoires
                for (int i = 0; i < popSize * dim; i++) {
                    positions[i] = (rand() / (double)RAND_MAX) * (ub - lb) + lb;
                }

                double *d_positions, *d_intensities, *d_betas;
                curandState *d_states;
                cudaMalloc(&d_positions, popSize * dim * sizeof(double));
                cudaMalloc(&d_intensities, popSize * sizeof(double));
                cudaMalloc(&d_betas, popSize * sizeof(double));
                cudaMalloc(&d_states, popSize * sizeof(curandState));

                cudaMemcpy(d_positions, positions.data(), popSize * dim * sizeof(double), cudaMemcpyHostToDevice);
                cudaMemcpy(d_betas, betas.data(), popSize * sizeof(double), cudaMemcpyHostToDevice);

                int threadsPerBlock = 256;
                int blocksPerGrid = (popSize + threadsPerBlock - 1) / threadsPerBlock;
                initCurand<<<blocksPerGrid, threadsPerBlock>>>(d_states, time(NULL), popSize);
                computeFitness<<<blocksPerGrid, threadsPerBlock>>>(d_positions, d_intensities, popSize, dim);
// fin initialisation de population
                // Création des événements CUDA pour la mesure du temps
                cudaEvent_t start, stop;
                cudaEventCreate(&start);
                cudaEventCreate(&stop);

                // Démarrer la mesure du temps
                cudaEventRecord(start);

                for (int t = 0; t < epochs; t++) {
                    updateFireflies<<<blocksPerGrid, threadsPerBlock>>>(d_positions, d_intensities, d_betas, beta_base,
                                                                        popSize, dim, gamma, alpha, lb, ub, d_states);
                    enforceBoundaries<<<blocksPerGrid, threadsPerBlock>>>(d_positions, popSize, dim, lb, ub);
                    computeFitness<<<blocksPerGrid, threadsPerBlock>>>(d_positions, d_intensities, popSize, dim);
                    cudaMemcpy(intensities.data(), d_intensities, popSize * sizeof(double), cudaMemcpyDeviceToHost);

                    // Trouver l'index du meilleur individu (celui qui a la plus petite fitness)
                    int best_index = distance(intensities.begin(), min_element(intensities.begin(), intensities.end()));

                    // Vérifier si la meilleure fitness trouvée est meilleure que l'actuelle
                    if (intensities[best_index] < bestFitness) {
                        bestFitness = intensities[best_index]; // Mise à jour de bestFitness

                        // Ajouter une perturbation à la meilleure solution pour éviter la stagnation
                        for (int i = 0; i < dim; i++) {
                            double u = ((rand() / (double) RAND_MAX) - 0.5) *
                                       alpha; // Générer une perturbation u ∈ [-0.5 * alpha, 0.5 * alpha]
                            positions[best_index * dim + i] += u; // Appliquer la perturbation
                        }

                        // Copier les positions mises à jour du CPU vers le GPU
                        cudaMemcpy(d_positions, positions.data(), popSize * dim * sizeof(double),
                                   cudaMemcpyHostToDevice);
                    }
                }

                cudaMemcpy(intensities.data(), d_intensities, popSize * sizeof(double), cudaMemcpyDeviceToHost);

                double currentBestFitness = *min_element(intensities.begin(), intensities.end());

                if (currentBestFitness < bestFitness) // Condition correcte (minimisation)
                {
                    bestFitness = currentBestFitness;
                }
                // Arrêter la mesure du temps
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);

                // Calcul du temps écoulé
                float milliseconds = 0;
                cudaEventElapsedTime(&milliseconds, start, stop);

                printf("Meilleure fitness obtenue après %d epochs: %f\n", epochs, bestFitness,milliseconds);
                outFile << bestFitness << endl;

                cudaEventDestroy(start);
                cudaEventDestroy(stop);

                cudaFree(d_positions);
                cudaFree(d_intensities);
                cudaFree(d_betas);
                cudaFree(d_states);
            }
        }
    }

    cout << "Calcul terminé." << endl;
    return 0;
}
