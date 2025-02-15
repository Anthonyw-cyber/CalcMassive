#ifndef FIREFLY_CUDA_H
#define FIREFLY_CUDA_H

#include <vector>

class Agent {
public:
    std::vector<double> x;
    double B;
    double I;
    double objectivFunc;

    Agent(int dim, double lb, double ub);
    ~Agent();
};

double sphereFunction(const std::vector<double>& x);
double rastriginFunction(const std::vector<double>& x);
double ackleyFunction(const std::vector<double>& x);
double rosenbrockFunction(const std::vector<double>& x);

void generate_population(int pop_size, int dim, Agent firefly[], double ub, double lb);
double uRandom(double a);

// Déclaration des fonctions CUDA
void runFireflyCUDA(int pop_size, int dim, int epochs, double lb, double ub, double *h_best_per_iteration);

#endif
