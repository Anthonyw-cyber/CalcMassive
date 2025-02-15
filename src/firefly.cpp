#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <random>
#include "firefly.h"
#include <cmath>

// ✅ Définir M_PI et M_E si elles ne sont pas disponibles
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_E
#define M_E 2.71828182845904523536
#endif

Agent::Agent(int dim, double lb, double ub) {
    B = 1.0;
    for (int i = 0; i < dim; i++) {
        double temp = (rand() / (double)RAND_MAX) * (ub - lb) + lb;
        x.push_back(temp);
    }
    objectivFunc = ackleyFunction(x);
    I = 1.0 / objectivFunc;
}

Agent::~Agent() {}

double sphereFunction(const std::vector<double> &x) {
    double sum = 0.0;
    for (double val : x) sum += val * val;
    return sum;
}

double rastriginFunction(const std::vector<double> &x) {
    double sum = 0.0;
    for (double val : x) sum += val * val - 10 * cos(2 * M_PI * val) + 10;
    return sum;
}

double ackleyFunction(const std::vector<double> &x) {
    double sum1 = 0.0, sum2 = 0.0;
    for (double val : x) {
        sum1 += val * val;
        sum2 += cos(2 * M_PI * val);
    }
    return -20 * exp(-0.2 * sqrt(sum1 / x.size())) - exp(sum2 / x.size()) + 20 + M_E;
}

double rosenbrockFunction(const std::vector<double> &x) {
    double sum = 0.0;
    for (size_t i = 0; i < x.size() - 1; i++) {
        sum += 100 * pow(x[i + 1] - pow(x[i], 2), 2) + pow(1 - x[i], 2);
    }
    return sum;
}

void generate_population(int pop_size, int dim, Agent firefly[], double ub, double lb) {
    for (int i = 0; i < pop_size; i++) {
        firefly[i] = Agent(dim, lb, ub);
    }
}

double uRandom(double a) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double E = distribution(gen);
    return a * (E - 0.5);
}
