#include <iostream>
#include <vector>
#include <fstream>
#include <cstdlib>  // ✅ Ajout nécessaire pour éviter les erreurs de compatibilité
#include "firefly.h"
using namespace std;
int main() {
    int popSize = 50, dim = 30, epochs = 5000;
    double lb = -10.0, ub = 10.0;
    int runs = 10;
    std::vector<double> best_intensities(runs);

    std::ofstream outFile("results_gpu_Ackley.txt");
    if (!outFile.is_open()) {
        std::cerr << "Erreur : Impossible d'ouvrir results_gpu.txt" << std::endl;
        return 1;
    }

    for (int run = 0; run < runs; run++) {
        std::cout << "Démarrage de l'algorithme Firefly sur GPU - Run " << run + 1 << "..." << std::endl;
        runFireflyCUDA(popSize, dim, epochs, lb, ub, &best_intensities[run]);
        std::cout << "🔥 Fin du Run " << run + 1 << " - Best Fitness: " << best_intensities[run] << std::endl;

        outFile << "Run " << run + 1 << " Best Fitness: " << best_intensities[run] << std::endl;
    }

    outFile.close();
    std::cout << "✅ Résultats GPU sauvegardés dans results_gpu.txt" << std::endl;

    return 0;
}
