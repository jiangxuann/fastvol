/**
 * @file example_neural.cpp
 * @brief Example usage of neural network pricing in fastvol.
 */

#include "fastvol/fastvol.hpp"
#include <iostream>
#include <vector>

int main() {
    // Check if neural support is available
    if (!fastvol_neural_available()) {
        std::cout << "Neural network support not available. Build with LibTorch to enable.\n";
        return 1;
    }

#ifdef FASTVOL_NEURAL_ENABLED
    try {
        // Single option pricing
        double price = fastvol::american::neural::price_fp64(
            100.0,  // spot price
            100.0,  // strike price
            1,      // call option
            0.25,   // 3 months
            0.2,    // 20% volatility
            0.05,   // 5% risk-free rate
            0.0     // no dividends
        );
        std::cout << "Neural price: $" << price << std::endl;

        // Batch pricing
        std::vector<double> S = {95.0, 100.0, 105.0};
        std::vector<double> K = {100.0, 100.0, 100.0};
        std::vector<char> cp = {1, 1, 1};
        std::vector<double> ttm = {0.25, 0.25, 0.25};
        std::vector<double> iv = {0.2, 0.2, 0.2};
        std::vector<double> r = {0.05, 0.05, 0.05};
        std::vector<double> q = {0.0, 0.0, 0.0};
        std::vector<double> results(3);

        fastvol::american::neural::price_fp64_batch(
            S.data(), K.data(), cp.data(), ttm.data(),
            iv.data(), r.data(), q.data(), 3, results.data()
        );

        for (size_t i = 0; i < 3; ++i) {
            std::cout << "Option " << (i+1) << ": $" << results[i] << std::endl;
        }

    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
        std::cout << "Requires neural model files (.pt) in current directory or ./models/\n";
        return 1;
    }
#endif

    return 0;
}
