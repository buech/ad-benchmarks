/*******************************************************************************
   CppAD Heston Stochastic Volatility benchmark - reverse mode, pathwise.

   8 inputs (S0, K, T, r, v0, kappa, theta, xi).
   Pathwise adjoint: tape one path at a time and accumulate gradients.

   Copyright (c) 2026 Xcelerit Computing Ltd.
   Licensed under the MIT License. See LICENSE file.
******************************************************************************/

#include "../src/benchmarks.hpp"
#include "../src/heston.hpp"
#include <cppad/cppad.hpp>

BenchmarkResult cppad_heston(int numPaths, size_t warmup, size_t iters)
{
    using ADdouble = CppAD::AD<double>;

    HestonInputs in;

    // Pre-generate samples for reproducibility
    auto samples = heston_generate_samples(numPaths);

    // Primal — pre-generated samples to match the gradient setup.
    double primal_ms = benchmark(
        [&]()
        {
            double sum = 0.0;
            for (int p = 0; p < numPaths; ++p)
                sum += heston_path(in.S0, in.K, in.T, in.r, in.v0, in.kappa,
                                   in.theta, in.xi, in.rho_corr,
                                   samples.z1[p], samples.z2[p]);
            volatile double v = sum / numPaths;
            (void)v;
        },
        warmup, iters);

    // Gradient via reverse mode (pathwise)
    double grad_ms = benchmark(
        [&]()
        {
            std::vector<double> total_grad(HESTON_INPUTS, 0.0);
            double total_price = 0.0;

            for (int path = 0; path < numPaths; ++path)
            {
                std::vector<ADdouble> ax = {in.S0, in.K, in.T, in.r,
                                            in.v0, in.kappa, in.theta, in.xi};
                CppAD::Independent(ax);

                std::vector<ADdouble> ay(1);
                ay[0] = heston_path(ax[0], ax[1], ax[2], ax[3],
                                    ax[4], ax[5], ax[6], ax[7],
                                    in.rho_corr,
                                    samples.z1[path], samples.z2[path]);

                CppAD::ADFun<double> f(ax, ay);

                std::vector<double> w(1, 1.0);
                std::vector<double> grad = f.Reverse(1, w);

                total_price += CppAD::Value(ay[0]);
                for (int i = 0; i < HESTON_INPUTS; ++i)
                    total_grad[i] += grad[i];
            }

            total_price /= numPaths;
            (void)total_price;
        },
        warmup, iters);

    return {"CppAD", "HestonMC", primal_ms, grad_ms};
}
