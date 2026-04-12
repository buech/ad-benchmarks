/*******************************************************************************
   XAD Heston Stochastic Volatility benchmark - adjoint mode, pathwise.

   Copyright (c) 2026 Xcelerit Computing Ltd.
   Licensed under the MIT License. See LICENSE file.
******************************************************************************/

#include "../src/benchmarks.hpp"
#include "../src/heston.hpp"
#include <XAD/XAD.hpp>

BenchmarkResult xad_heston(int numPaths, size_t warmup, size_t iters)
{
    using mode = xad::adj<double>;
    using tape_type = mode::tape_type;
    using AD = mode::active_type;

    HestonInputs in;
    const unsigned long long seed = 77777;
    auto samples = heston_generate_samples(numPaths, seed);

    // Primal — uses pre-generated samples to match the gradient benchmark
    // setup, so primal × (N+1) is comparable to FD gradient time.
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

    // Gradient via adjoint (pathwise)
    double grad_ms = benchmark(
        [&]()
        {
            tape_type tape;

            double d_S0 = 0.0, d_K = 0.0, d_T = 0.0, d_r = 0.0;
            double d_v0 = 0.0, d_kappa = 0.0, d_theta = 0.0, d_xi = 0.0;

            for (int path = 0; path < numPaths; ++path)
            {
                tape.clearAll();

                AD S0 = in.S0, K = in.K, T = in.T, r = in.r;
                AD v0 = in.v0, kappa = in.kappa, theta = in.theta, xi = in.xi;
                tape.registerInput(S0);
                tape.registerInput(K);
                tape.registerInput(T);
                tape.registerInput(r);
                tape.registerInput(v0);
                tape.registerInput(kappa);
                tape.registerInput(theta);
                tape.registerInput(xi);
                tape.newRecording();

                AD price = heston_path(S0, K, T, r, v0, kappa, theta, xi,
                                       in.rho_corr,
                                       samples.z1[path], samples.z2[path]);

                tape.registerOutput(price);
                derivative(price) = 1.0;
                tape.computeAdjoints();

                d_S0 += derivative(S0);
                d_K += derivative(K);
                d_T += derivative(T);
                d_r += derivative(r);
                d_v0 += derivative(v0);
                d_kappa += derivative(kappa);
                d_theta += derivative(theta);
                d_xi += derivative(xi);
            }

            // Average
            volatile double g1 = d_S0 / numPaths;
            volatile double g2 = d_K / numPaths;
            volatile double g3 = d_T / numPaths;
            volatile double g4 = d_r / numPaths;
            volatile double g5 = d_v0 / numPaths;
            volatile double g6 = d_kappa / numPaths;
            volatile double g7 = d_theta / numPaths;
            volatile double g8 = d_xi / numPaths;
            (void)g1; (void)g2; (void)g3; (void)g4;
            (void)g5; (void)g6; (void)g7; (void)g8;
        },
        warmup, iters);

    return {"XAD", "HestonMC", primal_ms, grad_ms};
}
