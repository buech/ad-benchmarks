/*******************************************************************************
   adventure Heston Stochastic Volatility benchmark - adjoint mode, pathwise.

   Copyright (c) 2026 Xcelerit Computing Ltd.
   Copyright (c) 2026 Adam Buechner
   Licensed under the MIT License. See LICENSE file.
******************************************************************************/

#include "../src/benchmarks.hpp"
#include "../src/heston.hpp"
#include "adventure/variable.hpp"

BenchmarkResult adventure_heston(int numPaths, size_t warmup, size_t iters)
{
    namespace ad = adventure;
    using AD = ad::Variable<double>;

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
            auto &tape = ad::get_tape<double>();
            //ad::Tape<double> tape;

            double d_S0 = 0.0, d_K = 0.0, d_T = 0.0, d_r = 0.0;
            double d_v0 = 0.0, d_kappa = 0.0, d_theta = 0.0, d_xi = 0.0;

            for (int path = 0; path < numPaths; ++path)
            {
                tape.clear();

                AD S0 = in.S0, K = in.K, T = in.T, r = in.r;
                AD v0 = in.v0, kappa = in.kappa, theta = in.theta, xi = in.xi;
                tape.register_input(S0);
                tape.register_input(K);
                tape.register_input(T);
                tape.register_input(r);
                tape.register_input(v0);
                tape.register_input(kappa);
                tape.register_input(theta);
                tape.register_input(xi);

                AD price = heston_path(S0, K, T, r, v0, kappa, theta, xi,
                                       in.rho_corr,
                                       samples.z1[path], samples.z2[path]);

                tape.register_output(price);
                price.grad() = 1.0;
                tape.backward();

                d_S0 += S0.grad();
                d_K += K.grad();
                d_T += T.grad();
                d_r += r.grad();
                d_v0 += v0.grad();
                d_kappa += kappa.grad();
                d_theta += theta.grad();
                d_xi += xi.grad();
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

    return {"adventure", "HestonMC", primal_ms, grad_ms};
}
