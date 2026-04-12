/*******************************************************************************
   Adept 2 Heston Stochastic Volatility benchmark - reverse mode, pathwise.

   8 inputs (S0, K, T, r, v0, kappa, theta, xi).
   Pathwise adjoint: tape one path at a time and accumulate gradients.

   Copyright (c) 2026 Xcelerit Computing Ltd.
   Licensed under the MIT License. See LICENSE file.
******************************************************************************/

#include "../src/benchmarks.hpp"
#include "../src/heston.hpp"
#include <adept.h>
#include <adept_arrays.h>

BenchmarkResult adept_heston(int numPaths, size_t warmup, size_t iters)
{
    using adept::adouble;

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
            adept::Stack stack;

            std::vector<double> total_d(HESTON_INPUTS, 0.0);
            double total_price = 0.0;

            for (int path = 0; path < numPaths; ++path)
            {
                adouble S0 = in.S0, K = in.K, T = in.T, r = in.r;
                adouble v0 = in.v0, kappa = in.kappa, theta = in.theta, xi = in.xi;

                stack.new_recording();

                adouble v = heston_path(S0, K, T, r, v0, kappa, theta, xi,
                                        in.rho_corr,
                                        samples.z1[path], samples.z2[path]);

                v.set_gradient(1.0);
                stack.reverse();

                total_price += adept::value(v);
                total_d[0] += S0.get_gradient();
                total_d[1] += K.get_gradient();
                total_d[2] += T.get_gradient();
                total_d[3] += r.get_gradient();
                total_d[4] += v0.get_gradient();
                total_d[5] += kappa.get_gradient();
                total_d[6] += theta.get_gradient();
                total_d[7] += xi.get_gradient();
            }

            total_price /= numPaths;
            (void)total_price;
        },
        warmup, iters);

    return {"Adept", "HestonMC", primal_ms, grad_ms};
}
