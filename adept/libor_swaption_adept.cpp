/*******************************************************************************
   Adept 2 LIBOR Swaption MC benchmark - reverse mode.

   Copyright (c) 2026 Xcelerit Computing Ltd.
   Licensed under the MIT License. See LICENSE file.
******************************************************************************/

#include "../src/benchmarks.hpp"
#include "../src/libor_swaption.hpp"
#include <adept.h>
#include <adept_arrays.h>

BenchmarkResult adept_libor_swaption(int numPaths, size_t warmup, size_t iters)
{
    using adept::adouble;

    auto portfolio = defaultPortfolio();
    auto market = defaultMarket();
    const unsigned long long SEED = 12354;

    double primal_ms = benchmark(
        [&]() {
            volatile double v = libor_price_mc(market, portfolio, numPaths, SEED);
            (void)v;
        },
        warmup, iters);

    double grad_ms = benchmark(
        [&]()
        {
            adept::Stack stack;
            std::mt19937 gen(SEED);
            std::vector<double> samples(market.lambda.size() / 2);

            SensitivityResults res;
            res.d_lambda.resize(market.lambda.size(), 0.0);
            res.d_L0.resize(market.L0.size(), 0.0);

            for (int path = 0; path < numPaths; ++path)
            {
                adouble delta = market.delta;
                std::vector<adouble> lambda(market.lambda.begin(), market.lambda.end());
                std::vector<adouble> L0(market.L0.begin(), market.L0.end());

                stack.new_recording();

                generateSamples(gen, samples);
                std::vector<adouble> L(L0.begin(), L0.end());
                std::vector<adouble> tmp1, tmp2;
                libor_path_gen(delta, L, lambda, samples);
                adouble v = libor_value_portfolio(delta, portfolio.maturities,
                                                  portfolio.swaprates, L, tmp1, tmp2);

                v.set_gradient(1.0);
                stack.reverse();

                res.price += adept::value(v);
                res.d_delta += delta.get_gradient();
                for (size_t i = 0; i < market.lambda.size(); ++i)
                {
                    res.d_lambda[i] += lambda[i].get_gradient();
                    res.d_L0[i] += L0[i].get_gradient();
                }
            }

            volatile double p = res.price / numPaths;
            (void)p;
        },
        warmup, iters);

    return {"Adept", "LiborSwaption", primal_ms, grad_ms};
}
