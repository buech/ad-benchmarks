/*******************************************************************************
   CppAD LIBOR Swaption MC benchmark - reverse mode.

   Copyright (c) 2026 Xcelerit Computing Ltd.
   Licensed under the MIT License. See LICENSE file.
******************************************************************************/

#include "../src/benchmarks.hpp"
#include "../src/libor_swaption.hpp"
#include <cppad/cppad.hpp>

BenchmarkResult cppad_libor_swaption(int numPaths, size_t warmup, size_t iters)
{
    using ADdouble = CppAD::AD<double>;

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
            std::mt19937 gen(SEED);
            std::vector<double> samples(market.lambda.size() / 2);
            const size_t totalInputs = 1 + market.lambda.size() + market.L0.size();

            double total_price = 0.0;
            std::vector<double> total_grads(totalInputs, 0.0);

            for (int path = 0; path < numPaths; ++path)
            {
                generateSamples(gen, samples);

                std::vector<ADdouble> ax(totalInputs);
                ax[0] = market.delta;
                for (size_t i = 0; i < market.lambda.size(); ++i)
                    ax[1 + i] = market.lambda[i];
                for (size_t i = 0; i < market.L0.size(); ++i)
                    ax[1 + market.lambda.size() + i] = market.L0[i];

                CppAD::Independent(ax);

                ADdouble delta = ax[0];
                std::vector<ADdouble> lambda(market.lambda.size());
                std::vector<ADdouble> L0(market.L0.size());
                for (size_t i = 0; i < market.lambda.size(); ++i)
                    lambda[i] = ax[1 + i];
                for (size_t i = 0; i < market.L0.size(); ++i)
                    L0[i] = ax[1 + market.lambda.size() + i];

                std::vector<ADdouble> L(L0.begin(), L0.end());
                std::vector<ADdouble> tmp1, tmp2;
                libor_path_gen(delta, L, lambda, samples);
                ADdouble v = libor_value_portfolio(delta, portfolio.maturities,
                                                   portfolio.swaprates, L, tmp1, tmp2);

                std::vector<ADdouble> ay = {v};
                CppAD::ADFun<double> f(ax, ay);

                std::vector<double> w(1, 1.0);
                std::vector<double> grad = f.Reverse(1, w);

                total_price += CppAD::Value(v);
                for (size_t i = 0; i < totalInputs; ++i)
                    total_grads[i] += grad[i];
            }

            volatile double p = total_price / numPaths;
            (void)p;
        },
        warmup, iters);

    return {"CppAD", "LiborSwaption", primal_ms, grad_ms};
}
