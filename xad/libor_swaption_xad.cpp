/*******************************************************************************
   XAD LIBOR Swaption MC benchmark - adjoint mode, tape-based.

   161 market inputs, pathwise adjoint. Re-records tape per path.

   Copyright (c) 2026 Xcelerit Computing Ltd.
   Licensed under the MIT License. See LICENSE file.
******************************************************************************/

#include "../src/benchmarks.hpp"
#include "../src/libor_swaption.hpp"
#include <XAD/XAD.hpp>

BenchmarkResult xad_libor_swaption(int numPaths, size_t warmup, size_t iters)
{
    using mode = xad::adj<double>;
    using tape_type = mode::tape_type;
    using AD = mode::active_type;

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
            tape_type tape;
            std::mt19937 gen(SEED);
            std::vector<double> samples(market.lambda.size() / 2);
            std::vector<AD> L, tmp1, tmp2, lambda, L0;

            SensitivityResults res;
            res.d_lambda.resize(market.lambda.size(), 0.0);
            res.d_L0.resize(market.L0.size(), 0.0);

            for (int path = 0; path < numPaths; ++path)
            {
                tmp1.clear(); tmp2.clear(); L.clear();
                lambda.clear(); L0.clear();
                tape.clearAll();

                AD delta = market.delta;
                lambda.assign(market.lambda.begin(), market.lambda.end());
                L0.assign(market.L0.begin(), market.L0.end());

                tape.registerInput(delta);
                tape.registerInputs(lambda);
                tape.registerInputs(L0);
                tape.newRecording();

                generateSamples(gen, samples);
                L.assign(L0.begin(), L0.end());
                libor_path_gen(delta, L, lambda, samples);
                AD v = libor_value_portfolio(delta, portfolio.maturities,
                                             portfolio.swaprates, L, tmp1, tmp2);

                tape.registerOutput(v);
                derivative(v) = 1.0;
                tape.computeAdjoints();

                res.price += value(v);
                res.d_delta += derivative(delta);
                for (size_t i = 0; i < market.lambda.size(); ++i)
                {
                    res.d_lambda[i] += derivative(lambda[i]);
                    res.d_L0[i] += derivative(L0[i]);
                }
            }

            volatile double p = res.price / numPaths;
            (void)p;
        },
        warmup, iters);

    return {"XAD", "LiborSwaption", primal_ms, grad_ms};
}
