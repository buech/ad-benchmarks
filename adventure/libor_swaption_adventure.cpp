/*******************************************************************************
   adventure LIBOR Swaption MC benchmark - adjoint mode, tape-based.

   161 market inputs, pathwise adjoint. Re-records tape per path.

   Copyright (c) 2026 Xcelerit Computing Ltd.
   Copyright (c) 2026 Adam Buechner
   Licensed under the MIT License. See LICENSE file.
******************************************************************************/

#include "../src/benchmarks.hpp"
#include "../src/libor_swaption.hpp"
#include "adventure/variable.hpp"

BenchmarkResult adventure_libor_swaption(int numPaths, size_t warmup, size_t iters)
{
    namespace ad = adventure;
    using AD = ad::Variable<double>;

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
            auto &tape = ad::get_tape<double>();
            //ad::Tape<double> tape;

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

                tape.clear();

                AD delta = market.delta;
                lambda.assign(market.lambda.begin(), market.lambda.end());
                L0.assign(market.L0.begin(), market.L0.end());

                tape.register_input(delta);
                for (std::size_t i = 0; i < lambda.size(); ++i)
                    tape.register_input(lambda[i]);
                for (std::size_t i = 0; i < L0.size(); ++i)
                    tape.register_input(L0[i]);

                generateSamples(gen, samples);
                L.assign(L0.begin(), L0.end());
                libor_path_gen(delta, L, lambda, samples);
                AD v = libor_value_portfolio(delta, portfolio.maturities,
                                             portfolio.swaprates, L, tmp1, tmp2);

                tape.register_output(v);
                v.grad() = 1.0;
                tape.backward();

                res.price += v.value();
                res.d_delta += delta.grad();
                for (size_t i = 0; i < market.lambda.size(); ++i)
                {
                    res.d_lambda[i] += lambda[i].grad();
                    res.d_L0[i] += L0[i].grad();
                }
            }

            volatile double p = res.price / numPaths;
            (void)p;
        },
        warmup, iters);

    return {"adventure", "LiborSwaption", primal_ms, grad_ms};
}
