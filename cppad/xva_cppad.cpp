/*******************************************************************************
   CppAD XVA benchmark - reverse mode. Re-tapes per path.

   Copyright (c) 2026 Xcelerit Computing Ltd.
   Licensed under the MIT License. See LICENSE file.
******************************************************************************/

#include "../src/benchmarks.hpp"
#include "../src/xva.hpp"
#include <cppad/cppad.hpp>

BenchmarkResult cppad_xva(size_t warmup, size_t iters)
{
    using ADdouble = CppAD::AD<double>;

    XVAMarketData market;
    auto portfolio = xva_default_portfolio();
    auto samples = xva_generate_samples(XVA_NUM_PATHS, 99999);

    double primal_ms = benchmark(
        [&]()
        {
            double total = 0.0;
            for (int path = 0; path < XVA_NUM_PATHS; ++path)
                total += xva_cva_double(market, samples[path].data(), portfolio);
            volatile double v = total / XVA_NUM_PATHS;
            (void)v;
        },
        warmup, iters);

    double grad_ms = benchmark(
        [&]()
        {
            double total_grads[XVA_NUM_MARKET_INPUTS] = {};

            for (int path = 0; path < XVA_NUM_PATHS; ++path)
            {
                std::vector<ADdouble> ax(XVA_NUM_MARKET_INPUTS);
                for (int i = 0; i < XVA_NUM_RATES; ++i)
                    ax[i] = market.rates[i];
                for (int i = 0; i < XVA_NUM_HAZARD; ++i)
                    ax[XVA_NUM_RATES + i] = market.hazard[i];
                for (int i = 0; i < XVA_NUM_VOLS; ++i)
                    ax[XVA_NUM_RATES + XVA_NUM_HAZARD + i] = market.vols[i];

                CppAD::Independent(ax);

                std::vector<ADdouble> ay(1);
                ay[0] = xva_compute_cva(ax.data(), ax.data() + XVA_NUM_RATES,
                                        ax.data() + XVA_NUM_RATES + XVA_NUM_HAZARD,
                                        samples[path].data(), portfolio);

                CppAD::ADFun<double> f(ax, ay);
                std::vector<double> w(1, 1.0);
                std::vector<double> grad = f.Reverse(1, w);

                for (int i = 0; i < XVA_NUM_MARKET_INPUTS; ++i)
                    total_grads[i] += grad[i];
            }

            for (int i = 0; i < XVA_NUM_MARKET_INPUTS; ++i)
            {
                volatile double g = total_grads[i] / XVA_NUM_PATHS;
                (void)g;
            }
        },
        warmup, iters);

    return {"CppAD", "XVA-CVA", primal_ms, grad_ms};
}
