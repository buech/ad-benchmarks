/*******************************************************************************
   Adept 2 XVA benchmark - reverse mode. Re-tapes per path.

   Copyright (c) 2026 Xcelerit Computing Ltd.
   Licensed under the MIT License. See LICENSE file.
******************************************************************************/

#include "../src/benchmarks.hpp"
#include "../src/xva.hpp"
#include <adept.h>
#include <adept_arrays.h>

BenchmarkResult adept_xva(size_t warmup, size_t iters)
{
    using adept::adouble;

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
            adept::Stack stack;
            double total_grads[XVA_NUM_MARKET_INPUTS] = {};

            for (int path = 0; path < XVA_NUM_PATHS; ++path)
            {
                adouble rates[XVA_NUM_RATES], hazard[XVA_NUM_HAZARD], vols[XVA_NUM_VOLS];
                for (int i = 0; i < XVA_NUM_RATES; ++i)
                    rates[i] = market.rates[i];
                for (int i = 0; i < XVA_NUM_HAZARD; ++i)
                    hazard[i] = market.hazard[i];
                for (int i = 0; i < XVA_NUM_VOLS; ++i)
                    vols[i] = market.vols[i];

                stack.new_recording();
                adouble cva = xva_compute_cva(rates, hazard, vols,
                                              samples[path].data(), portfolio);
                cva.set_gradient(1.0);
                stack.reverse();

                for (int i = 0; i < XVA_NUM_RATES; ++i)
                    total_grads[i] += rates[i].get_gradient();
                for (int i = 0; i < XVA_NUM_HAZARD; ++i)
                    total_grads[XVA_NUM_RATES + i] += hazard[i].get_gradient();
                for (int i = 0; i < XVA_NUM_VOLS; ++i)
                    total_grads[XVA_NUM_RATES + XVA_NUM_HAZARD + i] += vols[i].get_gradient();
            }

            for (int i = 0; i < XVA_NUM_MARKET_INPUTS; ++i)
            {
                volatile double g = total_grads[i] / XVA_NUM_PATHS;
                (void)g;
            }
        },
        warmup, iters);

    return {"Adept", "XVA-CVA", primal_ms, grad_ms};
}
