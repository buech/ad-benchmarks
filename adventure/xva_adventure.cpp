/*******************************************************************************
   adventure XVA benchmark - adjoint mode, tape-based.

   40 market inputs, 2000 MC paths. Re-records tape per path.

   Copyright (c) 2026 Xcelerit Computing Ltd.
   Copyright (c) 2026 Adam Buechner
   Licensed under the MIT License. See LICENSE file.
******************************************************************************/

#include "../src/benchmarks.hpp"
#include "../src/xva.hpp"
#include "adventure/variable.hpp"

BenchmarkResult adventure_xva(size_t warmup, size_t iters)
{
    namespace ad = adventure;
    using AD = ad::Variable<double>;

    XVAMarketData market;
    auto portfolio = xva_default_portfolio();
    auto samples = xva_generate_samples(XVA_NUM_PATHS, 99999);

    double primal_ms = benchmark(
        [&]()
        {
            double total = 0.0;
            for (int path = 0; path < XVA_NUM_PATHS; ++path)
            {
                total += xva_cva_double(market, samples[path].data(), portfolio);
            }
            volatile double v = total / XVA_NUM_PATHS;
            (void)v;
        },
        warmup, iters);

    double grad_ms = benchmark(
        [&]()
        {
            auto &tape = ad::get_tape<double>();
            //ad::Tape<double> tape;

            double total_grads[XVA_NUM_MARKET_INPUTS] = {};

            for (int path = 0; path < XVA_NUM_PATHS; ++path)
            {
                tape.clear();

                AD rates[XVA_NUM_RATES], hazard[XVA_NUM_HAZARD], vols[XVA_NUM_VOLS];
                for (int i = 0; i < XVA_NUM_RATES; ++i)
                    rates[i] = market.rates[i];
                for (int i = 0; i < XVA_NUM_HAZARD; ++i)
                    hazard[i] = market.hazard[i];
                for (int i = 0; i < XVA_NUM_VOLS; ++i)
                    vols[i] = market.vols[i];

                for (int i = 0; i < XVA_NUM_RATES; ++i)
                    tape.register_input(rates[i]);
                for (int i = 0; i < XVA_NUM_HAZARD; ++i)
                    tape.register_input(hazard[i]);
                for (int i = 0; i < XVA_NUM_VOLS; ++i)
                    tape.register_input(vols[i]);

                AD cva = xva_compute_cva(rates, hazard, vols,
                                         samples[path].data(), portfolio);

                tape.register_output(cva);
                cva.grad() = 1.0;
                tape.backward();

                for (int i = 0; i < XVA_NUM_RATES; ++i)
                    total_grads[i] += rates[i].grad();
                for (int i = 0; i < XVA_NUM_HAZARD; ++i)
                    total_grads[XVA_NUM_RATES + i] += hazard[i].grad();
                for (int i = 0; i < XVA_NUM_VOLS; ++i)
                    total_grads[XVA_NUM_RATES + XVA_NUM_HAZARD + i] += vols[i].grad();
            }

            for (int i = 0; i < XVA_NUM_MARKET_INPUTS; ++i)
            {
                volatile double g = total_grads[i] / XVA_NUM_PATHS;
                (void)g;
            }
        },
        warmup, iters);

    return {"adventure", "XVA-CVA", primal_ms, grad_ms};
}
