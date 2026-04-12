/*******************************************************************************
   XAD XVA benchmark - JIT codegen with AVX2 SIMD (4-wide).

   Records CVA computation graph once (80 inputs), compiles to AVX2
   native code, processes 4 MC paths simultaneously per kernel call.

   Copyright (c) 2026 Xcelerit Computing Ltd.
   Licensed under the MIT License. See LICENSE file.
******************************************************************************/

#include "../src/benchmarks.hpp"
#include "../src/xva.hpp"

#include <XAD/XAD.hpp>
#include <xad/CodegenBackendAVX.hpp>

/// JIT-compatible CVA: uses ABool::If for exposure = max(PV, 0).
template <class T>
T xva_compute_cva_jit(const T* rates, const T* hazard, const T* vols,
                      const T* z_ad, const std::vector<SwapDef>& portfolio)
{
    T cva = T(0.0);
    double dt = 0.5;

    T diffused_rates[XVA_NUM_RATES];
    for (int i = 0; i < XVA_NUM_RATES; ++i)
        diffused_rates[i] = rates[i];

    for (int bucket = 0; bucket < XVA_NUM_TIME_BUCKETS; ++bucket)
    {
        double t = (bucket + 1) * dt;

        // Diffuse rates using AD random samples
        {
            using std::sqrt;
            T sqrt_dt = T(sqrt(dt));
            for (int i = 0; i < XVA_NUM_RATES; ++i)
            {
                T vol = vols[i % XVA_NUM_VOLS];
                T shock = vol * sqrt_dt * z_ad[i % XVA_NUM_RANDOMS];
                diffused_rates[i] = diffused_rates[i] + shock;
            }
        }

        // Price portfolio
        T portfolio_pv = T(0.0);
        for (int s = 0; s < XVA_NUM_SWAPS; ++s)
        {
            SwapDef remaining = portfolio[s];
            int payments_elapsed = static_cast<int>(t / remaining.payment_freq);
            remaining.num_payments = remaining.num_payments - payments_elapsed;
            remaining.start_time = t;
            if (remaining.num_payments > 0)
                portfolio_pv = portfolio_pv + xva_price_swap(diffused_rates, remaining);
        }

        // Exposure via ABool::If for JIT
        T exposure = xad::less(portfolio_pv, T(0.0)).If(T(0.0), portfolio_pv);

        T surv_prev = xva_survival_prob(hazard, t - dt);
        T surv_curr = xva_survival_prob(hazard, t);
        T default_prob = surv_prev - surv_curr;
        T df = xva_discount_factor(rates, t);

        cva = cva + exposure * default_prob * df;
    }

    return cva;
}

BenchmarkResult xad_jit_xva(size_t warmup, size_t iters)
{
    using JitAD = xad::AD;
    constexpr int BATCH = xad::codegen::CodegenBackendAVX<double>::VECTOR_WIDTH;

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
            // --- Record and compile graph (once) ---
            xad::JITCompiler<double, 1> jit;

            JitAD rates[XVA_NUM_RATES], hazard[XVA_NUM_HAZARD], vols[XVA_NUM_VOLS];
            JitAD z_ad[XVA_NUM_RANDOMS];

            for (int i = 0; i < XVA_NUM_RATES; ++i)
                rates[i] = market.rates[i];
            for (int i = 0; i < XVA_NUM_HAZARD; ++i)
                hazard[i] = market.hazard[i];
            for (int i = 0; i < XVA_NUM_VOLS; ++i)
                vols[i] = market.vols[i];
            for (int i = 0; i < XVA_NUM_RANDOMS; ++i)
                z_ad[i] = samples[0][i];

            for (int i = 0; i < XVA_NUM_RATES; ++i)
                jit.registerInput(rates[i]);
            for (int i = 0; i < XVA_NUM_HAZARD; ++i)
                jit.registerInput(hazard[i]);
            for (int i = 0; i < XVA_NUM_VOLS; ++i)
                jit.registerInput(vols[i]);
            for (int i = 0; i < XVA_NUM_RANDOMS; ++i)
                jit.registerInput(z_ad[i]);

            jit.newRecording();

            JitAD cva = xva_compute_cva_jit(rates, hazard, vols, z_ad, portfolio);
            jit.registerOutput(cva);

            xad::codegen::CodegenBackendAVX<double> backend;
            backend.compile(jit.getGraph());

            // --- Execute AVX kernel: 4 paths per call ---
            const int numBatches = (XVA_NUM_PATHS + BATCH - 1) / BATCH;
            std::vector<double> outputBatch(BATCH);
            std::vector<double> inputGradients(XVA_TOTAL_INPUTS * BATCH);
            std::vector<double> inputBatch(BATCH);

            double total_grads[XVA_NUM_MARKET_INPUTS] = {};
            double total_price = 0.0;

            for (int b = 0; b < numBatches; ++b)
            {
                int batchStart = b * BATCH;
                int actualBatch = std::min(BATCH, XVA_NUM_PATHS - batchStart);

                // Set market inputs (same for all lanes)
                for (int i = 0; i < XVA_NUM_RATES; ++i)
                {
                    for (int l = 0; l < BATCH; ++l)
                        inputBatch[l] = market.rates[i];
                    backend.setInput(i, inputBatch.data());
                }
                for (int i = 0; i < XVA_NUM_HAZARD; ++i)
                {
                    for (int l = 0; l < BATCH; ++l)
                        inputBatch[l] = market.hazard[i];
                    backend.setInput(XVA_NUM_RATES + i, inputBatch.data());
                }
                for (int i = 0; i < XVA_NUM_VOLS; ++i)
                {
                    for (int l = 0; l < BATCH; ++l)
                        inputBatch[l] = market.vols[i];
                    backend.setInput(XVA_NUM_RATES + XVA_NUM_HAZARD + i, inputBatch.data());
                }

                // Set random draws (different per lane)
                for (int i = 0; i < XVA_NUM_RANDOMS; ++i)
                {
                    for (int l = 0; l < BATCH; ++l)
                    {
                        int pathIdx = batchStart + l;
                        inputBatch[l] = (pathIdx < XVA_NUM_PATHS) ? samples[pathIdx][i] : 0.0;
                    }
                    backend.setInput(XVA_NUM_MARKET_INPUTS + i, inputBatch.data());
                }

                backend.forwardAndBackward(outputBatch.data(), inputGradients.data());

                // Accumulate: gradient layout is [input0_lane0..3, input1_lane0..3, ...]
                for (int l = 0; l < actualBatch; ++l)
                {
                    total_price += outputBatch[l];
                    for (int i = 0; i < XVA_NUM_MARKET_INPUTS; ++i)
                        total_grads[i] += inputGradients[i * BATCH + l];
                }
            }

            volatile double v = total_price / XVA_NUM_PATHS;
            (void)v;
        },
        warmup, iters);

    return {"XAD-Codegen", "XVA-CVA", primal_ms, grad_ms};
}
