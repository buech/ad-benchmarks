/*******************************************************************************
   XAD LIBOR Swaption benchmark - JIT codegen with AVX2 SIMD.

   Ported from XAD samples/LiborSwaptionPricer/JIT.
   Records graph once, compiles to AVX2 native code, processes 4 paths
   per kernel call.

   Copyright (c) 2026 Xcelerit Computing Ltd.
   Licensed under the MIT License. See LICENSE file.
******************************************************************************/

#include "../src/benchmarks.hpp"
#include "../src/libor_swaption.hpp"

#include <XAD/XAD.hpp>
#include <xad/CodegenBackendAVX.hpp>

/// JIT-compatible path_gen: random samples are AD types (JIT inputs).
template <class ADT>
void libor_path_gen_jit(const ADT& delta, std::vector<ADT>& L,
                        const std::vector<ADT>& lambda, const std::vector<ADT>& z)
{
    using std::exp;
    using std::sqrt;

    for (size_t n = 0; n < z.size(); n++)
    {
        ADT sqez = sqrt(delta) * z[n];
        ADT v = ADT(0.0);
        for (size_t i = n + 1; i < L.size(); i++)
        {
            ADT lam = lambda[i - n - 1];
            ADT con1 = delta * lam;
            v += (con1 * L[i]) / (ADT(1.0) + delta * L[i]);
            L[i] *= exp(con1 * v + lam * (sqez - ADT(0.5) * con1));
        }
    }
}

/// JIT-compatible value_portfolio: uses ABool::If for branching.
inline xad::AD libor_value_portfolio_jit(
    const xad::AD& delta, const std::vector<int>& maturities,
    const std::vector<double>& swaprates, const std::vector<xad::AD>& L,
    std::vector<xad::AD>& Btmp, std::vector<xad::AD>& Stmp)
{
    const size_t NN = L.size();
    const size_t N = NN / 2;
    const size_t Nopt = swaprates.size();

    Btmp.resize(NN);
    Stmp.resize(NN);

    xad::AD b = 1.0;
    xad::AD s = 0.0;

    for (size_t n = N; n < NN; ++n)
    {
        b = b / (1.0 + delta * L[n]);
        s = s + delta * b;
        Btmp[n] = b;
        Stmp[n] = s;
    }

    xad::AD v = 0.0;
    for (size_t i = 0; i < Nopt; i++)
    {
        int m = maturities[i] + static_cast<int>(N) - 1;
        xad::AD swapval = Btmp[m] + swaprates[i] * Stmp[m] - 1.0;
        v += xad::less(swapval, xad::AD(0.0)).If(-100.0 * swapval, xad::AD(0.0));
    }

    for (size_t n = 0; n < N; n++)
        v = v / (1.0 + delta * L[n]);

    return v;
}

BenchmarkResult xad_jit_libor_swaption(int numPaths, size_t warmup, size_t iters)
{
    using JitAD = xad::AD;
    constexpr int BATCH = xad::codegen::CodegenBackendAVX<double>::VECTOR_WIDTH;

    auto portfolio = defaultPortfolio();
    auto market = defaultMarket();
    const unsigned long long SEED = 12354;
    const size_t numSamples = market.lambda.size() / 2;

    auto allSamples = libor_generate_all_samples(numPaths, numSamples, SEED);

    double primal_ms = benchmark(
        [&]() {
            volatile double v = libor_price_mc(market, portfolio, numPaths, SEED);
            (void)v;
        },
        warmup, iters);

    double grad_ms = benchmark(
        [&]()
        {
            // --- Record and compile graph (once) ---
            std::vector<JitAD> L, tmp1, tmp2, lambda, L0;
            std::vector<JitAD> jit_samples(numSamples);
            xad::JITCompiler<double, 1> jit;
            JitAD delta;

            delta = market.delta;
            lambda.assign(market.lambda.begin(), market.lambda.end());
            L0.assign(market.L0.begin(), market.L0.end());

            jit.registerInput(delta);
            jit.registerInputs(lambda);
            jit.registerInputs(L0);

            for (size_t i = 0; i < numSamples; ++i)
            {
                jit_samples[i] = JitAD(allSamples[0][i]);
                jit.registerInput(jit_samples[i]);
            }

            jit.newRecording();

            L.assign(L0.begin(), L0.end());
            libor_path_gen_jit(delta, L, lambda, jit_samples);
            JitAD v = libor_value_portfolio_jit(delta, portfolio.maturities,
                                                 portfolio.swaprates, L, tmp1, tmp2);
            jit.registerOutput(v);

            xad::codegen::CodegenBackendAVX<double> backend;
            backend.compile(jit.getGraph());

            // --- Execute AVX kernel: 4 paths per call ---
            const int numBatches = (numPaths + BATCH - 1) / BATCH;
            const size_t numInputs = 1 + market.lambda.size() + market.L0.size() + numSamples;

            std::vector<double> outputBatch(BATCH);
            std::vector<double> inputGradients(numInputs * BATCH);
            std::vector<double> inputBatch(BATCH);

            SensitivityResults res;
            res.d_lambda.resize(market.lambda.size(), 0.0);
            res.d_L0.resize(market.L0.size(), 0.0);

            for (int b = 0; b < numBatches; ++b)
            {
                int batchStart = b * BATCH;
                int actualBatch = std::min(BATCH, numPaths - batchStart);

                // Set delta (same for all lanes)
                for (int l = 0; l < BATCH; ++l)
                    inputBatch[l] = market.delta;
                backend.setInput(0, inputBatch.data());

                // Set lambda (same for all lanes)
                for (size_t k = 0; k < market.lambda.size(); ++k)
                {
                    for (int l = 0; l < BATCH; ++l)
                        inputBatch[l] = market.lambda[k];
                    backend.setInput(1 + k, inputBatch.data());
                }

                // Set L0 (same for all lanes)
                for (size_t k = 0; k < market.L0.size(); ++k)
                {
                    for (int l = 0; l < BATCH; ++l)
                        inputBatch[l] = market.L0[k];
                    backend.setInput(1 + market.lambda.size() + k, inputBatch.data());
                }

                // Set random samples (different per lane)
                size_t sampleOffset = 1 + market.lambda.size() + market.L0.size();
                for (size_t m = 0; m < numSamples; ++m)
                {
                    for (int l = 0; l < BATCH; ++l)
                    {
                        int pathIdx = batchStart + l;
                        inputBatch[l] = (pathIdx < numPaths) ? allSamples[pathIdx][m] : 0.0;
                    }
                    backend.setInput(sampleOffset + m, inputBatch.data());
                }

                backend.forwardAndBackward(outputBatch.data(), inputGradients.data());

                // Accumulate
                for (int l = 0; l < actualBatch; ++l)
                {
                    res.price += outputBatch[l];
                    res.d_delta += inputGradients[0 * BATCH + l];
                    for (size_t k = 0; k < market.lambda.size(); ++k)
                        res.d_lambda[k] += inputGradients[(1 + k) * BATCH + l];
                    for (size_t k = 0; k < market.L0.size(); ++k)
                        res.d_L0[k] += inputGradients[(1 + market.lambda.size() + k) * BATCH + l];
                }
            }

            volatile double p = res.price / numPaths;
            (void)p;
        },
        warmup, iters);

    return {"XAD-Codegen", "LiborSwaption", primal_ms, grad_ms};
}
