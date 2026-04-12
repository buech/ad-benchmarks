/*******************************************************************************
   XAD Heston MC benchmark - JIT codegen with AVX2 SIMD.

   Records one Heston path, compiles to AVX2 native code, processes 4
   paths per kernel call.

   Copyright (c) 2026 Xcelerit Computing Ltd.
   Licensed under the MIT License. See LICENSE file.
******************************************************************************/

#include "../src/benchmarks.hpp"
#include "../src/heston.hpp"

#include <XAD/XAD.hpp>
#include <xad/CodegenBackendAVX.hpp>

/// JIT-compatible Heston path using ABool::If for branching.
inline xad::AD heston_path_jit(
    const xad::AD& S0, const xad::AD& K, const xad::AD& expiry, const xad::AD& r,
    const xad::AD& v0, const xad::AD& kappa, const xad::AD& theta, const xad::AD& xi,
    double rho_corr,
    const std::vector<xad::AD>& z1, const std::vector<xad::AD>& z2)
{
    using std::exp;
    using std::sqrt;

    xad::AD dt = expiry / xad::AD(HESTON_TIME_STEPS);
    xad::AD sqrt_dt = sqrt(dt);
    xad::AD S = S0;
    xad::AD v = v0;
    double rho_orth = std::sqrt(1.0 - rho_corr * rho_corr);

    for (int i = 0; i < HESTON_TIME_STEPS; ++i)
    {
        xad::AD dW1 = z1[i];
        xad::AD dW2 = xad::AD(rho_corr) * z1[i] + xad::AD(rho_orth) * z2[i];

        xad::AD v_pos = xad::less(v, xad::AD(0.0)).If(-v, v);
        xad::AD sqrt_v = sqrt(v_pos + xad::AD(1e-10));

        S = S + r * S * dt + sqrt_v * S * sqrt_dt * dW1;
        v = v + kappa * (theta - v) * dt + xi * sqrt_v * sqrt_dt * dW2;
    }

    xad::AD intrinsic = S - K;
    xad::AD payoff = xad::less(intrinsic, xad::AD(0.0)).If(xad::AD(0.0), intrinsic);
    return exp(-r * expiry) * payoff;
}

BenchmarkResult xad_jit_heston(int numPaths, size_t warmup, size_t iters)
{
    using JitAD = xad::AD;
    constexpr int BATCH = xad::codegen::CodegenBackendAVX<double>::VECTOR_WIDTH;

    HestonInputs in;
    const unsigned long long SEED = 77777;
    auto samples = heston_generate_samples(numPaths, SEED);

    double primal_ms = benchmark(
        [&]() {
            double sum = 0.0;
            for (int p = 0; p < numPaths; ++p)
                sum += heston_path(in.S0, in.K, in.T, in.r, in.v0, in.kappa,
                                   in.theta, in.xi, in.rho_corr,
                                   samples.z1[p], samples.z2[p]);
            volatile double v = sum / numPaths;
            (void)v;
        },
        warmup, iters);

    double grad_ms = benchmark(
        [&]()
        {
            // --- Record and compile graph (once) ---
            xad::JITCompiler<double, 1> jit;

            JitAD S0 = in.S0, K = in.K, T = in.T, r = in.r;
            JitAD v0 = in.v0, kappa = in.kappa, theta = in.theta, xi = in.xi;

            jit.registerInput(S0); jit.registerInput(K);
            jit.registerInput(T);  jit.registerInput(r);
            jit.registerInput(v0); jit.registerInput(kappa);
            jit.registerInput(theta); jit.registerInput(xi);

            std::vector<JitAD> jit_z1(HESTON_TIME_STEPS), jit_z2(HESTON_TIME_STEPS);
            for (int i = 0; i < HESTON_TIME_STEPS; ++i)
            {
                jit_z1[i] = JitAD(samples.z1[0][i]);
                jit.registerInput(jit_z1[i]);
            }
            for (int i = 0; i < HESTON_TIME_STEPS; ++i)
            {
                jit_z2[i] = JitAD(samples.z2[0][i]);
                jit.registerInput(jit_z2[i]);
            }

            jit.newRecording();

            JitAD price = heston_path_jit(S0, K, T, r, v0, kappa, theta, xi,
                                          in.rho_corr, jit_z1, jit_z2);
            jit.registerOutput(price);

            xad::codegen::CodegenBackendAVX<double> backend;
            backend.compile(jit.getGraph());

            // --- Execute AVX kernel ---
            const size_t numInputs = HESTON_INPUTS + 2 * HESTON_TIME_STEPS;
            const int numBatches = (numPaths + BATCH - 1) / BATCH;

            std::vector<double> outputBatch(BATCH);
            std::vector<double> inputGradients(numInputs * BATCH);
            std::vector<double> inputBatch(BATCH);
            double total_grad[HESTON_INPUTS] = {};
            double total_price = 0.0;

            for (int b = 0; b < numBatches; ++b)
            {
                int batchStart = b * BATCH;
                int actualBatch = std::min(BATCH, numPaths - batchStart);

                // Market inputs (same for all lanes)
                double vals[] = {in.S0, in.K, in.T, in.r, in.v0, in.kappa, in.theta, in.xi};
                for (int j = 0; j < HESTON_INPUTS; ++j)
                {
                    for (int l = 0; l < BATCH; ++l)
                        inputBatch[l] = vals[j];
                    backend.setInput(j, inputBatch.data());
                }

                // Random z1 (different per lane)
                for (int i = 0; i < HESTON_TIME_STEPS; ++i)
                {
                    for (int l = 0; l < BATCH; ++l)
                    {
                        int p = batchStart + l;
                        inputBatch[l] = (p < numPaths) ? samples.z1[p][i] : 0.0;
                    }
                    backend.setInput(HESTON_INPUTS + i, inputBatch.data());
                }
                // Random z2
                for (int i = 0; i < HESTON_TIME_STEPS; ++i)
                {
                    for (int l = 0; l < BATCH; ++l)
                    {
                        int p = batchStart + l;
                        inputBatch[l] = (p < numPaths) ? samples.z2[p][i] : 0.0;
                    }
                    backend.setInput(HESTON_INPUTS + HESTON_TIME_STEPS + i, inputBatch.data());
                }

                backend.forwardAndBackward(outputBatch.data(), inputGradients.data());

                for (int l = 0; l < actualBatch; ++l)
                {
                    total_price += outputBatch[l];
                    for (int j = 0; j < HESTON_INPUTS; ++j)
                        total_grad[j] += inputGradients[j * BATCH + l];
                }
            }

            volatile double v = total_price / numPaths;
            (void)v;
        },
        warmup, iters);

    return {"XAD-Codegen", "HestonMC", primal_ms, grad_ms};
}
