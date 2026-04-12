/*******************************************************************************
   XAD SABR Surface Calibration benchmark - JIT codegen with AVX2 SIMD.

   Records the multi-expiry SABR calibration objective graph once
   (15 inputs), compiles to AVX2 native code, processes 4 evaluations
   per kernel call across optimizer iterations.

   Copyright (c) 2026 Xcelerit Computing Ltd.
   Licensed under the MIT License. See LICENSE file.
******************************************************************************/

#include "../src/benchmarks.hpp"
#include "../src/sabr.hpp"

#include <XAD/XAD.hpp>
#include <xad/CodegenBackendAVX.hpp>

BenchmarkResult xad_jit_sabr_calibration(size_t warmup, size_t iters)
{
    using JitAD = xad::AD;
    constexpr int BATCH = xad::codegen::CodegenBackendAVX<double>::VECTOR_WIDTH;

    SABRCalibData data;
    auto schedule = sabr_perturbation_schedule(SABR_CALIB_ITERS);

    // Primal
    double primal_ms = benchmark(
        [&]()
        {
            double params[SABR_NUM_PARAMS];
            for (int e = 0; e < SABR_NUM_EXPIRIES; ++e)
            {
                params[3*e+0] = data.init_alpha[e];
                params[3*e+1] = data.init_rho[e];
                params[3*e+2] = data.init_nu[e];
            }
            for (int iter = 0; iter < SABR_CALIB_ITERS; ++iter)
            {
                volatile double obj = sabr_objective_double(params, data);
                (void)obj;
                for (int p = 0; p < SABR_NUM_PARAMS; ++p)
                    params[p] += schedule[iter].dp[p];
            }
        },
        warmup, iters);

    // Gradient via JIT AVX2
    double grad_ms = benchmark(
        [&]()
        {
            // --- Record and compile graph (once) ---
            xad::JITCompiler<double, 1> jit;

            JitAD params[SABR_NUM_PARAMS];
            for (int e = 0; e < SABR_NUM_EXPIRIES; ++e)
            {
                params[3*e+0] = data.init_alpha[e];
                params[3*e+1] = data.init_rho[e];
                params[3*e+2] = data.init_nu[e];
            }
            for (int p = 0; p < SABR_NUM_PARAMS; ++p)
                jit.registerInput(params[p]);

            jit.newRecording();

            JitAD obj = sabr_calibration_objective(params, data);
            jit.registerOutput(obj);

            xad::codegen::CodegenBackendAVX<double> backend;
            backend.compile(jit.getGraph());

            // --- Execute AVX kernel: 4 evaluations per call ---
            double params_d[SABR_NUM_PARAMS];
            for (int e = 0; e < SABR_NUM_EXPIRIES; ++e)
            {
                params_d[3*e+0] = data.init_alpha[e];
                params_d[3*e+1] = data.init_rho[e];
                params_d[3*e+2] = data.init_nu[e];
            }

            std::vector<double> outputBatch(BATCH);
            std::vector<double> inputGradients(SABR_NUM_PARAMS * BATCH);
            std::vector<double> inputBatch(BATCH);

            // Process iterations in batches of 4
            const int numBatches = (SABR_CALIB_ITERS + BATCH - 1) / BATCH;
            for (int b = 0; b < numBatches; ++b)
            {
                int batchStart = b * BATCH;
                int actualBatch = std::min(BATCH, SABR_CALIB_ITERS - batchStart);

                // Compute params for each lane (they differ slightly due to perturbations)
                std::vector<std::vector<double>> lane_params(BATCH, std::vector<double>(SABR_NUM_PARAMS));
                for (int l = 0; l < BATCH; ++l)
                {
                    int iter = batchStart + l;
                    for (int e = 0; e < SABR_NUM_EXPIRIES; ++e)
                    {
                        lane_params[l][3*e+0] = data.init_alpha[e];
                        lane_params[l][3*e+1] = data.init_rho[e];
                        lane_params[l][3*e+2] = data.init_nu[e];
                    }
                    for (int s = 0; s < iter && s < SABR_CALIB_ITERS; ++s)
                        for (int p = 0; p < SABR_NUM_PARAMS; ++p)
                            lane_params[l][p] += schedule[s].dp[p];
                }

                // Set inputs: each input gets 4 lane values
                for (int p = 0; p < SABR_NUM_PARAMS; ++p)
                {
                    for (int l = 0; l < BATCH; ++l)
                        inputBatch[l] = lane_params[l][p];
                    backend.setInput(p, inputBatch.data());
                }

                backend.forwardAndBackward(outputBatch.data(), inputGradients.data());

                for (int l = 0; l < actualBatch; ++l)
                {
                    volatile double g = inputGradients[0 * BATCH + l];
                    (void)g;
                }
            }
        },
        warmup, iters);

    return {"XAD-Codegen", "SABRCalib", primal_ms, grad_ms};
}
