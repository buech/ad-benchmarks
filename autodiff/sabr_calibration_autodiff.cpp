/*******************************************************************************
   autodiff SABR Surface Calibration benchmark - forward mode (15 passes).

   With 15 inputs, forward mode requires 15 passes per evaluation vs
   1 reverse pass for adjoint libraries.

   Copyright (c) 2026 Xcelerit Computing Ltd.
   Licensed under the MIT License. See LICENSE file.
******************************************************************************/

#include "../src/benchmarks.hpp"
#include "../src/sabr.hpp"
#include <autodiff/forward/dual.hpp>

using namespace autodiff;

BenchmarkResult autodiff_sabr_calibration(size_t warmup, size_t iters)
{
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

    // Gradient via forward mode — 15 passes per iteration
    double grad_ms = benchmark(
        [&]()
        {
            double params_d[SABR_NUM_PARAMS];
            for (int e = 0; e < SABR_NUM_EXPIRIES; ++e)
            {
                params_d[3*e+0] = data.init_alpha[e];
                params_d[3*e+1] = data.init_rho[e];
                params_d[3*e+2] = data.init_nu[e];
            }

            for (int iter = 0; iter < SABR_CALIB_ITERS; ++iter)
            {
                // 15 forward passes, one per input
                for (int seed_idx = 0; seed_idx < SABR_NUM_PARAMS; ++seed_idx)
                {
                    dual params[SABR_NUM_PARAMS];
                    for (int q = 0; q < SABR_NUM_PARAMS; ++q)
                        params[q] = params_d[q];

                    // Seed derivative of seed_idx-th input
                    params[seed_idx].grad = 1.0;

                    dual obj = sabr_calibration_objective(params, data);
                    volatile double g = obj.grad;
                    (void)g;
                }

                for (int p = 0; p < SABR_NUM_PARAMS; ++p)
                    params_d[p] += schedule[iter].dp[p];
            }
        },
        warmup, iters);

    return {"autodiff", "SABRCalib", primal_ms, grad_ms};
}
