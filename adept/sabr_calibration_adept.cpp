/*******************************************************************************
   Adept 2 SABR Surface Calibration benchmark - reverse mode.

   500 optimizer iterations, 15 inputs, re-tape each iteration.

   Copyright (c) 2026 Xcelerit Computing Ltd.
   Licensed under the MIT License. See LICENSE file.
******************************************************************************/

#include "../src/benchmarks.hpp"
#include "../src/sabr.hpp"
#include <adept.h>
#include <adept_arrays.h>

BenchmarkResult adept_sabr_calibration(size_t warmup, size_t iters)
{
    using adept::adouble;

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

    // Gradient via reverse mode
    double grad_ms = benchmark(
        [&]()
        {
            adept::Stack stack;
            double params_d[SABR_NUM_PARAMS];
            for (int e = 0; e < SABR_NUM_EXPIRIES; ++e)
            {
                params_d[3*e+0] = data.init_alpha[e];
                params_d[3*e+1] = data.init_rho[e];
                params_d[3*e+2] = data.init_nu[e];
            }

            for (int iter = 0; iter < SABR_CALIB_ITERS; ++iter)
            {
                adouble params[SABR_NUM_PARAMS];
                for (int p = 0; p < SABR_NUM_PARAMS; ++p)
                    params[p] = params_d[p];

                stack.new_recording();
                adouble obj = sabr_calibration_objective(params, data);
                obj.set_gradient(1.0);
                stack.reverse();

                for (int p = 0; p < SABR_NUM_PARAMS; ++p)
                {
                    volatile double g = params[p].get_gradient();
                    (void)g;
                }

                for (int p = 0; p < SABR_NUM_PARAMS; ++p)
                    params_d[p] += schedule[iter].dp[p];
            }
        },
        warmup, iters);

    return {"Adept", "SABRCalib", primal_ms, grad_ms};
}
