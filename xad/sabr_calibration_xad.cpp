/*******************************************************************************
   XAD SABR Surface Calibration benchmark - adjoint mode, tape-based.

   Simulates an optimizer loop: evaluate the SABR calibration objective and
   its gradient (15 inputs = 5 expiries x 3 params) for 500 iterations.

   Copyright (c) 2026 Xcelerit Computing Ltd.
   Licensed under the MIT License. See LICENSE file.
******************************************************************************/

#include "../src/benchmarks.hpp"
#include "../src/sabr.hpp"
#include <XAD/XAD.hpp>

BenchmarkResult xad_sabr_calibration(size_t warmup, size_t iters)
{
    using mode = xad::adj<double>;
    using tape_type = mode::tape_type;
    using AD = mode::active_type;

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

    // Gradient via adjoint
    double grad_ms = benchmark(
        [&]()
        {
            tape_type tape;
            double params_d[SABR_NUM_PARAMS];
            for (int e = 0; e < SABR_NUM_EXPIRIES; ++e)
            {
                params_d[3*e+0] = data.init_alpha[e];
                params_d[3*e+1] = data.init_rho[e];
                params_d[3*e+2] = data.init_nu[e];
            }

            for (int iter = 0; iter < SABR_CALIB_ITERS; ++iter)
            {
                tape.clearAll();

                AD params[SABR_NUM_PARAMS];
                for (int p = 0; p < SABR_NUM_PARAMS; ++p)
                    params[p] = params_d[p];

                for (int p = 0; p < SABR_NUM_PARAMS; ++p)
                    tape.registerInput(params[p]);
                tape.newRecording();

                AD obj = sabr_calibration_objective(params, data);

                tape.registerOutput(obj);
                derivative(obj) = 1.0;
                tape.computeAdjoints();

                for (int p = 0; p < SABR_NUM_PARAMS; ++p)
                {
                    volatile double g = derivative(params[p]);
                    (void)g;
                }

                for (int p = 0; p < SABR_NUM_PARAMS; ++p)
                    params_d[p] += schedule[iter].dp[p];
            }
        },
        warmup, iters);

    return {"XAD", "SABRCalib", primal_ms, grad_ms};
}
