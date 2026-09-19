/*******************************************************************************
   adventure SABR Surface Calibration benchmark - adjoint mode, tape-based.

   Simulates an optimizer loop: evaluate the SABR calibration objective and
   its gradient (15 inputs = 5 expiries x 3 params) for 500 iterations.

   Copyright (c) 2026 Xcelerit Computing Ltd.
   Copyright (c) 2026 Adam Buechner
   Licensed under the MIT License. See LICENSE file.
******************************************************************************/

#include "../src/benchmarks.hpp"
#include "../src/sabr.hpp"
#include "adventure/variable.hpp"

BenchmarkResult adventure_sabr_calibration(size_t warmup, size_t iters)
{
    namespace ad = adventure;
    using AD = ad::Variable<double>;

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
            auto &tape = ad::get_tape<double>();
            //ad::Tape<double> tape;

            double params_d[SABR_NUM_PARAMS];
            for (int e = 0; e < SABR_NUM_EXPIRIES; ++e)
            {
                params_d[3*e+0] = data.init_alpha[e];
                params_d[3*e+1] = data.init_rho[e];
                params_d[3*e+2] = data.init_nu[e];
            }

            for (int iter = 0; iter < SABR_CALIB_ITERS; ++iter)
            {
                tape.clear();

                AD params[SABR_NUM_PARAMS];
                for (int p = 0; p < SABR_NUM_PARAMS; ++p)
                    params[p] = params_d[p];

                for (int p = 0; p < SABR_NUM_PARAMS; ++p)
                    tape.register_input(params[p]);

                AD obj = sabr_calibration_objective(params, data);

                tape.register_output(obj);
                obj.grad() = 1.0;
                tape.backward();

                for (int p = 0; p < SABR_NUM_PARAMS; ++p)
                {
                    volatile double g = params[p].grad();
                    (void)g;
                }

                for (int p = 0; p < SABR_NUM_PARAMS; ++p)
                    params_d[p] += schedule[iter].dp[p];
            }
        },
        warmup, iters);

    return {"adventure", "SABRCalib", primal_ms, grad_ms};
}
