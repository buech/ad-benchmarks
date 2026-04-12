/*******************************************************************************
   CppAD SABR Surface Calibration benchmark - reverse mode.

   500 optimizer iterations, 15 inputs, re-tape each iteration.

   Copyright (c) 2026 Xcelerit Computing Ltd.
   Licensed under the MIT License. See LICENSE file.
******************************************************************************/

#include "../src/benchmarks.hpp"
#include "../src/sabr.hpp"
#include <cppad/cppad.hpp>

BenchmarkResult cppad_sabr_calibration(size_t warmup, size_t iters)
{
    using ADdouble = CppAD::AD<double>;

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
            double params_d[SABR_NUM_PARAMS];
            for (int e = 0; e < SABR_NUM_EXPIRIES; ++e)
            {
                params_d[3*e+0] = data.init_alpha[e];
                params_d[3*e+1] = data.init_rho[e];
                params_d[3*e+2] = data.init_nu[e];
            }

            for (int iter = 0; iter < SABR_CALIB_ITERS; ++iter)
            {
                std::vector<ADdouble> ax(params_d, params_d + SABR_NUM_PARAMS);
                CppAD::Independent(ax);

                std::vector<ADdouble> ay(1);
                ay[0] = sabr_calibration_objective(ax.data(), data);

                CppAD::ADFun<double> f(ax, ay);

                std::vector<double> w(1, 1.0);
                std::vector<double> grad = f.Reverse(1, w);

                for (int p = 0; p < SABR_NUM_PARAMS; ++p)
                {
                    volatile double g = grad[p];
                    (void)g;
                }

                for (int p = 0; p < SABR_NUM_PARAMS; ++p)
                    params_d[p] += schedule[iter].dp[p];
            }
        },
        warmup, iters);

    return {"CppAD", "SABRCalib", primal_ms, grad_ms};
}
