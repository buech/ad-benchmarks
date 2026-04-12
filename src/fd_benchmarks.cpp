/*******************************************************************************
   Finite Difference (bump-and-revalue) benchmarks.

   These provide the baseline cost of computing sensitivities without AD.

   Copyright (c) 2026 Xcelerit Computing Ltd.
   Licensed under the MIT License. See LICENSE file.
******************************************************************************/

#include "benchmarks.hpp"
#include "heston.hpp"
#include "sabr.hpp"
#include "xva.hpp"
#include "libor_swaption.hpp"
#include <vector>

// ---------------------------------------------------------------------------
// Heston MC — 8 inputs, bump-and-revalue forward differences
// ---------------------------------------------------------------------------
BenchmarkResult fd_heston(int numPaths, size_t warmup, size_t iters)
{
    HestonInputs in;
    auto samples = heston_generate_samples(numPaths);
    const double eps = 1e-6;

    auto eval_primal = [&](const HestonInputs& cur) {
        double sum = 0.0;
        for (int p = 0; p < numPaths; ++p)
            sum += heston_path(cur.S0, cur.K, cur.T, cur.r, cur.v0, cur.kappa,
                               cur.theta, cur.xi, cur.rho_corr,
                               samples.z1[p], samples.z2[p]);
        return sum / numPaths;
    };

    double primal_ms = benchmark([&]() { volatile double v = eval_primal(in); (void)v; },
                                 warmup, iters);

    double grad_ms = benchmark(
        [&]() {
            double base = eval_primal(in);
            double grads[HESTON_INPUTS];
            double* fields[HESTON_INPUTS] = {
                &in.S0, &in.K, &in.T, &in.r, &in.v0, &in.kappa, &in.theta, &in.xi};
            for (int i = 0; i < HESTON_INPUTS; ++i)
            {
                double saved = *fields[i];
                *fields[i] = saved + eps;
                grads[i] = (eval_primal(in) - base) / eps;
                *fields[i] = saved;
            }
            volatile double g = grads[0];
            (void)g;
        },
        warmup, iters);

    return {"FD", "HestonMC", primal_ms, grad_ms};
}

// ---------------------------------------------------------------------------
// SABR Calibration — 15 inputs, bump-and-revalue forward differences
// ---------------------------------------------------------------------------
BenchmarkResult fd_sabr_calibration(size_t warmup, size_t iters)
{
    SABRCalibData data;
    data.init();
    double params[SABR_NUM_PARAMS];
    for (int e = 0; e < SABR_NUM_EXPIRIES; ++e)
    {
        params[3 * e + 0] = data.init_alpha[e];
        params[3 * e + 1] = data.init_rho[e];
        params[3 * e + 2] = data.init_nu[e];
    }
    auto schedule = sabr_perturbation_schedule(SABR_CALIB_ITERS);
    const double eps = 1e-6;

    // Primal — full 500-iteration optimizer trajectory (one objective per
    // iteration), matching the per-iteration cost the AAD libs measure.
    double primal_ms = benchmark(
        [&]() {
            double cur[SABR_NUM_PARAMS];
            for (int i = 0; i < SABR_NUM_PARAMS; ++i) cur[i] = params[i];
            double sink = 0.0;
            for (int it = 0; it < SABR_CALIB_ITERS; ++it)
            {
                sink += sabr_objective_double(cur, data);
                for (int i = 0; i < SABR_NUM_PARAMS; ++i)
                    cur[i] += schedule[it].dp[i];
            }
            volatile double s = sink;
            (void)s;
        },
        warmup, iters);

    double grad_ms = benchmark(
        [&]() {
            double cur[SABR_NUM_PARAMS];
            for (int i = 0; i < SABR_NUM_PARAMS; ++i) cur[i] = params[i];
            double sink = 0.0;
            for (int it = 0; it < SABR_CALIB_ITERS; ++it)
            {
                double base = sabr_objective_double(cur, data);
                double grads[SABR_NUM_PARAMS];
                for (int i = 0; i < SABR_NUM_PARAMS; ++i)
                {
                    double saved = cur[i];
                    cur[i] = saved + eps;
                    grads[i] = (sabr_objective_double(cur, data) - base) / eps;
                    cur[i] = saved;
                }
                for (int i = 0; i < SABR_NUM_PARAMS; ++i)
                    cur[i] += schedule[it].dp[i];
                sink += grads[0];
            }
            volatile double s = sink;
            (void)s;
        },
        warmup, iters);

    return {"FD", "SABRCalib", primal_ms, grad_ms};
}

// ---------------------------------------------------------------------------
// XVA CVA — 40 market inputs, bump-and-revalue forward differences
// ---------------------------------------------------------------------------
BenchmarkResult fd_xva(size_t warmup, size_t iters)
{
    XVAMarketData market;
    auto z = xva_generate_samples(XVA_NUM_PATHS, 12345);
    auto portfolio = xva_default_portfolio();
    const double eps = 1e-6;

    auto eval_primal = [&]() {
        double sum = 0.0;
        for (int p = 0; p < XVA_NUM_PATHS; ++p)
            sum += xva_cva_double(market, z[p].data(), portfolio);
        return sum / XVA_NUM_PATHS;
    };

    double primal_ms = benchmark([&]() { volatile double v = eval_primal(); (void)v; },
                                 warmup, iters);

    double grad_ms = benchmark(
        [&]() {
            double base = eval_primal();
            double grads[XVA_NUM_MARKET_INPUTS];
            double* fields[XVA_NUM_MARKET_INPUTS];
            int k = 0;
            for (int i = 0; i < XVA_NUM_RATES; ++i)  fields[k++] = &market.rates[i];
            for (int i = 0; i < XVA_NUM_HAZARD; ++i) fields[k++] = &market.hazard[i];
            for (int i = 0; i < XVA_NUM_VOLS; ++i)   fields[k++] = &market.vols[i];

            for (int i = 0; i < XVA_NUM_MARKET_INPUTS; ++i)
            {
                double saved = *fields[i];
                *fields[i] = saved + eps;
                grads[i] = (eval_primal() - base) / eps;
                *fields[i] = saved;
            }
            volatile double g = grads[0];
            (void)g;
        },
        warmup, iters);

    return {"FD", "XVA-CVA", primal_ms, grad_ms};
}

// ---------------------------------------------------------------------------
// LIBOR Swaption — 161 inputs (delta + 80 L0 + 80 lambda), bump-and-revalue
// ---------------------------------------------------------------------------
BenchmarkResult fd_libor_swaption(int numPaths, size_t warmup, size_t iters)
{
    auto market = defaultMarket();
    auto portfolio = defaultPortfolio();
    auto all_samples = libor_generate_all_samples(numPaths, market.lambda.size() / 2, 12354);
    const double eps = 1e-6;

    auto eval_primal = [&](const MarketParameters& m) {
        std::vector<double> L, tmp1, tmp2;
        double sum = 0.0;
        for (int p = 0; p < numPaths; ++p)
        {
            L.assign(m.L0.begin(), m.L0.end());
            libor_path_gen(m.delta, L, m.lambda, all_samples[p]);
            sum += libor_value_portfolio(m.delta, portfolio.maturities,
                                         portfolio.swaprates, L, tmp1, tmp2);
        }
        return sum / numPaths;
    };

    double primal_ms = benchmark([&]() { volatile double v = eval_primal(market); (void)v; },
                                 warmup, iters);

    double grad_ms = benchmark(
        [&]() {
            MarketParameters m = market;
            double base = eval_primal(m);
            double sink = 0.0;

            // bump delta
            double saved = m.delta;
            m.delta = saved + eps;
            sink += (eval_primal(m) - base) / eps;
            m.delta = saved;

            // bump each L0[i]
            for (size_t i = 0; i < m.L0.size(); ++i)
            {
                double s = m.L0[i];
                m.L0[i] = s + eps;
                sink += (eval_primal(m) - base) / eps;
                m.L0[i] = s;
            }
            // bump each lambda[i]
            for (size_t i = 0; i < m.lambda.size(); ++i)
            {
                double s = m.lambda[i];
                m.lambda[i] = s + eps;
                sink += (eval_primal(m) - base) / eps;
                m.lambda[i] = s;
            }
            volatile double v = sink;
            (void)v;
        },
        warmup, iters);

    return {"FD", "LiborSwaption", primal_ms, grad_ms};
}
