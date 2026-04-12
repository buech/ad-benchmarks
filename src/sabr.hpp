/*******************************************************************************
   SABR Implied Volatility (Hagan 2002) and Calibration Objective.

   Templated for use with any AD type. The Hagan approximation computes
   the Black implied volatility from the SABR stochastic volatility model.

   Multi-expiry surface calibration: fit (alpha_i, rho_i, nu_i) for 5 expiries
   simultaneously = 15 AD inputs. Minimise total SSE across 5 x 20 = 100
   strike/expiry pairs.

   Copyright (C) 2010-2026 Xcelerit Computing Ltd.
   Licensed under the MIT License. See LICENSE file.
******************************************************************************/

#pragma once

#include <cmath>
#include <vector>

constexpr int SABR_NUM_STRIKES = 20;
constexpr int SABR_NUM_EXPIRIES = 5;
constexpr int SABR_NUM_PARAMS = 3 * SABR_NUM_EXPIRIES;  // 15
constexpr int SABR_CALIB_ITERS = 500;

/// Market data for one expiry slice.
struct SABRSlice
{
    double F;
    double expiry;
    double beta;
    double strikes[SABR_NUM_STRIKES];
    double market_vols[SABR_NUM_STRIKES];
};

/// Full surface calibration data: 5 expiries.
struct SABRCalibData
{
    SABRSlice slices[SABR_NUM_EXPIRIES];

    // True parameters per expiry (used to generate synthetic market vols)
    double true_alpha[SABR_NUM_EXPIRIES];
    double true_rho[SABR_NUM_EXPIRIES];
    double true_nu[SABR_NUM_EXPIRIES];

    // Starting guesses
    double init_alpha[SABR_NUM_EXPIRIES];
    double init_rho[SABR_NUM_EXPIRIES];
    double init_nu[SABR_NUM_EXPIRIES];

    SABRCalibData() { init(); }
    void init();
};

/// Hagan 2002 SABR implied vol approximation.
/// F, K, beta, expiry are plain doubles (constants in the graph).
template <class T>
T hagan_implied_vol(const T& alpha, double beta, const T& rho, const T& nu,
                    double F, double K, double expiry)
{
    using std::log;
    using std::pow;
    using std::sqrt;

    const double one_m_beta = 1.0 - beta;
    const double one_m_beta2 = one_m_beta * one_m_beta;
    const double one_m_beta4 = one_m_beta2 * one_m_beta2;

    double FK_pow = pow(F * K, one_m_beta / 2.0);
    double FK_pow_full = pow(F * K, one_m_beta);
    double logFK = log(F / K);
    double logFK2 = logFK * logFK;
    double logFK4 = logFK2 * logFK2;

    // Volatility-of-variance correction (time-dependent)
    T alpha2 = alpha * alpha;
    T correction = T(1.0) + (one_m_beta2 / 24.0 * alpha2 / T(FK_pow_full)
                             + T(0.25) * rho * T(beta) * nu * alpha / T(FK_pow)
                             + (T(2.0) - T(3.0) * rho * rho) / T(24.0) * nu * nu)
                            * T(expiry);

    // ATM case: F == K (or very close) — branch on constants only
    double atm_tol = 1e-7 * F;
    if (std::abs(F - K) < atm_tol)
    {
        T sigma_atm = alpha / T(pow(F, one_m_beta)) * correction;
        return sigma_atm;
    }

    // Denominator correction for F != K
    double denom_geo = FK_pow * (1.0 + one_m_beta2 / 24.0 * logFK2
                                     + one_m_beta4 / 1920.0 * logFK4);

    // General case: F != K
    T z = nu / alpha * T(FK_pow) * T(logFK);
    T sqrt_term = sqrt(T(1.0) - T(2.0) * rho * z + z * z);
    T x_z = log((sqrt_term + z - rho) / (T(1.0) - rho));
    T z_over_xz = z / x_z;

    T sigma = alpha / T(denom_geo) * z_over_xz * correction;
    return sigma;
}

/// Multi-expiry calibration objective: sum of squared errors across all slices.
/// params layout: [alpha_0, rho_0, nu_0, alpha_1, rho_1, nu_1, ..., alpha_4, rho_4, nu_4]
template <class T>
T sabr_calibration_objective(const T* params, const SABRCalibData& data)
{
    T total = T(0.0);
    for (int e = 0; e < SABR_NUM_EXPIRIES; ++e)
    {
        const T& alpha = params[3 * e + 0];
        const T& rho   = params[3 * e + 1];
        const T& nu    = params[3 * e + 2];
        const SABRSlice& sl = data.slices[e];

        for (int i = 0; i < SABR_NUM_STRIKES; ++i)
        {
            T model_vol = hagan_implied_vol(alpha, sl.beta, rho, nu,
                                            sl.F, sl.strikes[i], sl.expiry);
            T diff = model_vol - T(sl.market_vols[i]);
            total = total + diff * diff;
        }
    }
    return total;
}

/// Pure double objective for primal timing.
inline double sabr_objective_double(const double* params, const SABRCalibData& data)
{
    return sabr_calibration_objective(params, data);
}

/// Deterministic perturbation schedule for optimizer simulation.
struct SABRPerturbation
{
    double dp[SABR_NUM_PARAMS];
};

inline std::vector<SABRPerturbation> sabr_perturbation_schedule(int n_iters)
{
    std::vector<SABRPerturbation> schedule(n_iters);
    for (int i = 0; i < n_iters; ++i)
    {
        double decay = 1.0 / (1.0 + 0.01 * i);
        double phase = 0.1 * i;
        for (int p = 0; p < SABR_NUM_PARAMS; ++p)
        {
            double freq = 1.0 + 0.3 * p;
            double scale = (p % 3 == 0) ? 0.0002 : 0.001;  // smaller for alpha
            schedule[i].dp[p] = scale * decay * std::cos(phase * freq);
        }
    }
    return schedule;
}

// --- Implementation of SABRCalibData::init ---
inline void SABRCalibData::init()
{
    // 5 expiries: 1Y, 2Y, 5Y, 10Y, 20Y
    double expiries[] = {1.0, 2.0, 5.0, 10.0, 20.0};
    double forwards[] = {0.025, 0.028, 0.030, 0.032, 0.035};

    // True SABR parameters vary by expiry (realistic term structure)
    double alphas[] = {0.040, 0.038, 0.035, 0.032, 0.030};
    double rhos[]   = {-0.15, -0.20, -0.25, -0.28, -0.30};
    double nus[]    = {0.50, 0.45, 0.40, 0.35, 0.30};

    for (int e = 0; e < SABR_NUM_EXPIRIES; ++e)
    {
        slices[e].F = forwards[e];
        slices[e].expiry = expiries[e];
        slices[e].beta = 0.5;

        true_alpha[e] = alphas[e];
        true_rho[e] = rhos[e];
        true_nu[e] = nus[e];

        // Starting guesses: perturbed from true values
        init_alpha[e] = alphas[e] + 0.005;
        init_rho[e] = rhos[e] + 0.10;
        init_nu[e] = nus[e] - 0.05;

        // Strike grid around forward
        double K_min = forwards[e] * 0.3;
        double K_max = forwards[e] * 2.0;
        for (int i = 0; i < SABR_NUM_STRIKES; ++i)
            slices[e].strikes[i] = K_min + (K_max - K_min) * i / (SABR_NUM_STRIKES - 1);

        // Generate synthetic market vols
        for (int i = 0; i < SABR_NUM_STRIKES; ++i)
            slices[e].market_vols[i] = hagan_implied_vol(
                alphas[e], slices[e].beta, rhos[e], nus[e],
                forwards[e], slices[e].strikes[i], expiries[e]);
    }
}
