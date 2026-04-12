/*******************************************************************************
   Simplified XVA (Credit Valuation Adjustment) Pricer.

   Prices a portfolio of interest rate swaps and computes CVA with
   sensitivities to all market risk factors.

   Market inputs (AD-active):
     - 20 yield curve zero rates (0.5Y, 1Y, 2Y, ..., 30Y)
     - 10 hazard rates for credit curve (1Y, 2Y, ..., 10Y)
     - 10 volatility term structure points
     Total: 40 market risk factors

   Random inputs (per MC path):
     - 40 Gaussian draws for yield curve diffusion

   Graph complexity: ~5000-10000 ops per path (10 swaps × 20 payments ×
   discount factor interp + CVA integration)

   Copyright (C) 2010-2026 Xcelerit Computing Ltd.
   Licensed under the MIT License. See LICENSE file.
******************************************************************************/

#pragma once

#include <cmath>
#include <random>
#include <vector>

// Dimensions
constexpr int XVA_NUM_RATES = 20;         // Yield curve points
constexpr int XVA_NUM_HAZARD = 10;        // Credit spread points
constexpr int XVA_NUM_VOLS = 10;          // Vol term structure
constexpr int XVA_NUM_MARKET_INPUTS = XVA_NUM_RATES + XVA_NUM_HAZARD + XVA_NUM_VOLS;  // 40
constexpr int XVA_NUM_RANDOMS = 40;       // Random draws per path
constexpr int XVA_TOTAL_INPUTS = XVA_NUM_MARKET_INPUTS + XVA_NUM_RANDOMS;  // 80
constexpr int XVA_NUM_SWAPS = 15;         // Portfolio size
constexpr int XVA_PAYMENTS_PER_SWAP = 20; // Semi-annual over 10Y
constexpr int XVA_NUM_TIME_BUCKETS = 20;  // CVA time grid
constexpr int XVA_NUM_PATHS = 10000;      // MC paths

// Yield curve tenors (in years)
inline const double* xva_rate_tenors()
{
    static const double t[] = {
        0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
        10.0, 12.0, 15.0, 17.0, 20.0, 22.0, 25.0, 27.0, 28.0, 30.0
    };
    return t;
}

// Hazard rate tenors
inline const double* xva_hazard_tenors()
{
    static const double t[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    return t;
}

/// Default market data
struct XVAMarketData
{
    double rates[XVA_NUM_RATES];
    double hazard[XVA_NUM_HAZARD];
    double vols[XVA_NUM_VOLS];

    XVAMarketData()
    {
        // Upward-sloping yield curve: 2% to 4%
        for (int i = 0; i < XVA_NUM_RATES; ++i)
            rates[i] = 0.02 + 0.001 * i;
        // Flat hazard rates at 1%
        for (int i = 0; i < XVA_NUM_HAZARD; ++i)
            hazard[i] = 0.01 + 0.001 * i;
        // Vol term structure: 15% to 25%
        for (int i = 0; i < XVA_NUM_VOLS; ++i)
            vols[i] = 0.15 + 0.01 * i;
    }
};

/// Swap definition
struct SwapDef
{
    double notional;
    double fixed_rate;
    int num_payments;
    double start_time;   // years
    double payment_freq; // years between payments (0.5 = semi-annual)
    bool is_payer;       // true = pay fixed
};

/// Default swap portfolio
inline std::vector<SwapDef> xva_default_portfolio()
{
    std::vector<SwapDef> swaps(XVA_NUM_SWAPS);
    for (int i = 0; i < XVA_NUM_SWAPS; ++i)
    {
        swaps[i].notional = 1000000.0;
        swaps[i].fixed_rate = 0.020 + 0.001 * i;
        swaps[i].num_payments = XVA_PAYMENTS_PER_SWAP;
        swaps[i].start_time = 0.0;
        swaps[i].payment_freq = 0.5;  // Semi-annual
        swaps[i].is_payer = (i % 2 == 0);
    }
    return swaps;
}

/// Pre-generate random samples for all MC paths.
inline std::vector<std::vector<double>> xva_generate_samples(int numPaths, unsigned long long seed)
{
    std::mt19937 gen(seed);
    std::normal_distribution<double> dist(0.0, 1.0);
    std::vector<std::vector<double>> z(numPaths);
    for (int p = 0; p < numPaths; ++p)
    {
        z[p].resize(XVA_NUM_RANDOMS);
        for (int i = 0; i < XVA_NUM_RANDOMS; ++i)
            z[p][i] = dist(gen);
    }
    return z;
}

/// Linear interpolation of discount factor from zero rates.
/// rates: array of XVA_NUM_RATES zero rates
/// t: time to discount
template <class T>
T xva_discount_factor(const T* rates, double t)
{
    using std::exp;
    const double* tenors = xva_rate_tenors();

    if (t <= tenors[0])
        return exp(-rates[0] * T(t));
    if (t >= tenors[XVA_NUM_RATES - 1])
        return exp(-rates[XVA_NUM_RATES - 1] * T(t));

    // Find interval
    int idx = 0;
    for (int i = 0; i < XVA_NUM_RATES - 1; ++i)
    {
        if (t < tenors[i + 1])
        {
            idx = i;
            break;
        }
    }

    // Linear interpolation of the zero rate
    double w = (t - tenors[idx]) / (tenors[idx + 1] - tenors[idx]);
    T rate = rates[idx] * T(1.0 - w) + rates[idx + 1] * T(w);
    return exp(-rate * T(t));
}

/// Interpolate survival probability from hazard rates.
template <class T>
T xva_survival_prob(const T* hazard, double t)
{
    using std::exp;
    const double* tenors = xva_hazard_tenors();

    if (t <= 0.0)
        return T(1.0);
    if (t <= tenors[0])
        return exp(-hazard[0] * T(t));
    if (t >= tenors[XVA_NUM_HAZARD - 1])
        return exp(-hazard[XVA_NUM_HAZARD - 1] * T(t));

    int idx = 0;
    for (int i = 0; i < XVA_NUM_HAZARD - 1; ++i)
    {
        if (t < tenors[i + 1])
        {
            idx = i;
            break;
        }
    }

    double w = (t - tenors[idx]) / (tenors[idx + 1] - tenors[idx]);
    T h = hazard[idx] * T(1.0 - w) + hazard[idx + 1] * T(w);
    return exp(-h * T(t));
}

/// Price a single swap given diffused rates.
template <class T>
T xva_price_swap(const T* rates, const SwapDef& swap)
{
    T fixed_leg = T(0.0);
    T float_leg = T(0.0);

    for (int i = 0; i < swap.num_payments; ++i)
    {
        double t = swap.start_time + (i + 1) * swap.payment_freq;
        T df = xva_discount_factor(rates, t);

        // Fixed leg: notional * fixed_rate * dt * df
        fixed_leg = fixed_leg + T(swap.notional * swap.fixed_rate * swap.payment_freq) * df;

        // Floating leg: notional * forward_rate * dt * df
        // Forward rate approximated from adjacent discount factors
        double t_prev = swap.start_time + i * swap.payment_freq;
        T df_prev = xva_discount_factor(rates, t_prev);
        T fwd_rate = (df_prev / df - T(1.0)) / T(swap.payment_freq);
        float_leg = float_leg + T(swap.notional * swap.payment_freq) * fwd_rate * df;
    }

    if (swap.is_payer)
        return float_leg - fixed_leg;  // Receive float, pay fixed
    else
        return fixed_leg - float_leg;  // Receive fixed, pay float
}

/// Diffuse yield curve: apply random shocks scaled by vol.
/// rates_out = rates_in + vol * sqrt(dt) * z (additive shift model)
template <class T>
void xva_diffuse_rates(const T* rates_in, const T* vols, const double* z,
                       double dt, T* rates_out)
{
    using std::sqrt;
    T sqrt_dt = T(sqrt(dt));
    for (int i = 0; i < XVA_NUM_RATES; ++i)
    {
        // Use vol[i % XVA_NUM_VOLS] since we have fewer vol points than rates
        T vol = vols[i % XVA_NUM_VOLS];
        T shock = vol * sqrt_dt * T(z[i % XVA_NUM_RANDOMS]);
        rates_out[i] = rates_in[i] + shock;
    }
}

/// Compute CVA for the entire portfolio across one MC path.
/// This is the main computation graph: heavy, many ops, templated.
///
/// rates: 20 yield curve points (AD inputs)
/// hazard: 10 hazard rates (AD inputs)
/// vols: 10 vol points (AD inputs)
/// z: random draws for this path (plain double or AD)
template <class T>
T xva_compute_cva(const T* rates, const T* hazard, const T* vols,
                  const double* z, const std::vector<SwapDef>& portfolio)
{
    T cva = T(0.0);
    double dt = 0.5; // Semi-annual time buckets

    // Working copy of rates for diffusion
    T diffused_rates[XVA_NUM_RATES];
    for (int i = 0; i < XVA_NUM_RATES; ++i)
        diffused_rates[i] = rates[i];

    for (int bucket = 0; bucket < XVA_NUM_TIME_BUCKETS; ++bucket)
    {
        double t = (bucket + 1) * dt;

        // Diffuse rates forward
        xva_diffuse_rates(diffused_rates, vols, z, dt, diffused_rates);

        // Price portfolio at this time point
        T portfolio_pv = T(0.0);
        for (int s = 0; s < XVA_NUM_SWAPS; ++s)
        {
            // Adjust swap: remaining payments only
            SwapDef remaining = portfolio[s];
            int payments_elapsed = static_cast<int>(t / remaining.payment_freq);
            remaining.num_payments = remaining.num_payments - payments_elapsed;
            remaining.start_time = t;
            if (remaining.num_payments > 0)
                portfolio_pv = portfolio_pv + xva_price_swap(diffused_rates, remaining);
        }

        // Expected Positive Exposure: max(PV, 0)
        // Use if/else branch — resolved at record time since PV depends on data
        T exposure;
        if (portfolio_pv > T(0.0))
            exposure = portfolio_pv;
        else
            exposure = T(0.0);

        // CVA contribution: exposure * default_prob * discount_factor
        T surv_prev = xva_survival_prob(hazard, t - dt);
        T surv_curr = xva_survival_prob(hazard, t);
        T default_prob = surv_prev - surv_curr;
        T df = xva_discount_factor(rates, t);

        cva = cva + exposure * default_prob * df;
    }

    return cva;
}

/// Pure double CVA for primal timing.
inline double xva_cva_double(const XVAMarketData& market, const double* z,
                             const std::vector<SwapDef>& portfolio)
{
    return xva_compute_cva(market.rates, market.hazard, market.vols, z, portfolio);
}
