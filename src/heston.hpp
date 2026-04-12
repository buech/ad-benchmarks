/*******************************************************************************
   Heston Stochastic Volatility pricing - templated for use with any AD type.

   Monte Carlo simulation of the Heston model for European option pricing.
   Uses Euler discretization of the SDE:
     dS = r*S*dt + sqrt(v)*S*dW1
     dv = kappa*(theta-v)*dt + xi*sqrt(v)*dW2
   with correlation rho between W1 and W2.

   8 inputs: S0, K, T, r, v0, kappa, theta, xi (rho fixed for simplicity)

   Copyright (C) 2010-2026 Xcelerit Computing Ltd.
   Licensed under the MIT License. See LICENSE file.
******************************************************************************/

#pragma once

#include <cmath>
#include <random>
#include <vector>

constexpr int HESTON_INPUTS = 8;
constexpr int HESTON_TIME_STEPS = 100;

struct HestonInputs
{
    double S0 = 100.0;
    double K = 105.0;
    double T = 1.0;
    double r = 0.05;
    double v0 = 0.04;       // initial variance
    double kappa = 2.0;     // mean reversion speed
    double theta = 0.04;    // long-run variance
    double xi = 0.3;        // vol of vol
    double rho_corr = -0.7; // correlation (fixed, not differentiated)
};

/// Simulate one Heston path and return discounted European call payoff.
/// z1, z2 are vectors of standard normal samples (HESTON_TIME_STEPS each).
template <class T>
T heston_path(const T& S0, const T& K, const T& expiry, const T& r,
              const T& v0, const T& kappa, const T& theta, const T& xi,
              double rho_corr,
              const std::vector<double>& z1, const std::vector<double>& z2)
{
    using std::exp;
    using std::sqrt;
    using std::abs;

    T dt = expiry / T(HESTON_TIME_STEPS);
    T sqrt_dt = sqrt(dt);

    T S = S0;
    T v = v0;

    for (int i = 0; i < HESTON_TIME_STEPS; ++i)
    {
        // Correlated Brownian motions
        double dW1 = z1[i];
        double dW2 = rho_corr * z1[i] + std::sqrt(1.0 - rho_corr * rho_corr) * z2[i];

        // Ensure variance stays positive (reflection)
        T v_pos = abs(v);
        T sqrt_v = sqrt(v_pos + T(1e-10));

        S = S + r * S * dt + sqrt_v * S * sqrt_dt * T(dW1);
        v = v + kappa * (theta - v) * dt + xi * sqrt_v * sqrt_dt * T(dW2);
    }

    // European call payoff
    T payoff = (S > K) ? (S - K) : T(0.0);
    return exp(-r * expiry) * payoff;
}

/// Pure primal MC pricing.
inline double heston_price_mc(const HestonInputs& in, int numPaths,
                              unsigned long long seed = 77777)
{
    std::mt19937 gen(seed);
    std::normal_distribution<double> dist(0.0, 1.0);

    std::vector<double> z1(HESTON_TIME_STEPS), z2(HESTON_TIME_STEPS);
    double sum = 0.0;

    for (int p = 0; p < numPaths; ++p)
    {
        for (auto& z : z1) z = dist(gen);
        for (auto& z : z2) z = dist(gen);
        sum += heston_path(in.S0, in.K, in.T, in.r, in.v0, in.kappa, in.theta, in.xi,
                           in.rho_corr, z1, z2);
    }
    return sum / numPaths;
}

/// Pre-generate random samples for reproducibility.
struct HestonSamples
{
    std::vector<std::vector<double>> z1, z2;
};

inline HestonSamples heston_generate_samples(int numPaths, unsigned long long seed = 77777)
{
    std::mt19937 gen(seed);
    std::normal_distribution<double> dist(0.0, 1.0);

    HestonSamples s;
    s.z1.resize(numPaths);
    s.z2.resize(numPaths);
    for (int p = 0; p < numPaths; ++p)
    {
        s.z1[p].resize(HESTON_TIME_STEPS);
        s.z2[p].resize(HESTON_TIME_STEPS);
        for (auto& z : s.z1[p]) z = dist(gen);
        for (auto& z : s.z2[p]) z = dist(gen);
    }
    return s;
}
