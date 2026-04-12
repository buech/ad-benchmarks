/*******************************************************************************
   LIBOR Swaption Portfolio Pricer - Shared templated functions.

   Adapted from Prof. Mike Giles' code:
   https://people.maths.ox.ac.uk/~gilesm/codes/libor_AD/testlinadj.cpp

   Copyright (C) 2010-2026 Xcelerit Computing Ltd.
   Licensed under the MIT License. See LICENSE file.
******************************************************************************/

#pragma once

#include <algorithm>
#include <random>
#include <vector>

struct SwaptionPortfolio
{
    std::vector<double> swaprates;
    std::vector<int> maturities;
};

struct MarketParameters
{
    double delta = 0.;
    std::vector<double> lambda;
    std::vector<double> L0;
};

struct SensitivityResults
{
    double price = 0.;
    double d_delta = 0.;
    std::vector<double> d_L0;
    std::vector<double> d_lambda;
};

constexpr int LIBOR_NUM_PATHS_DEFAULT = 10000;

inline SwaptionPortfolio defaultPortfolio()
{
    SwaptionPortfolio p;
    p.maturities = {4, 4, 4, 8, 8, 8, 20, 20, 20, 28, 28, 28, 40, 40, 40};
    p.swaprates = {.045, .05, .055, .045, .05, .055, .045, .05, .055, .045, .05, .055, .045, .05, .055};
    return p;
}

inline MarketParameters defaultMarket()
{
    MarketParameters market;
    market.delta = 0.05;
    market.L0.assign(80, 0.05);
    market.lambda.assign(80, 0.2);
    return market;
}

/// Path generation - calculates LIBOR rates at the given times.
template <class T>
void libor_path_gen(const T& delta, std::vector<T>& L, const std::vector<T>& lambda,
                    const std::vector<double>& z)
{
    using std::exp;
    using std::sqrt;

    for (size_t n = 0; n < z.size(); n++)
    {
        T sqez = sqrt(delta) * z[n];
        T v = T(0.0);
        for (size_t i = n + 1; i < L.size(); i++)
        {
            T lam = lambda[i - n - 1];
            T con1 = delta * lam;
            v += (con1 * L[i]) / (T(1.0) + delta * L[i]);
            L[i] *= exp(con1 * v + lam * (sqez - T(0.5) * con1));
        }
    }
}

/// Value the swap portfolio for the given LIBOR rates.
template <class T>
T libor_value_portfolio(const T& delta, const std::vector<int>& maturities,
                        const std::vector<double>& swaprates, const std::vector<T>& L,
                        std::vector<T>& Btmp, std::vector<T>& Stmp)
{
    const size_t NN = L.size();
    const size_t N = NN / 2;
    const size_t Nopt = swaprates.size();

    Btmp.resize(NN);
    Stmp.resize(NN);

    T b = T(1.0);
    T s = T(0.0);

    for (size_t n = N; n < NN; ++n)
    {
        b = b / (T(1.0) + delta * L[n]);
        s = s + delta * b;
        Btmp[n] = b;
        Stmp[n] = s;
    }

    T v = T(0.0);
    for (size_t i = 0; i < Nopt; i++)
    {
        int m = maturities[i] + static_cast<int>(N) - 1;
        T swapval = Btmp[m] + swaprates[i] * Stmp[m] - T(1.0);
        if (swapval < T(0.0))
            v += T(-100.0) * swapval;
    }

    for (size_t n = 0; n < N; n++)
        v = v / (T(1.0) + delta * L[n]);

    return v;
}

/// Generate random samples for one MC path.
inline void generateSamples(std::mt19937& gen, std::vector<double>& samples)
{
    std::normal_distribution<double> dist(0., 1.);
    std::generate(samples.begin(), samples.end(), [&]() { return dist(gen); });
}

/// Pure primal MC pricing (no AD).
inline double libor_price_mc(const MarketParameters& market, const SwaptionPortfolio& portfolio,
                             int numPaths, unsigned long long seed = 12354)
{
    std::mt19937 gen(seed);
    std::vector<double> samples(market.lambda.size() / 2);
    std::vector<double> L, tmp1, tmp2;
    double price = 0.0;
    for (int p = 0; p < numPaths; ++p)
    {
        generateSamples(gen, samples);
        L.assign(market.L0.begin(), market.L0.end());
        libor_path_gen(market.delta, L, market.lambda, samples);
        price += libor_value_portfolio(market.delta, portfolio.maturities, portfolio.swaprates, L, tmp1, tmp2);
    }
    return price / numPaths;
}

/// Pre-generate all random samples for reproducibility.
inline std::vector<std::vector<double>> libor_generate_all_samples(
    int numPaths, size_t numSamples, unsigned long long seed)
{
    std::mt19937 gen(seed);
    std::normal_distribution<double> dist(0., 1.);
    std::vector<std::vector<double>> all(numPaths);
    for (int p = 0; p < numPaths; ++p)
    {
        all[p].resize(numSamples);
        std::generate(all[p].begin(), all[p].end(), [&]() { return dist(gen); });
    }
    return all;
}
