/*******************************************************************************
   Benchmark timing harness.

   Copyright (C) 2010-2026 Xcelerit Computing Ltd.
   Licensed under the MIT License. See LICENSE file.
******************************************************************************/

#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <numeric>
#include <string>
#include <vector>

struct BenchmarkResult
{
    std::string library;
    std::string benchmark;
    double primal_ms = 0.0;
    double gradient_ms = 0.0;
    /// Returns overhead ratio, or -1 if primal too fast to measure meaningfully.
    double overhead() const
    {
        if (primal_ms < 0.001 || gradient_ms < 0)
            return -1.0;
        return gradient_ms / primal_ms;
    }
};

struct TimingStats
{
    double min_ms = 0.0;
    double median_ms = 0.0;
    double mean_ms = 0.0;
};

inline TimingStats computeStats(std::vector<double>& times)
{
    TimingStats s;
    if (times.empty()) return s;
    std::sort(times.begin(), times.end());
    s.min_ms = times.front();
    s.median_ms = times[times.size() / 2];
    s.mean_ms = std::accumulate(times.begin(), times.end(), 0.0) / static_cast<double>(times.size());
    return s;
}

/// Run a function with warmup and measured iterations, return median time in ms.
/// If innerLoop > 1, runs fn that many times per measurement and divides.
inline double benchmark(std::function<void()> fn, size_t warmup = 3, size_t iterations = 10,
                        size_t innerLoop = 1)
{
    for (size_t i = 0; i < warmup; ++i)
        for (size_t j = 0; j < innerLoop; ++j)
            fn();

    std::vector<double> times;
    times.reserve(iterations);
    for (size_t i = 0; i < iterations; ++i)
    {
        auto start = std::chrono::steady_clock::now();
        for (size_t j = 0; j < innerLoop; ++j)
            fn();
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        times.push_back(elapsed.count() / static_cast<double>(innerLoop));
    }

    auto stats = computeStats(times);
    return stats.median_ms;
}
