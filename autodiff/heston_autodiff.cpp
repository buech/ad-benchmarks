/*******************************************************************************
   autodiff Heston MC benchmark - N/A.

   100 time steps per path creates a deep expression tree. autodiff's
   reverse var mode traverses it per-variable, making it infeasible
   at scale (8 inputs x 10K paths x 100 steps).

   Copyright (c) 2026 Xcelerit Computing Ltd.
   Licensed under the MIT License. See LICENSE file.
******************************************************************************/

#include "../src/benchmarks.hpp"

BenchmarkResult autodiff_heston(int numPaths, size_t warmup, size_t iters)
{
    (void)numPaths; (void)warmup; (void)iters;
    return {"autodiff", "HestonMC", -1.0, -1.0};
}
