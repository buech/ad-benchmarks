/*******************************************************************************
   autodiff XVA benchmark - N/A.

   40 market inputs + deep computation (10 swaps × 20 time buckets)
   makes autodiff's expression-tree reverse mode infeasible.

   Copyright (c) 2026 Xcelerit Computing Ltd.
   Licensed under the MIT License. See LICENSE file.
******************************************************************************/

#include "../src/benchmarks.hpp"

BenchmarkResult autodiff_xva(size_t warmup, size_t iters)
{
    (void)warmup; (void)iters;
    return {"autodiff", "XVA-CVA", -1.0, -1.0};
}
