/*******************************************************************************
   autodiff LIBOR Swaption MC benchmark - N/A.

   161 inputs + deep nested loops makes autodiff's expression-tree
   reverse mode infeasible at scale.

   Copyright (c) 2026 Xcelerit Computing Ltd.
   Licensed under the MIT License. See LICENSE file.
******************************************************************************/

#include "../src/benchmarks.hpp"

BenchmarkResult autodiff_libor_swaption(int numPaths, size_t warmup, size_t iters)
{
    (void)numPaths; (void)warmup; (void)iters;
    return {"autodiff", "LiborSwaption", -1.0, -1.0};
}
