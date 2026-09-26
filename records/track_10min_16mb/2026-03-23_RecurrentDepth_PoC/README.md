# RecurrentDepth PoC (work in progress, no results)

Weight-sharing experiment on top of the 9L/512d Shah stack (`2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA`).

**Change:** only `NUM_UNIQUE_LAYERS` blocks are created (default 3) and cycled to fill the full depth (`block[i % NUM_UNIQUE_LAYERS]`), ALBERT-style. Fewer unique blocks means fewer parameters and more byte headroom in the 16 MB budget. Set it equal to `num_layers` to recover the unshared model.

**Status:** script only. There is no completed training run for this variant, so no score is claimed. See [EXPERIMENTS.md](../../EXPERIMENTS.md).
