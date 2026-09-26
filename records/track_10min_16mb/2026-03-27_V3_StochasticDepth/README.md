# V3 StochasticDepth (script only, no completed run)

Stochastic depth on the signalrush stack (`2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015`): blocks 3-8 are randomly skipped during training with probability `STOCHASTIC_DEPTH_PROB` (default 0.08), implemented as a tensor mask so it stays `torch.compile`-friendly. Disabled at evaluation. Also includes the FlashAttention -> SDPA fallback and the short-run warmdown fix.

**Status:** no completed run, so no score is claimed. See [EXPERIMENTS.md](../../EXPERIMENTS.md).
