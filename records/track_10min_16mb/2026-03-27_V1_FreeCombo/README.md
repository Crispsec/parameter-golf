# V1 FreeCombo (script only, no completed run)

Bundle of low-cost tweaks on the signalrush stack (`2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015`):

- `BIGRAM_VOCAB_SIZE` 2048 -> 4096
- `VE_LAYERS` 9,10 -> 7,8,9,10 (value embeddings on more layers)
- `EVAL_STRIDE` 64 -> 32 (finer sliding-window evaluation)
- Warmdown capped at 90% of the wall-clock budget; first 5 compile steps excluded from the step-time estimate
- FlashAttention fallback to PyTorch SDPA so it runs on non-Hopper GPUs

**Status:** no completed run, so no score is claimed. See [EXPERIMENTS.md](../../EXPERIMENTS.md).
