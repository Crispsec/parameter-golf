# V2 MemoryTokens

Learned persistent key/value "memory" tokens (`MEMORY_TOKENS`, default 4) prepended to the keys and values in every attention layer of the signalrush stack (`2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015`). The XSA step uses only the real values, not the memory. Includes a manual attention fallback so `torch.compile` works when q_len != kv_len on a non-Hopper GPU.

**Local result (1x RTX 5070 Ti, 4500 s cap, 49,152 tokens/step, 7,613 steps):** val_bpb 1.2884 after EMA, 1.2905 after the int6 round-trip, artifact 15.15 MB. Log: [`local_logs/v2_memory_tokens_4500s.txt`](../../local_logs/v2_memory_tokens_4500s.txt).

**Not comparable to the leaderboard** (about 7% of the tokens of a record run, no sliding-window eval), and there is no matched control without memory tokens, so the effect of the idea is unmeasured. See [EXPERIMENTS.md](../../EXPERIMENTS.md).
