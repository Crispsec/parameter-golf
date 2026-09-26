# My Parameter Golf experiments (Crispsec fork)

This is a fork of [openai/parameter-golf](https://github.com/openai/parameter-golf). The challenge: train the best language model that fits in a **16 MB artifact** and trains in **10 minutes on 8xH100**, scored in bits-per-byte (val_bpb) on FineWeb (lower is better).

**Short version:** I explored several ideas on top of existing leaderboard stacks. Every run documented here comes from a single consumer GPU and none is leaderboard-comparable. I also ran some experiments on rented 8xH100 pods, but I did not keep those logs, so **no H100 numbers are reported here** and I cannot claim a run that beats the official baseline (1.2244 val_bpb). Details and caveats below.

## Hardware and why this matters

| | Official record runs | My documented runs (logs in `local_logs/`) |
|---|---|---|
| GPU | 8x H100 SXM | 1x RTX 5070 Ti (16 GB, Blackwell) |
| Wall-clock | 600 s | 300 s to 4500 s |
| Tokens per step | 786,432 | 49,152 (long run), 524,288 (500-step screens) |
| Speed | ~85 ms/step | ~590 ms/step at 49k tokens/step |
| Tokens seen (best run) | ~5.6 B | ~0.37 B (7,613 steps x 49,152), about 7% |

A single 5070 Ti is roughly two orders of magnitude slower than the record setup, so locally I could only **screen ideas**. The rented H100 runs are not documented (logs not retained), so nothing below should be read as an H100 result.

## Procedure

1. **Pick a base stack** from existing records and change one thing at a time:
   - `2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA` (Raahil Shah, 1.1458): 9L/512d, int6 + zstd-22, 3x MLP, SmearGate, BigramHash, Muon WD, SWA.
   - `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015` (signalrush, 1.1233): 11L, XSA on last 4 layers, partial RoPE, EMA, GPTQ-lite, late QAT.
2. **Make the scripts run on my hardware.** FlashAttention 3 is Hopper-only, so I added fallbacks (FA2, then PyTorch SDPA; a manual attention path for the memory-token variant so `torch.compile` works when q_len differs from kv_len).
3. **Fix the schedule for short runs.** Warmdown is capped at 90% of the wall-clock budget, and the first 5 (compile-inflated) steps are excluded from the step-time estimate, so a time-capped run does not start with a decayed learning rate.
4. **Screen** each idea with 500-step single-GPU runs to check it trains, then run longer time-capped runs for the promising ones. Several long launches were stopped early (SIGTERM) while I iterated; only one long run completed.
5. **Compare against the same setup**, not against leaderboard numbers. (Step 5 is where this project is incomplete; see "What is missing".)

## The ideas

| Experiment | Idea | Code |
|---|---|---|
| **FactoredEmb** (R128, R128_Dim528, R64_Dim512, R64_Dim528) | Factor the tied embedding as vocab -> rank r -> model_dim to free parameter budget, optionally spending it on a wider model (dim 528). | [`records/track_10min_16mb/2026-03-22_FactoredEmb_R128`](records/track_10min_16mb/2026-03-22_FactoredEmb_R128/train_gpt.py) and the three siblings |
| **RecurrentDepth PoC** | ALBERT-style weight sharing: 3 unique blocks cycled across a 9-layer depth (`NUM_UNIQUE_LAYERS`), trading parameters for headroom. | [`2026-03-23_RecurrentDepth_PoC`](records/track_10min_16mb/2026-03-23_RecurrentDepth_PoC/) |
| **V1 FreeCombo** | Bundle of low-cost tweaks on the signalrush stack: BigramHash 2048 -> 4096, value-embeddings on layers 7-10, eval stride 32. | [`2026-03-27_V1_FreeCombo`](records/track_10min_16mb/2026-03-27_V1_FreeCombo/) |
| **V2 MemoryTokens** | Learned persistent key/value "memory" tokens prepended in each attention layer (a few parameters, extra context at every position). | [`2026-03-27_V2_MemoryTokens`](records/track_10min_16mb/2026-03-27_V2_MemoryTokens/) |
| **V3 StochasticDepth** | Randomly skip blocks 3-8 during training (p=0.08) as a regulariser, disabled at eval. | [`2026-03-27_V3_StochasticDepth`](records/track_10min_16mb/2026-03-27_V3_StochasticDepth/) |

## Results (single RTX 5070 Ti; every number is from a log in [`local_logs/`](local_logs/))

**500-step screens** (val_bpb before quantization, 1 train shard):

| Run | Params | Batch tokens / seq | val_bpb @500 |
|---|---:|---|---:|
| Naive baseline | 17.06 M | 524,288 / 1024 | 1.4589 |
| Naive baseline, repeat (same printed config) | 17.06 M | 524,288 / 1024 | 1.4782 |
| Shah stack (SmearGate/BigramHash/MLP3x/Muon WD) | 22.37 M | 786,432 / 2048 | 1.6224 |
| FactoredEmb R128 | 22.04 M | 524,288 / 1024 | 1.6538 |
| FactoredEmb R64, dim 528 | 23.29 M | 524,288 / 1024 | 1.6596 |

**Long run** (V2 MemoryTokens on the signalrush stack, 4500 s cap, 49,152 tokens/step, 7,613 steps):

| Metric | val_bpb |
|---|---:|
| Raw, at last step | 1.2900 |
| After EMA | 1.2884 |
| After int6 quantization round-trip | 1.2905 |
| Artifact size (int6 + zstd + code) | 15.15 MB (fits the 16 MB limit) |

### How to read these numbers

- **Not comparable to the leaderboard.** Roughly 7% of the tokens, a different batch size, one GPU, and the sliding-window eval used for record scores was not run for this experiment.
- **The 500-step screens say little.** The stacks use a 3000-step warmdown tuned for ~7k-step runs, so at 500 steps the learning rate is decayed from the start. That is why the fancier stacks look worse than the naive baseline here, and it is not evidence about the ideas themselves. The two baseline repeats differ by 0.019 bpb, which is larger than the differences I was trying to measure.
- **There is no matched control for any idea.** I never ran the unmodified base stack under exactly the same local settings as FactoredEmb, MemoryTokens or StochasticDepth, so I cannot say whether any of them helped or hurt.

## Did any attempt beat the baseline?

No, not in a way I can support. The official naive baseline is 1.2244 val_bpb on 8xH100. My best documented run reached 1.2884-1.2905 on much less compute, which neither beats nor properly compares against it. I did run some experiments on rented H100s, but without retained logs I make no claim about them. RecurrentDepth PoC, V1 and V3 have scripts but no documented completed runs.

## What is missing / next steps

- A **control run** of the unmodified signalrush stack in the same local configuration, then V1/V2/V3 against it (same seed, same wall-clock, ideally 3 seeds).
- Longer or multi-seed runs so the effect sizes exceed the run-to-run noise.
- Re-run the promising ideas on rented 8xH100 **and keep the logs** (`tee` to a file, copy off the pod before terminating); only then would a leaderboard or non-record PR make sense.
- Completing RecurrentDepth PoC and V1/V3 runs.

## Layout

```
EXPERIMENTS.md                     this file
local_logs/                        the logs behind every number above
records/track_10min_16mb/
  2026-03-22_FactoredEmb_R128/     \
  2026-03-23_FactoredEmb_*/         }  train_gpt.py variants (see table)
  2026-03-23_RecurrentDepth_PoC/   /
  2026-03-27_V1_FreeCombo/
  2026-03-27_V2_MemoryTokens/
  2026-03-27_V3_StochasticDepth/
```

Everything else in this repo is upstream OpenAI code and other participants' records; see the original [README](README.md) for the challenge rules and leaderboard.
