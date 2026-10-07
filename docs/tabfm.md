# How we ran Google TabFM 1.0 at full context (no OOM, ~1.5 GPU-hours per fold)

**Setup:**
- Model: TabFM 1.0, the PyTorch port `tabfm 1.0.1`, 1.64B parameters, native bf16, 8 estimators.
- Data: Playground S6E10, 5 folds, on one A100 80GB.
- Each fold: **559,708 context rows** (all of outer-train) and **439,771 query rows** (outer-valid + test).

**Result:**
- **OOF AUC 0.958871.**
- **7.5 GPU-hours for all 5 folds** (about 11 minutes per estimator per fold), **peak 45 GB.**

## The problem with stock TabFM

`TabFMClassifier.predict_proba` has **no context cache**. For every estimator and every call, it runs one forward pass over **[all context rows + the query rows]**. The 24-layer ICL transformer (d = 2048) then attends over that whole sequence, with a mask so rows only use the context.

That leaves two bad options at full context:
- **Pass all 440k queries in one call:** about 1M rows go through every stage and all 24 attention layers at once, which is where memory blows up.
- **Pass queries in batches:** every batch re-processes the full 560k-row context through the 1.64B model. By our timing that's roughly **a day per fold**.

## The key observation

Every TabFM stage that mixes rows only ever reads **context (training) rows**:

| Stage | Which rows it reads |
|---|---|
| Cell embedder | Per row |
| Two column set-transformers (3 induced-set blocks each) | 256 inducing points attend to context rows only; then every row attends to those 256 summaries |
| Row interactors | Attention across the features of one row |
| ICL predictor (24 blocks) | Every row attends to context rows only; context rows never see query rows |

So the context can be processed **once per estimator**, and query rows only need to *read* cached context states. This is exact, not an approximation.

## What we changed

We replaced only `TabFMClassifier._batch_forward`, the per-estimator forward. Preprocessing, the 8 estimators, class shifts, temperature 0.9 and logit averaging stay stock, and everything is one ordinary `predict_proba` call. For each estimator:

1. **Context rows, once:** run the embedding stages and **cache the column set-transformers' inducing-point summaries** (256 per block).
2. **Query rows, in chunks of 32,768:** run the same embedding stages, attending to the cached summaries instead of the context.
3. **ICL stage, one layer at a time (layer-major):** for each of the 24 layers,
   - compute the context keys/values once,
   - update the query rows (in chunks of 65,536) and the context rows against them,
   - free the K/V and move to the next layer.

   Only one layer's context K/V is ever in memory, and there is no (context + query)² attention matrix.
4. Decode the query rows in chunks.

```python
for blk in icl.tf_icl.blocks:                        # 24 ICL layers
    k, v = kv(blk.attn, blk.pre_attn_ln(x_ctx))      # context keys/values, computed once per layer
    x_q   = block(blk, x_q,   k, v, chunk=65536)     # query rows attend to context only
    x_ctx = block(blk, x_ctx, k, v, chunk=65536)     # context rows attend to context only
    del k, v                                          # memory stays at one layer's K/V
logits = decoder(ln(x_q))
```

## Verification

We ran our cached forward and the stock forward on identical inputs (fold 0, a holdout taken from outer-train).

| Precision | max \|Δp\| | mean \|Δp\| | AUC cached / stock |
|---|---:|---:|---|
| fp32 (20k context, 2k queries) | 3.3e-6 | 1e-7 | 0.962190 / 0.962191 |
| bf16, TabFM's native precision | 0.02–0.05 | 6e-4 to 1e-3 (3k–20k context runs) | 0.957981 / 0.957976 (50k context, 140k queries) |

The math is identical. In bf16 the two differ only by rounding (different attention kernels and summation order over 24 layers), which reaches 0.02–0.05 on a few rows.

**Speed** (fold 0, 50k context, 140k queries):

| | Time | Peak memory |
|---|---:|---:|
| Stock | 20.1 min | 13.8 GB |
| Cached | 2.9 min | 9.0 GB |

That's 7× faster, and the gap grows with context size and the number of query batches.

## Does full context help?

Fold-0 AUC by context rows:

| 25k | 50k | 100k | 200k | full (560k) |
|---:|---:|---:|---:|---:|
| 0.957520 | 0.957981 | 0.958462 | 0.958798 | 0.958875 |

Yes, though TabFM gains less beyond 200k than TabPFN-3.5 or Kumo-Tabular large.

## Notes

- **Inputs:** categoricals were passed as pandas `category`. NaN was left in; TabFM imputes it.
- **Smaller GPUs:** the 45 GB peak comes from the context-row embedding stages at 560k rows. A 16–24 GB GPU needs a smaller context or finer chunking of those stages.
