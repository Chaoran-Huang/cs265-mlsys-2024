# Activation Checkpointing & Tensor Swapping for DNN Training

A course project from **Harvard CS265 (Big Data Systems / ML Systems), Spring 2024**, where I dug into a question that anyone who has ever hit a CUDA out-of-memory error cares about: *do you really need to keep every activation around for the backward pass, or can you trade some compute to get that memory back?*

The short answer is you can, and this repo is my attempt to do it automatically at the graph level instead of by hand.

## The idea

During training, the forward pass produces a pile of intermediate activations that just sit in GPU memory until the backward pass needs them. On big models that pile is the thing that blows up your memory budget. There are two classic ways to shrink it:

1. **Recomputation (activation checkpointing)** — throw an activation away after the forward pass and recompute it on the fly when the backward pass asks for it. You pay in extra compute, you save memory.
2. **Swapping (offloading)** — move an activation off to CPU memory while it is not needed and prefetch it back before the backward pass. You pay in PCIe bandwidth, you save GPU memory.

The interesting part is deciding *which* activations to recompute, which to swap, and which to just keep — given a fixed memory limit. That is where the profiling comes in.

## How it works

Everything is built on `torch.fx`. I trace the combined forward + backward graph and then:

- **`graph_prof.py`** — a `GraphProfiler` (an `fx.Interpreter`) that runs the graph node by node and records per-node runtime, tensor sizes, and peak memory. It also classifies every tensor (parameter / activation / gradient) and, for each intermediate activation, figures out its *last forward use* and *first backward use* — the window during which it is dead weight. It can also swap those tensors out to pinned CPU memory and prefetch them back.
- **`recomputation.py`** — the policy that, given a memory limit, decides which activations are worth recomputing (cheap to redo, expensive to store) versus keeping.
- **`activation_checkpoint.py`** — the graph surgery: it extracts a sub-graph that recomputes the chosen activations, splices it into the original graph right before the first backward access, and rewires the downstream nodes to use the recomputed values.
- **`graph_tracer.py`** — tracing helpers plus the `SEPFunction` separator that marks the forward/backward boundary.

After the transformation, gradients are checked against the un-optimized run with `torch.allclose` so I know the rewrite is actually correct and not just smaller.

## Results

The headline was cutting peak activation memory substantially (roughly **70–85%** on the models I tested) for a modest, tunable increase in step time. The full memory–compute tradeoff curves and the per-node breakdowns live in [`analysis.ipynb`](analysis.ipynb).

## Running it

```bash
python activation_checkpoint.py
```

This traces a small two-layer example, profiles it, applies the recomputation policy, prints the before/after graphs, and verifies the gradients match. `benchmarks.py` drives the larger runs used for the analysis.

> Note: this is a research/course project, not a production library — it is meant to make the memory–compute tradeoff visible and measurable, not to replace `torch.utils.checkpoint`.
