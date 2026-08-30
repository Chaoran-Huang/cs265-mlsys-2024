# ZeRO: Memory Optimizations Toward Training Trillion Parameter Models

**Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, Yuxiong He — SC 2020**

> Keep data parallelism’s compute pattern but shard its redundant optimizer states, gradients, and parameters across workers.

## Read first

- [Paper PDF](paper.pdf)
- [Course discussion slides](discussion-slides.pdf)
- [Official paper page](https://arxiv.org/abs/1910.02054)

## The problem

Classic data parallelism replicates the full training state on every GPU. Mixed-precision Adam may store model parameters, master FP32 parameters, gradients, and two FP32 moment tensors. Adding GPUs increases compute capacity but does not let the model fit, because each GPU still holds essentially the same state.

## Key insight

ZeRO observes that these replicas are logically redundant. It partitions state across the data-parallel group and communicates pieces only when needed. The three stages progressively shard optimizer state, gradients, and parameters, trading more communication for a near-linear memory reduction.

## How it works

1. Stage 1 partitions optimizer states; each rank updates only its parameter partition.
2. Stage 2 additionally partitions reduced gradients using reduce-scatter.
3. Stage 3 also partitions parameters and all-gathers them before computation.
4. Overlap collectives with forward/backward compute where possible.
5. ZeRO-Offload can extend the hierarchy to CPU memory.

## What the results mean

The system enables dramatically larger models without abandoning data parallelism and became the basis of DeepSpeed’s large-model training stack. The paper shows that memory capacity, not raw FLOPs, was preventing scale.

Do not memorize only the headline speedup. Identify which resource stopped being the bottleneck, which resource became the new bottleneck, and whether the evaluation workload makes that transition visible.

## Limitations and critique

Stage 3 adds frequent communication and temporary all-gather peaks. Performance depends heavily on interconnect bandwidth, overlap, bucket sizing, and wrapping granularity. Sharding state does not eliminate activation memory, which still needs checkpointing or other techniques.

## Connection to CS265

D04 FSDP is the PyTorch-native production story for similar full sharding. Activation checkpointing trades compute for activation memory, while ZeRO removes replicated persistent state; they solve complementary slices of the memory budget.

## Check your understanding

1. List every tensor Adam training stores per parameter and its precision.
2. Why does Stage 3 communicate more than Stage 1?
3. When would activation checkpointing still be necessary with ZeRO-3?

If you cannot answer these without looking back, reread the design and evaluation sections rather than the introduction.

## One-paragraph review template

Write four sentences: **(1)** the resource bottleneck, **(2)** the paper’s mechanism, **(3)** the strongest experimental evidence, and **(4)** one workload or assumption where the result may not hold. This is more useful than restating the abstract.
