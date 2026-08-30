# S-LoRA: Serving Thousands of Concurrent LoRA Adapters

**Ying Sheng et al. — MLSys 2024**

> Share one base model while paging and batching thousands of small LoRA adapters with their KV caches.

## Read first

- [Paper PDF](paper.pdf)
- [Official paper page](https://arxiv.org/abs/2311.03285)

## The problem

LoRA makes customization cheap to train, but serving many tenants creates a new systems problem. Loading a separate base model per adapter wastes almost all memory. Keeping all adapters in GPU HBM does not scale, and batching requests that use different adapters requires irregular matrix operations.

## Key insight

S-LoRA shares the base model and manages adapter weights in a unified paging system alongside KV cache. Custom kernels efficiently apply heterogeneous low-rank updates for many adapters in one batch, while tensor parallelism handles larger base models.

## How it works

1. Keep one copy of the frozen base-model weights.
2. Store many small A/B adapter matrices in CPU/GPU memory tiers.
3. Page adapter weights into GPU memory for active requests.
4. Use custom batched LoRA kernels that tolerate different adapters and ranks.
5. Coordinate adapter paging with KV-cache memory and tensor parallelism.

## What the results mean

S-LoRA serves orders of magnitude more adapters concurrently than naive baselines while preserving high throughput, turning parameter-efficient fine-tuning into practical multi-tenant serving.

Do not memorize only the headline speedup. Identify which resource stopped being the bottleneck, which resource became the new bottleneck, and whether the evaluation workload makes that transition visible.

## Limitations and critique

The design assumes low-rank adapters. High ranks approach full-model cost. Workloads with rapidly changing, one-shot adapters can thrash the paging system, and adapter isolation/versioning remains an operational concern.

## Connection to CS265

S-LoRA connects the fine-tuning lectures to PagedAttention: both solve fragmentation and capacity with paging, but one pages per-request KV and the other pages per-tenant parameters. A self-designing system could jointly pick LoRA rank, placement, and batching.

## Check your understanding

1. Why can requests using different adapters still share base-model GEMMs?
2. What causes adapter-cache thrashing?
3. How do LoRA rank and serving cost interact?

If you cannot answer these without looking back, reread the design and evaluation sections rather than the introduction.

## One-paragraph review template

Write four sentences: **(1)** the resource bottleneck, **(2)** the paper’s mechanism, **(3)** the strongest experimental evidence, and **(4)** one workload or assumption where the result may not hold. This is more useful than restating the abstract.
