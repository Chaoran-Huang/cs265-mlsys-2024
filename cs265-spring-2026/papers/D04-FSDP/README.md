# PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel

**Yanli Zhao et al. — VLDB 2023**

> Turn ZeRO-3-style full sharding into a composable PyTorch feature and document the engineering choices that make it usable.

## Read first

- [Paper PDF](paper.pdf)
- [Course discussion slides](discussion-slides.pdf)
- [Official paper page](https://arxiv.org/abs/2304.11277)

## The problem

An algorithmic statement—shard parameters and gather them before use—is not yet a production training system. A framework must choose wrapping units, flatten tensors, manage mixed precision, schedule forward and backward prefetch, limit outstanding all-gathers, save checkpoints, and compose with activation checkpointing.

## Key insight

FSDP wraps module units whose parameters are flattened and sharded. Before a unit executes, ranks all-gather its full parameters; after use, full copies can be freed. Gradients are reduce-scattered in backward. The paper is an experience report about making those operations performant and understandable at scale.

## How it works

1. Wrap modules into FSDP units; unit size controls memory peaks versus communication overhead.
2. Flatten many small parameters to make collectives efficient.
3. All-gather parameters before forward/backward and reduce-scatter gradients.
4. Prefetch upcoming units and rate-limit outstanding allocations.
5. Support mixed precision, initialization, checkpointing, and hybrid sharding.

## What the results mean

FSDP scaled large PyTorch workloads while exposing a model-author-friendly API. The lessons about allocator pressure, communication ordering, and wrapping policy are more reusable than a single benchmark number.

Do not memorize only the headline speedup. Identify which resource stopped being the bottleneck, which resource became the new bottleneck, and whether the evaluation workload makes that transition visible.

## Limitations and critique

Users still make consequential configuration choices. Fine-grained units cause many small collectives; coarse units raise peak memory. Debugging distributed execution and producing portable checkpoints remain operational burdens.

## Connection to CS265

Read D03 first for the clean memory argument, then this paper for systems reality. A self-designing fine-tuner would choose wrapping policy, sharding strategy, precision, and checkpointing based on model, cluster, and memory constraints.

## Check your understanding

1. Why is wrapping every leaf module usually a bad idea?
2. What memory exists temporarily during an all-gather?
3. How can prefetch improve speed while making OOM risk worse?

If you cannot answer these without looking back, reread the design and evaluation sections rather than the introduction.

## One-paragraph review template

Write four sentences: **(1)** the resource bottleneck, **(2)** the paper’s mechanism, **(3)** the strongest experimental evidence, and **(4)** one workload or assumption where the result may not hold. This is more useful than restating the abstract.
