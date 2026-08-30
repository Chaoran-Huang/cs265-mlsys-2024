# Mirage: A Multi-Level Superoptimizer for Tensor Programs

**Mengdi Wu et al. — OSDI 2025**

> Search equivalent tensor programs across graph, thread-block, and thread levels rather than applying a fixed list of compiler heuristics.

## Read first

- [Paper PDF](paper.pdf)
- [Course discussion slides](discussion-slides.pdf)
- [Official paper page](https://arxiv.org/abs/2405.05751)

## The problem

Tensor compilers usually optimize one abstraction level at a time. Graph fusion may expose opportunities that require a different block algorithm; block scheduling may depend on an algebraic rewrite at the graph level. Greedy passes and single-level schedule search miss combinations across these boundaries.

## Key insight

Mirage represents tensor programs with a multi-level μGraph and systematically searches equivalent programs. A verifier checks candidate equivalence, while profiling identifies the fastest valid implementation. This is superoptimization: discover an implementation rather than merely tune parameters of one template.

## How it works

1. Represent computation at kernel, thread-block, and thread levels.
2. Enumerate algebraic and structural rewrites across levels.
3. Prune invalid or equivalent candidates aggressively.
4. Verify that optimized candidates preserve the original computation.
5. Compile and profile surviving programs on target hardware.

## What the results mean

Mirage finds fused implementations that established compilers and hand-written libraries miss, yielding strong speedups on important tensor programs. The result supports a broader search space than schedule tuning alone.

Do not memorize only the headline speedup. Identify which resource stopped being the bottleneck, which resource became the new bottleneck, and whether the evaluation workload makes that transition visible.

## Limitations and critique

Superoptimization can be computationally expensive, so pruning and verification determine practicality. Floating-point equivalence is subtle, and dynamic shapes multiply the search space. Vendor libraries remain formidable for isolated standard operators.

## Connection to CS265

D02 TVM supplies the compiler foundation; Mirage expands the searched object from schedules to equivalent programs. FlashAttention can be viewed as the kind of cross-operator, memory-aware algorithm a sufficiently capable superoptimizer should discover.

## Check your understanding

1. How does superoptimization differ from autotuning tile sizes?
2. Why is numerical equivalence harder than symbolic real-number equivalence?
3. Which FlashAttention transformations cross abstraction levels?

If you cannot answer these without looking back, reread the design and evaluation sections rather than the introduction.

## One-paragraph review template

Write four sentences: **(1)** the resource bottleneck, **(2)** the paper’s mechanism, **(3)** the strongest experimental evidence, and **(4)** one workload or assumption where the result may not hold. This is more useful than restating the abstract.
