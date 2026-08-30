# TVM: An Automated End-to-End Optimizing Compiler for Deep Learning

**Tianqi Chen et al. — OSDI 2018**

> Separate what a tensor program computes from how it is scheduled, then search for a fast schedule on each target.

## Read first

- [Paper PDF](paper.pdf)
- [Course discussion slides](discussion-slides.pdf)
- [Official paper page](https://arxiv.org/abs/1802.04799)

## The problem

A model must run on CPUs, NVIDIA GPUs, mobile GPUs, and custom accelerators. Each target wants different loop orders, tile sizes, vector widths, memory placement, and operator fusion. Hand-writing every model-by-hardware combination does not scale, while relying only on vendor libraries prevents optimization across operator boundaries.

## Key insight

TVM introduces an end-to-end compiler stack with a tensor-expression language and explicit scheduling primitives. The computation describes mathematical meaning; the schedule describes the physical implementation. A learned cost model guides search over schedules, allowing TVM to generate target-specific code rather than freezing one expert implementation.

## How it works

1. Import a graph from a framework and apply graph-level transformations such as fusion.
2. Describe operators as tensor expressions.
3. Transform schedules with tiling, reordering, vectorization, unrolling, and explicit cache placement.
4. Measure candidate programs and train a cost model to focus search on promising schedules.
5. Lower the selected schedule to target-specific code.

## What the results mean

TVM reaches or exceeds strong framework and vendor-library baselines on a variety of devices and workloads. More important than any one speedup is portability: one compiler architecture covers hardware that previously needed separate hand-tuned stacks.

Do not memorize only the headline speedup. Identify which resource stopped being the bottleneck, which resource became the new bottleneck, and whether the evaluation workload makes that transition visible.

## Limitations and critique

Search consumes compilation time and benchmark runs. The schedule space must be expressive but still searchable. Mature vendor libraries remain difficult to beat on common GEMMs, while dynamic shapes and end-to-end memory planning complicate the clean compute/schedule split.

## Connection to CS265

TVM is a self-designing system at the compiler layer. Compare it with D08 Mirage: TVM searches schedules for a mostly fixed computation, while Mirage explores equivalent computations across multiple IR levels. In the course project, you manually make choices TVM would search—tile dimensions, memory reuse, and fusion.

## Check your understanding

1. Give one computation and two valid schedules for it.
2. Why is measured performance needed in addition to an analytical cost model?
3. Which optimizations require graph-level visibility rather than one operator at a time?

If you cannot answer these without looking back, reread the design and evaluation sections rather than the introduction.

## One-paragraph review template

Write four sentences: **(1)** the resource bottleneck, **(2)** the paper’s mechanism, **(3)** the strongest experimental evidence, and **(4)** one workload or assumption where the result may not hold. This is more useful than restating the abstract.
