# Efficient Memory Management for Large Language Model Serving with PagedAttention

**Woosuk Kwon et al. — SOSP 2023**

> Manage the KV cache like virtual memory so variable-length requests do not waste contiguous GPU allocations.

## Read first

- [Paper PDF](paper.pdf)
- [Course discussion slides](discussion-slides.pdf)
- [Official paper page](https://arxiv.org/abs/2309.06180)

## The problem

During autoregressive generation, each sequence grows a key/value cache at an unpredictable rate. Reserving a contiguous maximum-length buffer wastes memory internally; allocating and moving variable buffers creates external fragmentation. Wasted HBM reduces batch size, and batch size is the main throughput lever in LLM serving.

## Key insight

PagedAttention divides each sequence’s KV cache into fixed-size logical blocks mapped to non-contiguous physical blocks. An attention kernel follows a block table, just as a process follows a page table. vLLM combines this with a scheduler and copy-on-write sharing.

## How it works

1. Allocate physical KV blocks on demand as tokens are generated.
2. Maintain a per-sequence logical-to-physical block table.
3. Modify the attention kernel to gather K/V through that table.
4. Use reference counts and copy-on-write to share prompt blocks among parallel samples.
5. Schedule many sequences continuously rather than waiting for fixed batches to finish together.

## What the results mean

The paper reports large throughput gains at comparable latency because much more of HBM becomes useful KV capacity. vLLM’s architecture subsequently became a standard open-source serving foundation.

Do not memorize only the headline speedup. Identify which resource stopped being the bottleneck, which resource became the new bottleneck, and whether the evaluation workload makes that transition visible.

## Limitations and critique

Paging adds address-translation logic and specialized kernels. Block size trades metadata and lookup overhead against internal fragmentation. The system still must handle prefill/decode interference and distributed prefix caching, addressed by later papers.

## Connection to CS265

This is a direct transfer of an operating-system data-structure idea into LLM serving. D06 disaggregates the two phases using this cache, D07 expands the cache into a distributed hierarchy, and D09 indexes reusable prefixes.

## Check your understanding

1. Distinguish internal from external KV-cache fragmentation.
2. Why does saving HBM increase throughput rather than only model capacity?
3. What data must copy-on-write duplicate during beam search?

If you cannot answer these without looking back, reread the design and evaluation sections rather than the introduction.

## One-paragraph review template

Write four sentences: **(1)** the resource bottleneck, **(2)** the paper’s mechanism, **(3)** the strongest experimental evidence, and **(4)** one workload or assumption where the result may not hold. This is more useful than restating the abstract.
