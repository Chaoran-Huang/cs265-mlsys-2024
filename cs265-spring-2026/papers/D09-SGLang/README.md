# SGLang: Efficient Execution of Structured Language Model Programs

**Lianmin Zheng et al. — NeurIPS 2024**

> Give multi-call LLM applications a programming model and reuse KV prefixes through a radix-tree cache.

## Read first

- [Paper PDF](paper.pdf)
- [Official paper page](https://arxiv.org/abs/2312.07104)

## The problem

Real LLM applications contain control flow, repeated prompts, branching samples, tool calls, and structured outputs. Expressing everything as independent `generate()` calls hides shared prefixes and forces redundant prefill. General-purpose Python orchestration also cannot expose enough structure for an efficient runtime.

## Key insight

SGLang combines a language for structured generation with a runtime. Its RadixAttention structure stores KV cache in a radix tree keyed by token prefixes, allowing requests and branches to share the longest available prefix. The runtime schedules these programs while managing cache reuse.

## How it works

1. Represent prompts, generation, branching, and parallelism as an LLM program.
2. Insert generated and prompted token sequences into a radix tree.
3. Associate tree nodes with paged KV-cache blocks.
4. Match each new request to its longest cached prefix.
5. Schedule requests while considering cache eviction and reuse.

## What the results mean

The paper reports large speedups on agent, JSON, and multi-sample workloads where shared structure is common. The system has since evolved into a widely used serving/runtime project.

Do not memorize only the headline speedup. Identify which resource stopped being the bottleneck, which resource became the new bottleneck, and whether the evaluation workload makes that transition visible.

## Limitations and critique

Flat, unrelated one-shot prompts offer little prefix reuse. Cached state consumes memory and eviction policy matters. Program-level optimizations are complicated by nondeterministic tool calls and changing model versions.

## Connection to CS265

RadixAttention combines a classic trie/radix data structure with PagedAttention’s KV blocks. It directly addresses the “glue engineering” problem from the AI-systems lectures and pairs naturally with D12’s end-to-end application optimization.

## Check your understanding

1. Why is a radix tree more appropriate than an exact-string hash table?
2. How should the runtime choose between keeping a long rare prefix and many short common ones?
3. Which parts of an agent program are safe to cache?

If you cannot answer these without looking back, reread the design and evaluation sections rather than the introduction.

## One-paragraph review template

Write four sentences: **(1)** the resource bottleneck, **(2)** the paper’s mechanism, **(3)** the strongest experimental evidence, and **(4)** one workload or assumption where the result may not hold. This is more useful than restating the abstract.
