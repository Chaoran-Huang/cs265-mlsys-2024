# Mooncake: Trading More Storage for Less Computation — A KVCache-centric Architecture for Serving LLM Chatbot

**Ruoyu Qin et al. — FAST 2025 (Best Paper)**

> Treat KV cache as durable, tiered serving data rather than disposable GPU scratch space.

## Read first

- [Paper PDF](paper.pdf)
- [Course discussion slides](discussion-slides.pdf)
- [Official paper page](https://arxiv.org/abs/2407.00079)

## The problem

Chat, RAG, and multi-turn applications repeatedly process common prefixes: system prompts, retrieved documents, and conversation history. Recomputing those prefixes wastes expensive GPU cycles, but GPU HBM is too small to retain every KV cache. A serving architecture centered only on model workers misses this storage problem.

## Key insight

Mooncake makes KV cache the center of the architecture. It disaggregates prefill and decode and uses a distributed CPU-memory/RDMA cache pool to retain and transfer KV. The system chooses when fetching cached state is cheaper than recomputing it—explicitly trading more storage for less GPU computation.

## How it works

1. Separate compute workers from a distributed KV-cache pool.
2. Store reusable prefix cache outside scarce GPU HBM.
3. Transfer cache via high-bandwidth networking to the worker that needs it.
4. Schedule requests with cache locality and compute/storage cost in mind.
5. Balance cache admission and eviction against expected prefix reuse.

## What the results mean

The paper demonstrates higher serving efficiency on chatbot-style workloads and won FAST 2025’s best-paper award. Its durable lesson is that LLM inference is also a storage system.

Do not memorize only the headline speedup. Identify which resource stopped being the bottleneck, which resource became the new bottleneck, and whether the evaluation workload makes that transition visible.

## Limitations and critique

Unique cold prompts have no reuse. Fetching KV can cost more than recomputation when network bandwidth is weak or prefixes are short. Cache correctness becomes operationally difficult when models, adapters, or prompt templates change.

## Connection to CS265

Mooncake applies Class 1’s STORE–MOVE–PROCESS framework literally: KV is the stored data, RDMA is movement, and prefill is the computation being avoided. Compare D05’s GPU-local pages and D09’s radix-tree prefix index.

## Check your understanding

1. What identifies a reusable KV prefix safely?
2. When is recomputation cheaper than remote cache retrieval?
3. How should model version and LoRA adapter identity enter the cache key?

If you cannot answer these without looking back, reread the design and evaluation sections rather than the introduction.

## One-paragraph review template

Write four sentences: **(1)** the resource bottleneck, **(2)** the paper’s mechanism, **(3)** the strongest experimental evidence, and **(4)** one workload or assumption where the result may not hold. This is more useful than restating the abstract.
