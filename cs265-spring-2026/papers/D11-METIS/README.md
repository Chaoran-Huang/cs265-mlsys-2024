# METIS: Fast Quality-Aware RAG Systems with Configuration Adaptation

**Siddhant Ray et al. — SOSP 2025**

> Adapt the RAG pipeline configuration per workload instead of manually freezing chunking, retrieval, reranking, and generation choices.

## Read first

- [Paper PDF](paper.pdf)
- [Official paper page](https://arxiv.org/abs/2412.10543)

## The problem

RAG quality and latency depend on interacting choices: chunking, embeddings, index settings, top-k, rerankers, context packing, and generator. The best configuration varies by corpus and query type. Exhaustively running every configuration is too expensive, while one global setting wastes cost or misses quality targets.

## Key insight

METIS treats RAG configuration as a quality-aware systems optimization problem. It evaluates or predicts promising configurations and adapts choices under quality and performance requirements, bringing a query-optimizer mindset to the end-to-end RAG pipeline.

## How it works

1. Expose a design space of retriever, reranker, and generation configurations.
2. Measure quality and latency on representative queries.
3. Use adaptation to avoid evaluating every full pipeline configuration.
4. Select configurations that satisfy a quality target at low cost/latency.
5. Update decisions as workload characteristics change.

## What the results mean

METIS demonstrates that adaptation can retain target answer quality while reducing RAG latency and resource use compared with static or exhaustive approaches.

Do not memorize only the headline speedup. Identify which resource stopped being the bottleneck, which resource became the new bottleneck, and whether the evaluation workload makes that transition visible.

## Limitations and critique

Quality proxies and samples may not generalize to new domains. Adaptation has its own measurement cost. A configuration that is optimal in aggregate may still fail rare but important query classes.

## Connection to CS265

This is the direct destination of Classes 3–4 and the Data Calculator philosophy: enumerate design primitives, model cost, and search rather than hand-tune. LOTUS optimizes semantic operators; METIS optimizes the RAG pipeline itself.

## Check your understanding

1. Which RAG knobs affect answer quality without affecting retrieval recall?
2. How would you detect workload drift safely?
3. What objective should replace average quality for high-stakes queries?

If you cannot answer these without looking back, reread the design and evaluation sections rather than the introduction.

## One-paragraph review template

Write four sentences: **(1)** the resource bottleneck, **(2)** the paper’s mechanism, **(3)** the strongest experimental evidence, and **(4)** one workload or assumption where the result may not hold. This is more useful than restating the abstract.
