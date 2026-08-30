# DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving

**Yinmin Zhong et al. — OSDI 2024**

> Run prefill and decode on different GPU groups because their resource profiles and latency objectives conflict.

## Read first

- [Paper PDF](paper.pdf)
- [Course discussion slides](discussion-slides.pdf)
- [Official paper page](https://arxiv.org/abs/2401.09670)

## The problem

Prefill processes a full prompt with large matrix multiplications and is relatively compute-intensive. Decode processes one token per request, repeatedly streams model weights and KV state, and is usually memory-bandwidth-bound. Mixing both phases on the same GPUs causes interference: a long prefill can damage inter-token latency, while decode work can reduce prefill efficiency.

## Key insight

DistServe separates prefill and decode into independently provisioned instances. After prefill, it transfers the request’s KV cache to a decode instance. Placement and parallelism are selected to maximize goodput—the number of requests meeting both time-to-first-token and time-per-output-token SLOs.

## How it works

1. Profile prefill and decode separately instead of treating inference as one workload.
2. Provision separate replicas and parallelism strategies for each phase.
3. Transfer KV cache between the two GPU pools.
4. Search configurations under TTFT and TPOT constraints.
5. Use goodput rather than raw tokens/second as the objective.

## What the results mean

Under latency SLOs, disaggregation serves substantially more successful requests than colocated baselines. Independent scaling is particularly valuable when prompt and output-length distributions change.

Do not memorize only the headline speedup. Identify which resource stopped being the bottleneck, which resource became the new bottleneck, and whether the evaluation workload makes that transition visible.

## Limitations and critique

KV transfer consumes network bandwidth and adds a stage boundary. Small prompts or a slow interconnect can eliminate the win. Load imbalance between pools requires admission control and careful provisioning.

## Connection to CS265

This is self-designing physical planning at cluster scale. D05 fixes local KV allocation; D06 decides where phases execute; D07 builds a storage hierarchy to move and reuse KV more efficiently.

## Check your understanding

1. Why is tokens/second an insufficient serving metric?
2. Under what workload would colocation beat disaggregation?
3. Estimate KV-transfer size from layers, heads, head dimension, tokens, and dtype.

If you cannot answer these without looking back, reread the design and evaluation sections rather than the introduction.

## One-paragraph review template

Write four sentences: **(1)** the resource bottleneck, **(2)** the paper’s mechanism, **(3)** the strongest experimental evidence, and **(4)** one workload or assumption where the result may not hold. This is more useful than restating the abstract.
