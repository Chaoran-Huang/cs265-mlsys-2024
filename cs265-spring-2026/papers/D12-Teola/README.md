# Teola: Towards End-to-End Optimization of LLM-based Applications

**Xin Tan, Yimin Jiang, Yitao Yang, Hong Xu — ASPLOS 2025**

> Optimize the entire DAG of LLM calls and tools rather than making each individual model invocation fast in isolation.

## Read first

- [Paper PDF](paper.pdf)
- [Official paper page](https://arxiv.org/abs/2407.00326)

## The problem

LLM applications are pipelines or DAGs: retrieve, call a model, parse, branch, invoke tools, and call another model. Serving engines optimize each call but ignore dependencies and cross-stage opportunities. Local batching or placement decisions can therefore harm end-to-end latency.

## Key insight

Teola captures the application’s dataflow and schedules it as a whole. Cross-stage batching, pipelining, parallel execution, and resource allocation become optimizer decisions. The application graph—not one forward pass—is the unit of optimization.

## How it works

1. Express an LLM application as operators and dependencies.
2. Track data readiness and schedule independent branches concurrently.
3. Batch compatible work across application instances.
4. Pipeline CPU/tool stages with GPU model work.
5. Allocate resources according to end-to-end critical paths.

## What the results mean

The evaluation shows end-to-end improvements over systems that optimize model serving alone, especially for multi-stage applications with parallelism and repeated operator patterns.

Do not memorize only the headline speedup. Identify which resource stopped being the bottleneck, which resource became the new bottleneck, and whether the evaluation workload makes that transition visible.

## Limitations and critique

The optimizer needs visibility into application structure, costs, and side effects. External tools have variable latency, and nondeterministic control flow makes static planning difficult. More aggressive batching may conflict with per-request latency.

## Connection to CS265

Teola is the system response to Class 7’s glue-engineering diagnosis. SGLang supplies a structured programming model; Teola emphasizes whole-application scheduling. Both move optimization above the single model server.

## Check your understanding

1. Give an example where locally optimal batching worsens end-to-end latency.
2. How should the scheduler model unpredictable tool latency?
3. Which application branches can execute speculatively?

If you cannot answer these without looking back, reread the design and evaluation sections rather than the introduction.

## One-paragraph review template

Write four sentences: **(1)** the resource bottleneck, **(2)** the paper’s mechanism, **(3)** the strongest experimental evidence, and **(4)** one workload or assumption where the result may not hold. This is more useful than restating the abstract.
