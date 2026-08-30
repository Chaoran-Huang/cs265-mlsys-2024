# Discussion papers and learning guides

Each folder contains the paper PDF, a plain-English learning article, and course discussion slides when Harvard published them.

| Discussion | Paper | Core idea |
| --- | --- | --- |
| D01 | [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](D01-FlashAttention/README.md) | Compute exact attention in SRAM-sized tiles so the quadratic attention matrix never has to be materialized in GPU HBM. |
| D02 | [TVM: An Automated End-to-End Optimizing Compiler for Deep Learning](D02-TVM/README.md) | Separate what a tensor program computes from how it is scheduled, then search for a fast schedule on each target. |
| D03 | [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](D03-ZeRO/README.md) | Keep data parallelism’s compute pattern but shard its redundant optimizer states, gradients, and parameters across workers. |
| D04 | [PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel](D04-FSDP/README.md) | Turn ZeRO-3-style full sharding into a composable PyTorch feature and document the engineering choices that make it usable. |
| D05 | [Efficient Memory Management for Large Language Model Serving with PagedAttention](D05-PagedAttention/README.md) | Manage the KV cache like virtual memory so variable-length requests do not waste contiguous GPU allocations. |
| D06 | [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving](D06-DistServe/README.md) | Run prefill and decode on different GPU groups because their resource profiles and latency objectives conflict. |
| D07 | [Mooncake: Trading More Storage for Less Computation — A KVCache-centric Architecture for Serving LLM Chatbot](D07-Mooncake/README.md) | Treat KV cache as durable, tiered serving data rather than disposable GPU scratch space. |
| D08 | [Mirage: A Multi-Level Superoptimizer for Tensor Programs](D08-Mirage/README.md) | Search equivalent tensor programs across graph, thread-block, and thread levels rather than applying a fixed list of compiler heuristics. |
| D09 | [SGLang: Efficient Execution of Structured Language Model Programs](D09-SGLang/README.md) | Give multi-call LLM applications a programming model and reuse KV prefixes through a radix-tree cache. |
| D10 | [Semantic Operators and Their Optimization: Enabling LLM-Based Data Processing with Accuracy Guarantees in LOTUS](D10-LOTUS/README.md) | Make LLM-powered map, filter, and join operations declarative, then optimize their cost while controlling accuracy. |
| D11 | [METIS: Fast Quality-Aware RAG Systems with Configuration Adaptation](D11-METIS/README.md) | Adapt the RAG pipeline configuration per workload instead of manually freezing chunking, retrieval, reranking, and generation choices. |
| D12 | [Teola: Towards End-to-End Optimization of LLM-based Applications](D12-Teola/README.md) | Optimize the entire DAG of LLM calls and tools rather than making each individual model invocation fast in isolation. |
| D13 | [S-LoRA: Serving Thousands of Concurrent LoRA Adapters](D13-SLoRA/README.md) | Share one base model while paging and batching thousands of small LoRA adapters with their KV caches. |
