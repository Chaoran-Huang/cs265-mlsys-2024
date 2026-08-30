# FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

**Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré — NeurIPS 2022**

> Compute exact attention in SRAM-sized tiles so the quadratic attention matrix never has to be materialized in GPU HBM.

## Read first

- [Paper PDF](paper.pdf)
- [Course discussion slides](discussion-slides.pdf)
- [Official paper page](https://arxiv.org/abs/2205.14135)

## The problem

Textbook attention computes `S = QKᵀ`, writes the full sequence-by-sequence matrix to HBM, reads it for softmax, writes it again, and then reads it for `softmax(S)V`. The FLOP count gets most of the attention, but on GPUs these repeated HBM transfers are often the real bottleneck. Approximate-attention papers reduce arithmetic yet may not improve wall-clock time because they ignore the memory hierarchy.

## Key insight

FlashAttention changes the execution schedule, not the mathematical function. It partitions Q, K, and V into blocks that fit in fast on-chip SRAM. Each Q block visits K/V blocks, updates a running softmax, and accumulates its output. The full score matrix never lands in HBM. The output is exact up to ordinary floating-point ordering effects.

## How it works

1. Tile Q, K, and V so working data fits in SRAM/shared memory.
2. Use an online softmax: maintain each row’s running maximum and normalization sum while new score blocks arrive.
3. Rescale the partial output whenever the running maximum changes.
4. During backward, recompute score blocks rather than storing the quadratic intermediates.
5. Prove an IO-complexity advantage over standard attention for a range of SRAM sizes.

## What the results mean

The paper reports end-to-end training speedups on BERT and GPT-2 and makes much longer contexts practical. Its most important result is conceptual: reducing HBM reads/writes can matter more than reducing FLOPs. Later FlashAttention versions mainly improve parallel work partitioning and target newer GPU features; they do not replace the core insight.

Do not memorize only the headline speedup. Identify which resource stopped being the bottleneck, which resource became the new bottleneck, and whether the evaluation workload makes that transition visible.

## Limitations and critique

The implementation is hardware-sensitive: tile shapes, SRAM capacity, register pressure, and sequence length affect the best schedule. It is not “free linear attention”; arithmetic remains quadratic in sequence length. For very short sequences, launch overhead or vendor kernels can erase the advantage.

## Connection to CS265

This is the purest example of CS265’s STORE–MOVE–PROCESS framework. It also connects directly to the LLM project’s tiled GEMM and stable softmax. The project’s simple attention materializes scores; a FlashAttention-style fused kernel is the next systems step.

## Check your understanding

1. Why can an algorithm with the same FLOP count run much faster?
2. Derive how online softmax combines two blocks with different row maxima.
3. What does backward recompute, and why is that a good trade?

If you cannot answer these without looking back, reread the design and evaluation sections rather than the introduction.

## One-paragraph review template

Write four sentences: **(1)** the resource bottleneck, **(2)** the paper’s mechanism, **(3)** the strongest experimental evidence, and **(4)** one workload or assumption where the result may not hold. This is more useful than restating the abstract.
