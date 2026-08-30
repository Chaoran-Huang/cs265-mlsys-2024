# Semantic Operators and Their Optimization: Enabling LLM-Based Data Processing with Accuracy Guarantees in LOTUS

**Liana Patel et al. — VLDB 2025**

> Make LLM-powered map, filter, and join operations declarative, then optimize their cost while controlling accuracy.

## Read first

- [Paper PDF](paper.pdf)
- [Official paper page](https://www.vldb.org/pvldb/vol18/p4171-patel.pdf)

## The problem

Developers increasingly use LLM calls as predicates or transformations over unstructured data. Naive loops are expensive, hard to compose, and provide no meaningful accuracy contract. Classical relational optimizers assume deterministic operators, an assumption an LLM violates.

## Key insight

LOTUS defines semantic operators for LLM-based data processing and gives the system enough structure to optimize their execution. It can use cheaper models, cascades, filters, batching, and sampling while reasoning about the accuracy of the resulting plan.

## How it works

1. Expose semantic map/filter/join-style operators over table columns.
2. Represent the application as a logical operator plan.
3. Estimate operator accuracy and cost using sampled data.
4. Select physical implementations such as model cascades or retrieval-assisted execution.
5. Provide probabilistic accuracy guarantees rather than pretending calls are deterministic.

## What the results mean

LOTUS makes semantic data processing substantially cheaper and faster than straightforward LLM execution while meeting specified quality targets on evaluated workloads.

Do not memorize only the headline speedup. Identify which resource stopped being the bottleneck, which resource became the new bottleneck, and whether the evaluation workload makes that transition visible.

## Limitations and critique

Guarantees inherit assumptions from samples and model stability. Distribution shift can invalidate estimates, and semantic predicates may be genuinely ambiguous. This is not the absolute correctness offered by normal relational algebra.

## Connection to CS265

LOTUS brings Class 1’s declarative interface to AI data processing: users state what semantic operation they want; the system selects how to execute it. Compare D11 METIS, which optimizes a RAG configuration rather than a table-operator plan.

## Check your understanding

1. What does an accuracy guarantee mean for a subjective semantic predicate?
2. Why can a cheap model followed by an expensive model beat either alone?
3. Which classical query-optimizer rules remain valid for probabilistic operators?

If you cannot answer these without looking back, reread the design and evaluation sections rather than the introduction.

## One-paragraph review template

Write four sentences: **(1)** the resource bottleneck, **(2)** the paper’s mechanism, **(3)** the strongest experimental evidence, and **(4)** one workload or assumption where the result may not hold. This is more useful than restating the abstract.
