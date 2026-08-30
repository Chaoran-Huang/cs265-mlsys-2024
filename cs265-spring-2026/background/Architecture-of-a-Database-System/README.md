# Architecture of a Database System

**Joseph M. Hellerstein, Michael Stonebraker, and James Hamilton — Foundations and Trends in Databases, 2007**

> A DBMS is not one algorithm. It is a server architecture: process model, storage, transactions, and a query processor that together decide how data is stored, moved, and processed.

## Read first

- [Paper PDF](paper.pdf)
- [Publisher page](https://www.nowpublishers.com/article/Details/DBS-002)

## The problem

Textbook algorithms (indexes, joins, locking) are not enough to explain why real database systems look the way they do. A DBMS is one of the earliest high-concurrency servers. Its architecture has to answer: who runs a query, how disks and memory are shared, how crashes recover, and how a declarative SQL statement becomes a physical plan.

## Key insight

The same logical database can be implemented with several process models (process-per-connection, thread-per-connection, process pool), several storage layouts, and several optimizer search strategies. Architecture is the set of those design choices and the interfaces between them. Once you see the components, later papers (column stores, LSM-trees, GPU kernels) look like replacements of one component rather than entirely new kinds of system.

## How it works

1. **Process and parallelism model.** Decide how connections map onto OS processes/threads and how shared-memory vs shared-nothing parallelism is used.
2. **Query processor.** Parse SQL, rewrite, optimize (cost-based search over access paths and join orders), then execute iterators or compiled plans.
3. **Storage.** Buffer pool, file organization, indexes, and logging. Pages and replacement policy dominate I/O.
4. **Transactions.** Concurrency (locks, latches, isolation) plus recovery (ARIES-style WAL, checkpoints, redo/undo).
5. **Shared services.** Catalog, admission control, utilities (backup, vacuum, bulk load).

## What to take from it

Read this as a map, not as a novel. After finishing it you should be able to point at any later CS265 paper and say which DBMS component it is replacing: storage layout, access path, optimizer, or execution engine.

## Limitations and critique

It is a 2007 snapshot of classical RDBMS architecture. It does not cover LSM-based KV stores, GPU inference, or RAG. Use it for vocabulary; use Monkey, column-store, and the 2026 papers for the modern instantiations.

## Connection to CS265

Class 1’s STORE–MOVE–PROCESS is this paper’s storage + buffer pool + executor. Self-designing systems later in the course try to *search* among the architectural alternatives this survey only describes.

## Check your understanding

1. Contrast process-per-connection with a thread pool: what fails under many idle clients vs many active queries?
2. Why does a buffer pool exist if the OS already caches files?
3. What is the difference between a latch and a lock?

## One-paragraph review

Name the four subsystems (process, storage, transactions, query processing), one alternative design inside each, and which CS265 topic replaces that subsystem in AI systems (e.g., KV cache ≈ buffer pool).
