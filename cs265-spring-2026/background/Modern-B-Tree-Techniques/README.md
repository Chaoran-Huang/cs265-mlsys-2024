# Modern B-Tree Techniques

**Goetz Graefe — Foundations and Trends in Databases, 2011**

> A production B-tree is far more than “balanced tree + binary search.” Concurrency, recovery, bulk utilities, and cache-conscious node layout are what make it the default index in databases.

## Read first

- Place your PDF here as [`paper.pdf`](paper.pdf) if it is not already present.
- [Publisher page](https://www.nowpublishers.com/article/Details/DBS-028)

A public copy of this monograph was not fetchable from this environment (Now Publishers / Microsoft Research hosts returned 403/404). If you uploaded `Modern B-Tree Techniques.pdf` locally, copy it into this folder as `paper.pdf`.

## The problem

Everyone knows B-trees from algorithms class. Industrial indexes still have to: search under many concurrent readers and writers, recover after a crash mid-split, bulk-load without locking the world, pack variable-length keys, and stay cache-conscious as node size and CPU caches change. Missing those techniques is why a homework B-tree is not a storage engine.

## Key insight

Separate **logical contents** from **physical representation**. Splitting a node, posting a separator, or moving a page in the buffer pool changes representation, not the set of keys. That split of concerns enables ghost records, physiological logging, non-logged page operations, and online index build.

## How it works (what to look for)

1. **Node layout.** Prefix compression, normalized keys, interpolation vs binary search, node size vs cache lines.
2. **Concurrency.** Latches vs locks, latch coupling, B-link trees, key-range locking, ghost (pseudo-deleted) records.
3. **Recovery.** Physiological logging: log physical page changes with enough logical context to redo/undo correctly.
4. **Query processing.** Covering indexes, index-only plans, index nested loops, ordered access vs heap fetches.
5. **Utilities.** Bulk load, rebuild, defragment, online create/drop.

## What to take from it

Whenever CS265 talks about “access path selection” or compares B-trees with LSM-trees, this survey is the B-tree side of the design space: read-optimized, in-place updates, rich concurrency.

## Limitations and critique

It is a B-tree-centric monograph. Write-heavy cloud KV stores often prefer LSM (see Monkey). Treat LSM vs B-tree as a continuum of merge vs in-place update, not a morality play.

## Connection to CS265

Class 2’s self-designing / periodic-table story: B-tree nodes are one composition of layout primitives (fanout, prefix, compression). GPU tiling in FlashAttention is the same idea one level down the memory hierarchy.

## Check your understanding

1. Why might interpolation search beat binary search inside a large node—and when does skew destroy that win?
2. What problem do ghost records solve for delete + concurrency?
3. Why can a node split be logged differently from inserting a user row?

## One-paragraph review

State one concurrency trick, one recovery trick, and one query-processing trick that a textbook B-tree omits, and why each exists.
