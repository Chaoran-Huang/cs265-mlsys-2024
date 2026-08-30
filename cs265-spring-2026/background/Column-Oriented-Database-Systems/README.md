# The Design and Implementation of Modern Column-Oriented Database Systems

**Daniel Abadi, Peter Boncz, Stavros Harizopoulos, Stratos Idreos, and Samuel Madden — Foundations and Trends in Databases, 2013**

> Store each column densely, then push that layout through compression, late materialization, and vectorized execution so scans become bandwidth-efficient sequential reads.

## Read first

- [Paper PDF](paper.pdf)
- [Publisher / authors’ copy](https://www.cs.umd.edu/~abadi/papers/abadi-column-stores.pdf)

## The problem

Row stores are good at touching a few attributes of one tuple (OLTP). Analytic queries scan few columns of many tuples. If those columns are interleaved in rows, the engine reads mostly unused bytes. Disk and RAM bandwidth, not CPU, become the bottleneck.

## Key insight

Physical layout is an optimizer decision. A column store makes sequential scans over compressed columns cheap, then delays stitching tuples back together (late materialization) until necessary. Compression is not only for space: it reduces I/O and can enable operating directly on encoded values.

## How it works

1. **Column files / disk layout.** One (or a few) files per column, often with block headers and dictionary/RLE/bit-vector encodings.
2. **Compression.** Dictionary, run-length, bit-packing, delta; chosen per column distribution.
3. **Late materialization.** Keep positions (row IDs) through selections and joins; reconstruct tuples late.
4. **Vectorized / block-at-a-time execution.** Operate on vectors that fit in cache rather than one-tuple iterators.
5. **Writes and updates.** Harder than reads: columns are not one-row-at-a-time friendly. Systems use delta stores, positional updates, or periodic merge.

## What to take from it

Column vs row is the same CS265 lesson as JPEG vs a model-specific image format: the storage format should match the access pattern. LLM inference later makes the same move (layout of QKV, KV cache pages, fused kernels).

## Limitations and critique

Pure column stores struggle with wide-tuple point updates and some OLTP. Hybrid and HTAP systems blur the line. Compression that helps scans can hurt if you always need full rows.

## Connection to CS265

This is the analytic sibling of Monkey (write-optimized LSM) and Graefe (B-tree). Self-designing systems would pick row, column, or hybrid given the query mix.

## Check your understanding

1. Why can late materialization reduce work even after you already paid to scan a column?
2. Give a compression scheme that also speeds selection, not just I/O.
3. What write path would you add so a column store can ingest a stream of single-row inserts?

## One-paragraph review

Name the access pattern column stores win on, the three techniques (layout, compression, late materialization) that implement that win, and one workload where you would still choose a row store.
