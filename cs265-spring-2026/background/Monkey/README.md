# Monkey: Optimal Navigable Key-Value Store

**Niv Dayan, Manos Athanassoulis, and Stratos Idreos — SIGMOD 2017**

> Worst-case LSM lookup I/O is proportional to the *sum* of Bloom-filter false-positive rates across levels. Allocate filter bits unevenly—tighter filters on larger levels—to minimize that sum.

## Read first

- [Paper PDF](paper.pdf)
- [Project page](http://daslab.seas.harvard.edu/monkey/)

## The problem

LSM-trees (LevelDB, RocksDB, Cassandra) buffer writes in memory, flush sorted runs, and merge them into levels. Lookups may probe many levels. Bloom filters skip empty levels, but every production store used the same bits-per-key at every level. That wastes memory where it would reduce I/O most, and it makes the lookup/update/memory tradeoff both suboptimal and hard to tune.

## Key insight

A point lookup’s extra I/Os come from false positives. The expected extra I/O is (roughly) the **sum of FPRs** over levels you must check. Because level sizes grow exponentially, giving every level the same FPR is the wrong optimization. Monkey sets FPR proportional to the number of entries in the level (exponentially smaller FPR at larger, colder levels), which removes an \(O(L)\) factor from worst-case lookup I/O.

It then writes closed-form costs for lookups, updates, and memory so you can **co-tune** merge policy, size ratio, buffer, and filters for a workload and device.

## How it works

1. Model LSM levels, merge frequency (leveling vs tiering), and Bloom filters.
2. Show lookup I/O ~ \(\sum_i \mathrm{FPR}_i\) (plus the run that actually contains the key).
3. Allocate a memory budget across filters to minimize that sum (not uniform bits-per-key).
4. Plug the optimal filters into cost formulas for updates vs lookups.
5. Navigate the design space: more merging (better reads, worse writes) vs more memory vs different size ratios.

## What to take from it

This is the Data Calculator idea in one artifact: **closed-form cost + design knobs ⇒ automatic tuning.** It is the NoSQL counterpart of METIS for RAG.

## Limitations and critique

The model is worst-case / analytical; skewed key distributions, range queries, and compaction storms need extra care. Implementation on LevelDB/RocksDB must still deal with real compaction scheduling. Range scans do not get the same Bloom-filter skip.

## Connection to CS265

Classes 1–2 and the NoSQL project. Self-designing storage: do not pick “RocksDB defaults”; pick a point on the LSM continuum. GPU papers later optimize a different hierarchy, but the method—name the scarce resource, write a cost, search knobs—is the same.

## Check your understanding

1. Why does a uniform bits-per-key policy overspend on small young levels?
2. Sketch why \(\sum \mathrm{FPR}\) appears in lookup I/O.
3. How would you retune Monkey if the workload shifted from 90% writes to 90% point reads?

## One-paragraph review

State the three-way tradeoff (lookup, update, memory), the FPR-allocation insight, and one what-if question the closed-form model can answer without reimplementing the store.
