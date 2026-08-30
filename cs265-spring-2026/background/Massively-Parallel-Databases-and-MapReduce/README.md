# Massively Parallel Databases and MapReduce Systems

**Shivnath Babu and Herodotos Herodotou — Foundations and Trends in Databases, 2013**

> Shared-nothing parallelism is the default at scale: partition the data, push operators to the partitions, and pay for shuffle only when the plan requires it.

## Read first

- [Paper PDF](paper.pdf)

## The problem

One machine cannot scan or join modern datasets. Parallel databases existed long before Hadoop; MapReduce then made cluster data processing widely programmable. The systems question is how to partition, schedule, tolerate failures, and optimize plans when communication is the scarce resource.

## Key insight

Both parallel RDBMSes and MapReduce-style systems implement the same skeleton: **partition**, **local compute**, **shuffle**, **reduce**. They differ in interface (SQL vs map/reduce functions), optimizer power, schema, and fault-tolerance granularity. Understanding the shared skeleton lets you see Spark, distributed training, and LLM serving as later instances.

## How it works

1. **Partitioning.** Hash, range, or random partitions of tables/files. Co-partitioning enables local joins.
2. **Parallel operator plans.** Each operator runs as many tasks; the optimizer (or programmer) inserts shuffles.
3. **Scheduling.** Assign tasks to machines that already hold the data (locality).
4. **Fault tolerance.** Restart tasks, replicate inputs, or lineage-recompute (MapReduce/Spark) vs clustered RDBMS recovery.
5. **Interfaces.** Declarative SQL with a cost-based optimizer vs user-defined map/reduce with limited optimization.

## What to take from it

When CS265 later talks about tensor/data parallelism, ZeRO, or DistServe’s two GPU pools, you are still looking at partitioning + communication vs compute. The vocabulary from this survey transfers.

## Limitations and critique

The 2013 landscape predates Spark SQL, cloud warehouses, and GPU clusters. The taxonomy remains useful; the product names do not.

## Connection to CS265

Distributed training (D03 ZeRO, D04 FSDP) is “massively parallel” applied to parameters and gradients instead of SQL tables. DistServe (D06) is partitioning by *phase* (prefill vs decode) rather than by key.

## Check your understanding

1. When does a hash join in a shared-nothing cluster require a shuffle?
2. Why is data locality a scheduling problem, not only a storage problem?
3. Contrast restarting a MapReduce task with ARIES-style DBMS recovery.

## One-paragraph review

Describe the partition–compute–shuffle loop, one place SQL engines automate it, one place MapReduce leaves it to the user, and how that shows up in ML systems.
