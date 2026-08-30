# Background readings

These are the core storage-systems surveys CS265 assumes (and the extra papers you added). Each folder has a learning article in `README.md` next to `paper.pdf`.

## Papers you added

| Folder | Paper | Core idea |
| --- | --- | --- |
| [Architecture-of-a-Database-System](Architecture-of-a-Database-System/README.md) | Hellerstein, Stonebraker, Hamilton | DBMS = process model + storage + transactions + query processor |
| [Modern-B-Tree-Techniques](Modern-B-Tree-Techniques/README.md) | Graefe | Production B-trees: concurrency, recovery, utilities, cache-conscious nodes |
| [Column-Oriented-Database-Systems](Column-Oriented-Database-Systems/README.md) | Abadi, Boncz, Harizopoulos, Idreos, Madden | Column layout, compression, late materialization |
| [Massively-Parallel-Databases-and-MapReduce](Massively-Parallel-Databases-and-MapReduce/README.md) | Babu and Herodotou | Shared-nothing partition / shuffle / fault tolerance |
| [Monkey](Monkey/README.md) | Dayan, Athanassoulis, Idreos | Optimal Bloom-filter allocation across LSM levels |

If `Modern-B-Tree-Techniques/paper.pdf` is missing, copy your local `Modern B-Tree Techniques.pdf` into that folder (the publisher copy was not reachable from this environment).

## Related DASlab papers already in this archive

| Folder | Paper |
| --- | --- |
| [Data-Calculator](Data-Calculator/) | Data structure primitives + learned cost models |
| [Design-Continuums](Design-Continuums/) | Design continuums between known structures |
| [Learning-Key-Value-Store-Design](Learning-Key-Value-Store-Design/) | Searching KV-store designs |
