# Baselines

Other vector indexes that the benchmark harness in `coldivf-bench/py` runs next to ColdIVF. This is its own Cargo
workspace with its own `target/`, so none of these dependencies reach ColdIVF's builds or checks.

```
cargo build --release
cargo test --release
cargo clippy --release --all-targets -- -D warnings
```

## diskann3

DiskANN from the `diskann` and `diskann-disk` 0.60.0 crates, with readers for macOS. It builds `libdiskann3.dylib`,
which `adapters/diskann.py` calls through ctypes, and the `diskann3` tool, which builds indexes.

| File | What It Holds |
|---|---|
| `src/builder.rs` | Builds a disk index from rows 0 to N of a base `.fbin` with DiskANN's builder, and writes `meta.json` |
| `src/prefix.rs` | A storage provider that serves rows 0 to N of a base file as a file of their own, so nothing is copied |
| `src/mac_reader.rs` | The readers that serve DiskANN's sector reads, and the threads of the pool reader |
| `src/search.rs` | Opens an index with one reader and searches it, one query or a batch on several threads |
| `src/ffi.rs` | The C interface for the Python adapter |
| `src/log.rs` | Prints DiskANN's log events and keeps the build's stage times |
| `src/usage.rs` | The process's CPU time, disk bytes and peak footprint for the build report |
| `src/bin/diskann3.rs` | `diskann3 build` and a quick `diskann3 search` check |

DiskANN's stock macOS reader copies the whole index file into memory every time the searcher makes a reader, and the
searcher makes several. So searches go through one of these readers, passed to `DiskVertexProviderFactory`.

| Reader | How It Reads a Beam |
|---|---|
| `pool` | Every 4 KiB sector at once, the first on the calling thread and the rest on a pool of threads, through a handle with `F_NOCACHE` |
| `advise` | `F_RDADVISE` for every sector, which starts them all at once, then a read of each in order, through the same kind of handle |
| `stock` | DiskANN's own reader, built once per index and shared. It copies the whole file at open, so it serves only as a control |

An index is a folder holding `index_disk.index`, `index_pq_pivots.bin`, `index_pq_compressed.bin` and `meta.json`.
`--build-pq-bytes N` builds the graph on N-byte PQ codes of the rows instead of the full vectors, which holds only the
codes in memory, and adds the build's own codebooks, `index_build_pq_pivots.bin`. Searches don't use them.

```
diskann3 build --data cohere-768-cosine.base.fbin --rows 50000 --out INDEX --threads 4
diskann3 search --index INDEX --queries cohere-768-cosine.query.fbin --truth truth.ibin --reader pool --list 100 --beam 4
```
