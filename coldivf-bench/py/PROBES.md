# Feasibility Probes

Measured on the M3 Pro (36 GiB of memory, 16 KiB pages) on 2026-10-08. Other jobs ran at the same time, so times are rough. Byte counts, page counts and footprints are exact. Unless a row says otherwise, the data is the first 1M rows of cohere10m-768-cosine. Recall is recall@10 on ids, against the prefix ground truth, on test-split queries (odd positions). GB means 10^9 bytes and MB 10^6. The probe scripts and their raw JSON lines are in runs/probes/, which git ignores.

## Installs

| Package | Version | Environment | Note |
|---|---|---|---|
| rabitqlib | 0.5.2 | Main venv | |
| usearch | 2.26.4 | Main venv | Brings numkong 7.8.5 |
| aster-vec | 0.2.2 | Main venv | Imports as `astervec` |
| turbovec | 1.1.2 | Main venv | |
| sqlite-vec | 0.1.9 | Main venv | |
| paimon-vindex | 0.5.0 | Main venv | |
| qdrant-edge-py | 0.8.0 | Main venv | Imports as `qdrant_edge` |
| lancedb, pylance | 0.40.0, 13.0.0 | Separate lance13 venv | They need lance-namespace 0.11.1, and the main venv has 0.13.0 |

Every requested version exists. Each dry run for the main venv listed only new packages, so nothing installed changed.

## Eviction and Residency

**msync with MS_INVALIDATE through a second mapping drops pages that a live process maps.** On a 1 GB file that a holder process mapped and touched, the driver's invalidation left 0 of 65,536 pages resident, seen from both mappings. The holder then read 1.07 GB back from disk (65.5K page-ins, ~1.0 s). This held for MAP_SHARED and MAP_PRIVATE holders and for a holder that only used pread. Invalidating one byte range dropped exactly that range (32,760 pages) and left the other 32,776 resident. A process can't drop pages through its own MAP_PRIVATE mapping, since that left all 65,536 resident.

**Mappings held for the whole batch keep page 0 out at GO, the driver's signal that starts a trial.** A fresh process mapped and read the file before each of 30 tries, as a trial would. Then the driver evicted, waited, evicted again and checked every page.

| Method | Tries With Pages Resident at GO | Tries Where Page 0 Came Back After GO |
|---|---|---|
| A fresh mapping for each eviction and check | 3 of 30 (1 GB file) | 30 of 30, 9 to 345 ms later |
| Held mappings, 0.5 s between evictions | 0 of 30 | 1 to 2 of 30, 0.54 to 0.70 s after the trial process ended |
| Held mappings, 1.0 s between evictions | 0 of 30 | 0 of 30, watched for 1 s |

Files were a 48 MB ColdIVF index and a 1 GB random file. Only the 1 GB file had pages resident at the check with fresh mappings. The late reads come from another process, likely the endpoint agent, which reads the first pages of a file soon after it is mapped.

**An APFS (Apple File System) clone starts 0% resident.** `cp -c` of a fully cached 1 GB file took 17 ms, and the clone had 0 of 65,536 pages cached. Reading it read 1.07 GB from disk, the same as an evicted file, while the source stayed cached. A new file doesn't stay cold on its own, though.
- In a normal directory, Spotlight read the first 2 MB of each new 8 MB file within ~0.5 s.
- In a directory whose name ends in `.noindex`, or below one, only pages 0 and 1 of each file came back.

**One open() costs ~120 µs today.** stat and a failing open cost 1.1 to 1.5 µs, so nearly all of it is the endpoint agent.

| Case | p50 (µs) | p99 (µs) |
|---|---|---|
| open and close of one existing file, 2,000 times | 120 | 536 |
| Existing ColdIVF index files | 115 | 140 |
| First open of a new small file | 653 | 1,252 |
| Second open of that file | 121 | 305 |

## RaBitQ-Library

IvfIndex.build of the IVF (inverted file index) takes the memmap without copying it, so the peak footprint is about the index size plus ~70 MB. It ran with 6,250 lists (160 vectors per list) from a shared faiss k-means partition, metric "ip" and 4 threads. The index loads at open, and its footprint grows by the file size.

| Bits | Build Seconds | Peak Footprint (GB) | File Bytes per Vector | Recall at nprobe 64 |
|---|---|---|---|---|
| 1 | 0.6 | 0.215 | 143.8 | 0.7250 |
| 5 | 25.3 | 0.607 | 535.8 | 0.9255 |
| 9 | 118.4 | 0.992 | 919.8 | 0.9445 |
| 32 (raw 32-bit float rerank) | 0.9 | 3.288 | 3,215.8 | 0.9460 |

By the same bytes per vector, 10M needs ~1.5 GB at 1 bit, ~5.5 GB at 5 bits and ~9.3 GB at 9 bits. 20M at 5 bits needs ~10.8 GB.

HnswIndex.build (M 16, ef_construction 200, 4 bits) on the first 200K rows took 74.5 s on 4 threads. It peaked at 0.22 GB, wrote 636.5 bytes per vector and reached recall 0.908 at ef 100. HnswIndex.build also takes centroids and cluster ids.

## AsterVec

bulk_build on 1M rows ran with defaults (m 8, ef_construction 32) plus cosine and 4 threads.
- **Time:** 721 s (1,388 vectors/s) and 2,705 CPU (central processing unit) seconds.
- **Memory:** a peak footprint of 5.58 GB, then 1.34 GB held after the build.
- **Disk:** 1,875 bytes per vector in 26 files: vectors.bin 1,028 B, lsm_vec_db.meta 416 B, and RocksDB's log and tables.
- **Recall:** 0.908 at ef 64 and 0.942 at ef 128.

At 50K it took 26.9 s and peaked at 0.253 GB. The peak is ~5.5 KB per vector, so 2M needs ~11 GB. open fails with "failed to open log file" unless the database directory exists or log_file_path is set. bulk_build logs at INFO level to stderr.

## faiss 1.15.1

The IVF indexes use the shared partition behind a 768-d random rotation (RR768), with 1-bit RaBitQ codes and 8-bit scalar quantization (SQ8) for refine. The HNSW (hierarchical navigable small world) indexes use HNSW32 on the first 200K rows. Each case evicted its files, then a fresh process opened the index and ran queries at 1 thread with nprobe 64, k_factor 4 or efSearch 64. "IFC" is IO_FLAG_MMAP_IFC and "MMAP" is IO_FLAG_MMAP.

| Index and Flags | Stays Mapped | Read at Open | Footprint Change at Open | First Query Read | First and Second Query (ms) |
|---|---|---|---|---|---|
| Rotation + IVF RaBitQ, IFC | Yes | 2.44 MB | +3.0 MB | 22.0 MB | 48, 17 |
| Rotation + IVF RaBitQ, MMAP | Lists only | 38.4 MB | +21.6 MB | 4.7 MB | 4.8, 4.4 |
| Rotation + IVF RaBitQ, no flag | No | 133.6 MB | +152 MB | 0 | 1.1, 1.1 |
| IndexRefine(that, SQ8), IFC | Yes, both parts | 2.48 MB | +2.8 MB | 22.7 MB | 63, 21 |
| IndexRefine(that, SQ8), MMAP | Lists only | 806 MB | +790 MB | 4.9 MB | 4.0, 2.6 |
| IndexRefine(that, SQ8), MMAP and IFC together | Fails to open | | | | |
| On-disk lists, plus a separate SQ8 index read with IFC | Lists and SQ8 | 21.7 MB | +21.6 MB | 5.3 MB | 7.8, 8.3 |
| Rotation + IVF RaBitQ FastScan, IFC | No | 126.6 MB | +162 MB | 19.2 MB | 33, 1.0 |
| HNSW32 + 4-bit RaBitQ + SQ8 refine, IFC | Codes only | 54.5 MB | +1.7 MB | 29.4 MB | 169, 79 |
| HNSW32 + SQ8 (IndexHNSWSQ), IFC | Codes only | 54.4 MB | +1.6 MB | 33.3 MB | 158, 122 |

- With IFC the rotation matrix (2.36 MB) is copied. The 19.2 MB coarse table stays mapped, and the first query faults all of it in one page at a time.
- MMAP turns the lists into OnDiskInvertedLists over the same file and copies the coarse table. It loads the SQ8 refine part.
- MMAP and IFC together fail in read_index with "mmap only supported for File objects".
- On-disk lists come from faiss.contrib.ondisk.merge_ondisk and open without flags. OnDiskInvertedLists prefetches the probed lists on up to 32 threads (prefetch_nthread), which is why the MMAP and on-disk rows answer in 4 to 8 ms. With IFC one thread faults the lists in, ~17 to 21 ms for 3 MB.
- FastScan copies its codes into BlockInvertedLists at open, with IFC or MMAP alike.
- The HNSW open touches the whole graph section (levels, offsets and neighbor lists, 272 bytes per vector at M 32), but keeps it file-backed. The codes after it stay untouched.
- Build at 1M, 4 threads: 15.8 s to add 1-bit codes, 14.6 s for FastScan, a lifetime peak of 2.86 GB with SQ8 held too. The 200K HNSW adds took 72 s with RaBitQ storage and 29 s with SQ8, at efConstruction 40.

## DiskANN Crates 0.60.0

`cargo check --target aarch64-apple-darwin` passes for diskann, diskann-disk, diskann-providers and diskann-vector 0.60.0. It checked 112 crates in 23.5 s with no warnings, including a test crate whose own pread reader plugs into the searcher. The reader traits are below.

```rust
// diskann_disk::search::provider::aligned_file_reader::traits
pub trait AlignedFileReader: Send + Sync {
    type Alignment: Alignment; // A1 or A512
    fn read(&mut self, read_requests: &mut [AlignedRead<u8, Self::Alignment>]) -> ANNResult<()>;
}
pub trait AlignedReaderFactory: Send + Sync {
    type AlignedReaderType: AlignedFileReader;
    fn build(&self) -> ANNResult<Self::AlignedReaderType>;
}
```

- AlignedRead has `offset()` and `aligned_buf_mut()`. `std::io::Error` converts to ANNError with `?`.
- `DiskVertexProviderFactory::new(reader_factory, CachingStrategy::None)` takes a custom factory.
- `DiskIndexSearcher::new(num_threads, search_io_limit, &DiskIndexReader, factory, Metric, Option<Runtime>)`, where `DiskIndexReader::new(pq_pivots, pq_compressed, &FileStorageProvider)` loads the PQ (product quantization) data. Searches go through `searcher.search(query, k, list_size, Some(beam_width), SearchMode::Graph { filter: None })`. The data type is `AdHoc<f32>`.
- On macOS the stock AlignedFileReaderFactory::build reads the whole index file into a Vec on every call. DiskIndexSearcher::new calls get_header twice, and each call builds a reader. Each search scratch builds one more.

## qdrant-edge-py 0.8.0

The shards came from the first 20K or 100K rows of the 1M dev set (cohere-768-cosine), with dot distance.

- **Placement:** only the older flags exist. These are on_disk on EdgeVectorParams and HnswIndexConfig, on_disk_payload on EdgeConfig, and always_ram on the scalar, product, binary and TurboQuant configs. There is no newer placement option.
- **Inline storage:** HnswIndexConfig(inline_storage=True) works with on-disk HNSW and 2-bit binary codes. At m 16 the graph file holds 8,032 bytes per vector.
- **One segment:** EdgeOptimizersConfig(default_segment_number=1, max_segment_size=10^9) puts every point in one indexed segment. qdrant keeps a second, empty appendable segment of 12 files (35.7 MB preallocated).
- **Indexing threads:** HnswIndexConfig(max_indexing_threads=N) is honored. At 100K, optimize used 0.99 CPU per wall second with 1 thread (16.7 s), and 3.52 with 4 threads (5.2 s). EdgeConfig also takes max_search_threads and search_pool_core.
- **WAL (write-ahead log):** it keeps every upsert, ~3.9 KB per point, and flush, optimize, close and reload never truncate it. load reads all of it. At 100K a load read 414 to 470 MB in ~0.53 s, ~387 MB of it WAL. After the final flush and close, removing wal/ and loading once gives a fresh empty WAL, with all points and the same recall.
- Points go in as Python lists through UpdateOperation.upsert_points, ~1.7 s per 100K points.

A cold load at 100K, after the WAL was removed:

| Configuration | Load (ms) | Read at Load | Footprint Change at Load | First Query |
|---|---|---|---|---|
| SQ8 codes always in RAM, vectors on disk | 128 | 83 MB (all codes, graph links, ID tracker) | +82 MB | 3.4 ms, 0.4 MB |
| 2-bit binary codes always in RAM, vectors on disk | 67 | 27 MB | +24 MB | 2.8 ms, 0.4 MB |
| Inline storage, 2-bit codes, everything on disk | 23 | 2.4 MB (ID tracker) | +4.8 MB | 8.7 ms, 2.1 MB in 131 page-ins |

Recall at 20K with ef 64 and rescoring at oversampling 2 was 0.956 for SQ8, 0.778 for 2-bit codes in RAM and 0.941 for inline storage.

## Lance 13 (lancedb 0.40.0, pylance 13.0.0)

The probe built a 50K table from the 1M dev set with IVF_RQ at 1 bit, 256 partitions and dot distance. Ingest took 0.07 s and the index 6.9 s. The table has 8 files: one 153.7 MB data file, index.idx (0.79 MB), auxiliary.idx (5.5 MB), two manifests, two transaction files and a version hint.

A cold query with nprobes 32 and refine_factor 4 touched 4 files: the manifest, index.idx, auxiliary.idx and the data file. Opening read 4 KB in 1.3 to 4.6 ms. The first query read ~3.8 MB in 14 to 33 ms, and the second 1.5 MB in ~5 ms. lancedb.connect plus open_table plus to_lance read the same bytes as lance.dataset. analyze_plan reported four stages.

| Plan Stage | bytes_read | iops |
|---|---|---|
| Ranking the 256 partitions (index.idx) | 797.2 K | 5 |
| Loading 32 partitions, parts_loaded 32 | 1.50 M | 215 |
| Refine rows (40 vectors) | 127.0 K | 41 |
| Fetching the id column | 18.1 K | 6 |

ScannerBuilder.nearest in pylance 13 takes the arguments below.
- `query_parallelism`: 0 is the automatic policy, which today means one sequential worker. -1 uses the CPU pool, and 2 or more use the partition-parallel path.
- `approx_mode`: "fast", "normal" (the default) or "accurate". It only affects RQ (RaBitQ quantization) indexes.
- `minimum_nprobes`, `maximum_nprobes`, `refine_factor`, `ef`, `use_index` and `distance_range`.

Warm, at nprobes 32 and refine 4, approx_mode left recall at 0.9605 and changed latency (1.98, 3.27 and 6.93 ms for fast, normal and accurate). Without refine, recall was 0.7465. lancedb 0.40's Table.create_index treats its first argument as the column name when config= is given, eg. `create_index("vector", config=IvfRq(num_bits=1))`. The legacy form defaults to num_bits 8.

## Paimon Vector Index 0.5.0

The build options were index.type ivf_rq, nlist 6250, metric inner_product, ivf.coarse-assignment exact and 4 Rayon threads. The trainer keeps a reservoir of max(65,536, 64 × nlist) vectors, 400K here.

| rq.bits | Peak Footprint (GB) | Footprint After Adds (GB) | Seconds | File Bytes per Vector | Recall at nprobe 64 |
|---|---|---|---|---|---|
| 4 | 3.01 (during training) | 0.71 | 49 | 425.4 | 0.8585 |
| 8 | 3.00 (during training) | 1.24 | 46 | 809.4 | 0.8785 |

The writer holds ~1.5 times its file size, so at 10M it needs ~6.5 GB at 4 bits and ~12 GB at 8 bits. Training at 10M needs a sample fed to the trainer, because 64 × nlist vectors at 160 per list is 4M vectors (12.3 GB).

The reader is `VectorIndexReader(input, memory_budget_bytes=4 GiB)`. The input object provides `pread_many(ranges)`, which returns one bytes-like object per (offset, length). It can also give the hints estimated_random_read_latency_nanos, preferred_window_bytes and max_ranges_per_read. Each round is one Python call, and the binding copies every chunk with memmove. A cold first query reads in three rounds.
1. Open reads the 64-byte header (3 ms).
2. The first query reads the centroids and offset table in one range (19.3 MB at 1M).
3. It then reads all probed lists in one round (64 ranges, 4.8 MB at 4 bits, 9.3 MB at 8 bits).

With a serial pread loop, the first query took 103 ms at 4 bits and the second 28 ms. read_plan only works for DiskANN indexes.

## Process Counters

- proc_pid_rusage reports times in mach ticks, 125/3 ns each on Apple silicon, not in nanoseconds.
- Its per-QoS (quality of service) CPU times put every second of CPU in the legacy bucket. This held for a USER_INTERACTIVE main thread and a USER_INITIATED worker in a command-line process, so they can't show which class a library's threads ran in.
- P-core (performance core) time in user_ptime, cycles and energy_nj do work.
- The lifetime peak footprint from rusage catches spikes that 1 s sampling misses, eg. 2.86 GB against 1.35 GB for the faiss build.
- A footprint sampler thread in Python sees nothing while a binding holds the GIL (global interpreter lock), as several RaBitQ-Library builds did.

## Changes for the Adapters

- **Evictor:** keep held mappings, and wait 1.0 s between the two evictions, not 0.5 s. C1, where a live process keeps its own state and only bulk data is evicted, can rely on the driver's range invalidation for every mapped system.
- **Index directories:** put every index under a `.noindex` directory, so Spotlight doesn't read 2 MB of each new file. The fresh-file control clones into such a directory, waits ~1 s, evicts and checks 0% before GO.
- **Open cost:** charge ~120 µs per open (p99 ~0.54 ms), and ~0.65 ms for the first open of a file created since the last trial.
- **faiss:**
  - The shipped setup is IndexRefine(RR768 + IVF RaBitQ, SQ8) with IFC. Its first cold query pays for the whole coarse table (19.2 MB at 1M, ~192 MB at 10M at 160 per list).
  - The on-disk setup (merge_ondisk lists plus SQ8 read with IFC) loads the coarse table at open and prefetches lists in parallel.
  - IO_FLAG_MMAP alone loads the SQ8 refine, and the two flags can't be combined.
  - FastScan is load-at-open.
  - HNSW with IFC reads its whole graph at open, ~272 MB at 1M, so its routing data is the graph section and C1 should keep it.
  - For C1 with IFC the adapter must copy the coarse table into RAM itself.
- **RaBitQ-Library:** IVF at 5 bits fits 10M under 8 GB. 9 bits at 10M and 5 bits at 20M need the 16 GB solo cap. HNSW to 5M fits 8 GB by footprint.
- **AsterVec:** 1M fits 8 GB, and 2M needs the 16 GB solo cap (~11 GB). Create the database directory before open.
- **qdrant-edge:**
  - Use the wheel. After optimize, flush and close, remove wal/ and load once before any cold run.
  - Set max_indexing_threads, default_segment_number 1, a large max_segment_size and max_search_threads 1.
  - In-RAM codes are copied at load, so those configurations are load-at-open for their codes. Inline storage reads only the ID tracker at load.
- **LanceDB 13:**
  - Pass query_parallelism -1 explicitly, since the default runs partitions one at a time.
  - Sweep approx_mode.
  - Create IVF_RQ with create_index("vector", config=IvfRq(...)).
  - analyze_plan gives bytes_read, iops and parts_loaded per stage.
- **Paimon:** serve pread_many from a thread pool, since a serial loop makes the list round slow. Report its recall ceiling, as it has no exact rerank. 10M at 4 bits fits ~8 GB only with a capped training sample. 8 bits at 10M needs the solo cap.
- **DiskANN:** implement AlignedReaderFactory and AlignedFileReader for both Mac readers, and pass the factory to DiskVertexProviderFactory::new with CachingStrategy::None. Never use AlignedFileReaderFactory on macOS outside the 1M control, since every build() copies the whole file.
- **Counters:** convert rusage ticks with mach_timebase_info, and don't rely on the per-QoS buckets. Report the build peak from the process's own lifetime maximum.
