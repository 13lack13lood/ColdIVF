# Benchmark Harness

This folder runs ColdIVF and other vector indexes through one driver, so every system is built, warmed and timed cold
the same way. The driver reads a suite file, starts one adapter process per job, and appends JSON lines (JSONL) to
`results/bench/<phase>/<system>.jsonl`. A rerun of a suite skips everything already measured.

| File | What It Holds |
|---|---|
| `driver.py` | The suite reader, the timing gate, the drift references, builds, warm grids, cold rounds and the batch summary |
| `schema.py` | The "coldivf-bench/2" line kinds and fields, the validator, the append-and-fsync writer, run keys and provenance |
| `rusage.py` | Process counters from macOS's `proc_pid_rusage`, the thread's quality of service (QoS) class and swap counts |
| `evictor.py` | Page cache eviction through mappings held for a whole batch, and residency checks with `mincore` |
| `data.py` | `.fbin` and `.ibin` memory maps, prefixes, query splits and recall with ties |
| `charts.py` | The repo README's charts and `docs/charts/chart-data.json`, drawn with matplotlib 3.11.2 from `venv-plot` |
| `pick.py` | Each system's cold cells from its warm grid and its screening trials, written into its section of a suite |
| `headtohead.py` | A phase's head-to-head: each system's pick for each recall target, its warm, cold and memory numbers on the test split, and how ColdIVF compares |
| `adapters/base.py` | The `Adapter` base class and the adapter's end of the wire |
| `adapters/flat.py` | Exact search with numpy, the smallest complete adapter |
| `adapters/coldivf.py` | ColdIVF through its release binaries |
| `adapters/coldivf_py.py` | ColdIVF through its Python module, in the adapter's process as every other system runs |
| `adapters/lancedb.py` | LanceDB through pylance's scanner, or through lancedb's query builder where pylance is missing |
| `adapters/paimon.py` | Paimon's IVF-RQ through a timed read input, a thread pool by default |
| `adapters/diskann.py` | DiskANN's disk index through `baselines/diskann3`, with two macOS readers |
| `adapters/usearch_hnsw.py` | usearch's graph, served from its file with view() |
| `adapters/qdrant_shard.py` | qdrant-edge's shard, with its codes in RAM or its graph and codes on disk |
| `adapters/astervec_lsm.py` | AsterVec's log-structured merge (LSM) graph, with the library's caches or none |
| `adapters/stores.py` | The sealed store of a system that writes to its own index whenever it opens or closes |
| `adapters/graphs.py` | Small helpers the three graph adapters share |
| `tests/` | Unit tests, a test adapter with failures on demand, and end-to-end batches |
| `../suites/` | Suite files. `example.toml` lists every key with its default |

## Running

```
cd coldivf-bench/py
python driver.py ../suites/<phase>.toml [--only build,warm,cold,refs] [--systems a,b:variant] [--groups g,h]
                                         [--gate enforce|warn|off] [--max-attempts N] [--dry-run]
python -m unittest discover -s tests -t .
```

`--systems` names systems by name, which takes every variant of one, or as `name:variant`, which takes one variant.
`--groups` runs only the cold cells of those groups, eg. one rival's group with the ColdIVF cells that run beside it.
`--dry-run` prints every job and how much of it is done, and writes nothing. Run the driver with the interpreter of the
environment its adapters need. Each system may name another interpreter in the suite, eg. for a library that needs its
own virtual environment. The driver never starts a process in a new session, so a memory cap on the driver's process
group covers every adapter and every program an adapter runs.

| Variable | Meaning |
|---|---|
| `COLDIVF_DATA` | The folder of datasets, `<name>.base.fbin`, `<name>.query.fbin` and `<name>.gt.ibin`. Default `data/datasets` |
| `COLDIVF_BIN` | The folder of ColdIVF's release binaries. Default `target/mine/release` |
| `COLDIVF_BENCH_LOCKS` | The folder where the machine's job gate keeps its lock files, read by the timing gate unless the suite's `[gate]` names it as `locks` |

Each system's cold cells come from its own results, by one rule for every system. For each recall target, eg. 0.90,
0.95 and 0.99, the cheapest settings are those with the lowest warm p50 at 1 thread whose recall@10 on the tune split
reaches it, a tie going to fewer bytes scanned. `pick.py screen` writes true-cold (C0) cells for the 3 cheapest, 30
trials each on the tune split. Once they have run, `pick.py verdict` writes, for each target, cells on the test split
for the screened setting with the lowest C0 p50, and for the cheaper setting just below the target, so a report can
interpolate latency to exactly the target. A target no setting reaches becomes a comment naming the best recall found.
Both print the cells, or with `--write` put them at the end of the system's section of the suite and check that it
still loads.

```
python pick.py screen ../suites/<phase>.toml --system NAME [--variant V] --config ID [--targets 0.90,0.95,0.99]
                      [--group G] [--threads T] [--dummy-n N | --dummy PATH] [--write]
python pick.py verdict ../suites/<phase>.toml --system NAME [--variant V] --config ID [--trials 100] [--warm] [--write]
```

`headtohead.py` then reads a phase's results by the same rule. For each build and target it takes the pick, and
reports on the test split its recall@10, warm p50 and queries per second, its C0 times, and its memory, the process
plus the pages of the index's own files in the page cache. A cold trial's memory is the footprint its open and first
query add plus the index pages they bring in, and ranges a system copies into its own memory count once. Each row also
gets its ratios to ColdIVF's, by default its Python module's: the ratio of C0 p50s, the median of per-query ratios over
the test queries both cells timed with a 95% bootstrap interval, and the ratio of C0 p50s at exactly the target where
both have cells around it.

```
python headtohead.py [--suite ../suites/bench-10m.toml] [--reference coldivf:python] [--md OUT.md] [--json OUT.json]
```

Ground truth for a prefix of n rows is `data/<name>-<n>.gt.ibin`, and a `.gt.dist.fbin` beside any truth file holds its
distances or scores. Indexes go to `data/bench.noindex/<phase>/<system>/<config_id>-n<n>-s<seed>/` unless a suite
says otherwise. A folder whose name ends in `.noindex` keeps Spotlight from reading the first 2 MB of every new file.

## Batch Flow

1. Check the timing gate and the disk floor. In "enforce" mode the driver waits for the gate and gives up after
   `max_wait_s`. In "warn" mode it records the gate and goes on.
2. Remove temporary build files and index clones under the index root whose process is gone.
3. Time `open()` of an existing file, the median of 21, for the open-cost line, and run the drift references. The first
   open of the new file goes in the batch line too, since it costs ~5 times more.
4. Build what isn't built, one build at a time, each after a disk check. A thread measures the build's folder every 20
   ms, so its peak disk use and its temporary files include files that live a fraction of a second.
5. Run the warm jobs in a random order. Before a system's first warm job the driver measures its per-call cost once,
   with its `overhead` subcommand, for each system, variant and interpreter. Every warm line of the system carries it
   as `overhead_us`, and the batch line keeps every result.
6. Run the cold groups. Each round shuffles the cells of a group, and each cell runs one trial per round.
7. Run the drift references again and write the batch summary line.

The timing gate needs no other job holding a slot of the job gate, no `cargo` or `rustc`, at least 85% idle CPU over
`top -l 3 -s 10`, AC power with low-power mode off, and `caffeinate` holding off idle sleep. In "enforce" mode it must
also see the job gate's lock folder, since without it no other job can show up. Every line's `gate` records the folder
and whether it was there, and `solo` is false whenever the gate can't see that this batch holds every slot.

A timed point is invalid when the machine's swap-ins or compressions move during it. Both are machine-wide, so on a
busy machine nearly every point is invalid. A suite that only checks the code can set `invalid_vm = []` in its gate,
which keeps the counts in every line without that rule.

The drift references are a raw read wave of 64 random 16 KiB pages and ColdIVF's C0 at one fixed setting. Each batch's
drift is measured against the median of earlier batches that enforced and passed the gate, and a batch that moves more
than `drift_pct` is flagged in its summary.

## Suite Files

A suite names one dataset, one prefix size `n` and any number of systems. `../suites/example.toml` holds every key with
its default and a comment. Paths are relative to the repo unless absolute, and `$VARS` and `~` expand. Unknown keys are
errors, so a typo never runs the default.

```toml
phase = "smoke"
dataset = "cohere-768-cosine"
n = 50000

[gate]
mode = "warn"

[[system]]
name = "faiss-ivf"            # the results file and every line's system
adapter = "faiss_ivf"         # adapters/faiss_ivf.py
variant = "shipped"           # every line carries it, and every subcommand gets --variant

  [[system.build]]
  config_id = "ivf640-rq1"
  config = { nlist = 640, bits = 1 }   # passed to the adapter's build as JSON
  seeds = [1]
  threads = 10
  disk_gb = 1

  [[system.warm]]
  config_id = "ivf640-rq1"
  split = "tune"
  grid = { nprobe = [16, 32, 64], k_factor = [1, 2] }   # the product, in this order

  [[system.cold]]
  config_id = "ivf640-rq1"
  regime = "c0"
  params = { nprobe = 32, k_factor = 2 }
  trials = 30
  dummy_n = 50000             # builds the same config at 50K, searched once before READY
```

A build entry with `index = "<path>"` uses an index that is already built. A build with its own `n` builds that prefix,
and its lines carry that n and are scored against the truth of that prefix. Warm and cold entries name a build by
`config_id`, and by `seed` when it has several. Cold cells share a round when they share a `group`. A group can't hold
a C1 cell and another cell on the same index, since C1 keeps part of that index cached.

## Result Lines

Every line has the common fields and the fields of its kind, all present, with null for what wasn't measured.
`schema.py` lists them with their types in `COMMON` and `FIELDS`, and `schema.validate` checks a line. Anything else a
system measures goes in `extra`, a dict. The driver checks every line before it appends it, and a line that fails is
kept as a failed line with the reason.

| Kind | One Line Per |
|---|---|
| `build` | Build of one config, seed and n |
| `warm` | Search setting of a warm job |
| `cold` | Attempt at one cold trial |
| `update` | Epoch of an update trace |
| `ref` | Drift reference, at the start or the end of a batch |
| `batch` | Batch, with its counts, drift, gate and disk use |

| Status | Meaning |
|---|---|
| `ok` | Measured |
| `rejected` | A cold trial that started with cached pages, from the driver's check or the adapter's own |
| `invalid` | Swap-ins or compressions happened during the timed part |
| `failed` | The adapter crashed, timed out, sent no result or wrote a bad line |
| `unsupported` | The system can't do this, eg. a regime it doesn't have |

A run key is a hash of what names a measurement, eg. a cold trial's system, variant, config, settings, regime, CPU
state, trial number and the index's size, inode and change time. A key is done once a line with it is `ok` or
`unsupported`, or once it has used `max_attempts` attempts. A rebuilt index gets new keys.

Every line also records the provenance of its code. That is the repo's staged tree hash, which equals the tree of a
later commit of the staged changes, a hash of everything not staged, a hash of `pip freeze` for the adapter's
interpreter, hashes of the system's packages or binaries, and the environment that sets threads. The machine record
holds the chip, each core type's counts and caches, page and line sizes, translation, graphics cores, the drive, the
operating system build, power and the Rust toolchain.

## Recall

A result is a hit if its id is among the true 10 nearest, or if its exact distance is at most the true 10th distance.
Each line is scored against the truth of its build's n, which is the suite's `truth` file for a build of the suite's n.
Both distances are computed in float64 from the base rows, so a duplicate of the 10th neighbor ties exactly, and ties
are judged within 2 float32 units in the last place. A distances file checks that the truth belongs to the base. The
warm lines also carry `recall10_ids_only`, recall@1, recall@100 when the search returned 100 ids, the 5th and 10th
percentiles of per-query recall and the share of queries at 0.9 or above. Repeated ids and ids outside the prefix are
never hits.

Query splits are fixed. Among the first 1,000 queries, even positions are `tune` and odd positions are `test`. `all` is
every query and `rest` every query after the first 1,000.

Each cold trial searches 1 + `next` queries of its cell's split, a block of positions of its own. The cells of a group
that search one index with one split run their trials in the same rounds, so they take turns at the split's blocks, as
`Driver.query_blocks` lays them out. In a round no two of them search the same query, and none searches one that
another searched in the round before, since a search of pages that a search just read can come out faster, although
none of them shows as cached. Cells that run as many trials, at least twice as many as there are cells, search the same
queries over them, in another order.

## Cold Protocol

**C0 (true cold).** Each trial is a fresh adapter process.
1. The adapter imports its libraries, loads its queries, sets the user-interactive QoS class, searches the dummy index
   once if there is one, turns off Python's garbage collection, and sends `ready`. Its warm-up searches use the
   queries of `--warmup-queries`, which no trial on the cell's index in its group searches. They come from the other
   split of a tune or test cell, so nothing a warm-up leaves in a cache can answer a timed query.
2. The driver holds a mapping of every file of the index from the start of the group. It fsyncs each file, drops its
   pages with `MS_INVALIDATE` through those mappings, waits `wait_s`, drops them again and checks every page with
   `mincore`. Any cached page rejects the trial, and the driver tries again with a new process. The wait is 1.0 s by
   default, since after 0.5 s another process still read page 0 back during 1 or 2 of 30 trials. It doubles for files
   mapped just before, whose first pages another process reads after a new mapping.
3. The driver sends `go`. The adapter runs the CPU state step, then times from just before the first call that touches
   the index until the ids are in hand. It searches the next queries without dropping anything, then the first query
   again with its pages cached.
4. The driver records how many files and bytes the trial left cached, and the machine's swap-ins over the trial. A file
   the adapter says it read in the trial's `files_read` counts as touched too, eg. one read through a handle that skips
   the page cache. The cached bytes of ranges marked "loaded", which the system copied into its own memory and counts
   in its footprint, go to `extra.resident_loaded_bytes_after` instead of `resident_file_bytes_after`.

With `fresh_clone = true` each attempt runs on a new copy-on-write clone of the index, a file no cache has seen, and the
clone is deleted after the trial. It is the control for files that were cached once and then dropped.

**Intent.** As C0, but the system may open the index as soon as the query text arrives, and the vector arrives
`delay_ms` later. The timer runs from the vector's arrival to the ids, so opening that takes longer than the delay
counts. The counters cover the whole trial, the open and the wait included.

**C1 (disk-resident steady state).** One adapter process per cell opens the index and runs one warm-up query before
`ready`, then serves every trial of the cell. The warm-up query is one no trial on the index in the group searches. The
cell's run keys name the index as it was before its process started, since a process that keeps a store open can change
it. Before each trial the driver drops the whole pages of the ranges the adapter marked "bulk", and at most 0.5% of them
may stay cached. Routing ranges stay as the system keeps them. The driver's drop reaches pages that a live process maps,
so mapped systems need no help.

The CPU state step is "prewarm" by default, which spins as many cores as the trial's threads for 150 ms. "idle" sleeps
200 ms instead, and "none" skips the step.

## Warm Protocol

At 1 thread an adapter opens the index, warms it, and for each setting runs `warmup` queries, then `passes` timed
passes over the split. A line holds each latency statistic's median over the passes, and the ids of the first pass go to
`results/bench/<phase>/ids/`, from which the driver computes recall. At more threads it measures queries per second
(QPS) for `qps_seconds`, `qps_repeats` times, and reports the median, through the library's batch call when it has one,
else threads or one process per worker. With worker processes, `cpu_ms_per_query` is the workers' CPU time over their
timed loops, and `footprint` the sum of the workers' footprints, each a process of its own. Mode "whole" reads every
file of the index first, apart from ranges marked "loaded", which the open copied into the system's memory. After each
setting's warm-up queries the base class drops the cached pages of those ranges, so the page cache holds no second
copy, and `resident_file_bytes` leaves them out. `extra.footprint_before_open` and `extra.footprint_since_open` split
the footprint into the process before the open and what the index added. Other modes are system-specific.

## Adapter Contract

The driver runs `<python> -m <module> <subcommand> ...` with `cwd` and `PYTHONPATH` set to this folder. The module
subclasses `adapters.base.Adapter` and ends with `main(MyAdapter)`. `--data` is the base `.fbin`, the queries are the
`.query.fbin` beside it, and `--rows N` is the prefix. Every vector's id is its row number in the base.

| Subcommand | Arguments | Events It Sends |
|---|---|---|
| `build` | `--data --rows --out DIR --config JSON --seed --threads --metric --variant --k [--clusters STEM]` | `build` |
| `warm` | `--data --rows --index PATH --grid JSON --split --mode --ids-dir DIR --passes --warmup --qps-seconds --qps-repeats --threads --metric --variant --k` | `setting` per setting, then `done` |
| `cold` | `--data --rows --index PATH --params JSON --queries IDS --regime c0\|c1\|intent --cpu-state --delay-ms --split --warmup-queries IDS --threads --metric --variant --k [--dummy PATH]` | `ready`, then `trial` per `go` |
| `files` | `--index PATH --variant` | `files` |
| `overhead` | `--data --rows --threads --metric --variant --k` | `overhead` |
| `info` | `[--variant]` | `info` |
| `qps-worker` | Internal, run by the base class for QPS with one process per worker | `ready`, then `count` with its queries, seconds, CPU time and footprints |

`--queries` is a comma-separated list of query ids. `--grid` is a JSON list of settings, each a dict. `--clusters` is a
stem of shared clusters, `<stem>.centroids.fbin` with one row per cluster and `<stem>.assign.ibin` with each base row's
cluster, for systems that can import a partition.

### Wire Format

Every message is one JSON object on one line, in UTF-8. The adapter writes events to the stdout it started with, and the
base class points file descriptor 1 at stderr first, so a library's prints can't reach the wire. Stderr goes to
`results/bench/<phase>/logs/<batch>/<system>.log`.

| Event | Fields |
|---|---|
| `info` | `system`, `system_version`, `binding`, `simd_level`, `regimes`, `batch_api`, `qps_mode`, `workers`, `worker_qos`, `wheel_sha256`, `venv`, `python` |
| `build` | `fields`, the build fields, and `info` |
| `setting` | `index`, the setting's place in `--grid`, `params`, `fields`, the warm fields, `ids_file` and `qids`, the query ids of its rows |
| `done` | Nothing, after the last setting |
| `ready` | `pid`, `regime`, and the `info` fields |
| `trial` | `trial` and `fields`, the cold fields |
| `files` | `files`, a list of `{path, bytes, ranges}`, each range `{offset, length, kind}` with kind `bulk` or `routing`, and `loaded` true for bytes the system copies into its own memory at open or first use |
| `overhead` | `fields` with `overhead_us`, the median cost of one call on a 1K-vector index |
| `unsupported` | `message`. The adapter then exits with status 0 |
| `error` | `message` and `traceback`. The adapter then exits with status 1 |

| Command | Fields |
|---|---|
| `go` | `trial`, `queries`, the query ids of the trial with the first one timed cold, and `self_evict` for C1 |
| `stop` | Nothing. The adapter cleans up and exits. The end of stdin means the same |

A C0 or intent process serves one `go` and exits. A C1 process serves `go` commands until `stop`. A process that gets
`stop` before `go` exits without a trial, which is how the driver ends a rejected trial.

### Fields an Adapter Reports

The base class fills these for an in-process adapter. An adapter of an outside program fills what it can, and leaves
the rest out or null.

| Event | Fields |
|---|---|
| `build` | `seconds`, `cpu_seconds`, `threads`, `seed`, `peak_footprint`, `bytes_read`, `bytes_written`, and from the hook `cluster_seconds`, `encode_seconds`, `stored_vector_bytes`, `exact_f32_stored`, `extra` |
| `setting` | `p50_us`, `p90_us`, `p99_us`, `mean_us`, `qps`, `queries`, `passes`, `batch_api`, `processes`, `cpu_ms_per_query`, `footprint`, `resident_file_bytes`, `memory_total`, `swapins_delta`, `compressions_delta`, `dist_evals`, `bytes_scanned` |
| `trial` | `open_us`, `search_us`, `total_us`, `next_us`, `resident_rerun_us`, `ids`, `stages`, `bytes_requested`, `disk_bytes`, `pageins`, `footprint_before`, `footprint_delta`, `lifetime_max_footprint`, `cpu_ms`, `cpu_qos_ms`, `p_share`, `ghz`, `energy_nj`, `qos`, `extra`, and `status` and `error` for a trial the adapter itself rejects. `extra.files_read` may name the bytes the trial read of each file, by path |

The driver adds the rest. That is every common field, recall from the ids, `resident_pages_at_go`, `files_in_index`,
`files_touched`, `resident_file_bytes_after`, `predecessor`, `swapins_delta`, `agents_cpu`, and for builds
`peak_disk_bytes`, `temporary_files`, `index_bytes`, `index_alloc_bytes`, `index_files` and `index_identity`.

`cpu_qos_ms` splits CPU time by the class the kernel ran each thread at. macOS caps processes started from a terminal
or an agent, so their user-interactive and user-initiated time lands in "legacy", while utility and background time
lands in its own class. `qos` is the class the timed thread asked for. Only the thread that calls the system asks for
the user-interactive class, in every system. The threads a system runs besides it, which `info` names in `workers`,
run at the class a new thread starts with, the default, whether the library or the adapter makes them. So `p_share`,
the share of CPU time on performance cores, is the signal of where they ran.

## Writing an Adapter

An in-process adapter subclasses `Adapter`, sets `system`, and implements the hooks below. `adapters/flat.py` is a
complete example in ~70 lines.

| Hook | Does |
|---|---|
| `load(self)` | Imports the library. Import heavy libraries here, not at the top of the module, so the thread environment is set first |
| `version(self) -> str` | The library's version |
| `wheels(self) -> dict` | Hashes of what the system runs, eg. `{"faiss-cpu": schema.dist_sha256("faiss-cpu")}` |
| `simd_level(self) -> str` | The SIMD level the library uses, eg. "NEON" |
| `thread_env(self, threads) -> dict` | Extra environment for the library's own thread setting. The common thread variables are set already |
| `build(self, ctx: BuildContext) -> dict` | Builds `ctx.data`, rows 0 to `ctx.rows` mapped, into the empty folder `ctx.out`. Returns the build fields it knows |
| `open(self, ctx: ServeContext, path) -> handle` | Opens the index at `path` with `ctx.params`. In C0 this is the first timed call |
| `search(self, handle, query, k, params) -> ids` | The ids of the k nearest rows, nearest first, padded with -1 |
| `search_batch(self, handle, queries, k, params, threads)` | Optional. A native batch call. Set `batch_api = True` |
| `configure(self, handle, params)` | Applies a warm setting to an open handle |
| `close(self, handle)` | Releases a handle |
| `warm_up(self, handle, ctx)` | Brings the index into memory for `ctx.mode`. The default reads every file for "whole" |
| `files(self, path) -> list` | The index's files and ranges, marked `bulk` or `routing`, and `loaded` where the open copies them. The default makes every file one bulk range, loaded when `loads_at_open` is set |
| `stats(self, handle) -> dict` | Counters of the last search, eg. `dist_evals`, `bytes_scanned`, `bytes_requested` or `stages` |
| `self_evict(self, handle)` | Optional. Drops the index's pages through the system's own mappings in C1 |
| `resident_rerun(self, ctx, handle, query)` | Times a cold trial's first query again with its pages in memory. A system whose reads skip the page cache fills the cache first |
| `supported_regimes(self) -> tuple` | The regimes it runs, `regimes` by default. A variant can narrow it |
| `overhead_config(self)`, `overhead_params(self)` | The build config and settings of the per-call cost index |

`BuildContext` has `config`, `data_path`, `rows`, `out`, `clusters`, `threads`, `seed`, `metric`, `variant`, `dataset`,
`data` and `dim`. `ServeContext` has `index`, `dataset`, `rows`, `k`, `metric`, `threads`, `variant`, `params`,
`queries` (query vectors by id), `warmup` (the vectors of the warm-up queries), `warmup_ids` (their ids), `regime`,
`cpu_state`, `delay_ms`, `dummy`, `split` and `mode`. A class sets `workers` to name the threads the system runs besides
the calling one, and `loads_at_open` when an open copies every file into memory. `self.variant` holds the variant before
`load` runs. A hook raises `adapters.base.Unsupported` for something the system can't do.

An adapter that drives an outside program, as `adapters/coldivf.py` does, overrides `run_build`, `cold_setup`,
`cold_trial`, `cold_teardown` and `warm_run`, and runs the program with `self.run_child`, which stops it when the
adapter gets SIGTERM. A program that speaks the wire itself, eg. a Rust binary, takes over with
`self.exec_program(cmd)` from `cold_setup`, before `ready`. It replaces the adapter process, keeps its pid, process
group and stdin, and writes its events to the adapter's original stdout. It then sends `ready` itself.

Rules every adapter follows:
- Ids are base row numbers, so recall and every comparison line up.
- Set threads explicitly, through the library's own call as well as the environment.
- Write nothing to the wire except through the base class.
- Start no process in a new session, and leave no process running after exit.
- Read and write only the index's own folder, `--out`, `--ids-dir` and the dataset.
- Report what the system measures beyond the schema in `extra`, never as new top-level fields.
- Move only the calling thread to the user-interactive class, never a pool's threads, so every system runs its own
  threads alike.
- Make a warm setting search as a fresh open with the same settings would. A key a setting leaves out gets its value
  from the open, not from the setting before it.

## ColdIVF Adapter

`adapters/coldivf.py` runs `build_index`, `warm` and `cold` from `COLDIVF_BIN`. An index is a folder that holds one
`.coldivf` file, or the file itself. The binaries take a dataset by name. For `warm` a split becomes a dataset under
`data/splits/`, with a link to the base, the split's queries and the split's truth. `cold` searches the dataset itself
and takes each trial's queries by their ids. Its `files` marks the header, rotation, centers and extents as routing,
the codes, rows and ids as bulk, and watches the `.delta` and `.lock` files beside the index.

Each C0 or intent trial is one `cold --one` run, a fresh process. With a dummy it first opens the dummy and searches it
once with the cell's first warm-up query, with a search context of its own, so the timed search finds its code and its
threads warm as other systems' do. It then drops the pages of the index and of the `.delta` and `.lock` files beside it
through its own mappings, checks them right before its timer, and reports the trial rejected if any page was cached. It
searches the trial's queries, then the first one again with its pages cached, which gives `resident_rerun_us`. In intent
the vector arrives `delay_ms` after the timer starts, while the open and `prepare` run, and the search starts once both
the vector is there and `prepare` is done. It waits for the vector as the base class does, and its counters cover the
wait too. `total_us` is then the time from the vector's arrival to the ids, so open time past the delay counts, as for
every other system. A C1 cell runs one `cold --one --c1` process for all its trials. It opens the index, searches the
first warm-up query, which loads the cluster tables it then keeps, and runs each trial from a line of its input. Before
each trial it drops the codes, rows and ids, and the timer covers the search alone. A C1 trial is rejected when more
than 0.5% of those pages were still cached, as the driver allows. Its `footprint_before` is the process's footprint
right before the search, which holds the cluster tables the process keeps, and `lifetime_max_footprint` the process's
peak by the search's end. The adapter refuses a `cold` that predates all of this, since such a binary would search
other queries than the trial's, or wait for the vector another way.

`cold` prints each query's ids, so the driver scores ColdIVF's cold trials with the same recall code as every other
system. Its own `recall10` counts ties the same way, from the base rows, and `recall10_ids_only` counts ids alone.
`warm` prints no ids. Each split's truth gets a distances file beside it, `<split>.<name>[-<n>].gt.dist.fbin`, with
each truth id's distance from its base row as the binaries measure it, minus the inner product for a cosine dataset and
the squared Euclidean distance for an `-euclidean` one. With it `warm` counts ties, reports `ties` true, and the adapter
takes recall@1, @10 and @100 and their percentiles from it, which `extra.recall_source` records. The binary counts a tie
only at most as far as the true 10th distance, where the driver allows two float32 units, so at a near tie between rows
that differ its recall can come out lower. `extra.near_ties` counts the split's queries with one. An older `warm`
scores recall by ids alone. That equals recall with ties wherever the truth has no tie at that rank, so then a recall
field is filled only when no query of the split has one, and the binary's own values go under `extra`. A `warm` that
takes `--mode t1`, passes and threads gets them from the adapter, which then fills p90 and QPS too. A warm line's
`footprint` is the warm binary's whole process at its peak, read from its record after it exits. Its swap-ins and
compressions are the machine's over the binary's whole run, its warm-up included, since the binary doesn't say when its
timed passes start, so the driver marks the line invalid when either moved, as it does for every other system.

The binaries have no CPU time per QoS class and no cluster import, so those report unsupported or null. They time
each search inside their own process, around the library's call, so their per-call cost is 0. The binaries ignore
flags they don't know, so the adapter refuses a build setting its `build_index` lacks, eg. `anchor` on an older build,
rather than build the default.

A build whose config sets `export_clusters = true` also saves the partition it encodes, i.e. its clusters and their
groups, for other systems to import as shared clusters. The files go in a folder beside the index folder,
`<index folder>.clusters/partition.*`, so a suite's `clusters` for another system is
`<index folder>.clusters/partition`. The index folder holds the index alone, and a dummy build of the same config saves
its own partition beside its own folder. Build lines record the stem in `extra.export_clusters`.

## LanceDB Adapter

`adapters/lancedb.py` runs LanceDB with or without pylance. With lancedb 0.40 and pylance 13 a search goes through
pylance's scanner, which takes `approx_mode` and `query_parallelism`. With lancedb alone, eg. the 0.39 reference row, it
goes through lancedb's query builder, and those two settings report unsupported. An index folder holds the table `t`
and `meta.json`. The rows go in as one stream of batches, and nothing compacts the table.

| Build Setting | Meaning |
|---|---|
| `index` | `ivf_rq`, the default, or `ivf_pq`, made through lancedb's config object |
| `bits` | RaBitQ bits, 1 by default, or PQ bits, 8 by default |
| `num_sub_vectors` | PQ sub-vectors, 96 by default |
| `partitions`, `vpl` | The partition count, or the rows per partition it comes from. Lance's default without either |
| `sample_rate` | Training rows per partition, 256 lowered so partitions × rate stays within the rows and 1M |
| `max_iterations` | k-means iterations, 50 by default |
| `distance` | Lance's distance, `dot` for a cosine dataset of unit vectors |
| `batch_rows` | Rows per ingest batch, 65,536 by default |

| Search Setting | Meaning |
|---|---|
| `nprobes` | Both the minimum and the maximum nprobes. It is required |
| `refine_factor` | Rescored candidates as a multiple of k, from the table's vectors. None or 0 for no refine |
| `approx_mode` | pylance's "fast", "normal" or "accurate" estimates for RaBitQ codes |
| `query_parallelism` | pylance's setting, -1 unless set, which runs the partitions on the compute pool |
| `open` | `lance`, the default with pylance, opens with pylance's dataset alone, as a user of its scanner would. `lancedb` opens with connect, open_table and to_lance, which opens the dataset twice. Without pylance it is always `lancedb` |
| `analyze` | A cold cell that times analyze_plan instead of a search, for its reads per stage |

`LANCE_CPU_THREADS` is the job's threads, so a cold cell at 10 threads runs Lance's default pool on this Mac, and a cell
at 1 thread is the 1-thread variant. A trial's `bytes_requested` and `extra.iops` are pylance's read counters over the
open and the first query. lancedb has a Lance of its own, so with `open = "lancedb"` the reads of its connect and
open_table aren't in them. Every line names its open in `extra.open_path`. `files` marks `meta.json`, the manifests and
`index.idx` as routing, and also loaded, since Lance keeps what it reads of them in its caches. The partitions in
`auxiliary.idx`, the data files and the transaction files are bulk.

In C1 the process holds one session whose index cache is about the smallest that keeps the model, its centroids and
metadata, loaded. A fresh session holds the model plus one partition after a query that probes one partition, and one
query that probes more shows the bytes of a partition. The capacity starts at the model plus one partition and grows
one partition at a time, over the cell's warm-up queries, until queries stop loading the index again, and then gets
one partition more. A cache that size still keeps a few partitions between queries. So each C1 trial reports, in
`extra`, the capacity, the start, the model's and a partition's bytes, whether the index stayed loaded, the bytes held,
and `partitions_cached_max` and `partitions_cached`, how many partitions the cache can hold beside the model and holds
after the trial.
Each warm setting ends with analyze_plan over 8 queries of its split. The mean per role (ranking, partitions, refine and
ids) goes in `extra.analyze`, and the index comparisons fill `dist_evals`. Shared clusters become pylance's
`ivf_centroids`, and lancedb alone can't import them. Warm mode "index" reads only the files under the table's
`_indices` folder before the first setting and leaves the data files to the queries, for a table too large to cache
whole. The warm-up searches then cache the refine rows and ids of the queries they search, and no other row.

## Paimon Adapter

`adapters/paimon.py` builds Paimon's IVF-RQ with exact coarse assignment and reads it through an input whose rounds it
times. Paimon trains its own clusters, has no rerank and takes no seed. The build measures the routing span, the 64-byte
header plus the centroids and offsets that the first search reads in one range, checks it against the format and keeps
it in `meta.json`. `files` marks that span as routing and the lists after it as bulk, which C1 drops before each trial.

| Setting | Meaning |
|---|---|
| `bits` | RaBitQ bits, 1 to 8, 4 by default |
| `nlist`, `vpl` | The list count, or the rows per list it comes from. The library's own choice without either |
| `train_rows` | Evenly spaced rows fed to the trainer, for a capped training sample. Every row by default |
| `options` | Writer options that add to or override the others, eg. `ivf.train.max-points-per-centroid` |
| `nprobe` | The lists a search probes. It is required |
| `read_threads` | The read pool's size, 64 by default |

| Variant | Input |
|---|---|
| (none) | A thread pool, one pread per range, so the ranges of a round are read together |
| `rdadvise` | F_RDADVISE for every range of the round, then preads in turn |
| `serial` | Plain preads in turn, the control |

A cold trial's `stages` hold each round's ranges, bytes and time. A cold first query reads three rounds, i.e. the header
at open, then the routing range, then the probed lists. The reader keeps the routing range in memory, so `files` marks
it loaded. The read pool's threads run at the default QoS class, as every library's own threads do. `search_batch` runs
on the library's Rayon pool, so QPS at several threads uses it.

## DiskANN Adapter

`adapters/diskann.py` builds DiskANN's disk index with the `diskann3` tool and searches it through `libdiskann3.dylib`
over ctypes. Both come from `cargo build --release` in `baselines/`, a Cargo workspace of its own. The build reads rows
0 to N of the base through a storage provider that serves them as a file of their own, so nothing is copied. DiskANN
builds its own graph, so it imports no clusters. An index is a folder with `index_disk.index`, which holds the graph and
the full vectors in 4 KiB sectors, the two product quantization (PQ) files and `meta.json`.

| Setting | Meaning |
|---|---|
| `degree` | The graph's degree after pruning, 64 by default |
| `build_list` | The search list size while building, 100 by default |
| `pq_bytes` | Bytes per vector of the PQ codes that searches keep in memory, 96 by default |
| `alpha`, `saturate` | DiskANN's pruning parameter, 1.2 by default, and whether pruned lists are filled back up, off by default |
| `metric` | `l2` by default. A metric other than the dataset's ranks alike only on unit vectors, which the build checks |
| `build_ram_gb` | The memory DiskANN plans its build for, 7 by default. A build that needs more is split into overlapping shards |
| `list`, `beam` | A search's list size and beam width. Both are required |
| `cache_nodes` | The nodes nearest the entry point kept in memory from the open on. 10,000 in C1 and none elsewhere by default |

| Variant | Reader |
|---|---|
| `pool`, the default | Every sector of a beam at once, one on the search thread and the rest on a pool of threads, through a handle with `F_NOCACHE` |
| `advise` | `F_RDADVISE` for every sector of a beam, then a read of each in order, through the same kind of handle |
| `stock` | DiskANN's own macOS reader, built once and shared. It copies the whole index file at open, so it runs only up to 1M vectors at 1 thread, and has no C1 |

The read pool runs as many threads as the beam width times the search threads, at the default QoS class, as every
library's own threads do. They start before READY, along with the searcher's runtime, so a C0 open times DiskANN's own
work, i.e. loading both PQ files, opening the index file and reading its header, which DiskANN does twice. A trial's
`bytes_requested` covers the open and the first query, i.e. both PQ files whole, the header sectors and every node
sector, and its `stages` split the open into its PQ, reader and searcher steps. `files` marks the PQ files, `meta.json`
and the index file's header sector as routing, and the other sectors as bulk, which C1 drops before each trial. The PQ
files and the meta are also loaded, since an open copies them, and with the stock reader so is the whole index file.
Warm runs at several threads use the library's batch search, which spreads the queries over that many threads.
`dist_evals` counts PQ distance computations, and `bytes_scanned` the node sector bytes a query read. A setting with
another `cache_nodes` opens the index again.

The pool reader's reads skip the page cache, so mincore can't see them. A trial's `extra.files_read` names the bytes it
read of each file, which the driver counts in `files_touched`. Nothing stays cached for a rerun of the first query
either, so the pool reader's rerun first runs the query once with its prefill switch on, which asks for each sector
with `F_RDADVISE` and leaves it in the page cache. The timed rerun then reads them from there through the same reader,
and the adapter drops them again after it, as the pool reader would leave them. `extra.rerun_prefilled` marks it.

## usearch Adapter

`adapters/usearch_hnsw.py` builds usearch's hierarchical navigable small world (HNSW) graph. It adds row i with key i
on the build's threads and saves `index.usearch` beside `meta.json`. An open is `Index.restore` with `view=True` and
`enable_key_lookups=False`, which maps the file without loading it. The file holds an 8-byte size header and the
vectors, then a 64-byte header and the graph, i.e. a 40-byte header, a 16-bit level per node and each node's key and
neighbor lists. `files` reads that layout from the headers and marks the vectors as bulk and the rest as routing. So C1
drops the vectors before each trial and leaves the graph as the process found it.

| Setting | Meaning |
|---|---|
| `dtype` | `i8`, the default, with cosine, or `f16` or `f32` with the inner product where the rows have unit length |
| `connectivity` | M, the links per node, 16 by default |
| `expansion_add` | A build's candidate list, usearch's 128 by default |
| `expansion_search` | A search's candidate list, usearch's 64 by default |

A trial's `extra` holds `dist_evals`, `visited` and `bytes_scanned`, which is the distances times the bytes of one
vector. usearch takes no seed. Its batch call runs its own threads outside Python's lock, so QPS at several threads uses
it.

## qdrant-edge Adapter

`adapters/qdrant_shard.py` upserts row i as point i into one shard, with the original vectors on disk. It then
optimizes the shard into one segment with an HNSW graph, beside the empty segment qdrant keeps for new points.
Thresholds of 1 KB make qdrant index and search the graph at any size. Searches run on one thread, which the shard's
config holds. qdrant's write-ahead log keeps every upsert, and a load would read all of it. So the build removes the log
after its final flush and loads the shard once, which starts an empty log, and build lines record `extra.wal_removed`.

| Setting | Meaning |
|---|---|
| `codes` | `sq8`, 8-bit scalar quantization (SQ8) and the default, or `bq2`, 2-bit binary codes |
| `inline` | The graph on disk with each node's neighbors' codes beside its links, and the codes on disk too. Off by default, which keeps the codes and the graph in RAM |
| `m`, `ef_construct` | qdrant's 16 and 100 by default |
| `quantile` | The SQ8 quantile, qdrant's own without it |
| `hnsw_ef` | A search's candidate list, qdrant's own without it |
| `oversampling` | The candidates rescored with the original vectors, as a multiple of k |
| `rescore` | Whether to rescore, on by default |

`files` marks the original vectors as bulk, and with `inline` the graph and the codes too. Without `inline` the codes
in `quantized.data` are also loaded, since a load copies them, as its footprint grows by their size. Searches from
several Python threads don't run in parallel in this binding, so QPS at several threads runs one process per worker,
and each worker loads its own codes.

## AsterVec Adapter

`adapters/astervec_lsm.py` loads every row with bulk_build, which gives row i the id i. The store holds `db/`, a
RocksDB store of the graph's bottom layer, and `vectors.bin`, the 8-bit codes. The upper layers load into RAM at open.
The first open after bulk_build moves the build's log into tables, so the build opens and closes the store once more.
The build's seed is AsterVec's random seed.

| Setting | Meaning |
|---|---|
| `m`, `m_max`, `m_level`, `ef_construction` | AsterVec's graph settings, 8, 24, 1 and 32 by default |
| `ef` | A search's candidate list, 128 by default |
| `caches` | `default`, AsterVec's 32 MiB block cache, 8,192 vector pages of 4 KiB and 100,000 edge lists, or `minimal`, all three at 0 |
| `block_cache_bytes`, `edge_cache_size`, `paged_max_cached_pages` | One cache's size, over the preset |

Caches are set at open, so a warm setting with other caches reopens the store. `files` marks the vector file and the
RocksDB tables as bulk. Searches from several Python threads don't run in parallel in this binding either, so QPS at
several threads runs one process per worker.

## Sealed Stores

qdrant-edge rewrites its config and segment flags at every load and close. AsterVec writes a new manifest, options file
and log at every open, and rewrites its graph's metadata at every close. Opened in place, their indexes would change
with every trial, so `adapters/stores.py` seals each one. The build runs one session on a copy made of clones, notes the
files the session changes or removes, and keeps a clone of each in `stash/`. After every session the adapter deletes
what the session made and puts the stashed files back. A clone on the Apple File System (APFS) shares its disk blocks
and keeps its size and change time. So a restore copies nothing, and between sessions the index has the identity its
build recorded. The restored files are new files, which a cold trial drops through new mappings like any file it
hasn't seen.

- One process at a time holds `store.lock` and opens the store in place. A cold trial takes it before READY, so the
  timed open reads no record and writes nothing first.
- A process that can't take it, i.e. a QPS worker or a second C1 process on the same index, opens a private copy under
  `<index>.sessions/`, beside the index folder. The copy hard-links the bulk files, so its pages are the store's, and
  clones the rest. Its trials carry `extra.private_copy`.
- A C1 process keeps its session open for its whole cell, so the index differs from the build until the process ends.
  The driver keys the cell's trials on the index as it was before the process started, so a rerun of the batch skips
  them like any other.
- `index_bytes` counts the stash, whose clones share their disk blocks with the store, and a build's `peak_disk_bytes`
  counts the copy of its learning session. `extra.store_bytes` is the store alone.
- Mode "whole" reads the data of the store's files and skips holes, eg. those of qdrant's sparse 32 MB log and payload
  files.
