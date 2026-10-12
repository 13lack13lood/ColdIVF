//! Builds a small index of random unit vectors and searches it with every reader.

use std::path::{Path, PathBuf};

use diskann_vector::distance::Metric;
use diskann3::builder::{self, Spec};
use diskann3::mac_reader::ReadPool;
use diskann3::search::{Index, Options, Reader, prepare_runtime};

const ROWS: usize = 3000;
const DIM: usize = 32;
const K: usize = 10;

/// A folder under the system's temporary folder, removed when dropped.
struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// SplitMix64, a small seeded generator.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A value from -1 to 1.
    fn value(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }

    fn unit(&mut self) -> Vec<f32> {
        let v: Vec<f32> = (0..DIM).map(|_| self.value()).collect();
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.into_iter().map(|x| x / norm).collect()
    }
}

fn write_base(path: &Path, rows: &[Vec<f32>]) {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(rows.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&(DIM as u32).to_le_bytes());

    for row in rows {
        row.iter().for_each(|x| bytes.extend_from_slice(&x.to_le_bytes()));
    }

    std::fs::write(path, bytes).unwrap();
}

fn exact(rows: &[Vec<f32>], query: &[f32]) -> Vec<u32> {
    let mut scored: Vec<(f32, u32)> = rows
        .iter()
        .enumerate()
        .map(|(i, r)| (r.iter().zip(query).map(|(a, b)| (a - b) * (a - b)).sum(), i as u32))
        .collect();
    scored.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    scored.into_iter().take(K).map(|(_, i)| i).collect()
}

#[test]
fn every_reader_finds_the_same_neighbors() {
    let dir = Scratch(std::env::temp_dir().join(format!("diskann3-e2e-{}", std::process::id())));
    let _ = std::fs::remove_dir_all(&dir.0);
    std::fs::create_dir_all(dir.0.join("index")).unwrap();
    let mut rng = Rng(7);
    // A base of 2,000 more rows than the prefix, so the build must stop at the prefix.
    let rows: Vec<Vec<f32>> = (0..ROWS + 2000).map(|_| rng.unit()).collect();
    let base = dir.0.join("base.fbin");
    write_base(&base, &rows);
    let rows = &rows[..ROWS];

    let spec = Spec {
        base: base.clone(),
        rows: ROWS,
        out: dir.0.join("index"),
        degree: 16,
        list: 50,
        pq_bytes: 8,
        alpha: None,
        saturate: false,
        metric: Metric::L2,
        threads: 2,
        seed: 3,
        ram_gb: 1.0,
        build_pq_bytes: 0,
    };
    let meta = builder::build(&spec).unwrap();
    assert_eq!(meta["rows"], ROWS);
    assert_eq!(meta["dim"], DIM);
    assert_eq!(meta["metric"], "l2");
    assert_eq!(meta["build_pq_bytes"], 0);
    // Only a build on PQ codes trains a quantizer of its own.
    assert!(!spec.out.join("index_build_pq_pivots.bin").exists());

    for file in builder::files(&spec.out).unwrap() {
        assert!(Path::new(&file).exists(), "{file}");
    }

    let mut queries: Vec<Vec<f32>> = (0..20).map(|_| rng.unit()).collect();
    queries.extend((0..10).map(|i| rows[i * 97].clone()));
    let truth: Vec<Vec<u32>> = queries.iter().map(|q| exact(rows, q)).collect();
    let pool = ReadPool::global();
    pool.reserve(4).unwrap();
    let prefix = builder::prefix(&spec.out).unwrap();
    let mut answers = Vec::new();

    for reader in [Reader::Pool, Reader::Advise, Reader::Stock] {
        prepare_runtime().unwrap();
        let options = Options {
            reader,
            metric: Metric::L2,
            threads: 1,
            cache_nodes: 0,
        };
        let index = Index::open(&prefix, &options, pool).unwrap();
        assert_eq!((index.points(), index.dim()), (ROWS, DIM));
        let disk = std::fs::metadata(&builder::files(&spec.out).unwrap()[2]).unwrap().len();
        let whole = if reader == Reader::Stock { disk } else { 0 };
        assert!(
            index.opened.bytes >= whole + 4096,
            "{reader:?} read {} bytes at open",
            index.opened.bytes
        );
        assert!(index.opened.bytes < whole + 64 * 4096);
        let mut ids = Vec::new();
        let mut hits = 0;

        for (q, query) in queries.iter().enumerate() {
            let found = index.search(query, K, 50, 4).unwrap();
            assert_eq!(found.ids.len(), K);
            assert!(found.stats.ios > 0 && found.stats.hops > 0);
            assert_eq!(
                found.stats.reads.reads, found.stats.ios as u64,
                "{reader:?} reads a sector per node"
            );
            assert!(found.distances.windows(2).all(|w| w[0] <= w[1]));
            hits += found.ids.iter().filter(|id| truth[q].contains(id)).count();
            ids.push(found.ids);
        }

        let recall = hits as f64 / (queries.len() * K) as f64;
        assert!(recall >= 0.9, "{reader:?} recall {recall}");

        for (i, q) in (20..30).enumerate() {
            assert_eq!(ids[q][0], (i * 97) as u32, "a base row finds itself first");
        }

        let flat: Vec<f32> = queries.iter().flatten().copied().collect();
        let batch = index.search_batch(&flat, K, 50, 4, 3).unwrap();

        for (q, found) in ids.iter().enumerate() {
            let row: Vec<u32> = batch[q * K..(q + 1) * K].iter().map(|&x| x as u32).collect();
            assert_eq!(&row, found, "the batch search of query {q} differs");
        }

        assert!(index.search(&queries[0][..DIM - 1], K, 50, 4).is_err());
        assert!(index.search(&queries[0], K, K - 1, 4).is_err());
        // Only the pool reader has the prefill switch, and a search that fills the cache finds the same ids.
        assert_eq!(index.set_prefill(true), reader == Reader::Pool);
        assert_eq!(index.search(&queries[0], K, 50, 4).unwrap().ids, ids[0]);
        index.set_prefill(false);
        answers.push(ids);
    }

    assert_eq!(answers[0], answers[1]);
    assert_eq!(answers[0], answers[2]);

    let threaded = Options {
        reader: Reader::Stock,
        metric: Metric::L2,
        threads: 2,
        cache_nodes: 0,
    };
    let refused = Index::open(&prefix, &threaded, pool).err().unwrap();
    assert_eq!(refused.kind(), std::io::ErrorKind::Unsupported);

    let cached = Options {
        reader: Reader::Pool,
        metric: Metric::L2,
        threads: 1,
        cache_nodes: 300,
    };
    let index = Index::open(&prefix, &cached, pool).unwrap();
    let mut fewer = 0;

    for (q, query) in queries.iter().enumerate() {
        let found = index.search(query, K, 50, 4).unwrap();
        assert_eq!(found.ids, answers[0][q], "the node cache changes no answer");
        fewer += (found.stats.reads.reads < found.stats.ios as u64) as usize;
    }

    assert!(fewer > 0, "the cached nodes are never read");
}

#[test]
fn a_bad_spec_is_refused_before_any_work() {
    let dir = Scratch(std::env::temp_dir().join(format!("diskann3-spec-{}", std::process::id())));
    let _ = std::fs::remove_dir_all(&dir.0);
    std::fs::create_dir_all(&dir.0).unwrap();
    let mut rng = Rng(1);
    let rows: Vec<Vec<f32>> = (0..300).map(|_| rng.unit()).collect();
    let base = dir.0.join("base.fbin");
    write_base(&base, &rows);
    let good = Spec {
        base,
        rows: 300,
        out: dir.0.join("index"),
        degree: 8,
        list: 20,
        pq_bytes: 8,
        alpha: None,
        saturate: false,
        metric: Metric::L2,
        threads: 1,
        seed: 1,
        ram_gb: 1.0,
        build_pq_bytes: 0,
    };
    assert_eq!(builder::check(&good).unwrap(), DIM);
    assert_eq!(
        builder::check(&Spec {
            build_pq_bytes: DIM,
            ..good.clone()
        })
        .unwrap(),
        DIM
    );

    for bad in [
        Spec {
            rows: 301,
            ..good.clone()
        },
        Spec {
            rows: 0,
            ..good.clone()
        },
        Spec {
            pq_bytes: DIM + 1,
            ..good.clone()
        },
        Spec {
            degree: 0,
            ..good.clone()
        },
        Spec {
            ram_gb: 0.0,
            ..good.clone()
        },
        Spec {
            ram_gb: f64::NAN,
            ..good.clone()
        },
        Spec {
            build_pq_bytes: DIM + 1,
            ..good.clone()
        },
    ] {
        assert_eq!(
            builder::check(&bad).unwrap_err().kind(),
            std::io::ErrorKind::InvalidInput
        );
    }

    assert_eq!(builder::metric("ip").unwrap(), Metric::InnerProduct);
    assert!(builder::metric("hamming").is_err());

    for m in ["l2", "ip", "cosine", "cosine-normalized"] {
        assert_eq!(builder::metric_name(builder::metric(m).unwrap()), m);
    }
}

#[test]
fn a_graph_built_on_pq_codes_finds_the_neighbors() {
    let dir = Scratch(std::env::temp_dir().join(format!("diskann3-pq-build-{}", std::process::id())));
    let _ = std::fs::remove_dir_all(&dir.0);
    std::fs::create_dir_all(dir.0.join("index")).unwrap();
    let mut rng = Rng(11);
    let rows: Vec<Vec<f32>> = (0..ROWS).map(|_| rng.unit()).collect();
    let base = dir.0.join("base.fbin");
    write_base(&base, &rows);
    let spec = Spec {
        base,
        rows: ROWS,
        out: dir.0.join("index"),
        degree: 16,
        list: 50,
        pq_bytes: 8,
        alpha: None,
        saturate: false,
        metric: Metric::L2,
        threads: 2,
        seed: 3,
        ram_gb: 1.0,
        build_pq_bytes: DIM,
    };
    let meta = builder::build(&spec).unwrap();
    assert_eq!(meta["build_pq_bytes"], DIM);
    assert!(spec.out.join("index_build_pq_pivots.bin").exists());
    prepare_runtime().unwrap();
    let pool = ReadPool::global();
    pool.reserve(4).unwrap();
    let options = Options {
        reader: Reader::Pool,
        metric: Metric::L2,
        threads: 1,
        cache_nodes: 0,
    };
    let index = Index::open(&builder::prefix(&spec.out).unwrap(), &options, pool).unwrap();
    let queries: Vec<Vec<f32>> = (0..30).map(|_| rng.unit()).collect();
    let hits: usize = queries
        .iter()
        .map(|q| {
            let truth = exact(&rows, q);
            let found = index.search(q, K, 50, 4).unwrap();
            found.ids.iter().filter(|id| truth.contains(id)).count()
        })
        .sum();
    let recall = hits as f64 / (queries.len() * K) as f64;
    assert!(recall >= 0.9, "recall {recall}");
}
