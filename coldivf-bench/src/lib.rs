//! Benchmarks of ColdIVF. The library holds what every harness shares: the datasets, the metrics and result files,
//! the tools that empty the operating system's cache and measure what a process used, and the streams of changes
//! that the maintenance benchmark replays, with exact neighbors kept up to date as they change.

pub mod args;
pub mod coldcache;
pub mod data;
pub mod metrics;
pub mod stream;
pub mod truth;

#[cfg(test)]
#[path = "../tests/unit/testutil.rs"]
mod testutil;
