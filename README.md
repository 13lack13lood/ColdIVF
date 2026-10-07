# ColdIVF

A read-optimized vector index in Rust, built for cold reads: it answers a query against an index that isn't in memory by fetching only the data that query needs in a few batched disk reads.
