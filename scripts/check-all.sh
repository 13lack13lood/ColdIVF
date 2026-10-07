#!/usr/bin/env bash
# Run before every commit.
# macOS targets get every feature and the tests. The three cross targets get only the pure-Rust crates
# (--no-default-features), because crates with C/C++ build scripts can't cross-compile from the Mac.
set -euo pipefail
cd "$(dirname "$0")/.."

minor=$(rustc --version | sed -E 's/^rustc 1\.([0-9]+).*/\1/')
[ "$minor" -ge 98 ] || { echo "rustc >= 1.98 required"; exit 1; }

cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
cargo clippy --workspace --all-targets --all-features --target x86_64-apple-darwin -- -D warnings
ROSETTA_ADVERTISE_AVX=1 cargo test -p coldivf --all-features --target x86_64-apple-darwin
for t in x86_64-unknown-linux-gnu aarch64-unknown-linux-gnu x86_64-pc-windows-msvc; do
  cargo clippy --workspace --all-targets --no-default-features --target "$t" -- -D warnings
done
echo "check-all: OK"
