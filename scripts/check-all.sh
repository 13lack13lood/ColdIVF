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

# The Arm release builds must not use instructions that M1 Macs lack (i8mm, bf16 and SME), since no test on a newer
# Mac would notice. The audit reads the compiler's assembly, and first checks that it sees the dot-product kernels.
# It builds in its own folder and clears the crate there first, so the compiler always runs and writes the assembly.
for t in aarch64-apple-darwin aarch64-unknown-linux-gnu; do
  cargo clean -q -p coldivf --release --target "$t" --target-dir target/asm-audit
  cargo rustc -q --release -p coldivf --lib --no-default-features --target "$t" --target-dir target/asm-audit -- --emit=asm
  asm=$(cat target/asm-audit/"$t"/release/deps/coldivf-*.s)
  grep -qw sdot <<<"$asm" || { echo "asm audit: no dot-product code in the $t build"; exit 1; }
  if grep -qEw 'usdot|sudot|smmla|ummla|usmmla|bfdot|bfmlalb|bfmlalt|bfmmla|bfcvt|bfcvtn|bfcvtn2|smstart|smstop' <<<"$asm" ||
    grep -qE '\bz[0-9]+\.[bhsdq]\b|\bza\b|\bza[0-9]' <<<"$asm"; then
    echo "asm audit: the $t build uses an instruction that M1 Macs lack"
    exit 1
  fi
done

echo "check-all: OK"
