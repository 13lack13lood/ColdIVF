#!/usr/bin/env bash
# Run before every commit.
# macOS targets get every feature and the tests. The three cross targets get only the pure-Rust crates
# (--no-default-features), because crates with C/C++ build scripts can't cross-compile from the Mac.
# coldivf-bench turns on coldivf's bench and inmem features in every workspace build. Apps build coldivf without them,
# so it is also linted alone that way on every target, and tested that way on the Mac.
set -euo pipefail
cd "$(dirname "$0")/.."

minor=$(rustc --version | sed -E 's/^rustc 1\.([0-9]+).*/\1/')
[ "$minor" -ge 98 ] || { echo "rustc >= 1.98 required"; exit 1; }

# The kernel tests fail unless the fastest tier they find is the one named here, NEON with dotprod on this Mac, and
# AVX2 under Rosetta, which offers it when ROSETTA_ADVERTISE_AVX is set.
export COLDIVF_EXPECT_TIER=neon

cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
cargo clippy -p coldivf --all-targets --no-default-features -- -D warnings
cargo test -p coldivf --no-default-features
cargo clippy --workspace --all-targets --all-features --target x86_64-apple-darwin -- -D warnings
cargo clippy -p coldivf --all-targets --no-default-features --target x86_64-apple-darwin -- -D warnings
COLDIVF_EXPECT_TIER=avx2 ROSETTA_ADVERTISE_AVX=1 cargo test -p coldivf --all-features --target x86_64-apple-darwin
for t in x86_64-unknown-linux-gnu aarch64-unknown-linux-gnu x86_64-pc-windows-msvc; do
  cargo clippy --workspace --all-targets --no-default-features --target "$t" -- -D warnings
  cargo clippy -p coldivf --all-targets --no-default-features --target "$t" -- -D warnings
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

# The x86 release build must hold both x86 tiers with the instructions they were written for. No test on the Mac can
# run the AVX-512 tier, so this audit is its check here, and the Linux machine runs its tests. Each tier's functions
# are read apart. The AVX2 tier must use 256-bit byte shuffles, multiply-adds and fused multiply-adds, and nothing that
# needs AVX-512. The AVX-512 tier must use 512-bit byte shuffles, fused multiply-adds, compresses and the dot product of
# its Vector Neural Network Instructions (VNNI), and nothing from a part beyond F, BW, VL and VNNI: DQ, VBMI, VBMI2,
# BITALG, VPOPCNTDQ, IFMA, BF16, FP16 or the 256-bit VNNI (AVX-VNNI) that Ice Lake lacks. No other function of the
# crate may use those registers.
t=x86_64-unknown-linux-gnu
cargo clean -q -p coldivf --release --target "$t" --target-dir target/asm-audit
cargo rustc -q --release -p coldivf --lib --no-default-features --target "$t" --target-dir target/asm-audit \
  -- --emit=asm
asm=$(cat target/asm-audit/"$t"/release/deps/coldivf-*.s)
# The assembly of the functions whose symbols match $1 but not $2, each from its label to the next function's.
part() {
  awk -v m="$1" -v x="${2:-^$}" '/^[^.[:space:]][^[:space:]]*:/ { inside = $0 ~ m && $0 !~ x } inside' <<<"$asm"
}
avx2=$(part 7kernels4avx2)
avx512=$(part 7kernels6avx512)
# The rest of the crate's own functions, and any intrinsic that wasn't inlined, must not use 256-bit or 512-bit
# registers at all. Their symbols start with the crate's or core_arch's path, unlike eg. faer's code instantiated here.
own='^_(R[A-Za-z]*Cs[0-9A-Za-z]+_|ZN)(7coldivf|4core9core_arch)'
if grep -qE '%[yz]mm|%k[0-7]' <<<"$(part "$own" '7kernels(4avx2|6avx512)')"; then
  echo "asm audit: SIMD code outside the x86 kernels, eg. an intrinsic that wasn't inlined"
  exit 1
fi
for want in 'vpshufb.*%ymm' 'vpmaddubsw.*%ymm' 'vpmaddwd.*%ymm' 'vfmadd[0-9]+ps.*%ymm'; do
  grep -qE "$want" <<<"$avx2" || { echo "asm audit: the AVX2 kernels have no $want"; exit 1; }
done
if grep -qE '%zmm|%k[0-7]|\{vex\}|vpdpbusd' <<<"$avx2"; then
  echo "asm audit: the AVX2 kernels use AVX-512 or VNNI"
  exit 1
fi
for want in 'vpshufb.*%zmm' 'vpdpbusd.*%zmm' 'vfmadd[0-9]+ps.*%zmm' 'vcompressps' 'vpcompressd'; do
  grep -qE "$want" <<<"$avx512" || { echo "asm audit: the AVX-512 kernels have no $want"; exit 1; }
done
# DQ, then VBMI, VBMI2, BITALG and VPOPCNTDQ, then IFMA, BF16, FP16 and AVX-VNNI.
beyond=(
  'vpmullq' 'vpmovm2[dq]' 'vpmov[dq]2m' 'v(extract|insert)[fi](32x8|64x2)' 'vbroadcast[fi](32x2|32x8|64x2)'
  'vcvtt?p[sd]2u?qq' 'vcvtu?qq2p[sd]' 'vfpclass' 'vrange[ps]' 'vreduce[ps]' '\bk(add|test)w\b'
  '\bk(add|and|andn|mov|not|or|ortest|shiftl|shiftr|test|xnor|xor)b\b' 'v(and|andn|or|xor)p[sd]\b.*(%zmm|\{%k)'
  '\bvperm[it]?2?b\b' 'vpmultishiftqb' '\bvp(compress|expand)[bw]\b' '\bvpsh[lr]dv?[wdq]\b' '\bvpopcnt[bwdq]\b'
  'vpshufbitqmb' 'vpmadd52[hl]uq' 'vdpbf16ps' 'vcvtne2?ps2bf16' '\bv(add|sub|mul|div|min|max|sqrt)ph\b'
  '\bvf(n?m(add|sub)|maddsub|msubadd)[0-9]+ph\b' '\{vex\}'
)
if grep -qE "$(IFS='|' && echo "${beyond[*]}")" <<<"$avx512"; then
  echo "asm audit: the AVX-512 kernels use an instruction outside AVX-512 F, BW, VL and VNNI"
  exit 1
fi

echo "check-all: OK"
