//! ColdIVF's Python module, `coldivf`. `Index` opens an index file and searches it, and `build` writes one from an
//! array of vectors. Build it into a Python environment with `maturin develop --release`, or a wheel with
//! `maturin build --release`.

#[cfg(feature = "python")]
mod module;
