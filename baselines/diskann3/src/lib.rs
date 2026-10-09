#[cfg(not(target_os = "macos"))]
compile_error!("diskann3's readers use macOS file controls, so it builds only for macOS");

pub mod builder;
mod ffi;
pub mod log;
pub mod mac_reader;
pub mod prefix;
pub mod search;
pub mod usage;

#[cfg(test)]
#[path = "../tests/unit/testutil.rs"]
mod testutil;

use std::io::Error;

use diskann::ANNError;

/// The version of the DiskANN crates this library runs.
pub const DISKANN_VERSION: &str = "0.60.0";

/// A DiskANN error as an I/O error. DiskANN's errors carry their own kinds, so they all map to `Other` with their
/// message.
pub fn ann(e: ANNError) -> Error {
    Error::other(e.to_string())
}
