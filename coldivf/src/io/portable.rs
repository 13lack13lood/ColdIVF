//! The plainest backend. It reads each range when asked for it, one at a time, and every other backend must give the
//! same bytes.

use std::fs::File;
use std::io::Error;

use super::{InOrder, Memory, Range, Wave};
use crate::trace::ReadPath;

#[derive(Default)]
pub(crate) struct PortableWave(InOrder);

impl Wave for PortableWave {
    fn page(&self) -> usize {
        4096
    }

    fn submit(&mut self, lane: usize, _file: &File, ranges: &[Range], _arena: Memory) -> Result<(), Error> {
        self.0.start(lane, ranges.len());
        Ok(())
    }

    fn next(&mut self, lane: usize, file: &File, ranges: &[Range], arena: Memory) -> Result<Option<usize>, Error> {
        self.0.read(lane, file, ranges, arena)
    }

    fn read_path(&self) -> ReadPath {
        ReadPath {
            backend: "portable",
            ..ReadPath::default()
        }
    }
}
