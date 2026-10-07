//! An index file opened for search. Opening reads the first 16 KiB and maps the whole file without loading it. The
//! cluster tables load on the first search.

use std::fs::File;
use std::io::{Error, ErrorKind, Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::OnceLock;

use crate::format::{self, FIRST_READ, Layout, Part, Span, Tables};
use crate::io::map::Map;
use crate::kernels::Tier;
use crate::quantization::rotation::Rotation;
use crate::quantization::sq8;
use crate::search::route::Centers;
use crate::search::{RowPages, View, run, run_route};
use crate::{Hit, LeafId, SearchCtx, SearchParams};

pub(crate) struct Inner {
    map: Map,
    layout: Layout,
    mean: Vec<f32>,
    rotation: Rotation,
    tables: OnceLock<Tables>,
    tier: Tier,
}

impl Inner {
    /// Opens `path`. A damaged file fails with `InvalidData`.
    pub(crate) fn open(path: &Path) -> Result<Inner, Error> {
        let mut file = File::open(path)?;
        let len = usize::try_from(file.metadata()?.len()).map_err(|_| damaged("the file is too large"))?;
        let mut first = vec![0u8; FIRST_READ.min(len)];
        file.read_exact(&mut first)?;
        let layout = format::read_header(&first, len)?;

        // Above 2806 dimensions the rotation runs past the first read.
        let span = layout.span(Part::Rotation);
        let rotation_bytes = match first.get(span.range()) {
            Some(bytes) => bytes.to_vec(),
            None => read_span(&mut file, span)?,
        };
        let (rotation, mean) = format::read_rotation(&rotation_bytes, &layout)?;
        let map = Map::new(&file)?;

        if map.bytes().len() != len {
            return Err(damaged("the file changed while it was being opened"));
        }

        Ok(Inner {
            map,
            layout,
            mean,
            rotation,
            tables: OnceLock::new(),
            tier: Tier::best(),
        })
    }

    /// Asks the operating system (OS) to start reading the cluster tables, so the read overlaps the embedding model.
    pub(crate) fn prepare(&self) -> Result<(), Error> {
        let (centers, extents) = (self.layout.span(Part::Centers), self.layout.span(Part::Extents));
        self.map.will_need(centers.offset, extents.end() - centers.offset);
        Ok(())
    }

    pub(crate) fn search<'a>(
        &self,
        query: &[f32],
        params: &SearchParams,
        ctx: &'a mut SearchCtx,
    ) -> Result<&'a [Hit], Error> {
        run(&self.view(self.tables()?), query, params, ctx)
    }

    pub(crate) fn route(
        &self,
        query: &[f32],
        nprobe: usize,
        ctx: &mut SearchCtx,
        out: &mut Vec<LeafId>,
    ) -> Result<(), Error> {
        run_route(&self.view(self.tables()?), query, nprobe, ctx, out)
    }

    /// The cluster tables, read and checked the first time. Two threads may both read them, and one copy wins.
    fn tables(&self) -> Result<&Tables, Error> {
        if let Some(tables) = self.tables.get() {
            return Ok(tables);
        }

        let tables = format::read_tables(self.map.bytes(), &self.layout)?;

        Ok(self.tables.get_or_init(|| tables))
    }

    fn view<'a>(&'a self, tables: &'a Tables) -> View<'a> {
        let (bytes, layout) = (self.map.bytes(), &self.layout);
        let row_bytes = sq8::row_bytes(layout.dim);
        View {
            dim: layout.dim,
            metric: layout.metric,
            mean: &self.mean,
            rotation: &self.rotation,
            centers: Centers {
                codes: &bytes[tables.codes.range()],
                factors: &tables.factors,
            },
            lists: &tables.lists,
            blocks: &bytes[layout.span(Part::Codes).range()],
            rows: RowPages {
                bytes: &bytes[layout.span(Part::Rows).range()],
                row_bytes,
                per_page: layout.page_size / row_bytes,
                page_size: layout.page_size,
            },
            tier: self.tier,
        }
    }

    /// The same index with another kernel tier, so tests can check that every tier gives the same results.
    #[cfg(test)]
    pub(crate) fn with_tier(mut self, tier: Tier) -> Inner {
        self.tier = tier;
        self
    }

    /// The whole file, and what its header says, for checks.
    #[cfg(test)]
    pub(crate) fn parts(&self) -> (&[u8], &Layout) {
        (self.map.bytes(), &self.layout)
    }
}

fn read_span(file: &mut File, span: Span) -> Result<Vec<u8>, Error> {
    let mut bytes = vec![0u8; span.len];
    file.seek(SeekFrom::Start(span.offset as u64))?;
    file.read_exact(&mut bytes)?;
    Ok(bytes)
}

fn damaged(message: &str) -> Error {
    Error::new(ErrorKind::InvalidData, message)
}

#[cfg(test)]
#[path = "../tests/unit/index.rs"]
mod tests;
