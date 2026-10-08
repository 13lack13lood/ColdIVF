//! An index file opened for search. Opening reads the first 16 KiB, maps the whole file without loading it and opens a
//! second handle for cold reads, which every later read goes through. The cluster tables load on the first search.

use std::fs::File;
use std::io::{Error, ErrorKind};
use std::path::Path;
use std::sync::{Arc, OnceLock};

use crate::format::{self, FIRST_READ, Layout, Part, Tables};
use crate::io::map::Map;
use crate::io::{self, Cold, Span};
use crate::kernels::Tier;
use crate::quantization::rotation::Rotation;
use crate::quantization::sq8;
use crate::search::cold::{Reads, reads_for};
use crate::search::rerank::rerank;
use crate::search::route::Centers;
use crate::search::{InPlace, RowPages, View, check_search, invalid, prepare_query, run, run_route, scratch_for};
use crate::{Hit, LeafId, Limits, SearchCtx, SearchParams};

pub(crate) struct Inner {
    cold: Cold,
    map: Map,
    len: usize,
    layout: Layout,
    mean: Vec<f32>,
    rotation: Rotation,
    tables: OnceLock<Tables>,
    tier: Tier,
}

impl Inner {
    /// Opens `path`. A damaged file fails with `InvalidData`.
    pub(crate) fn open(path: &Path) -> Result<Inner, Error> {
        let file = File::open(path)?;
        let len = usize::try_from(file.metadata()?.len()).map_err(|_| damaged("the file is too large"))?;
        let mut first = vec![0u8; FIRST_READ.min(len)];
        io::read_at(&file, &mut first, 0)?;
        let layout = format::read_header(&first, len)?;

        // Above 2806 dimensions the rotation runs past the first read.
        let span = layout.span(Part::Rotation);
        let rotation_bytes = match first.get(span.range()) {
            Some(bytes) => bytes.to_vec(),
            None => read_span(&file, span)?,
        };
        let (rotation, mean) = format::read_rotation(&rotation_bytes, &layout)?;
        let map = Map::new(&file)?;

        if map.bytes().len() != len {
            return Err(damaged("the file changed while it was being opened"));
        }

        Ok(Inner {
            cold: io::open_cold(path, &first)?,
            map,
            len,
            layout,
            mean,
            rotation,
            tables: OnceLock::new(),
            tier: Tier::best(),
        })
    }

    /// Asks the operating system (OS) to start reading the cluster tables, so the read overlaps the embedding model.
    /// Once the tables are loaded, it does nothing.
    pub(crate) fn prepare(&self) {
        if self.tables.get().is_none() {
            let span = format::tables_span(&self.layout);
            io::will_need(&self.cold.file, span.offset, span.len);
        }
    }

    /// Checks a search's arguments against this index and `limits`.
    pub(crate) fn check(&self, query: &[f32], params: &SearchParams, limits: &Limits) -> Result<(), Error> {
        check_search(query, self.layout.dim, params, limits)
    }

    /// Searches through the memory map, which leaves the reads to page faults. Tests check the cold search against it.
    pub(crate) fn search_mapped<'a>(
        &self,
        query: &[f32],
        params: &SearchParams,
        ctx: &'a mut SearchCtx,
    ) -> Result<&'a [Hit], Error> {
        run(&self.view(self.tables()?), &self.in_place(), query, params, ctx)
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

    /// Starts reading `leaves` in one wave, through `ctx`'s reads of this index, which the first fetch makes. Returns
    /// the reads' slot in `ctx`.
    pub(crate) fn fetch(self: &Arc<Inner>, leaves: &[LeafId], ctx: &mut SearchCtx) -> Result<usize, Error> {
        let tables = self.tables()?;

        if leaves.len() > ctx.limits.max_nprobe {
            return Err(invalid("more leaves than the search context's max_nprobe"));
        }

        if leaves.iter().any(|&l| l as usize >= tables.lists.len()) {
            return Err(invalid("a leaf is past the index's clusters"));
        }

        let SearchCtx { limits, reads, .. } = ctx;
        let make = || Reads::new(self, (self.cold.wave)(), &self.layout, self.len, &tables.lists, limits);
        let slot = reads_for(reads, self, make);
        reads[slot].fetch(&self.cold.file, &tables.lists, leaves)?;
        Ok(slot)
    }

    /// Scores the fetched clusters for `query` and starts reading the rows of the best `n`, using the reads in `slot`.
    pub(crate) fn prefetch_rows(&self, query: &[f32], n: usize, ctx: &mut SearchCtx, slot: usize) -> Result<(), Error> {
        let params = SearchParams {
            k: 0,
            nprobe: 0,
            rerank: n,
        };
        self.check(query, &params, &ctx.limits)?;

        let view = self.view(self.tables()?);
        let SearchCtx {
            limits, scratch, reads, ..
        } = ctx;
        let s = scratch_for(scratch, view.dim, view.lists.len(), limits);
        prepare_query(&view, query, s)?;
        s.candidates.reset(n);
        let r = &mut reads[slot];
        r.scan(&self.cold.file, &view, s)?;
        r.prefetch(&self.cold.file, s.candidates.as_slice())
    }

    /// Ranks the fetched clusters for `query`, with the reads in `slot`, and returns the `params.k` nearest. The hits
    /// live in `ctx`.
    pub(crate) fn search_fetched<'a>(
        &self,
        query: &[f32],
        params: &SearchParams,
        ctx: &'a mut SearchCtx,
        slot: usize,
    ) -> Result<&'a [Hit], Error> {
        self.check(query, params, &ctx.limits)?;

        let view = self.view(self.tables()?);
        let SearchCtx {
            limits,
            scratch,
            reads,
            hits,
            ..
        } = ctx;
        let s = scratch_for(scratch, view.dim, view.lists.len(), limits);
        prepare_query(&view, query, s)?;
        s.candidates.reset(params.rerank.max(params.k));
        let r = &mut reads[slot];
        r.scan(&self.cold.file, &view, s)?;
        r.read_rows(&self.cold.file, s.candidates.as_slice())?;
        let row = |id: u32| r.row(id);
        rerank(
            s.candidates.as_slice(),
            row,
            &s.offset,
            params.k,
            view.metric,
            &mut s.rescored,
            hits,
        )?;

        Ok(&hits[..])
    }

    /// The cluster tables, read and checked the first time. Two threads may both read them, and one copy wins.
    fn tables(&self) -> Result<&Tables, Error> {
        if let Some(tables) = self.tables.get() {
            return Ok(tables);
        }

        let bytes = read_span(&self.cold.file, format::tables_span(&self.layout))?;
        let tables = format::read_tables(bytes, &self.layout)?;

        Ok(self.tables.get_or_init(|| tables))
    }

    fn view<'a>(&'a self, tables: &'a Tables) -> View<'a> {
        View {
            dim: self.layout.dim,
            metric: self.layout.metric,
            mean: &self.mean,
            rotation: &self.rotation,
            centers: Centers {
                codes: &tables.codes,
                factors: &tables.factors,
            },
            lists: &tables.lists,
            tier: self.tier,
        }
    }

    fn in_place(&self) -> InPlace<'_> {
        let (bytes, layout) = (self.map.bytes(), &self.layout);
        let row_bytes = sq8::row_bytes(layout.dim);
        InPlace {
            blocks: &bytes[layout.span(Part::Codes).range()],
            rows: RowPages {
                bytes: &bytes[layout.span(Part::Rows).range()],
                row_bytes,
                per_page: layout.page_size / row_bytes,
                page_size: layout.page_size,
            },
        }
    }

    /// The same index with another kernel tier, so tests can check that every tier gives the same results.
    #[cfg(test)]
    pub(crate) fn with_tier(mut self, tier: Tier) -> Inner {
        self.tier = tier;
        self
    }

    /// The same index with another read backend, so tests can check that every backend gives the same results.
    #[cfg(test)]
    pub(crate) fn with_wave(mut self, wave: io::NewWave) -> Inner {
        self.cold.wave = wave;
        self
    }

    /// The whole file, and what its header says, for checks.
    #[cfg(test)]
    pub(crate) fn parts(&self) -> (&[u8], &Layout) {
        (self.map.bytes(), &self.layout)
    }
}

fn read_span(file: &File, span: Span) -> Result<Vec<u8>, Error> {
    let mut bytes = vec![0u8; span.len];
    io::read_at(file, &mut bytes, span.offset)?;
    Ok(bytes)
}

fn damaged(message: &str) -> Error {
    Error::new(ErrorKind::InvalidData, message)
}

#[cfg(test)]
#[path = "../tests/unit/index.rs"]
mod tests;
