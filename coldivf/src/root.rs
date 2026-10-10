//! Where an index's files are, and the commit that moves the index to a new generation.
//!
//! An index at `name` starts as one data file, `name`, with its log `name.delta`. That is generation 0. Its first
//! commit makes the folder `name.d`. From then on generation G is the data file `g<G>-<r>.data` and its log
//! `g<G>-<r>.log` there, where r is a random tag, so no name is ever used twice. Two root slots, the files `root.0` and
//! `root.1`, name the current generation, each in a 4 KiB block with a checksum, and the valid one with the higher
//! generation wins. No file that a reader maps is ever changed, renamed over or cut short.
//!
//! A commit of generation G + 1:
//! 1. writes its data file and syncs it fully
//! 2. writes its log with the changes made since the run's snapshot, and syncs it fully
//! 3. syncs the folder, so both new names survive a power loss
//! 4. writes slot (G + 1) mod 2, the one that doesn't name G, and syncs it, which is the commit
//! 5. deletes generation G's files. Readers that have them open go on reading them.
//!
//! A crash before step 4 leaves G current with every change in its log, and the next writer's open deletes the files
//! that no slot names. A crash in step 4 leaves a slot whose checksum fails, so the other slot stays current, and its
//! files still exist. A slot damaged later is made again from the newest data file and log that verify against each
//! other. When the files of the generation that the other slot names are still there, eg. because a power cut undid
//! their deletes, the newer generation wins only once its log holds changes past that one's, since a writer appends
//! those only after the commit. A build at the path of an index that has a root commits its file the same way, with an
//! empty log.

use std::fs::{self, File};
use std::hash::{BuildHasher, RandomState};
use std::io::{Error, ErrorKind, Read};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::format::{self, HEADER_BYTES, Layout, Root, read_header};
use crate::io;
use crate::maintain::job;
use crate::update::{self, Version, log};

/// The two root slots' file names, by slot.
pub(crate) const SLOTS: [&str; 2] = ["root.0", "root.1"];

/// The most times a lookup reads the slots again because a commit deleted the files they named in between.
const TRIES: usize = 3;

/// The folder that holds the generations and the root of the index at `path`, `name.d`.
pub(crate) fn folder(path: &Path) -> PathBuf {
    update::with_suffix(path, ".d")
}

/// Where a generation of an index is.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Located {
    pub(crate) generation: u64,
    pub(crate) data: PathBuf,
    pub(crate) log: PathBuf,
    /// What the root says of it, or `None` for an index before its first commit, whose data file is its path and whose
    /// log is `name.delta`.
    pub(crate) root: Option<Root>,
}

impl Located {
    /// The version of the generation's log.
    pub(crate) fn version(&self) -> Version {
        match &self.root {
            None => Version::One,
            Some(r) => Version::Two {
                index_id: r.index_id,
                generation: r.generation,
                base: r.base,
            },
        }
    }

    /// The index's id, or 0 before its first commit.
    pub(crate) fn index_id(&self) -> u64 {
        self.root.as_ref().map_or(0, |r| r.index_id)
    }
}

/// The index at `path` before its first commit.
fn legacy(path: &Path) -> Located {
    Located {
        generation: 0,
        data: path.to_path_buf(),
        log: update::log_path(path),
        root: None,
    }
}

/// The generation that `root` names, in the folder `dir`.
fn named(dir: &Path, root: Root) -> Located {
    Located {
        generation: root.generation,
        data: dir.join(&root.data),
        log: dir.join(&root.log),
        root: Some(root),
    }
}

/// The root slots in `dir`, each one that is there and whole, and whether one is there but damaged, i.e. neither whole
/// nor the zeros of a slot never written. A slot that can't be read for another reason than being missing fails.
fn read_slots(dir: &Path) -> Result<([Option<Root>; 2], bool), Error> {
    let (mut slots, mut damaged) = ([None, None], false);

    for (slot, name) in slots.iter_mut().zip(SLOTS) {
        let bytes = match fs::read(dir.join(name)) {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == ErrorKind::NotFound => continue,
            Err(e) => return Err(e),
        };
        *slot = format::read_root(&bytes).ok();
        damaged |= slot.is_none() && bytes.iter().any(|&b| b != 0);
    }

    Ok((slots, damaged))
}

/// The slot of `slots` with the higher generation.
fn newest(slots: &[Option<Root>; 2]) -> Option<&Root> {
    slots.iter().flatten().max_by_key(|r| r.generation)
}

/// Finds the current generation of the index at `path`, through symbolic links. Without a valid slot it is the data
/// file at `path`. When the files that the newest slot names are gone, it reads the slots again, since a commit may
/// have deleted them in between, and when they are still gone it takes the newest generation that the folder holds
/// whole.
pub(crate) fn locate(path: &Path) -> Result<Located, Error> {
    let path = &update::real_path(path);
    let dir = folder(path);
    let mut last: Option<Root> = None;

    for _ in 0..TRIES {
        let (slots, damaged) = read_slots(&dir)?;
        let Some(root) = newest(&slots).cloned() else {
            // Before the first commit, or after a failed one, the data file is the index's path. Only when it is gone
            // too did both slots break after a commit.
            if path.exists() || !dir.is_dir() {
                return Ok(legacy(path));
            }
            return Ok(recover(&dir, None, 0)?.unwrap_or_else(|| legacy(path)));
        };
        let at = named(&dir, root.clone());

        // A damaged slot may have named a newer generation, whose files are there too.
        if at.data.exists() && at.log.exists() {
            if damaged && let Some(newer) = committed_after(&dir, &root) {
                return Ok(newer);
            }
            return Ok(at);
        }

        if last.as_ref() == Some(&root) {
            let found = recover(&dir, Some(root.index_id), root.generation + 1)?;
            return found.ok_or_else(|| {
                Error::new(
                    ErrorKind::InvalidData,
                    "the files of the index's current generation are missing",
                )
            });
        }

        last = Some(root);
    }

    Err(Error::new(
        ErrorKind::NotFound,
        "the index kept changing while it was being opened",
    ))
}

/// One generation's data file and log, found in the folder without a slot.
struct Pair {
    located: Located,
    /// The sequence number after its log's last record.
    end: u64,
}

/// The newest generation from `from` on in `dir` whose data file and log verify against each other, and whose log
/// holds every record that the logs of older generations there hold past its first one. With `index_id` the log must
/// be that index's. It is what the root named before a slot broke. A generation whose files are damaged doesn't count.
fn recover(dir: &Path, index_id: Option<u64>, from: u64) -> Result<Option<Located>, Error> {
    let mut pairs: Vec<Pair> = generation_files(dir)?
        .into_iter()
        .filter_map(|(name, generation)| name.strip_suffix(".data").map(|stem| (stem.to_owned(), generation)))
        .filter_map(|(stem, generation)| verify_pair(dir, &stem, generation, index_id))
        .collect();
    pairs.sort_by_key(|p| std::cmp::Reverse(p.located.generation));

    for (i, pair) in pairs.iter().enumerate() {
        let holds_older = pairs[i + 1..].iter().all(|older| older.end <= pair.end);
        if pair.located.generation >= from && holds_older {
            return Ok(Some(pair.located.clone()));
        }
    }

    Ok(None)
}

/// The generation after the one that `root` names in `dir`, when its data file and log verify against each other and
/// its log holds changes past the end of `root`'s log. A writer appends those only once that generation is committed,
/// so it was, and a slot that named it broke. At the commit both logs end at the same change.
fn committed_after(dir: &Path, root: &Root) -> Option<Located> {
    let files = generation_files(dir).ok()?;
    let pair = |generation: u64| {
        let stems = files.iter().filter(|(_, g)| *g == generation);
        let stems = stems.filter_map(|(name, _)| name.strip_suffix(".data"));
        stems
            .filter_map(|stem| verify_pair(dir, stem, generation, Some(root.index_id)))
            .collect::<Vec<Pair>>()
    };
    let end = pair(root.generation)
        .into_iter()
        .find(|p| p.located.data == dir.join(&root.data))?
        .end;
    let newer = pair(root.generation + 1).into_iter().filter(|p| p.end > end);
    newer.max_by_key(|p| p.end).map(|p| p.located)
}

/// The header of the data file at `path`. A damaged one fails with `InvalidData`.
pub(crate) fn header_of(path: &Path) -> Result<Layout, Error> {
    let mut file = File::open(path)?;
    let len = usize::try_from(file.metadata()?.len())
        .map_err(|_| Error::new(ErrorKind::InvalidData, "the file is too large"))?;
    let mut first = vec![0u8; HEADER_BYTES.min(len)];
    file.read_exact(&mut first)?;
    read_header(&first, len)
}

/// Generation `generation`'s data file `<stem>.data` and log `<stem>.log` in `dir`, when the log names the data file
/// and is whole, and with `index_id` when it is that index's.
fn verify_pair(dir: &Path, stem: &str, generation: u64, index_id: Option<u64>) -> Option<Pair> {
    let (data, log_path) = (dir.join(format!("{stem}.data")), dir.join(format!("{stem}.log")));
    let layout = header_of(&data).ok()?;
    let bytes = fs::read(&log_path).ok()?;
    let head = log::read_header_2(&bytes)?;
    let fits = head.generation == generation && head.check == layout.check;

    if !fits || index_id.is_some_and(|id| id != head.index_id) {
        return None;
    }

    let version = Version::Two {
        index_id: head.index_id,
        generation,
        base: head.base,
    };
    let (records, _) = log::records(&bytes[log::HEADER_2_BYTES..], version, layout.dim).ok()?;
    let end = match records.last() {
        Some(r) => log::seq_of(r)? + 1,
        None => head.base,
    };
    let root = Root {
        index_id: head.index_id,
        generation,
        data: format!("{stem}.data"),
        log: format!("{stem}.log"),
        check: layout.check,
        base: head.base,
        writer: 0,
    };

    Some(Pair {
        located: named(dir, root),
        end,
    })
}

/// The generation files in `dir`, as their names with their generations. Nothing else in the folder counts, so a
/// cleanup never touches another file.
fn generation_files(dir: &Path) -> Result<Vec<(String, u64)>, Error> {
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(e),
    };
    let mut out = Vec::new();

    for entry in entries {
        let name = entry?.file_name().to_string_lossy().into_owned();
        if let Some(generation) = generation_of(&name) {
            out.push((name, generation));
        }
    }

    Ok(out)
}

/// The generation of a generation file's name, `g<G>-<r>.data` or `g<G>-<r>.log` with r 16 hexadecimal digits.
fn generation_of(name: &str) -> Option<u64> {
    let stem = name.strip_suffix(".data").or_else(|| name.strip_suffix(".log"))?;
    let (generation, tag) = stem.strip_prefix('g')?.split_once('-')?;
    let digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());

    if !digits(generation) || tag.len() != 16 || !tag.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }

    generation.parse().ok()
}

/// The resume files in `dir`, by name, which jobs leave next to their data files.
fn resume_files(dir: &Path) -> Vec<String> {
    let Ok(entries) = fs::read_dir(dir) else {
        return Vec::new();
    };
    let names = entries.flatten().map(|e| e.file_name().to_string_lossy().into_owned());
    names.filter(|n| n.ends_with(".resume")).collect()
}

/// A random number, eg. an id or a file's tag.
pub(crate) fn random() -> u64 {
    RandomState::new().hash_one(std::time::SystemTime::now())
}

/// An open file's device and inode, which tell whether a path still leads to it.
#[cfg(unix)]
fn identity(meta: &fs::Metadata) -> (u64, u64) {
    use std::os::unix::fs::MetadataExt;
    (meta.dev(), meta.ino())
}

/// Elsewhere every file looks the same.
#[cfg(not(unix))]
fn identity(_meta: &fs::Metadata) -> (u64, u64) {
    (0, 0)
}

/// The root slots, open, with the device and inode of each.
struct Slots {
    files: [File; 2],
    ids: [(u64, u64); 2],
}

/// An index's files as its writer holds them: the writer's lock, the root slots, and the current generation.
pub(crate) struct Owner {
    path: PathBuf,
    dir: PathBuf,
    lock: Arc<File>,
    /// The slots, once the folder exists.
    slots: Option<Slots>,
    current: Located,
    /// This writer's id, which each slot it writes records.
    writer: u64,
    /// Set while a slot write is under way, so a failure then leaves unknown whether it committed.
    unsure: bool,
}

impl Owner {
    /// Takes the writer's lock of the index at `path` and finds its current generation. A slot that names no whole
    /// generation is made again. When another writer holds the lock, it fails with `WouldBlock`.
    pub(crate) fn open(path: &Path) -> Result<Owner, Error> {
        Owner::with_lock(path, Arc::new(update::lock(path)?))
    }

    /// The same, with the lock already taken.
    pub(crate) fn with_lock(path: &Path, lock: Arc<File>) -> Result<Owner, Error> {
        let dir = folder(path);
        let mut owner = Owner {
            path: path.to_path_buf(),
            current: locate(path)?,
            dir,
            lock,
            slots: None,
            writer: random(),
            unsure: false,
        };

        if let Some(root) = owner.current.root.clone() {
            owner.slots = Some(owner.open_slots()?);
            let slot = (root.generation % 2) as usize;
            if read_slots(&owner.dir)?.0[slot].as_ref() != Some(&root) {
                owner.check_slot(slot)?;
                owner.write_slot(&root)?;
            }
        }

        Ok(owner)
    }

    pub(crate) fn lock(&self) -> Arc<File> {
        Arc::clone(&self.lock)
    }

    /// The index's path.
    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    /// This writer's id, which each slot it writes records.
    pub(crate) fn writer(&self) -> u64 {
        self.writer
    }

    pub(crate) fn current(&self) -> &Located {
        &self.current
    }

    /// Whether the last commit failed in its slot write, so whether it committed is unknown.
    pub(crate) fn unsure(&self) -> bool {
        self.unsure
    }

    /// Deletes the files in the folder that belong to no generation but the current one: what a crash or a failed run
    /// left, and the old generations that a commit couldn't delete, eg. on Windows while a reader mapped them. Before
    /// the first commit every generation file is a leftover. It is best effort, so a file that can't go stays.
    pub(crate) fn clean(&self) {
        let Ok(files) = generation_files(&self.dir) else {
            return;
        };
        // A job of the current generation that stopped early leaves its data file and resume file for the next one.
        let kept = job::kept(self);
        let keep = |name: &str| {
            let current = self.current.root.as_ref();
            current.is_some_and(|r| name == r.data || name == r.log) || kept.iter().any(|k| k == name)
        };

        for (name, _) in files.iter().filter(|(name, _)| !keep(name)) {
            let _ = io::remove(&self.dir.join(name));
        }

        for name in resume_files(&self.dir).iter().filter(|name| !keep(name)) {
            let _ = io::remove(&self.dir.join(name));
        }

        // The data file and log from before the first commit, if the commit couldn't delete them.
        if self.current.root.is_some() {
            let _ = io::remove(&self.path);
            let _ = io::remove(&update::log_path(&self.path));
        }
    }

    /// Makes the folder and both root slots unless they exist, then the data file of the next generation, empty.
    /// Returns the generation's files, which are the caller's until it commits them, and the data file open.
    pub(crate) fn next(&mut self) -> Result<(Next, File), Error> {
        self.make_folder()?;
        let (generation, index_id) = (
            self.current.generation + 1,
            self.current.root.as_ref().map_or_else(random, |r| r.index_id),
        );

        loop {
            let stem = format!("g{generation}-{:016x}", random());
            let data = self.dir.join(format!("{stem}.data"));
            match io::create_new(&data) {
                Ok(file) => {
                    let next = Next {
                        generation,
                        index_id,
                        log: self.dir.join(format!("{stem}.log")),
                        data,
                        dir: self.dir.clone(),
                        committed: false,
                    };
                    return Ok((next, file));
                }
                Err(e) if e.kind() == ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e),
            }
        }
    }

    /// The next generation's files for the data file `data` that an earlier job of the current generation left, with
    /// the data file open, so a job goes on with it.
    pub(crate) fn reopen(&mut self, data: &Path) -> Result<(Next, File), Error> {
        self.make_folder()?;
        let name = file_name(data);
        let (generation, index_id) = (
            self.current.generation + 1,
            self.current.root.as_ref().map_or_else(random, |r| r.index_id),
        );

        if generation_of(&name) != Some(generation) || !name.ends_with(".data") {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "the file isn't the next generation's data file",
            ));
        }

        let file = fs::OpenOptions::new().read(true).write(true).open(data)?;
        let next = Next {
            generation,
            index_id,
            log: data.with_extension("log"),
            data: data.to_path_buf(),
            dir: self.dir.clone(),
            committed: false,
        };
        Ok((next, file))
    }

    /// The same, for the data file `tmp`, which a build wrote next to the index and synced. It renames it into the
    /// folder as the next generation's data file.
    pub(crate) fn adopt(&mut self, tmp: &Path) -> Result<Next, Error> {
        let (next, file) = self.next()?;
        drop(file);
        fs::rename(tmp, &next.data)?;
        Ok(next)
    }

    /// Commits `next`, whose data file and log are written, synced and named in a synced folder. Its data file's header
    /// checksum is `check`, and its log starts at sequence number `base`. It writes the slot that doesn't name the
    /// current generation, and syncs it. Returns the generation that was current. A failure before the write leaves
    /// that generation current. A failure in it leaves the outcome unknown, so `unsure` then says so.
    pub(crate) fn commit(&mut self, next: &mut Next, check: u32, base: u64) -> Result<Located, Error> {
        let root = Root {
            index_id: next.index_id,
            generation: next.generation,
            data: file_name(&next.data),
            log: file_name(&next.log),
            check,
            base,
            writer: self.writer,
        };
        self.check_slot((root.generation % 2) as usize)?;
        // From the write on, the files may be the current generation's, so nothing deletes them but a later cleanup.
        next.committed = true;
        self.write_slot(&root)?;
        io::step("committed", &self.dir);
        Ok(std::mem::replace(&mut self.current, named(&self.dir, root)))
    }

    /// Checks that slot `slot`'s name still leads to the file that the writer holds open.
    fn check_slot(&self, slot: usize) -> Result<(), Error> {
        let slots = self
            .slots
            .as_ref()
            .ok_or_else(|| Error::other("the index has no root slots"))?;

        if identity(&fs::metadata(self.dir.join(SLOTS[slot]))?) != slots.ids[slot] {
            return Err(Error::other("a root slot was replaced while the writer had it open"));
        }

        Ok(())
    }

    /// Writes `root` into its slot, generation mod 2, and syncs it.
    fn write_slot(&mut self, root: &Root) -> Result<(), Error> {
        let slot = (root.generation % 2) as usize;
        let path = self.dir.join(SLOTS[slot]);
        let file = &self
            .slots
            .as_ref()
            .ok_or_else(|| Error::other("the index has no root slots"))?
            .files[slot];
        self.unsure = true;
        io::write_all_at(file, &format::encode_root(root), 0, &path)?;
        io::sync_file(file, &path)?;
        self.unsure = false;
        Ok(())
    }

    /// Deletes the files of `old`, a generation that the root no longer names. On Windows a file that a reader maps
    /// can't be deleted, so it stays until a writer's open cleans the folder.
    pub(crate) fn retire(&self, old: &Located) {
        let _ = io::remove(&old.data);
        let _ = io::remove(&old.log);
    }

    /// Makes the folder and both slots, each a zeroed block that no lookup takes, unless the folder has them, and syncs
    /// what it made. Then it opens the slots.
    fn make_folder(&mut self) -> Result<(), Error> {
        if self.slots.is_some() {
            return Ok(());
        }

        // A folder or a slot left by an earlier try may not be synced yet, so they are synced either way.
        io::create_folder(&self.dir)?;
        io::sync_parent(&self.dir)?;

        for name in SLOTS {
            let path = self.dir.join(name);
            match io::create_new(&path) {
                Ok(file) => {
                    io::write_all_at(&file, &[0u8; format::ROOT_BYTES], 0, &path)?;
                    io::sync_file(&file, &path)?;
                }
                Err(e) if e.kind() == ErrorKind::AlreadyExists => {}
                Err(e) => return Err(e),
            }
        }

        io::sync_folder(&self.dir)?;
        self.slots = Some(self.open_slots()?);
        Ok(())
    }

    fn open_slots(&self) -> Result<Slots, Error> {
        let open = |name: &str| fs::OpenOptions::new().read(true).write(true).open(self.dir.join(name));
        let files = [open(SLOTS[0])?, open(SLOTS[1])?];
        let ids = [identity(&files[0].metadata()?), identity(&files[1].metadata()?)];
        Ok(Slots { files, ids })
    }
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// The files of a generation that a run writes. Until a commit makes them current they are the run's own, and dropping
/// them deletes them.
pub(crate) struct Next {
    pub(crate) generation: u64,
    pub(crate) index_id: u64,
    pub(crate) data: PathBuf,
    pub(crate) log: PathBuf,
    dir: PathBuf,
    committed: bool,
}

impl Next {
    /// The version of the generation's log, whose first record has the sequence number `base`.
    pub(crate) fn version(&self, base: u64) -> Version {
        Version::Two {
            index_id: self.index_id,
            generation: self.generation,
            base,
        }
    }

    /// The generation as a lookup would find it once its log starts at `base` and its data file's header checksum is
    /// `check`.
    pub(crate) fn located(&self, check: u32, base: u64) -> Located {
        let root = Root {
            index_id: self.index_id,
            generation: self.generation,
            data: file_name(&self.data),
            log: file_name(&self.log),
            check,
            base,
            writer: 0,
        };
        named(&self.dir, root)
    }

    /// Syncs the folder, so the names of the data file and the log survive a power loss.
    pub(crate) fn seal(&self) -> Result<(), Error> {
        io::sync_folder(&self.dir)
    }

    /// Leaves the files where they are, eg. for a job that goes on with them later.
    pub(crate) fn keep(mut self) {
        self.committed = true;
    }
}

impl Drop for Next {
    fn drop(&mut self) {
        if !self.committed {
            let _ = io::remove(&self.data);
            let _ = io::remove(&self.log);
        }
    }
}

/// Puts the data file `tmp`, which a build wrote next to `out`, in place as the index at `out`. It first syncs it
/// fully. Before the index's first commit it renames `tmp` over `out`, as builds always have. After it, it commits
/// `tmp` as a new generation with an empty log, so the root names it. It takes the writer's lock for that, so it fails
/// with `WouldBlock` while a writer has the index open.
pub(crate) fn publish(out: &Path, tmp: &Path) -> Result<(), Error> {
    io::sync_file(&fs::OpenOptions::new().read(true).write(true).open(tmp)?, tmp)?;
    let path = update::real_path(out);
    let mut owner = Owner::open(&path)?;

    if owner.current.root.is_none() {
        fs::rename(tmp, out)?;
        return io::sync_parent(out);
    }

    let mut next = owner.adopt(tmp)?;
    let layout = header_of(&next.data)?;
    let base = 1;
    drop(log::Log::create(
        &next.log,
        next.version(base),
        &layout,
        &[],
        owner.lock(),
    )?);
    next.seal()?;
    let old = owner.commit(&mut next, layout.check, base)?;
    owner.retire(&old);
    Ok(())
}

#[cfg(test)]
#[path = "../tests/unit/root.rs"]
mod tests;
