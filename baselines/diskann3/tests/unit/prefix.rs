use std::io::{ErrorKind, Read, Seek, SeekFrom, Write};

use super::*;
use crate::testutil::{Scratch, base_file};

/// The bytes a prefix of `rows` rows of width `width` holds, from `base_file`.
fn expected(rows: usize, width: usize) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&(rows as u32).to_le_bytes());
    out.extend_from_slice(&(width as u32).to_le_bytes());

    for v in 0..rows * width {
        out.extend_from_slice(&(v as f32).to_le_bytes());
    }

    out
}

#[test]
fn the_header_gives_the_rows_and_the_width() {
    let dir = Scratch::new("header");
    let path = dir.path().join("base.fbin");
    base_file(&path, 7, 3);
    assert_eq!(header(&path).unwrap(), (7, 3));
    std::fs::OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap()
        .write_all(&[0; 4])
        .unwrap();
    assert_eq!(header(&path).unwrap_err().kind(), ErrorKind::InvalidData);
}

#[test]
fn the_prefix_reads_as_a_file_of_its_own() {
    let dir = Scratch::new("prefix");
    let path = dir.path().join("base.fbin");
    base_file(&path, 1000, 5);
    let storage = PrefixStorage::new(&path, 300).unwrap();
    let want = expected(300, 5);
    assert_eq!(storage.get_length(storage.name()).unwrap(), want.len() as u64);
    assert!(storage.exists(storage.name()));

    let mut all = Vec::new();
    storage
        .open_reader(storage.name())
        .unwrap()
        .read_to_end(&mut all)
        .unwrap();
    assert_eq!(all, want);

    // Odd-sized reads that cross the header and the chunks give the same bytes.
    let mut reader = storage.open_reader(storage.name()).unwrap();
    let mut pieces = Vec::new();
    let mut piece = [0u8; 7];

    loop {
        let n = reader.read(&mut piece).unwrap();

        if n == 0 {
            break;
        }

        pieces.extend_from_slice(&piece[..n]);
    }

    assert_eq!(pieces, want);
}

#[test]
fn seeks_move_the_prefix_reader() {
    let dir = Scratch::new("seek");
    let path = dir.path().join("base.fbin");
    base_file(&path, 50, 4);
    let storage = PrefixStorage::new(&path, 20).unwrap();
    let want = expected(20, 4);
    let mut reader = storage.open_reader(storage.name()).unwrap();

    for at in [100u64, 4, 0, 319, 8, 200] {
        reader.seek(SeekFrom::Start(at)).unwrap();
        let mut got = [0u8; 9];
        let n = reader.read(&mut got).unwrap();
        let end = (at as usize + 9).min(want.len());
        assert!(n > 0 && n <= end - at as usize, "a read at {at} gave {n} bytes");
        assert_eq!(&got[..n], &want[at as usize..at as usize + n], "at {at}");
    }

    assert_eq!(reader.seek(SeekFrom::End(-4)).unwrap(), want.len() as u64 - 4);
    let mut tail = Vec::new();
    reader.read_to_end(&mut tail).unwrap();
    assert_eq!(tail, &want[want.len() - 4..]);
    assert_eq!(reader.seek(SeekFrom::Current(-2)).unwrap(), want.len() as u64 - 2);
    assert!(reader.seek(SeekFrom::Current(-1000)).is_err());
    reader.seek(SeekFrom::Start(10_000)).unwrap();
    assert_eq!(reader.read(&mut [0u8; 4]).unwrap(), 0);
}

#[test]
fn a_prefix_must_fit_the_base() {
    let dir = Scratch::new("fit");
    let path = dir.path().join("base.fbin");
    base_file(&path, 10, 2);
    assert_eq!(
        PrefixStorage::new(&path, 0).err().unwrap().kind(),
        ErrorKind::InvalidInput
    );
    assert_eq!(
        PrefixStorage::new(&path, 11).err().unwrap().kind(),
        ErrorKind::InvalidInput
    );
    assert!(PrefixStorage::new(&path, 10).is_ok());
}

#[test]
fn other_paths_go_to_the_file_system_and_the_prefix_stays_read_only() {
    let dir = Scratch::new("files");
    let path = dir.path().join("base.fbin");
    base_file(&path, 10, 2);
    let storage = PrefixStorage::new(&path, 5).unwrap();
    let other = dir.path().join("other.bin");
    let other = other.to_str().unwrap();

    assert!(!storage.exists(other));
    storage.create_for_write(other).unwrap().write_all(b"hello").unwrap();
    assert!(storage.exists(other));
    assert_eq!(storage.get_length(other).unwrap(), 5);
    let mut text = String::new();
    storage.open_reader(other).unwrap().read_to_string(&mut text).unwrap();
    assert_eq!(text, "hello");
    storage.delete(other).unwrap();
    assert!(!storage.exists(other));

    assert_eq!(
        storage.create_for_write(storage.name()).err().unwrap().kind(),
        ErrorKind::PermissionDenied
    );
    assert_eq!(
        storage.open_writer(storage.name()).err().unwrap().kind(),
        ErrorKind::PermissionDenied
    );
    assert_eq!(
        storage.delete(storage.name()).err().unwrap().kind(),
        ErrorKind::PermissionDenied
    );
    assert!(path.exists());
}
