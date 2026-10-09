//! A small `tracing` subscriber for the build tool. It prints DiskANN's log events to stderr and keeps the stage times
//! that DiskANN's builder logs at its checkpoints.

use std::fmt::Write as _;
use std::io::Error;
use std::sync::{Mutex, PoisonError};
use std::time::Instant;

use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id, Record};
use tracing::{Event, Level, Metadata, Subscriber};

/// The stage times logged so far, each a stage name and its seconds, in order.
static STAGES: Mutex<Vec<(String, f64)>> = Mutex::new(Vec::new());

/// Prints events up to `level` with the seconds since it was installed.
pub struct Log {
    start: Instant,
    level: Level,
}

impl Log {
    /// Installs the subscriber for the whole process. It fails if another one is installed.
    pub fn install(level: Level) -> Result<(), Error> {
        tracing::subscriber::set_global_default(Log {
            start: Instant::now(),
            level,
        })
        .map_err(Error::other)
    }
}

impl Subscriber for Log {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        *metadata.level() <= self.level
    }

    fn new_span(&self, _span: &Attributes<'_>) -> Id {
        Id::from_u64(1)
    }

    fn record(&self, _span: &Id, _values: &Record<'_>) {}

    fn record_follows_from(&self, _span: &Id, _follows: &Id) {}

    fn event(&self, event: &Event<'_>) {
        let mut message = Message::default();
        event.record(&mut message);

        if let Some(stage) = checkpoint(&message.0) {
            STAGES.lock().unwrap_or_else(PoisonError::into_inner).push(stage);
        }

        eprintln!(
            "[{:9.2} s] {} {}",
            self.start.elapsed().as_secs_f64(),
            event.metadata().target(),
            message.0
        );
    }

    fn enter(&self, _span: &Id) {}

    fn exit(&self, _span: &Id) {}
}

/// An event's message, then its other fields as name=value.
#[derive(Default)]
struct Message(String);

impl Visit for Message {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            let _ = write!(self.0, "{value:?}");
        } else {
            let _ = write!(self.0, " {}={value:?}", field.name());
        }
    }
}

/// The stage and the seconds in a checkpoint message of DiskANN's builder, which reads eg. "Time for DiskIndexBuild
/// [Checkpoint: PqConstruction] completed: 12.345 seconds, ...".
pub fn checkpoint(message: &str) -> Option<(String, f64)> {
    let (_, rest) = message.split_once("[Checkpoint: ")?;
    let (stage, rest) = rest.split_once(']')?;
    let (_, rest) = rest.split_once("completed: ")?;
    let (seconds, _) = rest.split_once(" seconds")?;
    Some((stage.trim().to_string(), seconds.trim().parse().ok()?))
}

/// The stage times logged so far.
pub fn stages() -> Vec<(String, f64)> {
    STAGES.lock().unwrap_or_else(PoisonError::into_inner).clone()
}

#[cfg(test)]
#[path = "../tests/unit/log.rs"]
mod tests;
