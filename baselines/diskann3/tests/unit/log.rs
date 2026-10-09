use super::*;

#[test]
fn a_checkpoint_message_gives_its_stage_and_seconds() {
    let message = "Time for DiskIndexBuild [Checkpoint: PqConstruction] completed: 12.345 seconds, 1.234B cycles, \
                   99.000% CPU time, peak memory 1.234 GBs";
    assert_eq!(checkpoint(message), Some(("PqConstruction".to_string(), 12.345)));
}

#[test]
fn other_messages_give_nothing() {
    assert_eq!(checkpoint("Sector #100000 written"), None);
    assert_eq!(
        checkpoint("Time for DiskIndexBuild [Checkpoint: DiskLayout] completed: soon"),
        None
    );
    assert_eq!(checkpoint(""), None);
}
