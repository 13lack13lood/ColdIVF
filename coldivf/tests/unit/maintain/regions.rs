//! Unit tests for regions and the rules that make them due.

use super::*;
use crate::MaintainParams;

/// 40 clusters of 100 vectors in 4 regions of 10, all just clustered.
fn regions() -> Regions {
    Regions {
        of: (0..40).map(|l| l / 10).collect(),
        last: vec![1000; 4],
        changes: vec![0; 4],
        list_last: vec![100; 40],
        per_region: 10.0,
        major_count: 4000,
        major_inserts: 0,
        major_deletes: 0,
        pass_changes: 0,
    }
}

/// Every rule on, with the change share at 10%.
fn rules() -> MinorParams {
    MinorParams {
        change: 0.1,
        shrink: 0.2,
        hot_list: 8.0,
        list_loss: 0.5,
        region_lists: 0,
        split: 2.0,
        merge: 0.4,
        ring: 16,
        pool: None,
        global_pass: None,
        refresh: false,
    }
}

#[test]
fn each_rule_makes_a_region_due_on_its_own() {
    let sizes = vec![100; 40];
    assert!(regions().due(&sizes, &rules()).is_empty(), "nothing changed");

    // 100 inserts plus deletes since region 2 was clustered, a tenth of its 1000 then.
    let mut r = regions();
    r.changes[2] = 100;
    assert_eq!(r.due(&sizes, &rules()), vec![2]);
    r.changes[2] = 99;
    assert!(r.due(&sizes, &rules()).is_empty(), "99 changes are under a tenth");

    // Region 1 falls to 800 of its 1000, while no cluster loses half.
    let mut shrunk = sizes.clone();
    (10..20).for_each(|l| shrunk[l] = 80);
    assert_eq!(regions().due(&shrunk, &rules()), vec![1]);
    shrunk[10] = 81;
    assert!(
        regions().due(&shrunk, &rules()).is_empty(),
        "801 of 1000 is above 0.8 of it"
    );

    // One cluster of region 3 holds over 8 times the mean, which it is part of.
    let mut hot = sizes.clone();
    hot[35] = 1000;
    assert_eq!(regions().due(&hot, &rules()), vec![3]);
    hot[35] = 970;
    assert!(
        regions().due(&hot, &rules()).is_empty(),
        "970 is under 8 times the mean of 121.75"
    );

    // One cluster of region 0 lost half of its vectors.
    let mut lost = sizes.clone();
    lost[4] = 50;
    assert_eq!(regions().due(&lost, &rules()), vec![0]);
    lost[4] = 51;
    assert!(regions().due(&lost, &rules()).is_empty(), "51 of 100 is above half");
}

#[test]
fn a_rule_at_0_is_off_and_the_most_changed_region_comes_first() {
    let sizes = vec![100; 40];
    let off = MinorParams {
        change: 0.0,
        shrink: 0.0,
        hot_list: 0.0,
        list_loss: 0.0,
        ..rules()
    };
    let mut r = regions();
    r.changes = vec![300, 0, 500, 200];
    let (mut hot, mut lost) = (sizes.clone(), sizes.clone());
    hot[15] = 2000;
    lost[4] = 10;
    assert!(r.due(&hot, &off).is_empty() && r.due(&lost, &off).is_empty());
    assert_eq!(r.due(&sizes, &rules()), vec![2, 0, 3]);

    // A region without vectors is never due, since its clusters drop out anyway.
    let mut empty = sizes.clone();
    (30..40).for_each(|l| empty[l] = 0);
    assert_eq!(r.due(&empty, &rules()), vec![2, 0]);
}

#[test]
fn regions_hold_the_set_number_of_clusters() {
    let with = |region_lists| MaintainParams {
        minor: MinorParams {
            region_lists,
            ..rules()
        },
        ..MaintainParams::default()
    };
    assert_eq!(regions_for(6250, &with(0)), 79);
    assert_eq!(regions_for(6250, &with(160)), 39);
    assert_eq!(regions_for(3, &with(0)), 2);
    assert_eq!(regions_for(1, &with(0)), 1);
    assert_eq!(regions_for(5, &with(100)), 1);
    assert_eq!(renumber(&[7, 3, 7, 9, 3]), vec![1, 0, 1, 2, 0]);
}
