//! Regression for #1963: the planned contraction path is a function of the
//! subscripts, shapes and options alone, identical in every process.
//!
//! `HashMap` iteration order is seeded per process, so a planner whose tie
//! break follows it agrees with itself inside one process and only diverges
//! across processes. The test therefore re-executes this test binary several
//! times and compares the paths the child processes print.

use std::process::Command;

use tenferro_einsum::{ContractionOptimizerOptions, ContractionTree, Subscripts};

const CHILD_ENV: &str = "TENFERRO_EINSUM_DETERMINISTIC_PATH_CHILD";
const TEST_NAME: &str = "deterministic_path::contraction_path_is_identical_across_processes";

/// Specs with several equal-cost candidate pairs; the first is the #1963
/// reproducer.
fn paths() -> String {
    let specs: [(&str, &[&[usize]]); 3] = [
        (
            "abcdef,bf,cf,df,ef->f",
            &[&[2, 3, 2, 4, 3, 9], &[3, 9], &[2, 9], &[4, 9], &[3, 9]],
        ),
        (
            "ab,bc,cd,de,ea->",
            &[&[4, 4], &[4, 4], &[4, 4], &[4, 4], &[4, 4]],
        ),
        (
            "ai,bi,ci,di,ei,fi->abcdef",
            &[&[2, 5], &[2, 5], &[2, 5], &[2, 5], &[2, 5], &[2, 5]],
        ),
    ];
    let mut out = String::new();
    for (spec, shapes) in specs {
        let subs = Subscripts::parse(spec).unwrap();
        for tree in [
            ContractionTree::optimize(&subs, shapes).unwrap(),
            ContractionTree::optimize_with_options(
                &subs,
                shapes,
                &ContractionOptimizerOptions {
                    ntrials: 3,
                    ..ContractionOptimizerOptions::default()
                },
            )
            .unwrap(),
        ] {
            let pairs = (0..tree.step_count())
                .map(|step| format!("{:?}", tree.step_pair(step).unwrap()))
                .collect::<Vec<_>>()
                .join(" ");
            out.push_str(&format!("{spec}: {pairs}\n"));
        }
    }
    out
}

#[test]
fn contraction_path_is_identical_across_processes() {
    if std::env::var_os(CHILD_ENV).is_some() {
        println!("PATHS-BEGIN\n{}PATHS-END", paths());
        return;
    }
    let exe = std::env::current_exe().unwrap();
    let expected = paths();
    for run in 0..8 {
        let output = Command::new(&exe)
            .args([TEST_NAME, "--exact", "--nocapture", "--test-threads=1"])
            .env(CHILD_ENV, "1")
            .output()
            .unwrap();
        assert!(output.status.success(), "child run {run} failed");
        let stdout = String::from_utf8(output.stdout).unwrap();
        let begin = stdout
            .find("PATHS-BEGIN\n")
            .expect("child printed no paths")
            + 12;
        let end = stdout.find("PATHS-END").unwrap();
        assert_eq!(
            &stdout[begin..end],
            expected,
            "child run {run} planned a different contraction path"
        );
    }
}
