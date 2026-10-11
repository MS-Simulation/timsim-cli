//! **`--ms2-gain` scales rendered fragment signal, and only fragment signal — on real output.**
//!
//! The unit test next to `apply_ms2_gain` proves the helper; this proves the wiring: that every DIA
//! render path routes fragments through it, that MS1 and the precursor-survival bleed into MS2 stay
//! unscaled, and that the binary refuses the gain where it would be ignored.
//!
//! Renders the golden DIA fixture (`tests/golden/gaussian_golden.json`, case `bruker_dia`) at floor 1,
//! so the reporting floor does not clip the comparison, and reads per-level totals from the
//! `Frames.SummedIntensities` column of `analysis.tdf`. Fixtures are machine-local: absent, the test
//! says so loudly and passes; with `TIMSIM_GOLDEN=1` absence is a hard failure, as in the golden test.
#![cfg(feature = "tdf")]

use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::Command;

const MANIFEST: &str = include_str!("golden/gaussian_golden.json");

fn bin() -> PathBuf {
    match std::env::var_os("CARGO_TARGET_DIR") {
        Some(d) => PathBuf::from(d).join("release"),
        None => PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target").join("release"),
    }
    .join("timsim-render")
}

/// The golden DIA case's argv with `{fixture}` / `{out}` substituted, or None if a fixture is absent.
fn dia_args(out: &Path) -> Option<Vec<String>> {
    let m: Value = serde_json::from_str(MANIFEST).unwrap();
    let fx = m["fixtures"].as_object().unwrap();
    let case = m["cases"].as_array().unwrap().iter().find(|c| c["name"] == "bruker_dia").unwrap();
    let mut args = Vec::new();
    for a in case["args"].as_array().unwrap() {
        let mut s = a.as_str().unwrap().replace("{out}", out.to_str().unwrap());
        for (k, v) in fx {
            let key = format!("{{{k}}}");
            if s.contains(&key) {
                let p = v.as_str().unwrap();
                if !Path::new(p).exists() {
                    return None;
                }
                s = s.replace(&key, p);
            }
        }
        args.push(s);
    }
    Some(args)
}

/// Render the fixture with `extra` flags into a fresh dir; returns (MS1 total, MS2 total, tdf_bin bytes).
fn render(base: &Path, tag: &str, extra: &[&str]) -> (f64, f64, Vec<u8>) {
    let out = base.join(tag);
    let _ = std::fs::remove_dir_all(&out);
    std::fs::create_dir_all(&out).unwrap();
    let mut args = dia_args(&out).unwrap();
    args.extend(["--min-peak-intensity", "1"].iter().map(|s| s.to_string()));
    args.extend(extra.iter().map(|s| s.to_string()));
    let r = Command::new(bin()).args(&args).output().unwrap();
    assert!(r.status.success(), "{tag}: {}", String::from_utf8_lossy(&r.stderr));
    let d = out.join("data.d");
    let con = rusqlite::Connection::open(d.join("analysis.tdf")).unwrap();
    let total = |t: i64| -> f64 {
        con.query_row(
            "SELECT COALESCE(SUM(SummedIntensities), 0) FROM Frames WHERE MsMsType = ?1",
            [t],
            |row| row.get::<_, f64>(0),
        )
        .unwrap()
    };
    let (ms1, ms2) = (total(0), total(9));
    (ms1, ms2, std::fs::read(d.join("analysis.tdf_bin")).unwrap())
}

fn fixtures_present(base: &Path) -> bool {
    if bin().exists() && dia_args(base).is_some() {
        return true;
    }
    let strict = std::env::var_os("TIMSIM_GOLDEN").is_some_and(|v| v != "0");
    assert!(!strict, "TIMSIM_GOLDEN is set but the ms2-gain fixtures or the release binary are missing");
    eprintln!("ms2_gain: SKIPPED — golden fixtures or {} absent; nothing was checked", bin().display());
    false
}

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("timsim-ms2-gain-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn gain_scales_ms2_only_on_every_render_path() {
    let base = scratch("paths");
    if !fixtures_present(&base) {
        return;
    }
    let (m1_a, m2_a, _) = render(&base, "g1", &[]);
    let (m1_b, m2_b, bin_b) = render(&base, "g2", &["--ms2-gain", "2"]);
    assert_eq!(m1_a, m1_b, "MS1 must not move with --ms2-gain");
    let ratio = m2_b / m2_a;
    assert!((ratio - 2.0).abs() < 0.02, "MS2 ratio at gain 2 is {ratio}, expected ~2 (quantisation only)");
    // The serial and the chunked-streaming paths build ions through the same projector; prove it on
    // output rather than by reading the code.
    for (tag, flags) in [("serial", vec!["--no-parallel"]), ("chunked", vec!["--render-chunks", "3"])] {
        let mut f = vec!["--ms2-gain", "2"];
        f.extend(flags);
        let (_, _, b) = render(&base, tag, &f);
        assert!(b == bin_b, "{tag} render at gain 2 differs from the default path");
    }
    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn precursor_survival_bleed_is_not_scaled() {
    // MS2(g) = g * F + B, where F is fragment signal and B the intact-precursor bleed. If the bleed were
    // scaled too, the intercept B would be zero. Three gains give two equal slopes and a positive
    // intercept.
    let base = scratch("survival");
    if !fixtures_present(&base) {
        return;
    }
    let s = ["--precursor-survival-min", "0.3", "--precursor-survival-max", "0.3"];
    let at = |g: &str, tag: &str| {
        let mut f = s.to_vec();
        f.extend(["--ms2-gain", g]);
        render(&base, tag, &f).1
    };
    let (m1, m2, m3) = (at("1", "s1"), at("2", "s2"), at("3", "s3"));
    let (d21, d32) = (m2 - m1, m3 - m2);
    assert!((d21 - d32).abs() / d21 < 0.02, "MS2 not linear in the gain: {m1} {m2} {m3}");
    let bleed = m1 - d21;
    assert!(bleed > 0.05 * m1, "survival bleed {bleed} of MS2 {m1} vanished — is it being scaled?");
    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn gain_is_refused_where_it_would_be_ignored() {
    if !bin().exists() {
        eprintln!("ms2_gain: SKIPPED refusal test — {} absent", bin().display());
        return;
    }
    let run = |args: &[&str]| {
        let r = Command::new(bin())
            .args(["--precursors", "x", "--peptide-rt", "x", "--ion-spectra", "x", "--precursor-ccs", "x"])
            .args(["--peptide-quantities", "x", "--sample", "A", "--out", "/nonexistent/out"])
            .args(args)
            .output()
            .unwrap();
        assert!(!r.status.success(), "{args:?} was accepted");
        String::from_utf8_lossy(&r.stderr).to_string()
    };
    assert!(run(&["--ms2-gain", "2"]).contains("--dia (without --dda)"));
    assert!(run(&["--dia", "--dda", "--ms2-gain", "2"]).contains("--dia (without --dda)"));
    for bad in ["0", "-1", "NaN", "inf", "1001"] {
        // `=` form: clap would read a bare `-1` as a flag.
        let flag = format!("--ms2-gain={bad}");
        assert!(run(&["--dia", &flag]).contains("--ms2-gain must be finite"), "{bad}");
    }
}
