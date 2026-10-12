//! **`--noise-mz-draw`, on real output.** The unit tests next to `timsim_cli::ms2::ObsNoise` prove the
//! draw and the deposit; this proves the wiring through the binary and the run's real m/z calibrator:
//! the default is byte-identical whether the flag is spelled out or omitted, `per-observation` at ppm 0 is
//! a no-op, the per-observation render is the same on every render path, the level totals do not move,
//! and the binary refuses the mode where it would be ignored.
//!
//! Renders the golden DIA fixture (`tests/golden/gaussian_golden.json`, case `bruker_dia`) at floor 1.
//! Fixtures are machine-local: absent, the test says so loudly and passes; with `TIMSIM_GOLDEN=1`
//! absence is a hard failure, as in the golden test.
#![cfg(feature = "tdf")]

use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::Command;

const MANIFEST: &str = include_str!("golden/gaussian_golden.json");
const PPM: [&str; 4] = ["--noise-mz-ppm", "6.5", "--noise-frag-ppm", "6.5"];

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
    assert!(!strict, "TIMSIM_GOLDEN is set but the mz-noise-draw fixtures or the release binary are missing");
    eprintln!("mz_noise_draw: SKIPPED — golden fixtures or {} absent; nothing was checked", bin().display());
    false
}

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("timsim-mz-noise-draw-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn with(base: &[&'static str], more: &[&'static str]) -> Vec<&'static str> {
    base.iter().chain(more).copied().collect()
}

#[test]
fn the_default_is_byte_identical_spelled_out_or_omitted() {
    let base = scratch("default");
    if !fixtures_present(&base) {
        return;
    }
    let (_, _, omitted) = render(&base, "omitted", &PPM);
    let (_, _, explicit) = render(&base, "explicit", &with(&PPM, &["--noise-mz-draw", "per-peak"]));
    assert!(omitted == explicit, "--noise-mz-draw per-peak differs from omitting the flag");
    // ppm 0 makes the scope irrelevant: per-observation must then be the noiseless render, bit for bit.
    let (_, _, noiseless) = render(&base, "noiseless", &[]);
    let (_, _, obs0) = render(&base, "obs0", &["--noise-mz-draw", "per-observation"]);
    assert!(noiseless == obs0, "per-observation at ppm 0 is not the noiseless render");
    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn per_observation_is_the_same_on_every_render_path_and_conserves_level_totals() {
    let base = scratch("paths");
    if !fixtures_present(&base) {
        return;
    }
    let obs = with(&PPM, &["--noise-mz-draw", "per-observation"]);
    let (m1_obs, m2_obs, bin_obs) = render(&base, "obs", &obs);
    let (m1_peak, m2_peak, bin_peak) = render(&base, "peak", &PPM);
    assert!(bin_obs != bin_peak, "per-observation rendered the same bytes as per-peak");
    // Positions move, intensities do not: only bin merging and integer rounding can shift the totals.
    for (what, a, b) in [("MS1", m1_obs, m1_peak), ("MS2", m2_obs, m2_peak)] {
        assert!(a > 0.0 && ((a - b) / b).abs() < 0.005, "{what} total moved: {a} vs {b}");
    }
    for (tag, flags) in [("serial", vec!["--no-parallel"]), ("chunked", vec!["--render-chunks", "3"])] {
        let (_, _, b) = render(&base, tag, &with(&obs, &flags));
        assert!(b == bin_obs, "{tag} per-observation render differs from the default path");
    }
    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn per_observation_is_refused_where_it_would_be_ignored() {
    if !bin().exists() {
        eprintln!("mz_noise_draw: SKIPPED refusal test — {} absent", bin().display());
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
    let want = "--noise-mz-draw per-observation is implemented for --dia (without --dda) only";
    assert!(run(&["--noise-mz-draw", "per-observation", "--noise-mz-ppm", "6.5"]).contains(want));
    assert!(run(&["--dia", "--dda", "--noise-mz-draw", "per-observation", "--noise-frag-ppm", "6.5"]).contains(want));
}
