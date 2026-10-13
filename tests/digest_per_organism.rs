//! **`timsim-digest --max-peptides-per-organism`, through the binaries.** The unit tests next to
//! `sample_per_organism` prove the draw; this proves the wiring: the `organism` column read from a real
//! `timsim-proteome` artifact, occurrences mapped to organisms, both outputs filtered exactly as the global
//! cap filters them, the per-organism report, and the refusals that need a real proteome.
//!
//! The proteome is generated: tryptic 8-mers, so with no missed cleavages every protein digests into
//! exactly the peptides it was built from. HUMAN and YEAST share one peptide.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use arrow::array::{Array, StringArray, UInt64Array};
use timsim_schema::tables::{peptide_occurrences as PO, proteome as PROT};

/// A distinct 7-residue stem + K: base-14 digits of `i` over residues other than K, R, P (cleavage) and
/// M (an N-terminal Met could add a clipped variant).
fn peptide(i: usize) -> String {
    const AA: &[u8] = b"ACDEFGHILNQSTV";
    let mut s: Vec<u8> = (0..7).map(|k| AA[(i / AA.len().pow(k)) % AA.len()]).collect();
    s.push(b'K');
    String::from_utf8(s).unwrap()
}

/// Four peptides per protein.
fn fasta(dir: &Path, name: &str, peptides: &[String]) {
    let text: String = peptides
        .chunks(4)
        .enumerate()
        .map(|(i, c)| format!(">sp|{name}{i}|{name}{i}_TEST test protein\n{}\n", c.concat()))
        .collect();
    std::fs::write(dir.join(format!("{name}.fasta")), text).unwrap();
}

fn run(bin: &str, args: &[&str]) -> Output {
    Command::new(bin).args(args).output().unwrap()
}

fn digest(dir: &Path, proteome: &str, tag: &str, extra: &[&str]) -> Output {
    let (p, o, c) = (dir.join(format!("{tag}.pep.parquet")), dir.join(format!("{tag}.occ.parquet")), dir.join(format!("{tag}.cs.parquet")));
    let mut args = vec![
        "--proteome", proteome, "--out-peptides", p.to_str().unwrap(), "--out-occurrences", o.to_str().unwrap(),
        "--out-cleavage-sites", c.to_str().unwrap(), "--max-missed-cleavages", "0", "--min-length", "7", "--seed", "41",
    ];
    args.extend_from_slice(extra);
    run(env!("CARGO_BIN_EXE_timsim-digest"), &args)
}

fn bytes(dir: &Path, tag: &str, kind: &str) -> Vec<u8> {
    std::fs::read(dir.join(format!("{tag}.{kind}.parquet"))).unwrap()
}

/// `(peptide_id, protein_id)` rows of an occurrence table, in file order.
fn occurrences(dir: &Path, tag: &str) -> Vec<(u64, String)> {
    let mut out = Vec::new();
    for b in timsim_schema::read(dir.join(format!("{tag}.occ.parquet")), PO::TABLE).unwrap() {
        let pid = b.column_by_name(PO::PEPTIDE_ID).unwrap().as_any().downcast_ref::<UInt64Array>().unwrap();
        let prot = b.column_by_name(PO::PROTEIN_ID).unwrap().as_any().downcast_ref::<StringArray>().unwrap();
        out.extend((0..b.num_rows()).map(|i| (pid.value(i), prot.value(i).to_string())));
    }
    out
}

fn workdir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("timsim-digest-per-organism-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn per_organism_quotas_through_the_binaries() {
    let d = workdir("hye");
    let shared = peptide(999);
    let human: Vec<String> = (0..19).map(peptide).chain([shared.clone()]).collect();
    let yeast: Vec<String> = (100..129).map(peptide).chain([shared]).collect();
    let ecoli: Vec<String> = (200..212).map(peptide).collect();
    for (name, peps) in [("HUMAN", &human), ("YEAST", &yeast), ("ECOLI", &ecoli)] {
        fasta(&d, name, peps);
    }
    let spec = "[[source]]\npath = \"HUMAN.fasta\"\norganism = \"HUMAN\"\n\
                [[source]]\npath = \"YEAST.fasta\"\norganism = \"YEAST\"\n\
                [[source]]\npath = \"ECOLI.fasta\"\norganism = \"ECOLI\"\n";
    std::fs::write(d.join("hye.toml"), spec).unwrap();
    let proteome = d.join("proteome.parquet");
    let r = run(env!("CARGO_BIN_EXE_timsim-proteome"), &["--spec", d.join("hye.toml").to_str().unwrap(), "--out", proteome.to_str().unwrap()]);
    assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
    let proteome = proteome.to_str().unwrap();

    // The organism of each peptide, from the UNCAPPED digest and the proteome itself.
    assert!(digest(&d, proteome, "full", &[]).status.success());
    let mut org_of: HashMap<String, String> = HashMap::new();
    for b in timsim_schema::read(proteome, PROT::TABLE).unwrap() {
        let id = b.column_by_name(PROT::PROTEIN_ID).unwrap().as_any().downcast_ref::<StringArray>().unwrap();
        let org = b.column_by_name(PROT::ORGANISM).unwrap().as_any().downcast_ref::<StringArray>().unwrap();
        for i in 0..b.num_rows() {
            org_of.insert(id.value(i).to_string(), org.value(i).to_string());
        }
    }
    let full = occurrences(&d, "full");
    let mut organisms_of: BTreeMap<u64, BTreeSet<&str>> = BTreeMap::new();
    for (pid, prot) in &full {
        organisms_of.entry(*pid).or_default().insert(org_of[prot].as_str());
    }
    let count = |org: &str| organisms_of.values().filter(|o| o.contains(org)).count();
    assert_eq!((count("HUMAN"), count("YEAST"), count("ECOLI")), (20, 30, 12), "the generated proteome is not what this test assumes");

    // HUMAN draws first (sorted after ECOLI) and takes all 20, the shared one included, so YEAST's
    // pool is its 29 own peptides.
    let r = digest(&d, proteome, "quota", &["--max-peptides-per-organism", "YEAST=5,HUMAN=20,ECOLI=4"]);
    let err = String::from_utf8_lossy(&r.stderr);
    assert!(r.status.success(), "{err}");
    let report: Vec<Vec<&str>> = err.lines().filter(|l| l.starts_with("    ")).map(|l| l.split_whitespace().collect()).collect();
    let rows: Vec<(&str, &str, &str, &str)> = report.iter().map(|w| (w[0], w[1], w[4], w[6])).collect();
    assert_eq!(rows, [("ECOLI", "4", "12", "(0"), ("HUMAN", "20", "20", "(1"), ("YEAST", "5", "29", "(0")], "{err}");

    // Occurrences: the uncapped rows of the kept peptides, in the uncapped order. Cleavage sites: all.
    let kept: Vec<(u64, String)> = occurrences(&d, "quota");
    let kept_ids: BTreeSet<u64> = kept.iter().map(|o| o.0).collect();
    assert_eq!(kept_ids.len(), 29);
    assert_eq!(kept, full.iter().filter(|o| kept_ids.contains(&o.0)).cloned().collect::<Vec<_>>());
    assert_eq!(bytes(&d, "quota", "cs"), bytes(&d, "full", "cs"));
    let per_org = |org: &str| kept_ids.iter().filter(|p| organisms_of[*p].contains(org)).count();
    assert_eq!((per_org("HUMAN"), per_org("YEAST"), per_org("ECOLI")), (20, 6, 4)); // the shared one is in both

    // Refusals that need the proteome: an organism left out, one that is not there, too few peptides.
    for (quotas, want) in [
        ("HUMAN=1,YEAST=1", r#"missing ["ECOLI"]"#),
        ("HUMAN=1,YEAST=1,ECOLI=1,MOUSE=0", r#"not in the proteome ["MOUSE"]"#),
        ("HUMAN=20,YEAST=30,ECOLI=0", "YEAST=30, but only 29 of its 30 peptides are available (1 already kept"),
    ] {
        let r = digest(&d, proteome, "refused", &["--max-peptides-per-organism", quotas]);
        let err = String::from_utf8_lossy(&r.stderr);
        assert!(!r.status.success() && err.contains(want), "{quotas}: {err}");
    }
    let r = digest(&d, proteome, "refused", &["--max-peptides", "5", "--max-peptides-per-organism", "HUMAN=1,YEAST=1,ECOLI=1"]);
    assert!(!r.status.success() && String::from_utf8_lossy(&r.stderr).contains("exclusive"));
    std::fs::remove_dir_all(&d).unwrap();
}

/// With one organism the quota IS `--max-peptides`: the same files, byte for byte. A protein without an
/// organism has no quota, so it is refused rather than dropped.
#[test]
fn one_organism_is_the_global_cap_and_untagged_proteins_are_refused() {
    let d = workdir("one");
    fasta(&d, "HUMAN", &(0..40).map(peptide).collect::<Vec<_>>());
    let fa = d.join("HUMAN.fasta");
    let (tagged, untagged) = (d.join("tagged.parquet"), d.join("untagged.parquet"));
    for (out, org) in [(&tagged, Some("HUMAN")), (&untagged, None)] {
        let mut args = vec!["--fasta", fa.to_str().unwrap(), "--out", out.to_str().unwrap()];
        if let Some(o) = org {
            args.extend(["--organism", o]);
        }
        assert!(run(env!("CARGO_BIN_EXE_timsim-proteome"), &args).status.success());
    }
    assert!(digest(&d, tagged.to_str().unwrap(), "cap", &["--max-peptides", "7"]).status.success());
    assert!(digest(&d, tagged.to_str().unwrap(), "quota", &["--max-peptides-per-organism", "HUMAN=7"]).status.success());
    for kind in ["pep", "occ", "cs"] {
        assert_eq!(bytes(&d, "cap", kind), bytes(&d, "quota", kind), "{kind}");
    }
    let r = digest(&d, untagged.to_str().unwrap(), "untagged", &["--max-peptides-per-organism", "HUMAN=7"]);
    let err = String::from_utf8_lossy(&r.stderr);
    assert!(!r.status.success() && err.contains("10 proteins have no organism"), "{err}");
    std::fs::remove_dir_all(&d).unwrap();
}
