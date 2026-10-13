//! `timsim-digest` — which peptides exist. STRUCTURE.
//!
//! There is **no `--digestion-efficiency` here, and no seed**. Efficiency is a *quantity*: it
//! says how much, not what exists. And cleavage-blocking modifications (acetyl-K, ubiquitin-GG)
//! are regulated, so cleavage probabilities differ between conditions — putting a yield here
//! would make the structure condition-dependent and silently destroy sharing across samples.
//! Yields are emitted per-sample by `timsim-yield`.
//!
//! No seed because the digest is **analytic**: an exact expectation, not a sample.

use anyhow::{anyhow, bail, Result};
use arrow::array::{Array, ArrayRef, Float64Array, StringArray, UInt16Array, UInt32Array, UInt64Array};
use clap::Parser;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use timsim_chem::{ids, mass, Enumerator, Protocol};
use timsim_cli::{batch, print_schema, producer};
use timsim_schema::tables::{cleavage_sites as CS, peptide_occurrences as PO, peptides as PEP, proteome as PROT};

#[derive(Parser)]
#[command(name = "timsim-digest", about = "proteome.parquet -> peptides + occurrences + cleavage sites (structure)")]
struct Args {
    #[arg(long)]
    proteome: Option<PathBuf>,
    #[arg(long)]
    enzyme: Option<String>,
    /// Numerical truncation, NOT a model parameter. Its cost is measured and reported.
    #[arg(long, default_value_t = 2)]
    max_missed_cleavages: u32,
    #[arg(long, default_value_t = 7)]
    min_length: u32,
    #[arg(long, default_value_t = 50)]
    max_length: u32,
    #[arg(long)]
    out_peptides: Option<PathBuf>,
    #[arg(long)]
    out_occurrences: Option<PathBuf>,
    #[arg(long)]
    out_cleavage_sites: Option<PathBuf>,
    /// OPT-IN cap on the number of unique peptides. `0` (default) keeps the full ANALYTIC digest (every
    /// peptide — the exact expectation). A positive value draws a DETERMINISTIC, seed-keyed RANDOM sample
    /// of that many peptides (and filters occurrences to match) — for a tractable run on a large proteome
    /// while keeping the full FASTA as the search space (v1's `num_sample_peptides`). Cleavage sites are
    /// protein-level and kept in full.
    #[arg(long, default_value_t = 0)]
    max_peptides: u64,
    /// OPT-IN per-organism form of `--max-peptides`: `ORG=N[,ORG=N...]` (e.g.
    /// `HUMAN=3089,YEAST=3528,ECOLI=3369`), keyed on the proteome's `organism` column. One global sample
    /// hands the largest proteome most of the peptides; a dataset that sampled each FASTA on its own and
    /// pooled them (v1's HYE) needs a count per organism. Exclusive with `--max-peptides`; empty = off.
    ///
    /// Every organism in the proteome must be named, and only those, so none is dropped or misspelt
    /// silently; N may be 0. A peptide belongs to every organism it occurs in. Organisms draw in sorted
    /// name order, each walking its peptides in `--max-peptides`' seed-keyed order and keeping the first N
    /// not already kept by an earlier organism: each contributes exactly N new peptides, and a shared
    /// peptide counts for the first organism that takes it. Fewer than N available is an error.
    /// Occurrences are filtered to the kept peptides; cleavage sites are protein-level and kept in full.
    #[arg(long)]
    max_peptides_per_organism: Option<String>,
    /// Seed for `--max-peptides` / `--max-peptides-per-organism` sampling (ignored without either; the
    /// analytic digest has no seed).
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long)]
    schema: bool,
}

/// splitmix64 — a stable, dep-free hash so the peptide sample is reproducible across runs (unlike the
/// std hasher, whose output is not guaranteed stable).
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// `--max-peptides-per-organism`: `ORG=N[,ORG=N...]` -> quotas; empty (or blank) = off. Strict, because a
/// quota read as something other than what was meant changes the experiment without failing anything.
fn parse_organism_quotas(spec: &str) -> Result<BTreeMap<String, u64>> {
    let mut quotas = BTreeMap::new();
    if spec.trim().is_empty() {
        return Ok(quotas);
    }
    for entry in spec.split(',') {
        let Some((name, n)) = entry.split_once('=') else {
            bail!("--max-peptides-per-organism: {entry:?} is not ORG=N");
        };
        let (name, n) = (name.trim(), n.trim());
        if name.is_empty() {
            bail!("--max-peptides-per-organism: {entry:?} names no organism");
        }
        // Digits only: `u64::from_str` would also accept a leading `+`.
        let n: u64 = Some(n)
            .filter(|n| n.bytes().all(|b| b.is_ascii_digit()))
            .and_then(|n| n.parse().ok())
            .ok_or_else(|| anyhow!("--max-peptides-per-organism: {name}={n:?} is not a non-negative 64-bit integer"))?;
        if quotas.insert(name.to_string(), n).is_some() {
            bail!("--max-peptides-per-organism: {name:?} is named twice");
        }
    }
    Ok(quotas)
}

/// One organism's line of the `--max-peptides-per-organism` report.
#[derive(Debug, PartialEq)]
struct OrganismDraw {
    organism: String,
    /// Peptides this organism added to the kept set (its quota).
    kept: usize,
    /// Its peptides that no earlier organism had already kept: the pool it drew from.
    available: usize,
    /// Of the kept ones, how many also occur in another organism.
    shared: usize,
}

/// The `--max-peptides-per-organism` draw: the kept peptide ids, and one report line per organism.
///
/// `organisms` is every organism in the proteome, taken from the PROTEINS, so an organism whose proteins
/// yield no peptide still has to be named. `occurrences` pairs each occurrence's peptide id with its
/// protein's organism (any order, repeats allowed). Pure, so the draw is testable without a proteome.
fn sample_per_organism<'a>(
    quotas: &BTreeMap<String, u64>,
    organisms: &BTreeSet<&'a str>,
    occurrences: impl IntoIterator<Item = (u64, &'a str)>,
    seed: u64,
) -> Result<(HashSet<u64>, Vec<OrganismDraw>)> {
    // Names first: an organism left out would silently lose every peptide, a misspelt one keep none.
    let unnamed: Vec<&str> = organisms.iter().copied().filter(|o| !quotas.contains_key(*o)).collect();
    let unknown: Vec<&str> = quotas.keys().map(String::as_str).filter(|q| !organisms.contains(q)).collect();
    if !unnamed.is_empty() || !unknown.is_empty() {
        bail!(
            "--max-peptides-per-organism must name every organism in the proteome, and only those: \
             missing {unnamed:?}, not in the proteome {unknown:?} (the proteome has {organisms:?})"
        );
    }

    // Deduplicated (peptide, organism) pairs. Sorted by peptide, a peptide that occurs in several
    // organisms is a run of equal ids.
    let index: HashMap<&str, usize> = organisms.iter().enumerate().map(|(i, &o)| (o, i)).collect();
    let mut pairs: Vec<(u64, usize)> = Vec::new();
    for (id, org) in occurrences {
        let i = *index.get(org).ok_or_else(|| anyhow!("an occurrence's organism {org:?} is not in the organism list"))?;
        pairs.push((id, i));
    }
    pairs.sort_unstable();
    pairs.dedup();
    let shared: HashSet<u64> = pairs.windows(2).filter(|w| w[0].0 == w[1].0).map(|w| w[0].0).collect();
    let mut by_organism: Vec<Vec<u64>> = vec![Vec::new(); organisms.len()];
    for (id, i) in pairs {
        by_organism[i].push(id);
    }

    let mut kept: HashSet<u64> = HashSet::new();
    let mut report = Vec::with_capacity(organisms.len());
    for (&name, mut ids) in organisms.iter().zip(by_organism) {
        let n = quotas[name];
        // `--max-peptides`' order. splitmix64 is a bijection, so distinct ids never tie and the order
        // depends on (ids, seed) alone.
        ids.sort_unstable_by_key(|&id| splitmix64(id ^ seed));
        let available = ids.iter().filter(|id| !kept.contains(*id)).count();
        if (available as u64) < n {
            bail!(
                "--max-peptides-per-organism: {name}={n}, but only {available} of its {} peptides are available \
                 ({} already kept by an organism earlier in sorted order)",
                ids.len(),
                ids.len() - available
            );
        }
        let take: Vec<u64> = ids.iter().copied().filter(|id| !kept.contains(id)).take(n as usize).collect();
        let n_shared = take.iter().filter(|id| shared.contains(*id)).count();
        kept.extend(&take);
        report.push(OrganismDraw { organism: name.to_string(), kept: take.len(), available, shared: n_shared });
    }
    Ok((kept, report))
}

fn main() -> Result<()> {
    let a = Args::parse();
    if a.schema {
        for t in [PEP::TABLE, PO::TABLE, CS::TABLE] {
            print_schema(t)?;
            println!();
        }
        return Ok(());
    }

    let proteome = a.proteome.ok_or_else(|| anyhow::anyhow!("--proteome is required"))?;
    let enzyme = a.enzyme.unwrap_or_else(|| "trypsin".into());
    let (op, oo, oc) = (
        a.out_peptides.ok_or_else(|| anyhow::anyhow!("--out-peptides is required"))?,
        a.out_occurrences.ok_or_else(|| anyhow::anyhow!("--out-occurrences is required"))?,
        a.out_cleavage_sites.ok_or_else(|| anyhow::anyhow!("--out-cleavage-sites is required"))?,
    );
    let quotas = parse_organism_quotas(a.max_peptides_per_organism.as_deref().unwrap_or(""))?;
    if !quotas.is_empty() && a.max_peptides > 0 {
        bail!("--max-peptides and --max-peptides-per-organism are exclusive: one global sample, or one per organism");
    }

    // Validate-on-read: a wrong or renamed column fails HERE, not three stages downstream.
    let batches = timsim_schema::read(&proteome, PROT::TABLE)?;
    let mut input: Vec<(String, String)> = Vec::new();
    let mut organism: Vec<Option<String>> = Vec::new(); // parallel to `input`; only the quotas read it
    for b in &batches {
        let id = b.column_by_name(PROT::PROTEIN_ID).unwrap().as_any().downcast_ref::<StringArray>().unwrap();
        let sq = b.column_by_name(PROT::SEQUENCE).unwrap().as_any().downcast_ref::<StringArray>().unwrap();
        let org = b.column_by_name(PROT::ORGANISM).unwrap().as_any().downcast_ref::<StringArray>().unwrap();
        for i in 0..b.num_rows() {
            input.push((id.value(i).to_string(), sq.value(i).to_string()));
            organism.push(org.is_valid(i).then(|| org.value(i).to_string()));
        }
    }

    let e = Enumerator::new(
        Protocol::parse(&enzyme).map_err(anyhow::Error::msg)?,
        a.max_missed_cleavages,
        a.min_length,
        a.max_length,
    )
    .map_err(anyhow::Error::msg)?;

    let t = std::time::Instant::now();
    let digests = e.enumerate_all(&input);
    let elapsed = t.elapsed();

    // ── peptides (unique) + occurrences ──
    let seqs: std::collections::HashMap<&str, &str> =
        input.iter().map(|(i, s)| (i.as_str(), s.as_str())).collect();

    let mut coll = ids::CollisionCheck::new();
    let mut pep_seen: std::collections::HashMap<u64, (String, u16, f64)> = Default::default();
    let (mut o_pid, mut o_prot, mut o_start, mut o_end, mut o_mc) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut c_prot, mut c_pos) = (Vec::new(), Vec::new());
    let mut skipped_mass = 0usize;

    for d in &digests {
        let pseq = seqs[&*d.protein_id];
        for &pos in &d.cleavage_sites {
            c_prot.push(d.protein_id.to_string());
            c_pos.push(pos);
        }
        for o in &d.occurrences {
            let s = o.sequence(pseq);
            let id = ids::peptide_id(s);
            coll.insert(id, s).map_err(anyhow::Error::msg)?;
            if !pep_seen.contains_key(&id) {
                match mass::monoisotopic(s) {
                    Ok(m) => {
                        pep_seen.insert(id, (s.to_string(), s.len() as u16, m));
                    }
                    Err(_) => {
                        // Non-standard residue (X/B/Z). We refuse to guess a mass: a quietly
                        // wrong mass corrupts the mass balance without failing anything.
                        skipped_mass += 1;
                        continue;
                    }
                }
            }
            o_pid.push(id);
            o_prot.push(d.protein_id.to_string());
            o_start.push(o.start);
            o_end.push(o.end);
            o_mc.push(o.n_missed_cleavages);
        }
    }

    let mut peps: Vec<(u64, String, u16, f64)> =
        pep_seen.into_iter().map(|(k, (s, l, m))| (k, s, l, m)).collect();
    peps.sort_unstable_by_key(|p| p.0); // deterministic row order

    // Opt-in peptide cap (v1's num_sample_peptides): a deterministic seed-keyed random sample for a
    // tractable run on a large proteome, keeping the full FASTA as the search space. Occurrences are
    // filtered to the kept peptides; cleavage sites are protein-level and kept in full.
    let (mut o_pid, mut o_prot, mut o_start, mut o_end, mut o_mc) = (o_pid, o_prot, o_start, o_end, o_mc);
    let full_peptides = peps.len();
    let mut kept: Option<HashSet<u64>> = None;
    if a.max_peptides > 0 && (peps.len() as u64) > a.max_peptides {
        peps.sort_by_key(|p| splitmix64(p.0 ^ a.seed)); // random order keyed by (id, seed)
        peps.truncate(a.max_peptides as usize);
        peps.sort_unstable_by_key(|p| p.0); // restore id order
        kept = Some(peps.iter().map(|p| p.0).collect());
        eprintln!("  --max-peptides       : sampled {} of {full_peptides} unique peptides (seed {})", a.max_peptides, a.seed);
    }
    // The same draw, per organism (`--max-peptides-per-organism`; semantics in its help). v1's HYE sampled
    // each FASTA on its own and pooled them; one global sample cannot reproduce that organism mix.
    if !quotas.is_empty() {
        let mut org_of: HashMap<&str, &str> = HashMap::with_capacity(input.len());
        let mut untagged: Vec<&str> = Vec::new();
        for ((id, _), org) in input.iter().zip(&organism) {
            match org {
                Some(o) => {
                    org_of.insert(id, o);
                }
                None => untagged.push(id),
            }
        }
        if let Some(first) = untagged.first() {
            // No organism, no quota: its peptides would vanish without anyone having asked for that.
            bail!(
                "--max-peptides-per-organism: {} proteins have no organism (e.g. {first:?}); tag every source of the proteome",
                untagged.len()
            );
        }
        let present: BTreeSet<&str> = org_of.values().copied().collect();
        let occurrences = o_pid.iter().zip(&o_prot).map(|(&id, prot)| (id, org_of[prot.as_str()]));
        let (keep, draws) = sample_per_organism(&quotas, &present, occurrences, a.seed)?;
        peps.retain(|p| keep.contains(&p.0)); // never reordered, so still in id order
        eprintln!("  --max-peptides-per-organism: kept {} of {full_peptides} unique peptides (seed {})", peps.len(), a.seed);
        for d in &draws {
            eprintln!(
                "    {:<12} {:>9} kept / {:>9} available   ({} of them also occur in another organism)",
                d.organism, d.kept, d.available, d.shared
            );
        }
        kept = Some(keep);
    }
    if let Some(kept) = &kept {
        let keep: Vec<usize> = (0..o_pid.len()).filter(|&i| kept.contains(&o_pid[i])).collect();
        let (np, npr): (Vec<u64>, Vec<String>) = keep.iter().map(|&i| (o_pid[i], o_prot[i].clone())).unzip();
        let ns: Vec<u32> = keep.iter().map(|&i| o_start[i]).collect();
        let ne: Vec<u32> = keep.iter().map(|&i| o_end[i]).collect();
        let nm: Vec<u16> = keep.iter().map(|&i| o_mc[i]).collect();
        o_pid = np; o_prot = npr; o_start = ns; o_end = ne; o_mc = nm;
    }

    timsim_schema::write(&op, PEP::TABLE, &batch(PEP::TABLE, vec![
        Arc::new(UInt64Array::from(peps.iter().map(|p| p.0).collect::<Vec<_>>())) as ArrayRef,
        Arc::new(StringArray::from(peps.iter().map(|p| p.1.clone()).collect::<Vec<_>>())),
        Arc::new(UInt16Array::from(peps.iter().map(|p| p.2).collect::<Vec<_>>())),
        Arc::new(Float64Array::from(peps.iter().map(|p| p.3).collect::<Vec<_>>())),
    ])?, &producer("timsim-digest"), None)?;

    // The bounds travel WITH the structure. A consumer must never re-enter them: yielding at
    // a smaller bound than the digest used silently zeroes every occurrence beyond it.
    let bounds_meta: Vec<(&str, String)> = vec![
        (timsim_schema::meta::MAX_MISSED_CLEAVAGES, a.max_missed_cleavages.to_string()),
        (timsim_schema::meta::MIN_LENGTH, a.min_length.to_string()),
        (timsim_schema::meta::MAX_LENGTH, a.max_length.to_string()),
    ];

    let n_occ = o_pid.len();
    timsim_schema::write_with(&oo, PO::TABLE, &batch(PO::TABLE, vec![
        Arc::new(UInt64Array::from(o_pid)) as ArrayRef,
        Arc::new(StringArray::from(o_prot)),
        Arc::new(UInt32Array::from(o_start)),
        Arc::new(UInt32Array::from(o_end)),
        Arc::new(UInt16Array::from(o_mc)),
    ])?, &producer("timsim-digest"), None, &bounds_meta)?;

    let n_sites = c_pos.len();
    timsim_schema::write(&oc, CS::TABLE, &batch(CS::TABLE, vec![
        Arc::new(StringArray::from(c_prot)) as ArrayRef,
        Arc::new(UInt32Array::from(c_pos)),
    ])?, &producer("timsim-digest"), None)?;

    println!("  enzyme               : {enzyme}");
    println!("  max missed cleavages : {}  (numerical truncation; cost measured by timsim-yield)", a.max_missed_cleavages);
    println!("  length filter        : {}-{}", a.min_length, a.max_length);
    println!("  proteins             : {}", input.len());
    println!("  cleavage sites       : {n_sites}");
    println!("  unique peptides      : {}", peps.len());
    println!("  occurrences          : {n_occ}   (+{} — repeats and shared peptides, which a protein_ids list column would destroy)",
             n_occ.saturating_sub(peps.len()));
    if skipped_mass > 0 {
        eprintln!("  warning: {skipped_mass} peptides skipped (non-standard residues; we refuse to guess a mass)");
    }
    println!("  wall time            : {elapsed:.2?}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quotas(spec: &str) -> BTreeMap<String, u64> {
        parse_organism_quotas(spec).unwrap()
    }

    /// `(peptide, organism)` occurrences: one per id in `ids`.
    fn occ(org: &'static str, ids: impl IntoIterator<Item = u64>) -> Vec<(u64, &'static str)> {
        ids.into_iter().map(|id| (id, org)).collect()
    }

    fn draw(spec: &str, organisms: &[&'static str], occurrences: &[(u64, &'static str)], seed: u64)
        -> Result<(HashSet<u64>, Vec<OrganismDraw>)> {
        let organisms: BTreeSet<&str> = organisms.iter().copied().collect();
        sample_per_organism(&quotas(spec), &organisms, occurrences.iter().copied(), seed)
    }

    fn lines(report: &[OrganismDraw]) -> Vec<(&str, usize, usize, usize)> {
        report.iter().map(|d| (d.organism.as_str(), d.kept, d.available, d.shared)).collect()
    }

    /// Anything but `ORG=N`, with N a plain non-negative integer, is refused, and so is naming an organism
    /// twice. Empty means off, not an error.
    #[test]
    fn the_quota_spelling_is_strict() {
        assert!(quotas("").is_empty() && quotas("  ").is_empty());
        let want = BTreeMap::from([("ECOLI".to_string(), 0), ("HUMAN".to_string(), 3089), ("YEAST".to_string(), 3528)]);
        assert_eq!(quotas("HUMAN=3089, YEAST=3528,ECOLI=0"), want);
        for bad in ["HUMAN", "HUMAN=", "=5", "HUMAN=-1", "HUMAN=+5", "HUMAN=1.5", "HUMAN=1e3", "HUMAN=1_000",
                    "HUMAN=1,", ",HUMAN=1", "HUMAN=1,,YEAST=2", "HUMAN=1,HUMAN=2", "HUMAN=18446744073709551616"] {
            assert!(parse_organism_quotas(bad).is_err(), "{bad:?} must be refused");
        }
    }

    /// Each organism adds EXACTLY its quota to the kept set: the per-organism counts are the point.
    #[test]
    fn each_organism_contributes_exactly_its_quota() {
        let o = [occ("HUMAN", 0..500), occ("YEAST", 1000..1300), occ("ECOLI", 2000..2200)].concat();
        let (kept, report) = draw("HUMAN=50,YEAST=70,ECOLI=0", &["HUMAN", "YEAST", "ECOLI"], &o, 41).unwrap();
        assert_eq!(lines(&report), [("ECOLI", 0, 200, 0), ("HUMAN", 50, 500, 0), ("YEAST", 70, 300, 0)]);
        assert_eq!(kept.iter().filter(|&&id| id < 1000).count(), 50);
        assert_eq!(kept.iter().filter(|&&id| (1000..2000).contains(&id)).count(), 70);
        assert_eq!(kept.len(), 120);
    }

    /// For one organism the draw IS `--max-peptides`: the first N ids in `splitmix64(id ^ seed)` order. It is
    /// fixed by the seed, and neither the order nor the repetition of occurrences moves it.
    #[test]
    fn the_draw_is_the_global_caps_order_and_is_deterministic() {
        let o = occ("HUMAN", 0..1000);
        let cap = |seed: u64| -> HashSet<u64> {
            let mut ids: Vec<u64> = (0..1000).collect();
            ids.sort_by_key(|&id| splitmix64(id ^ seed));
            ids.into_iter().take(100).collect()
        };
        let kept = draw("HUMAN=100", &["HUMAN"], &o, 41).unwrap().0;
        assert_eq!(kept, cap(41));
        assert_eq!(kept, draw("HUMAN=100", &["HUMAN"], &o, 41).unwrap().0);
        assert_ne!(kept, draw("HUMAN=100", &["HUMAN"], &o, 7).unwrap().0, "the seed must matter");
        let mut reordered: Vec<_> = o.iter().rev().copied().collect();
        reordered.extend_from_slice(&o[..10]); // repeats: one peptide, however often it occurs
        assert_eq!(kept, draw("HUMAN=100", &["HUMAN"], &reordered, 41).unwrap().0);
    }

    /// A peptide shared between organisms counts for the first organism in SORTED name order that takes
    /// it, whatever order the input lists them in; a later organism draws only from what is left, and an
    /// organism with quota 0 still lets another organism keep a peptide they share.
    #[test]
    fn shared_peptides_go_to_the_first_organism_in_sorted_order() {
        // 1..=4 occur in HUMAN and YEAST; 10 in YEAST and ECOLI. Listed YEAST first, on purpose.
        let o = [occ("YEAST", [1, 2, 3, 4, 10, 11, 12, 13]), occ("HUMAN", [1, 2, 3, 4]), occ("ECOLI", [10, 20])].concat();
        let organisms = ["YEAST", "HUMAN", "ECOLI"];
        // HUMAN (before YEAST) takes all four shared ones, so YEAST must take exactly its own four. Had
        // YEAST drawn first, it would have taken some of 1..=4 and left HUMAN short (an error).
        let (kept, report) = draw("YEAST=4,HUMAN=4,ECOLI=0", &organisms, &o, 41).unwrap();
        assert_eq!(kept, HashSet::from([1, 2, 3, 4, 10, 11, 12, 13]));
        assert_eq!(lines(&report), [("ECOLI", 0, 2, 0), ("HUMAN", 4, 4, 4), ("YEAST", 4, 4, 1)]);
        // And YEAST cannot have a fifth: the shared four are already kept.
        let e = draw("YEAST=5,HUMAN=4,ECOLI=0", &organisms, &o, 41).unwrap_err().to_string();
        assert!(e.contains("YEAST=5, but only 4 of its 8 peptides are available (4 already kept"), "{e}");
    }

    /// Every organism in the proteome must be named, and only those; case matters. An organism whose
    /// proteins yield no peptide is still in the proteome: quota 0 is fine, anything more is not.
    #[test]
    fn every_organism_must_be_named_and_only_those() {
        let o = [occ("HUMAN", 0..10), occ("YEAST", 10..20)].concat();
        let both = ["HUMAN", "YEAST"];
        let e = draw("HUMAN=1", &both, &o, 0).unwrap_err().to_string();
        assert!(e.contains(r#"missing ["YEAST"], not in the proteome []"#), "{e}");
        let e = draw("HUMAN=1,YEAST=1,ECOLI=1", &both, &o, 0).unwrap_err().to_string();
        assert!(e.contains(r#"missing [], not in the proteome ["ECOLI"]"#), "{e}");
        let e = draw("HUMAN=1,yeast=1", &both, &o, 0).unwrap_err().to_string();
        assert!(e.contains(r#"missing ["YEAST"], not in the proteome ["yeast"]"#), "{e}");

        let three = ["HUMAN", "YEAST", "ECOLI"];
        assert!(draw("HUMAN=1,YEAST=1", &three, &o, 0).is_err());
        assert_eq!(draw("HUMAN=1,YEAST=1,ECOLI=0", &three, &o, 0).unwrap().0.len(), 2);
        let e = draw("HUMAN=1,YEAST=1,ECOLI=1", &three, &o, 0).unwrap_err().to_string();
        assert!(e.contains("ECOLI=1, but only 0 of its 0 peptides are available"), "{e}");
    }

    /// Fewer peptides than the quota is an error, not a quietly smaller sample. Exactly all is fine.
    #[test]
    fn a_quota_above_the_available_peptides_is_refused() {
        let o = occ("HUMAN", 0..10);
        assert_eq!(draw("HUMAN=10", &["HUMAN"], &o, 0).unwrap().0.len(), 10);
        let e = draw("HUMAN=11", &["HUMAN"], &o, 0).unwrap_err().to_string();
        assert!(e.contains("HUMAN=11, but only 10 of its 10 peptides are available"), "{e}");
    }
}
