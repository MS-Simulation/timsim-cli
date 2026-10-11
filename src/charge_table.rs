//! Empirical charge-state model: resample a peptide's charge distribution from a reference table.
//!
//! # Why a table and not a formula
//!
//! Reproducing a published v1 dataset needs v1's charge-state mix, and v1 did not have a formula: it
//! ran a GRU (`DeepChargeStateDistribution`) on the sequence, then dropped states below
//! `min_charge_contrib` without renormalising. Neither v2 model reproduces it — the site-specific
//! model makes 56% of a deposited HeLa sample dominantly 4+ where v1 made 72% dominantly 2+, and the
//! binomial makes a third of clean tryptic peptides 1+. Porting the GRU means porting its weights.
//!
//! The deposited `synthetic_data.db` already holds v1's answer for 10,000 real peptides. On that
//! sample three sequence features — K/R count, H count and length — explain 73% of the variance of
//! the mean charge, so a peptide is given the **share vector of a reference peptide with the same
//! features**, picked by an identity-keyed draw. Resampling whole vectors (rather than averaging
//! them per cell) keeps v1's per-peptide ion COUNT as well as its mix: an average of vectors that
//! each carry two states carries four.
//!
//! # Table format
//!
//! Tab-separated, one reference peptide per row, header required:
//!
//! ```text
//! n_kr  n_h  length  f1  f2  f3  f4
//! 2     0    12      0   0.71  0.29  0
//! ```
//!
//! `n_kr`/`n_h` count K+R and H anywhere in the stripped sequence; `f1..f4` are the fractions of the
//! peptide's ions at 1+..4+ as the reference recorded them. They may sum to less than 1 (v1's cut
//! states are simply absent) but never more.
//!
//! # Cells and fallback
//!
//! Cell = (min(n_kr, 4), min(n_h, 2), length bin [≤10, 11–14, 15–18, 19–23, ≥24]). A peptide whose
//! cell is empty falls back to (n_kr, n_h), then to n_kr alone, then to the whole table — so the
//! model is total over any sequence, and how often each level was used is reported.

use anyhow::{anyhow, bail, Context, Result};
use std::collections::HashMap;
use std::path::Path;

/// Highest charge a table row can carry.
pub const TABLE_MAX_CHARGE: usize = 4;

type Shares = [f32; TABLE_MAX_CHARGE];

/// (K+R, H, length) features of a stripped sequence, capped as the cells are.
pub fn features(sequence: &str) -> (u8, u8, u8) {
    let n_kr = sequence.bytes().filter(|&b| b == b'K' || b == b'R').count().min(4) as u8;
    let n_h = sequence.bytes().filter(|&b| b == b'H').count().min(2) as u8;
    (n_kr, n_h, length_bin(sequence.len()))
}

fn length_bin(len: usize) -> u8 {
    match len {
        0..=10 => 0,
        11..=14 => 1,
        15..=18 => 2,
        19..=23 => 3,
        _ => 4,
    }
}

/// The fallback level a lookup was served from (0 = full cell … 3 = whole table).
pub const LEVELS: [&str; 4] = ["(K+R, H, length)", "(K+R, H)", "(K+R)", "whole table"];

pub struct ChargeTable {
    full: HashMap<(u8, u8, u8), Vec<Shares>>,
    kr_h: HashMap<(u8, u8), Vec<Shares>>,
    kr: HashMap<u8, Vec<Shares>>,
    all: Vec<Shares>,
}

impl ChargeTable {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| format!("read charge table {}", path.display()))?;
        Self::parse(&text).with_context(|| format!("charge table {}", path.display()))
    }

    pub fn parse(text: &str) -> Result<Self> {
        let mut lines = text.lines().filter(|l| !l.trim().is_empty());
        let header: Vec<&str> = lines.next().ok_or_else(|| anyhow!("empty"))?.split('\t').map(str::trim).collect();
        let want = ["n_kr", "n_h", "length", "f1", "f2", "f3", "f4"];
        if header != want {
            bail!("header must be {:?} (tab-separated), got {:?}", want, header);
        }
        let mut t = ChargeTable { full: HashMap::new(), kr_h: HashMap::new(), kr: HashMap::new(), all: Vec::new() };
        for (i, line) in lines.enumerate() {
            let row = i + 2;
            let v: Vec<&str> = line.split('\t').map(str::trim).collect();
            if v.len() != want.len() {
                bail!("row {row}: expected {} columns, got {}", want.len(), v.len());
            }
            let int = |s: &str| s.parse::<usize>().map_err(|e| anyhow!("row {row}: bad integer {s:?}: {e}"));
            let (n_kr, n_h, len) = (int(v[0])?, int(v[1])?, int(v[2])?);
            if len == 0 {
                bail!("row {row}: length must be >= 1");
            }
            let mut f: Shares = [0.0; TABLE_MAX_CHARGE];
            for (k, s) in v[3..].iter().enumerate() {
                let x: f32 = s.parse().map_err(|e| anyhow!("row {row}: bad fraction {s:?}: {e}"))?;
                if !(x.is_finite() && (0.0..=1.0).contains(&x)) {
                    bail!("row {row}: fraction f{} = {x} outside [0, 1]", k + 1);
                }
                f[k] = x;
            }
            let sum: f32 = f.iter().sum();
            if sum > 1.0 + 1e-4 {
                bail!("row {row}: fractions sum to {sum} > 1");
            }
            if sum <= 0.0 {
                bail!("row {row}: a reference peptide with no ions carries no charge information");
            }
            let key = (n_kr.min(4) as u8, n_h.min(2) as u8, length_bin(len));
            t.full.entry(key).or_default().push(f);
            t.kr_h.entry((key.0, key.1)).or_default().push(f);
            t.kr.entry(key.0).or_default().push(f);
            t.all.push(f);
        }
        if t.all.is_empty() {
            bail!("no rows");
        }
        Ok(t)
    }

    pub fn len(&self) -> usize {
        self.all.len()
    }

    pub fn is_empty(&self) -> bool {
        self.all.is_empty()
    }

    /// The share vector for `sequence` and the fallback level it came from. Deterministic in
    /// (sequence, seed): every modform of a peptide gets the same donor, matching the other models,
    /// which compute charge from the bare sequence.
    pub fn shares(&self, sequence: &str, seed: u64) -> (Shares, usize) {
        let (n_kr, n_h, lb) = features(sequence);
        let (pool, level) = if let Some(p) = self.full.get(&(n_kr, n_h, lb)) {
            (p, 0)
        } else if let Some(p) = self.kr_h.get(&(n_kr, n_h)) {
            (p, 1)
        } else if let Some(p) = self.kr.get(&n_kr) {
            (p, 2)
        } else {
            (&self.all, 3)
        };
        let h = splitmix64(fnv1a(sequence.as_bytes()) ^ splitmix64(seed ^ 0xC4A6_E5EE_D0D0_0001));
        // Unbiased index (multiply-high, not modulo).
        let idx = ((h as u128 * pool.len() as u128) >> 64) as usize;
        (pool[idx], level)
    }
}

/// Replace the charge states of one modform's precursors with `shares`.
///
/// `template` is any precursor of the modform (the model's own output); its m/z fixes the
/// monoisotopic mass, and every other field is per-modform. States with share 0 or above
/// `max_charge` are not emitted. Returns nothing if `template` is empty (the modform produced no ion).
pub fn with_shares(
    template: Vec<timsim_chem::Precursor>,
    shares: &Shares,
    max_charge: u8,
) -> Vec<timsim_chem::Precursor> {
    let Some(t) = template.into_iter().next() else {
        return Vec::new();
    };
    let mono = t.mz * t.charge as f64 - t.charge as f64 * timsim_chem::mass::PROTON;
    (1..=TABLE_MAX_CHARGE.min(max_charge as usize))
        .filter(|z| shares[z - 1] > 0.0)
        .map(|z| {
            let z = z as u8;
            timsim_chem::Precursor {
                precursor_id: timsim_chem::ids::precursor_id(t.modform_id, z),
                charge: z,
                mz: timsim_chem::isotope::mz(mono, z, 0),
                charge_fraction: shares[z as usize - 1],
                isotope_intensity: t.isotope_intensity.clone(),
                ..t.clone()
            }
        })
        .collect()
}

fn fnv1a(b: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &x in b {
        h ^= x as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    const T: &str = "n_kr\tn_h\tlength\tf1\tf2\tf3\tf4\n\
                     1\t0\t8\t0\t1\t0\t0\n\
                     1\t0\t9\t0\t0.7\t0.3\t0\n\
                     3\t1\t25\t0\t0.3\t0.4\t0.3\n";

    #[test]
    fn lookup_is_deterministic_and_stays_in_the_cell() {
        let t = ChargeTable::parse(T).unwrap();
        let (a, la) = t.shares("PEPTIDEK", 7);
        assert_eq!(t.shares("PEPTIDEK", 7), (a, la));
        assert_eq!(la, 0);
        assert!(a == [0.0, 1.0, 0.0, 0.0] || a == [0.0, 0.7, 0.3, 0.0], "{a:?} is not a cell member");
    }

    #[test]
    fn seeds_and_sequences_spread_over_the_cell() {
        let t = ChargeTable::parse(T).unwrap();
        let picks: std::collections::HashSet<u32> =
            (0..64).map(|s| t.shares("PEPTIDEK", s).0[1].to_bits()).collect();
        assert_eq!(picks.len(), 2, "both cell members must be drawn across seeds");
    }

    #[test]
    fn empty_cells_fall_back_level_by_level() {
        let t = ChargeTable::parse(T).unwrap();
        assert_eq!(t.shares("PEPTIDEKAAAAAAAAAAAAK", 1).1, 3); // no row has 2 K/R → whole table
        assert_eq!(t.shares("AAAAAAAAAAAAAAAK", 1).1, 1); // (1, 0) present, length bin 2 absent
        assert_eq!(t.shares("HAAAAAAAK", 1).1, 2); // (1, 1) absent, n_kr 1 present
        assert_eq!(t.shares("AAAAAAAA", 1).1, 3); // no K/R at all → whole table
    }

    #[test]
    fn malformed_tables_are_refused() {
        assert!(ChargeTable::parse("n_kr\tn_h\tlength\tf1\tf2\tf3\n1\t0\t8\t0\t1\t0\n").is_err());
        assert!(ChargeTable::parse("n_kr\tn_h\tlength\tf1\tf2\tf3\tf4\n1\t0\t8\t0\t0.8\t0.4\t0\n").is_err());
        assert!(ChargeTable::parse("n_kr\tn_h\tlength\tf1\tf2\tf3\tf4\n1\t0\t8\t0\t0\t0\t0\n").is_err());
        assert!(ChargeTable::parse("n_kr\tn_h\tlength\tf1\tf2\tf3\tf4\n1\t0\t8\t0\tNaN\t0\t0\n").is_err());
        assert!(ChargeTable::parse("n_kr\tn_h\tlength\tf1\tf2\tf3\tf4\n").is_err());
    }

    #[test]
    fn with_shares_rebuilds_mz_and_ids_from_the_template() {
        use timsim_chem::ionize::{ChargeModel, Flyability, Ionizer};
        let io = Ionizer { charge: ChargeModel::realistic(), flyability: Flyability::Uniform, isotope_depth: 3, seed: 1 };
        let ps = io.precursors_of(11, "PEPTIDEK").unwrap();
        let mono = ps[0].mz * ps[0].charge as f64 - ps[0].charge as f64 * timsim_chem::mass::PROTON;
        let out = with_shares(ps.clone(), &[0.0, 0.5, 0.3, 0.15], 4);
        assert_eq!(out.iter().map(|p| p.charge).collect::<Vec<_>>(), vec![2, 3, 4]);
        for p in &out {
            assert!((p.mz - timsim_chem::isotope::mz(mono, p.charge, 0)).abs() < 1e-9);
            assert_eq!(p.precursor_id, timsim_chem::ids::precursor_id(p.modform_id, p.charge));
            assert_eq!(p.modform_id, ps[0].modform_id);
            assert_eq!(p.ionization_propensity, ps[0].ionization_propensity);
        }
        // An m/z this function rebuilt must equal the model's own m/z where both exist.
        let own_2 = ps.iter().find(|p| p.charge == 2).unwrap();
        let ours_2 = out.iter().find(|p| p.charge == 2).unwrap();
        assert_eq!(own_2.mz.to_bits(), ours_2.mz.to_bits());
        assert_eq!(own_2.precursor_id, ours_2.precursor_id);
        assert_eq!(with_shares(ps, &[0.0, 0.5, 0.3, 0.15], 3).len(), 2, "max_charge caps the table");
        assert!(with_shares(Vec::new(), &[0.0, 1.0, 0.0, 0.0], 4).is_empty());
    }
}
