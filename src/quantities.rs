//! The per-sample abundance axis (`peptide_quantities`), as the renders read it.
//!
//! One loader for `timsim-render`, `timsim-render-thermo` and `timsim-render-sciex`. Each used to carry
//! its own copy, and all three copies shared the silent unit-amount fallback refused below.

use anyhow::{anyhow, bail, Result};
use arrow::array::{Array, Float64Array, StringArray, UInt64Array};
use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use timsim_schema::tables::peptide_quantities as PQ;

/// `peptide_id -> amount_amol` for one sample of the design (the first sorted `sample_id` if `sample`
/// is None). Empty map when no `peptide_quantities` path is given — the caller then falls back to a
/// unit amount, i.e. abundance driven by charge/ionisation propensities only.
///
/// A `sample` with no rows is an ERROR, not that fallback: a misspelt sample name would otherwise render
/// a different experiment, at unit amounts, with at most a warning to show for it.
pub fn load_amounts(path: &Option<PathBuf>, sample: &Option<String>) -> Result<HashMap<u64, f64>> {
    let mut out = HashMap::new();
    let Some(path) = path else { return Ok(out) };

    // Resolve the sample: the caller's choice, or the first sorted id present.
    let chosen = match sample {
        Some(s) => s.clone(),
        None => sample_ids(path)?.into_iter().next().ok_or_else(|| anyhow!("{} has no samples", path.display()))?,
    };

    for b in timsim_schema::read(path, PQ::TABLE)? {
        let pid: &UInt64Array = b.column_by_name(PQ::PEPTIDE_ID).unwrap().as_any().downcast_ref().unwrap();
        let sid: &StringArray = b.column_by_name(PQ::SAMPLE_ID).unwrap().as_any().downcast_ref().unwrap();
        let amt: &Float64Array = b.column_by_name(PQ::AMOUNT_AMOL).unwrap().as_any().downcast_ref().unwrap();
        for i in 0..b.num_rows() {
            if sid.value(i) == chosen {
                out.insert(pid.value(i), amt.value(i));
            }
        }
    }
    if out.is_empty() {
        bail!("--sample {chosen:?} has no rows in {}; its samples are {:?}", path.display(), sample_ids(path)?);
    }
    Ok(out)
}

/// The distinct `sample_id`s of a `peptide_quantities` artifact, sorted.
fn sample_ids(path: &Path) -> Result<Vec<String>> {
    let mut ids = BTreeSet::new();
    for b in timsim_schema::read(path, PQ::TABLE)? {
        let s: &StringArray = b.column_by_name(PQ::SAMPLE_ID).unwrap().as_any().downcast_ref().unwrap();
        for i in 0..b.num_rows() {
            if !ids.contains(s.value(i)) {
                ids.insert(s.value(i).to_string());
            }
        }
    }
    Ok(ids.into_iter().collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::ArrayRef;
    use std::sync::Arc;

    /// A misspelt `--sample` used to load NOTHING, and the render then ran at unit amounts: a different
    /// experiment that finished normally. It must fail, and say which samples exist. The two legitimate
    /// cases stay as they were: no artifact at all (the caller's unit-amount fallback), and no `--sample`
    /// (the first sorted one).
    #[test]
    fn a_sample_with_no_rows_is_refused_and_the_real_ones_are_named() {
        let dir = std::env::temp_dir().join(format!("timsim_quantities_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("peptide_quantities.parquet");
        let rows = crate::batch(PQ::TABLE, vec![
            Arc::new(UInt64Array::from(vec![1u64, 2, 1, 2])) as ArrayRef,
            Arc::new(StringArray::from(vec!["B_R1", "B_R1", "A_R1", "A_R1"])),
            Arc::new(Float64Array::from(vec![10.0, 20.0, 1.0, 2.0])),
        ])
        .unwrap();
        timsim_schema::write(&path, PQ::TABLE, &rows, &crate::producer("test"), None).unwrap();
        let path = Some(path);

        assert_eq!(load_amounts(&path, &Some("B_R1".into())).unwrap(), HashMap::from([(1, 10.0), (2, 20.0)]));
        assert_eq!(load_amounts(&path, &None).unwrap(), HashMap::from([(1, 1.0), (2, 2.0)]));
        assert!(load_amounts(&None, &Some("A_R1".into())).unwrap().is_empty());

        let e = load_amounts(&path, &Some("A_R2".into())).unwrap_err().to_string();
        assert!(e.contains(r#"--sample "A_R2" has no rows"#) && e.ends_with(r#"its samples are ["A_R1", "B_R1"]"#), "{e}");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
