//! The MS2 (DIA fragment) emission + its **independent** conservation oracle — built before the
//! production sweep wires MS2 into the projector, exactly as the MS1 oracle was.
//!
//! An ion deposits its fragment spectrum into every MS2 frame whose window group transmits its
//! *precursor* m/z at that scan (the diagonal), scaled by `abundance × elution × mobility ×
//! transmission`. Fragments are gated by the **parent**, never re-gated at their own m/z.
//!
//! The oracle discipline (Codex): the reference must share *nothing* with the production path except
//! pure physics it can't get wrong. So [`ms2_render`] gates with mscore's [`TimsTransmissionDIA`]
//! (frame-major), while [`ms2_reference`] is **ion-major**, rebuilds the frame→group map from the
//! schedule pattern, rebuilds the `(group, scan)→window` map from the raw windows, and recomputes the
//! transmission with its OWN logistic — replicating mscore's exact edge convention (soft sigmoids
//! centred at `midpoint ± (half_window + 0.25)`, `k = 15`). If the two agree at every bin, the
//! emission loop, the diagonal lookup, AND the transmission maths are all corroborated independently.

use crate::dia::DiaSchedule;
use crate::render::{elution_frac, gauss_frac, Geometry};
use mscore::timstof::quadrupole::IonTransmission;
use std::collections::{BTreeMap, HashMap};

/// The soft-edge steepness the DIA transmission uses by default (mscore `TimsTransmissionDIA` `k`).
const K: f64 = 15.0;
/// Transmission below this is treated as zero — the logistic never reaches 0, so a cutoff is required
/// (Codex). BOTH paths use the same `EPS`, so it never causes a disagreement.
const EPS: f64 = 1e-3;

/// One ion to render: its elution/mobility placement, abundance, the **precursor** m/z the quad gates
/// on, and both already-projected spectra as `(tof, relative intensity)` — `ms1_peaks` (precursor
/// isotopes, deposited into MS1 frames) and `ms2_peaks` (fragments, into transmission-gated MS2 frames).
#[derive(Clone, Debug)]
pub struct DiaIon {
    pub apex_frame: f64,
    pub scan_center: f64,
    pub abundance: f64,
    pub precursor_mz: f64,
    pub ms1_peaks: Vec<(u32, f32)>,
    pub ms2_peaks: Vec<(u32, f32)>,
    /// `--noise-mz-draw per-observation` only (empty otherwise): the same spectra UNPROJECTED, with each
    /// peak's draw identity, so every observation can take its own m/z offset at deposit time. See
    /// [`ObsNoise`]. A level whose ppm is 0 deposits `ms1_peaks` / `ms2_peaks` instead.
    pub ms1_obs: Vec<ObsPeak>,
    pub ms2_obs: Vec<ObsPeak>,
    /// Fraction of the precursor that survives fragmentation *intact* (incomplete fragmentation). When
    /// `> 0`, that fraction of the precursor's isotope envelope is deposited into the MS2 windows too,
    /// through the same quad diagonal — modelling residual precursor bleed-through. `0.0` (default) is
    /// full fragmentation. Mirrors v1's `precursor_survival`.
    pub survival: f64,
    /// THIS ion's chromatographic peak — see [`crate::render::Ion::elution`].
    pub elution: crate::render::Elution,
    /// THIS ion's mobility width in scans — see [`crate::render::Ion::sigma_scans`].
    pub sigma_scans: f64,
}

/// One peak of a `--noise-mz-draw per-observation` ion: its NOISELESS m/z, its relative intensity, and
/// `h`, the peak's immutable draw identity (`timsim-render`'s `noise_state(precursor, level, raw peak
/// index, seed)`, computed once at projection). Each observation's offset mixes `h` with its own
/// `(frame, scan)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ObsPeak {
    pub mz: f64,
    pub inten: f32,
    pub h: u64,
}

/// v1's A1 m/z noise scope (`--noise-mz-draw per-observation`): an independent offset for EVERY
/// observation of every peak, i.e. per `(frame, scan)` it is deposited in, applied to the m/z before the
/// run's own m/z→TOF conversion.
///
/// v1 (rustdf `sim/precursor.rs`, `sim/dia.rs`) calls `add_mz_noise_normal(ppm)` inside its
/// `(frame, ion, scan)` loop, so a peak's offsets average out across its elution and dither the TOF
/// rounding: a search engine integrating the peak sees a centroid far more precise than one draw. The
/// default `per-peak` scope keeps ONE offset per peak for the whole run, which a search engine cannot
/// average away (measured on the DIA-H01 pilot: DIA-NN true-hit error sd 0.6 ppm on v1's runs vs 2.9 ppm
/// on `per-peak` renders; real timsTOF HeLa runs sit at about 3 ppm).
///
/// Compatibility with v1 is statistical, not bitwise: v1 draws from a thread RNG and rounds the noisy m/z
/// to 1e-6 Th (below 0.002 ppm). The draw here is identity-keyed, so it does not depend on chunking,
/// threading, ion order or `--limit`.
pub struct ObsNoise<'a> {
    /// The 3σ envelope (Gaussian) or half-width (uniform) in ppm, per level. 0 = that level uses the
    /// pre-projected, noiseless `ms*_peaks`.
    pub ppm_ms1: f64,
    pub ppm_ms2: f64,
    /// v1's `mz_noise_uniform` (symmetric; `right_drag` is not ported).
    pub uniform: bool,
    /// The acquisition m/z range. Applied to each observation AFTER its draw, as the `per-peak` path
    /// applies it after its own draw.
    pub mz_min: f64,
    pub mz_max: f64,
    /// One past the highest valid TOF index; deposits are clamped to `tof_max - 1`, as on the default path.
    pub tof_max: u32,
    /// The run's own m/z→TOF converter, so the rounding policy and calibration are the default path's.
    pub to_tof: &'a (dyn Fn(f64) -> u32 + Sync),
}

impl ObsNoise<'_> {
    /// The relative m/z offset of the observation of peak `h` in `(frame, scan)`: Gaussian with
    /// sd `ppm/3 · 1e-6` (v1 `add_mz_noise_normal`), or uniform on `±ppm · 1e-6` (v1
    /// `add_mz_noise_uniform`, `right_drag = false`).
    pub fn rel(&self, h: u64, frame: u32, scan: u32, ppm: f64) -> f64 {
        let z = obs_state(h, frame, scan);
        if self.uniform {
            (2.0 * to01(splitmix64(z ^ 0x2545_F491_4F6C_DD1D)) - 1.0) * ppm * 1e-6
        } else {
            let u1 = to01(splitmix64(z)).max(1e-12); // clamp away from ln(0)
            let u2 = to01(splitmix64(z ^ 0xD1B5_4A32_D192_ED03));
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos() * ppm * 1e-6 / 3.0
        }
    }

    /// The TOF index this observation lands in, or `None` when its perturbed m/z leaves the acquisition
    /// range (the instrument would not record it).
    pub fn tof(&self, p: &ObsPeak, frame: u32, scan: u32, ppm: f64) -> Option<u32> {
        let m = p.mz * (1.0 + self.rel(p.h, frame, scan, ppm));
        if m < self.mz_min || m > self.mz_max {
            return None;
        }
        Some((self.to_tof)(m).min(self.tof_max.saturating_sub(1)))
    }
}

/// Whether a precursor becomes a [`DiaIon`]: it needs an MS1 spectrum and something to deposit, i.e. a
/// projected peak or (`per-observation`) a candidate that can scatter into the acquisition range. With no
/// candidates (`per-peak`, or ppm 0) this is the original rule, unchanged.
pub fn keeps_ion(had_ms1: bool, ms1: &[(u32, f32)], ms2: &[(u32, f32)], ms1_obs: &[ObsPeak], ms2_obs: &[ObsPeak]) -> bool {
    had_ms1 && !(ms1.is_empty() && ms2.is_empty() && ms1_obs.is_empty() && ms2_obs.is_empty())
}

/// The `per-observation` projection of one spectrum. Returns:
/// - `fixed`: the NOISELESS `(tof, intensity)` peaks inside `[mz_min, mz_max]` — exactly what the
///   `per-peak` projection gives at ppm 0, and what a level whose ppm is 0 deposits;
/// - `obs`: when `ppm > 0`, the candidates for per-observation deposits: every peak within the range
///   WIDENED by `3 · ppm · 1e-6` (9σ Gaussian, 3× the uniform bound), so an observation can cross the
///   edge inward as well as outward. Each carries `key(k)`, the draw identity of raw peak index `k`
///   (indexed BEFORE any filtering, as the `per-peak` draw is).
///
/// `gain` maps the stored intensity (the `--ms2-gain` hook); it is applied identically to both lists.
#[allow(clippy::too_many_arguments)]
pub fn project_per_observation(
    peaks: &[(f64, f32)],
    ppm: f64,
    mz_min: f64,
    mz_max: f64,
    tof_max: u32,
    to_tof: &dyn Fn(f64) -> u32,
    key: impl Fn(usize) -> u64,
    gain: impl Fn(f32) -> f32,
) -> (Vec<(u32, f32)>, Vec<ObsPeak>) {
    let widen = 3.0 * ppm * 1e-6;
    let (lo, hi) = (mz_min * (1.0 - widen), mz_max * (1.0 + widen));
    let mut fixed = Vec::with_capacity(peaks.len());
    let mut obs = Vec::new();
    for (k, &(m, inten)) in peaks.iter().enumerate() {
        let inten = gain(inten);
        // Same comparison as the `per-peak` path, so the two agree exactly at ppm 0.
        if !(m < mz_min || m > mz_max) {
            fixed.push((to_tof(m).min(tof_max.saturating_sub(1)), inten));
        }
        if ppm > 0.0 && m >= lo && m <= hi {
            obs.push(ObsPeak { mz: m, inten, h: key(k) });
        }
    }
    (fixed, obs)
}

/// Per-observation draw state: the peak identity `h` folded with the observation's frame and scan
/// through successive avalanches, under a salt of its own so it never coincides with the `per-peak`
/// stream of the same peak. `h` is avalanched BEFORE the frame is folded in: XOR-ing a raw `h` with the
/// frame would make `(h, f)` and `(h ^ d, f ^ d)` the same draw.
fn obs_state(h: u64, frame: u32, scan: u32) -> u64 {
    splitmix64(splitmix64(splitmix64(h ^ 0x0B5E_7A71_0B5E_7A71) ^ frame as u64) ^ scan as u64)
}

/// splitmix64 finaliser: a full-avalanche `u64 -> u64` bijection (the same one `timsim-render` keys its
/// noise with).
fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[inline]
fn to01(x: u64) -> f64 {
    (x >> 11) as f64 / (1u64 << 53) as f64
}

/// Elution frame window `[start, end]` (1-based, to match the DIA schedule / `Frames.Id`). Public so the
/// chunked streaming render can decide which ions are active in a given frame range without duplicating
/// the window maths.
pub fn active_frames(apex: f64, e: &crate::render::Elution, g: &Geometry) -> (u32, u32) {
    // Takes the Elution rather than an ion so DiaIon and the DIA staging struct can share it — and
    // so the window is always THIS peptide's, never the run's.
    let (left, right) = e.half_widths(g.n_sigma);
    let start = (apex - left).max(1.0) as u32;
    let end = ((apex + right) as u32).min(g.n_frames);
    (start, end)
}

fn scan_window(scan_center: f64, sigma_scans: f64, g: &Geometry) -> (u32, u32) {
    let lo = (scan_center - g.n_sigma * sigma_scans).max(0.0) as u32;
    let hi = ((scan_center + g.n_sigma * sigma_scans) as u32).min(g.n_scans - 1);
    (lo, hi)
}

/// PRODUCTION MS2 emission: frame-major, gated by mscore's `TimsTransmissionDIA`. Only MS2 frames emit
/// fragments; the gate returns 0 for a precursor outside the frame's window at that scan.
///
/// Like [`ms2_reference`], this pins the `per-peak` emission only: it reads `ms2_peaks` and never draws
/// per-observation noise, so it must not be handed ions carrying `ms*_obs`.
pub fn ms2_render(ions: &[DiaIon], sched: &DiaSchedule, g: &Geometry) -> BTreeMap<(u32, u32, u32), f64> {
    debug_assert!(ions.iter().all(|io| io.ms1_obs.is_empty() && io.ms2_obs.is_empty()), "per-observation ions");
    let mut out: BTreeMap<(u32, u32, u32), f64> = BTreeMap::new();
    for frame in 1..=g.n_frames {
        if sched.ms_level(frame) != 2 {
            continue; // MS1 frames carry no fragments
        }
        let f = frame as f64;
        for ion in ions {
            let (fs, fe) = active_frames(ion.apex_frame, &ion.elution, g);
            if frame < fs || frame > fe {
                continue;
            }
            let ew = ion.elution.frac(f - 0.5, f + 0.5, ion.apex_frame);
            if ew <= 0.0 {
                continue;
            }
            let (s_lo, s_hi) = scan_window(ion.scan_center, ion.sigma_scans, g);
            for scan in s_lo..=s_hi {
                let t = sched
                    .transmission
                    .apply_transmission(frame as i32, scan as i32, &vec![ion.precursor_mz])[0];
                if t <= EPS {
                    continue;
                }
                let mw = gauss_frac(scan as f64 - 0.5, scan as f64 + 0.5, ion.scan_center, ion.sigma_scans);
                if mw <= 0.0 {
                    continue;
                }
                let base = ion.abundance * ew * mw * t;
                for &(tof, iv) in &ion.ms2_peaks {
                    let v = base * iv as f64;
                    if v <= 0.0 {
                        continue;
                    }
                    *out.entry((frame, scan, tof)).or_insert(0.0) += v;
                }
            }
        }
    }
    out
}

/// mscore's exact quad transfer for one window, reimplemented independently: two sigmoids `k` steep,
/// centred `0.25 Th` outside each window edge, differenced. Matches `apply_transmission(mid, width, k)`.
fn logistic(mid: f64, width: f64, mz: f64) -> f64 {
    let hw = width / 2.0;
    let up_mid = mid - hw - 0.25;
    let down_mid = mid + hw + 0.25;
    let up = 1.0 / (1.0 + (-K * (mz - up_mid)).exp());
    let down = 1.0 / (1.0 + (-K * (mz - down_mid)).exp());
    up - down
}

/// Independent `(window_group, scan) -> (isolation m/z, width)`, rebuilt from the raw windows with the
/// same inclusive expansion and last-write-wins order mscore uses — so overlapping per-scan windows
/// resolve identically without touching `TimsTransmissionDIA`.
fn independent_window_map(sched: &DiaSchedule) -> HashMap<(u32, u32), (f64, f64)> {
    let mut m = HashMap::new();
    for w in &sched.windows {
        for scan in w.scan_num_begin..=w.scan_num_end {
            m.insert((w.window_group, scan), (w.isolation_mz, w.isolation_width));
        }
    }
    m
}

/// INDEPENDENT reference MS2 emission: ion-major, own frame→group (from the pattern), own window map,
/// own logistic. Shares only `gauss_frac`. `per-peak` emission only, as [`ms2_render`].
pub fn ms2_reference(ions: &[DiaIon], sched: &DiaSchedule, g: &Geometry) -> BTreeMap<(u32, u32, u32), f64> {
    debug_assert!(ions.iter().all(|io| io.ms1_obs.is_empty() && io.ms2_obs.is_empty()), "per-observation ions");
    let wmap = independent_window_map(sched);
    let mut out: BTreeMap<(u32, u32, u32), f64> = BTreeMap::new();
    for ion in ions {
        let (fs, fe) = active_frames(ion.apex_frame, &ion.elution, g);
        for frame in fs..=fe {
            let Some(group) = sched.pattern[((frame - 1) % sched.cycle_len) as usize] else {
                continue; // MS1 frame — no fragments
            };
            let f = frame as f64;
            let ew = ion.elution.frac(f - 0.5, f + 0.5, ion.apex_frame);
            if ew <= 0.0 {
                continue;
            }
            let (s_lo, s_hi) = scan_window(ion.scan_center, ion.sigma_scans, g);
            for scan in s_lo..=s_hi {
                let Some(&(iso_mz, iso_w)) = wmap.get(&(group, scan)) else {
                    continue; // no window at this (group, scan) — mscore blocks it too
                };
                let t = logistic(iso_mz, iso_w, ion.precursor_mz);
                if t <= EPS {
                    continue;
                }
                let mw = gauss_frac(scan as f64 - 0.5, scan as f64 + 0.5, ion.scan_center, ion.sigma_scans);
                if mw <= 0.0 {
                    continue;
                }
                let base = ion.abundance * ew * mw * t;
                for &(tof, iv) in &ion.ms2_peaks {
                    let v = base * iv as f64;
                    if v <= 0.0 {
                        continue;
                    }
                    *out.entry((frame, scan, tof)).or_insert(0.0) += v;
                }
            }
        }
    }
    out
}

/// The full DIA render: a single 1-based frame-major pass in which MS1 frames receive each ion's
/// precursor isotopes (ungated) and MS2 frames receive its fragments gated by the diagonal
/// transmission. Calls `emit(frame, ms_ms_type, &triples)` once per non-empty frame — the same
/// `MsMsType` (0 / 9) the writer stamps. The MS2 half is exactly the emission the oracle pins; the MS1
/// half is the same isotope placement the MS1 sweep uses, restricted to MS1 frames.
///
/// Active-set sweep-line (the deferred "active-set-optimised" speed-up; the frame-major version rescanned
/// every ion for every frame — O(frames × ions), which dominated render time at scale). Ions are entered
/// in `frame_start` order against a moving frame cursor and dropped once past `frame_end`, so each frame
/// touches only its active ions: ~O(frames + Σ active). The output is identical to the frame-major
/// version — the same ions are active at each frame, and each frame's active set is visited in original
/// ion-index order, so the per-frame deposit sequence (summed downstream, order-independent anyway) is
/// unchanged.
///
/// The `per-peak` render (no [`ObsNoise`]); `--noise-mz-draw per-observation` goes through
/// [`dia_render_range`].
pub fn dia_render<F: FnMut(u32, u8, &[(u32, u32, f64)])>(
    ions: &[DiaIon],
    sched: &DiaSchedule,
    g: &Geometry,
    emit: F,
) {
    dia_render_range(ions, sched, g, 1, g.n_frames, false, None, emit);
}

/// The sweep restricted to frames `[frame_lo, frame_hi]` — the primitive the chunked streaming render
/// calls once per apex-bucket, so only that bucket's ions are ever resident. `ions` must already contain
/// every ion active anywhere in the range (frame_start ≤ frame_hi and frame_end ≥ frame_lo); ions whose
/// window opened before `frame_lo` are correctly entered at `frame_lo`. Emits only frames in the range,
/// so a caller stitching ranges together writes each frame exactly once.
/// `emit_all`: when true, call `emit` for EVERY frame in the range (with an empty buffer where no ion
/// deposits), so a caller that adds per-frame background (A2 real-data noise / spike-into-real) reaches
/// frames with no synthetic signal too. When false (the default render), only frames with a non-empty
/// deposit are emitted — byte-identical to the frame-major render.
/// `obs`: `Some` for `--noise-mz-draw per-observation` — each observation of a peak at a level whose ppm
/// is > 0 takes its own m/z offset ([`ObsNoise`]). `None` (the default) deposits the pre-projected
/// `ms*_peaks` exactly as before.
#[allow(clippy::too_many_arguments)]
pub fn dia_render_range<F: FnMut(u32, u8, &[(u32, u32, f64)])>(
    ions: &[DiaIon],
    sched: &DiaSchedule,
    g: &Geometry,
    frame_lo: u32,
    frame_hi: u32,
    emit_all: bool,
    obs: Option<&ObsNoise>,
    mut emit: F,
) {
    let (ppm_ms1, ppm_ms2) = obs.map_or((0.0, 0.0), |o| (o.ppm_ms1, o.ppm_ms2));
    let n = ions.len();
    let windows: Vec<(u32, u32)> = ions.iter().map(|io| active_frames(io.apex_frame, &io.elution, g)).collect();
    // Enter in frame_start order; sort indices, not the ions (peaks stay put).
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by_key(|&i| windows[i].0);
    let mut cursor = 0usize;
    let mut active: Vec<usize> = Vec::new();
    // ONE triples buffer for the whole range, cleared per frame: `Vec::clear` keeps the capacity, so the
    // deposit loop allocates once per sub-range instead of once per frame (a busy MS2 frame pushes
    // millions of triples, and the parallel render runs `threads * 8` of these sweeps). The buffer is
    // handed to `emit` as a slice and never escapes, so reuse is invisible to the caller and the emitted
    // CONTENTS are unchanged — byte-identity is untouched.
    let mut buf: Vec<(u32, u32, f64)> = Vec::new();

    for frame in frame_lo..=frame_hi {
        // Enter ions whose window has opened.
        while cursor < n && windows[order[cursor]].0 <= frame {
            active.push(order[cursor]);
            cursor += 1;
        }
        // Leave ions whose window has closed.
        active.retain(|&i| windows[i].1 >= frame);
        let ms_level = sched.ms_level(frame);
        let ms_type = if ms_level == 1 { 0 } else { 9 };
        if active.is_empty() {
            // No synthetic deposit here. Still emit (empty) when the caller wants every frame, so its
            // per-frame background reaches this frame; otherwise skip it (unchanged render).
            if emit_all {
                emit(frame, ms_type, &[]);
            }
            continue;
        }
        active.sort_unstable(); // visit in original-index order -> deterministic, frame-major-identical
        let f = frame as f64;
        buf.clear();
        for &idx in &active {
            let ion = &ions[idx];
            let ew = ion.elution.frac(f - 0.5, f + 0.5, ion.apex_frame);
            if ew <= 0.0 {
                continue;
            }
            let (s_lo, s_hi) = scan_window(ion.scan_center, ion.sigma_scans, g);
            for scan in s_lo..=s_hi {
                // MS1 frames pass everything; MS2 frames gate the precursor through the quad diagonal.
                let t = if ms_level == 1 {
                    1.0
                } else {
                    let tt = sched.transmission.apply_transmission(frame as i32, scan as i32, &vec![ion.precursor_mz])[0];
                    if tt <= EPS {
                        continue;
                    }
                    tt
                };
                let mw = gauss_frac(scan as f64 - 0.5, scan as f64 + 0.5, ion.scan_center, ion.sigma_scans);
                if mw <= 0.0 {
                    continue;
                }
                let base = ion.abundance * ew * mw * t;
                // `ppm` is the deposited peaks' own level: the intact-precursor bleed into MS2 frames
                // below carries the PRECURSOR's m/z noise, not the fragments'.
                let mut deposit = |peaks: &[(u32, f32)], obs_peaks: &[ObsPeak], ppm: f64, gain: f64| {
                    match obs {
                        Some(o) if ppm > 0.0 => {
                            // The candidate set is the acquisition range WIDENED by the noise envelope,
                            // so it is never smaller than the noiseless projection.
                            debug_assert!(obs_peaks.len() >= peaks.len(), "per-observation ion without its obs peaks");
                            for p in obs_peaks {
                                let v = base * gain * p.inten as f64;
                                if v > 0.0 {
                                    if let Some(tof) = o.tof(p, frame, scan, ppm) {
                                        buf.push((scan, tof, v));
                                    }
                                }
                            }
                        }
                        _ => {
                            for &(tof, iv) in peaks {
                                let v = base * gain * iv as f64;
                                if v > 0.0 {
                                    buf.push((scan, tof, v));
                                }
                            }
                        }
                    }
                };
                if ms_level == 1 {
                    // MS1 is a separate survey scan: it always shows the whole intact precursor,
                    // regardless of what the same ions do downstream in MS2. Not part of the partition.
                    deposit(&ion.ms1_peaks, &ion.ms1_obs, ppm_ms1, 1.0);
                } else {
                    // Incomplete fragmentation PARTITIONS the selected precursor current: a `survival`
                    // fraction passes the quad intact (deposited at the precursor isotopes) and the
                    // remaining `1 - survival` fragments. The two must sum to the whole (conservation) —
                    // emitting full fragments + survival would over-count (survival=1 → 200%). Same gate
                    // `t` for both: the quad selects the parent before fragmentation-or-survival.
                    deposit(&ion.ms2_peaks, &ion.ms2_obs, ppm_ms2, 1.0 - ion.survival);
                    if ion.survival > 0.0 {
                        deposit(&ion.ms1_peaks, &ion.ms1_obs, ppm_ms1, ion.survival);
                    }
                }
            }
        }
        if emit_all || !buf.is_empty() {
            emit(frame, ms_type, &buf);
        }
        // Bound what the reuse RETAINS. Without this, one pathological frame (an ion whose elution
        // window spans the run, a dense mid-gradient MS2 frame) would pin its own worst-case capacity
        // for the rest of the sub-range — times the thread count, since every thread runs its own
        // sweep. Dropping the buffer costs one re-grow on the next frame and caps the retention.
        if buf.capacity() > MAX_RETAINED_TRIPLES {
            buf = Vec::new();
        }
    }
}

/// Upper bound on the triples capacity carried between frames: 16M triples x 16 B = 256 MiB per sweep.
/// Sized to sit well inside the ~1.75 GiB/thread the render is already budgeted for (see
/// `timsim_cli::memguard`), while being far above any ordinary frame so the reuse actually pays off.
const MAX_RETAINED_TRIPLES: usize = 16 * 1024 * 1024;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::cube_diff;
    use ms_io::data::meta::{DiaMsMisInfo, DiaMsMsWindow};

    fn geom() -> Geometry {
        Geometry { n_frames: 9, n_scans: 3, sigma_frames: 2.0, sigma_scans: 1.0, n_sigma: 2.0, shape: crate::render::PeakShape::Gaussian }
    }

    /// 3-frame cycle [MS1, g1(iso 500±5), g2(iso 600±5)], windows over scans 0..=2.
    fn schedule() -> DiaSchedule {
        let info = vec![
            DiaMsMisInfo { frame_id: 2, window_group: 1 },
            DiaMsMisInfo { frame_id: 3, window_group: 2 },
            DiaMsMisInfo { frame_id: 5, window_group: 1 },
            DiaMsMisInfo { frame_id: 6, window_group: 2 },
        ];
        let win = |g: u32, mz: f64| DiaMsMsWindow {
            window_group: g, scan_num_begin: 0, scan_num_end: 2, isolation_mz: mz, isolation_width: 10.0, collision_energy: 25.0,
        };
        DiaSchedule::build(&info, vec![win(1, 500.0), win(2, 600.0)], 9, 3).unwrap()
    }

    fn ion(precursor_mz: f64) -> DiaIon {
        DiaIon {
            apex_frame: 5.0,
            scan_center: 1.0,
            abundance: 100.0,
            precursor_mz,
            ms1_peaks: vec![],
            ms2_peaks: vec![(300, 1.0), (450, 0.4)],
            ms1_obs: vec![],
            ms2_obs: vec![],
            survival: 0.0,
            elution: crate::render::Elution::global(2.5, crate::render::PeakShape::Gaussian),
            sigma_scans: 1.5,
        }
    }

    /// The load-bearing test: production (mscore-gated, frame-major) and the independent reference
    /// (own logistic, ion-major) agree at every bin.
    #[test]
    fn production_matches_independent_reference_every_bin() {
        let (s, g) = (schedule(), geom());
        // Include a precursor AT the g1 window edge (505 = midpoint+half_window), where transmission
        // is partial (~0.5) and the exact soft-edge convention matters — otherwise the every-bin test
        // can't distinguish edge conventions (verified: mutating the 0.25 offset then fails here).
        let ions = vec![ion(500.0), ion(600.0), ion(505.0)];
        let prod = ms2_render(&ions, &s, &g);
        let refr = ms2_reference(&ions, &s, &g);
        assert!(!prod.is_empty(), "fixture emitted nothing — test would be vacuous");
        let (worst, only_p, only_r) = cube_diff(&prod, &refr);
        assert_eq!((only_p, only_r), (0, 0), "bin set differs ({only_p}, {only_r})");
        assert!(worst < 1e-12, "worst per-bin diff {worst:.3e}");
    }

    /// A precursor outside every window at its scan emits ZERO MS2 signal.
    #[test]
    fn precursor_outside_all_windows_emits_nothing() {
        let (s, g) = (schedule(), geom());
        let prod = ms2_render(&[ion(1000.0)], &s, &g); // 1000 is far from both 500 and 600
        assert!(prod.is_empty(), "an untransmitted precursor produced {} bins", prod.len());
    }

    /// Duplicating an ion exactly doubles its fragment bins (linearity / no cross-ion interference).
    #[test]
    fn duplicating_an_ion_doubles_its_bins() {
        let (s, g) = (schedule(), geom());
        let one = ms2_render(&[ion(500.0)], &s, &g);
        let two = ms2_render(&[ion(500.0), ion(500.0)], &s, &g);
        assert_eq!(one.keys().collect::<Vec<_>>(), two.keys().collect::<Vec<_>>(), "bin set changed");
        for (k, &v1) in &one {
            assert!((two[k] - 2.0 * v1).abs() < 1e-9, "bin {k:?} not doubled");
        }
    }

    /// Fragments are gated by the PRECURSOR, never their own m/z: moving a fragment's tof changes only
    /// where its peaks land, never WHICH (frame, scan) cells emit.
    #[test]
    fn fragments_are_not_re_gated_at_their_own_mz() {
        let (s, g) = (schedule(), geom());
        let mut moved = ion(500.0);
        moved.ms2_peaks = vec![(99_999, 1.0), (1, 0.4)]; // wildly different tof (=m/z)
        let base_cells: std::collections::BTreeSet<(u32, u32)> =
            ms2_render(&[ion(500.0)], &s, &g).keys().map(|&(f, sc, _)| (f, sc)).collect();
        let moved_cells: std::collections::BTreeSet<(u32, u32)> =
            ms2_render(&[moved], &s, &g).keys().map(|&(f, sc, _)| (f, sc)).collect();
        assert_eq!(base_cells, moved_cells, "fragment m/z changed the emitting (frame, scan) set");
    }

    /// `dia_render`'s MS2 half must reproduce the oracle-pinned `ms2_render` exactly, and its MS1
    /// frames must carry the precursor isotopes (never fragments).
    #[test]
    fn dia_render_ms2_matches_the_oracle_and_ms1_carries_isotopes() {
        let (s, g) = (schedule(), geom());
        let mut ions = vec![ion(500.0), ion(600.0), ion(505.0)];
        for io in &mut ions {
            io.ms1_peaks = vec![(700, 1.0), (704, 0.5)]; // precursor isotopes at some tof
        }
        let mut ms2_cube: BTreeMap<(u32, u32, u32), f64> = BTreeMap::new();
        let mut ms1_frames = 0u32;
        dia_render(&ions, &s, &g, |frame, ms_type, tri| {
            if ms_type == 9 {
                for &(scan, tof, v) in tri {
                    *ms2_cube.entry((frame, scan, tof)).or_insert(0.0) += v;
                }
            } else {
                ms1_frames += 1;
                // MS1 frames carry only the precursor isotope tofs (700, 704), never fragment tofs.
                assert!(tri.iter().all(|&(_, tof, _)| tof == 700 || tof == 704), "MS1 frame has non-isotope tof");
            }
        });
        let oracle = ms2_render(&ions, &s, &g);
        let (worst, a, b) = cube_diff(&ms2_cube, &oracle);
        assert_eq!((a, b), (0, 0), "dia_render MS2 bins differ from ms2_render");
        assert!(worst < 1e-12, "dia_render MS2 values differ (worst {worst:.3e})");
        assert!(ms1_frames >= 1, "expected at least one MS1 frame with content");
    }

    /// Incomplete fragmentation PARTITIONS the MS2 current: fragments scale by `1 - survival` and the
    /// intact precursor by `survival`, so the surviving-precursor : fragment ratio is `s/(1-s)` (not `s`)
    /// and the two sum to the whole. `survival = 0` (default) leaves the MS2 frames fragment-only.
    #[test]
    fn survival_bleeds_precursor_into_ms2() {
        let (s, g) = (schedule(), geom());
        let mut base = ion(500.0); // group-1 precursor; ms2 tof 300 (iv 1.0), 450 (0.4)
        base.ms1_peaks = vec![(700, 1.0)]; // one precursor isotope at a tof distinct from any fragment

        // survival = 0: the precursor tof must never appear in an MS2 frame.
        let mut off = base.clone();
        off.survival = 0.0;
        dia_render(&[off], &s, &g, |_f, ms_type, tri| {
            if ms_type == 9 {
                assert!(tri.iter().all(|&(_, tof, _)| tof != 700), "survival=0 leaked precursor into MS2");
            }
        });

        // survival > 0: the precursor tof appears at survival × the co-located fragment (tof 300, iv 1.0).
        let mut on = base;
        on.survival = 0.3;
        let mut seen = false;
        dia_render(&[on], &s, &g, |_f, ms_type, tri| {
            if ms_type != 9 {
                return;
            }
            let mut by: HashMap<u32, (Option<f64>, Option<f64>)> = HashMap::new();
            for &(scan, tof, v) in tri {
                let e = by.entry(scan).or_insert((None, None));
                if tof == 300 { e.0 = Some(v); }
                if tof == 700 { e.1 = Some(v); }
            }
            for (_, (frag, prec)) in by {
                if let (Some(fr), Some(pr)) = (frag, prec) {
                    seen = true;
                    // s=0.3 -> precursor/fragment = 0.3/0.7; fragments carry (1-s), precursor s.
                    let want = 0.3 / 0.7;
                    assert!((pr / fr - want).abs() < 1e-9, "precursor:fragment should be s/(1-s)={want}; got {}", pr / fr);
                }
            }
        });
        assert!(seen, "survival>0 produced no precursor bleed-through in MS2");
    }

    // ── `--noise-mz-draw per-observation` ───────────────────────────────────────────────────────────

    /// 0.01 Th bins, rounding: a stand-in for the run's calibrator (the integration test uses the real one).
    fn lin_tof(m: f64) -> u32 {
        (m * 100.0).round().max(0.0) as u32
    }

    fn obs_cfg(ppm_ms1: f64, ppm_ms2: f64, uniform: bool) -> ObsNoise<'static> {
        ObsNoise { ppm_ms1, ppm_ms2, uniform, mz_min: 100.0, mz_max: 1700.0, tof_max: 1_000_000, to_tof: &lin_tof }
    }

    /// 90 frames of the same [MS1, g1, g2] cycle, 40 scans, windows over every scan.
    fn big() -> (DiaSchedule, Geometry) {
        let info = vec![
            DiaMsMisInfo { frame_id: 2, window_group: 1 },
            DiaMsMisInfo { frame_id: 3, window_group: 2 },
            DiaMsMisInfo { frame_id: 5, window_group: 1 },
            DiaMsMisInfo { frame_id: 6, window_group: 2 },
        ];
        let win = |g: u32, mz: f64| DiaMsMsWindow {
            window_group: g, scan_num_begin: 0, scan_num_end: 39, isolation_mz: mz, isolation_width: 10.0, collision_energy: 25.0,
        };
        let s = DiaSchedule::build(&info, vec![win(1, 500.0), win(2, 600.0)], 90, 40).unwrap();
        let g = Geometry { n_frames: 90, n_scans: 40, sigma_frames: 2.0, sigma_scans: 1.0, n_sigma: 3.0, shape: crate::render::PeakShape::Gaussian };
        (s, g)
    }

    /// One MS1 peak and one fragment, both lists built by `project_per_observation` exactly as the
    /// render builds them. Wide elution and mobility, so each peak is observed hundreds of times.
    fn obs_ion(precursor_mz: f64, ms1_mz: f64, ms2_mz: f64, ppm_ms1: f64, ppm_ms2: f64) -> DiaIon {
        let (m1, o1) = project_per_observation(&[(ms1_mz, 1.0)], ppm_ms1, 100.0, 1700.0, 1_000_000, &lin_tof, |k| 0xA1_0000 + k as u64, |x| x);
        let (m2, o2) = project_per_observation(&[(ms2_mz, 1.0)], ppm_ms2, 100.0, 1700.0, 1_000_000, &lin_tof, |k| 0xA2_0000 + k as u64, |x| x);
        DiaIon {
            apex_frame: 45.0,
            scan_center: 20.0,
            abundance: 1000.0,
            precursor_mz,
            ms1_peaks: m1,
            ms2_peaks: m2,
            ms1_obs: o1,
            ms2_obs: o2,
            survival: 0.0,
            elution: crate::render::Elution::global(8.0, crate::render::PeakShape::Gaussian),
            sigma_scans: 5.0,
        }
    }

    type Dep = (u32, u8, u32, u32, f64); // frame, MsMsType, scan, tof, value

    fn deposits(ions: &[DiaIon], s: &DiaSchedule, g: &Geometry, lo: u32, hi: u32, obs: Option<&ObsNoise>) -> Vec<Dep> {
        let mut out = Vec::new();
        dia_render_range(ions, s, g, lo, hi, false, obs, |f, t, tri| {
            out.extend(tri.iter().map(|&(sc, tof, v)| (f, t, sc, tof, v)));
        });
        out
    }

    fn mean_sd(x: &[f64]) -> (f64, f64) {
        let n = x.len() as f64;
        let m = x.iter().sum::<f64>() / n;
        (m, (x.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / (n - 1.0)).sqrt())
    }

    fn corr(a: &[f64], b: &[f64]) -> f64 {
        let ((ma, sa), (mb, sb)) = (mean_sd(a), mean_sd(b));
        a.iter().zip(b).map(|(x, y)| (x - ma) * (y - mb)).sum::<f64>() / ((a.len() as f64 - 1.0) * sa * sb)
    }

    /// The continuous draws, before any TOF rounding: v1's Gaussian (sd = ppm/3 · 1e-6) and uniform
    /// (±ppm · 1e-6) distributions, over 10^5 observations.
    #[test]
    fn obs_draws_have_v1s_distribution() {
        let keys: Vec<(u64, u32, u32)> =
            (0..100u64).flat_map(|h| (0..100u32).flat_map(move |f| (0..10u32).map(move |s| (h, f, s)))).collect();
        let n = keys.len() as f64;
        let g = obs_cfg(6.5, 6.5, false);
        let z: Vec<f64> = keys.iter().map(|&(h, f, s)| g.rel(h, f, s, 6.5) / (6.5e-6 / 3.0)).collect();
        let (m, sd) = mean_sd(&z);
        assert!(m.abs() < 3.0 / n.sqrt(), "Gaussian mean {m}");
        assert!((sd - 1.0).abs() < 0.02, "Gaussian sd {sd} (want 1, i.e. ppm/3)");
        let u = obs_cfg(6.5, 6.5, true);
        let x: Vec<f64> = keys.iter().map(|&(h, f, s)| u.rel(h, f, s, 6.5) / 6.5e-6).collect();
        assert!(x.iter().all(|v| v.abs() <= 1.0), "uniform draw outside ±ppm");
        let (m, sd) = mean_sd(&x);
        assert!(m.abs() < 3.0 / (3.0 * n).sqrt(), "uniform mean {m}");
        assert!((sd - 1.0 / 3f64.sqrt()).abs() < 0.02 / 3f64.sqrt(), "uniform sd {sd}");
    }

    /// Every observation is its own draw: neighbouring frames and neighbouring scans are uncorrelated.
    #[test]
    fn adjacent_observations_are_independent() {
        let g = obs_cfg(6.5, 6.5, false);
        let keys: Vec<(u64, u32, u32)> =
            (0..100u64).flat_map(|h| (0..100u32).flat_map(move |f| (0..10u32).map(move |s| (h, f, s)))).collect();
        let at = |dh: u64, df: u32, ds: u32| -> Vec<f64> { keys.iter().map(|&(h, f, s)| g.rel(h + dh, f + df, s + ds, 6.5)).collect() };
        let base = at(0, 0, 0);
        for (what, other) in [("frame", at(0, 1, 0)), ("scan", at(0, 0, 1)), ("peak", at(1000, 0, 0))] {
            let r = corr(&base, &other);
            assert!(r.abs() < 0.02, "adjacent {what} correlation {r}");
        }
    }

    /// Each deposited observation sits exactly where ITS OWN draw sends it through the converter: the
    /// right peak identity, the right `(frame, scan)`, and the level's own ppm. The intact-precursor
    /// bleed into MS2 frames carries the PRECURSOR's ppm and identity, not the fragments'.
    #[test]
    fn each_observation_lands_where_its_own_draw_sends_it() {
        let (s, g) = big();
        let (ppm1, ppm2) = (30.0, 3.0);
        let mut io = obs_ion(500.0, 700.0, 300.0, ppm1, ppm2);
        io.survival = 0.3;
        let cfg = obs_cfg(ppm1, ppm2, false);
        let (h1, h2) = (io.ms1_obs[0].h, io.ms2_obs[0].h);
        let (mut n_ms1, mut n_frag, mut n_bleed, mut bleed_moved) = (0, 0, 0, 0);
        for (f, t, sc, tof, _) in deposits(&[io], &s, &g, 1, 90, Some(&cfg)) {
            let want = |m: f64, h: u64, ppm: f64| lin_tof(m * (1.0 + cfg.rel(h, f, sc, ppm)));
            if t == 0 {
                n_ms1 += 1;
                assert_eq!(tof, want(700.0, h1, ppm1), "MS1 observation (frame {f}, scan {sc})");
            } else if tof < 50_000 {
                n_frag += 1;
                assert_eq!(tof, want(300.0, h2, ppm2), "fragment observation (frame {f}, scan {sc})");
            } else {
                n_bleed += 1;
                assert_eq!(tof, want(700.0, h1, ppm1), "bleed observation (frame {f}, scan {sc})");
                bleed_moved += (tof != 70_000) as u32;
            }
        }
        assert!(n_ms1 > 100 && n_frag > 100 && n_bleed > 100, "too few observations: {n_ms1} {n_frag} {n_bleed}");
        assert!(bleed_moved > 10, "the bleed did not scatter at the MS1 ppm");
    }

    /// ppm 0 makes the mode a no-op: byte-identical to the per-peak render, per level.
    #[test]
    fn zero_ppm_is_the_per_peak_render() {
        let (s, g) = big();
        // Half-bin phases, so any draw at all would move some observations across a rounding boundary.
        let ions = vec![obs_ion(500.0, 700.005, 300.005, 6.5, 6.5), obs_ion(600.0, 800.005, 400.005, 6.5, 6.5)];
        let none = deposits(&ions, &s, &g, 1, 90, None);
        assert!(!none.is_empty());
        assert_eq!(deposits(&ions, &s, &g, 1, 90, Some(&obs_cfg(0.0, 0.0, false))), none, "both levels at 0");
        // One level at 0: that level is untouched, the other one scatters.
        let ms2_only = deposits(&ions, &s, &g, 1, 90, Some(&obs_cfg(0.0, 6.5, false)));
        let by_level = |d: &[Dep], t: u8| -> Vec<Dep> { d.iter().copied().filter(|x| x.1 == t).collect() };
        assert_eq!(by_level(&ms2_only, 0), by_level(&none, 0), "MS1 at ppm 0 must be untouched");
        assert_ne!(by_level(&ms2_only, 9), by_level(&none, 9), "MS2 at ppm 6.5 did not scatter");
        let ms1_only = deposits(&ions, &s, &g, 1, 90, Some(&obs_cfg(6.5, 0.0, false)));
        assert_eq!(by_level(&ms1_only, 9), by_level(&none, 9), "MS2 at ppm 0 must be untouched");
        assert_ne!(by_level(&ms1_only, 0), by_level(&none, 0), "MS1 at ppm 6.5 did not scatter");
    }

    /// The point of the mode. Per peak, every observation rounds to the same TOF bin, so the integrated
    /// centroid keeps the rounding offset. Per observation, the draws dither the rounding and the
    /// intensity-weighted centroid converges on the continuous position, within its standard error.
    #[test]
    fn per_observation_dithers_away_the_rounding_offset() {
        let (s, g) = big();
        for phase in [0.0, 0.25, 0.5, 0.75] {
            let c = 50_000.0 + phase; // continuous TOF position, in 0.01 Th bins
            let m = c / 100.0;
            let ppm = 3.0 * 0.4 / c * 1e6; // sd 0.4 bin
            let io = obs_ion(500.0, m, 300.0, ppm, 0.0);
            let ms1 = |d: Vec<Dep>| -> Vec<(f64, f64)> { d.into_iter().filter(|x| x.1 == 0).map(|x| (x.3 as f64, x.4)).collect() };
            let centroid = |o: &[(f64, f64)]| o.iter().map(|(t, v)| t * v).sum::<f64>() / o.iter().map(|x| x.1).sum::<f64>();
            let obs = ms1(deposits(std::slice::from_ref(&io), &s, &g, 1, 90, Some(&obs_cfg(ppm, 0.0, false))));
            let (sw, sw2) = (obs.iter().map(|x| x.1).sum::<f64>(), obs.iter().map(|x| x.1 * x.1).sum::<f64>());
            let n_eff = sw * sw / sw2;
            assert!(n_eff > 100.0, "fixture too thin: n_eff {n_eff}");
            // Per-observation spread: the draw (0.4 bin) plus the dithered rounding (1/12 bin^2).
            let se = (0.4f64.powi(2) + 1.0 / 12.0).sqrt() / n_eff.sqrt();
            let got = centroid(&obs);
            assert!((got - c).abs() < 4.0 * se, "phase {phase}: centroid {got} vs {c} (4 SE = {})", 4.0 * se);
            if phase > 0.0 {
                let fixed = centroid(&ms1(deposits(&[io], &s, &g, 1, 90, None)));
                assert!((fixed - c).abs() > 4.0 * se, "phase {phase}: per-peak centroid {fixed} should keep the rounding offset");
            }
        }
    }

    /// The acquisition range applies to each observation after its draw: a peak just outside can deposit
    /// inward, and a peak just inside loses the observations that leave.
    #[test]
    fn observations_cross_the_acquisition_edges_both_ways() {
        let (s, g) = big();
        let ppm = 24.0; // sd 8 ppm; the peaks sit 5 ppm from the edge
        let ms1_count = |m: f64, obs: Option<&ObsNoise>| -> usize {
            let io = obs_ion(500.0, m, 300.0, ppm, 0.0);
            deposits(&[io], &s, &g, 1, 90, obs).iter().filter(|x| x.1 == 0).count()
        };
        let cfg = obs_cfg(ppm, 0.0, false);
        let outside = 1700.0 * (1.0 + 5e-6);
        let (fixed, cand) = project_per_observation(&[(outside, 1.0)], ppm, 100.0, 1700.0, 1_000_000, &lin_tof, |k| k as u64, |x| x);
        assert!(fixed.is_empty() && cand.len() == 1, "a peak 5 ppm outside must be a candidate, not a projected peak");
        assert_eq!(ms1_count(outside, None), 0);
        assert!(ms1_count(outside, Some(&cfg)) > 0, "no observation crossed inward");
        let inside = 1700.0 * (1.0 - 5e-6);
        let (all, kept) = (ms1_count(inside, None), ms1_count(inside, Some(&cfg)));
        assert!(kept > 0 && kept < all, "inside peak: {kept} of {all} observations kept");
    }

    /// An ion whose noiseless peaks all fall just outside the acquisition range is still rendered when it
    /// has candidates that can scatter in; without candidates the rule is the original one.
    #[test]
    fn ions_with_only_edge_candidates_are_kept() {
        let c = [ObsPeak { mz: 1700.01, inten: 1.0, h: 1 }];
        let p = [(1u32, 1.0f32)];
        assert!(keeps_ion(true, &[], &[], &c, &[]), "MS1 candidate only");
        assert!(keeps_ion(true, &[], &[], &[], &c), "MS2 candidate only");
        assert!(!keeps_ion(false, &p, &p, &c, &c), "no MS1 spectrum");
        for (m1, m2) in [(&p[..], &[][..]), (&[][..], &p[..]), (&p[..], &p[..]), (&[][..], &[][..])] {
            let original = !(m1.is_empty() && m2.is_empty());
            assert_eq!(keeps_ion(true, m1, m2, &[], &[]), original, "per-peak rule changed");
        }
    }

    /// Candidates keep their RAW peak index as the draw identity, even when earlier peaks are filtered out.
    #[test]
    fn candidates_keep_their_raw_peak_index() {
        let (fixed, cand) = project_per_observation(
            &[(50.0, 1.0), (500.0, 0.5)], 6.5, 100.0, 1700.0, 1_000_000, &lin_tof, |k| 1000 + k as u64, |x| 2.0 * x,
        );
        assert_eq!(fixed, vec![(50_000, 1.0)], "noiseless projection, gain applied");
        assert_eq!(cand, vec![ObsPeak { mz: 500.0, inten: 1.0, h: 1001 }], "raw index 1, not 0");
        let (_, none) = project_per_observation(&[(500.0, 1.0)], 0.0, 100.0, 1700.0, 1_000_000, &lin_tof, |k| k as u64, |x| x);
        assert!(none.is_empty(), "ppm 0 carries no candidates");
    }

    /// Identity-keyed draws: the same render twice is identical, and so is a render split into frame
    /// ranges (what the chunked and parallel paths do).
    #[test]
    fn per_observation_is_deterministic_and_range_split_invariant() {
        let (s, g) = big();
        let ions = vec![obs_ion(500.0, 700.3, 300.2, 6.5, 6.5), obs_ion(600.0, 650.1, 410.7, 6.5, 6.5), obs_ion(503.0, 900.9, 250.4, 6.5, 6.5)];
        let cfg = obs_cfg(6.5, 6.5, false);
        let whole = deposits(&ions, &s, &g, 1, 90, Some(&cfg));
        assert_eq!(deposits(&ions, &s, &g, 1, 90, Some(&cfg)), whole, "two renders differ");
        let mut split = deposits(&ions, &s, &g, 1, 37, Some(&cfg));
        split.extend(deposits(&ions, &s, &g, 38, 90, Some(&cfg)));
        assert_eq!(split, whole, "frame-range split changed the render");
    }

    /// Only positions move: away from the edges, every frame deposits the same values in the same order,
    /// so the pre-quantisation intensity per frame is identical between the modes.
    #[test]
    fn pre_quantisation_intensity_is_conserved() {
        let (s, g) = big();
        let ions = vec![obs_ion(500.0, 700.3, 300.2, 6.5, 6.5), obs_ion(600.0, 650.1, 410.7, 6.5, 6.5)];
        let sums = |d: Vec<Dep>| -> BTreeMap<u32, f64> {
            let mut m = BTreeMap::new();
            for (f, _, _, _, v) in d {
                *m.entry(f).or_insert(0.0) += v;
            }
            m
        };
        let a = sums(deposits(&ions, &s, &g, 1, 90, None));
        let b = sums(deposits(&ions, &s, &g, 1, 90, Some(&obs_cfg(6.5, 6.5, false))));
        assert_eq!(a, b, "per-frame intensity moved with the draw scope");
    }

    /// Uniform mode: every quantised observation within the uniform bound plus half a bin, and spread.
    #[test]
    fn uniform_observations_stay_inside_their_bound() {
        let (s, g) = big();
        let (m, ppm) = (500.0025, 24.0);
        let io = obs_ion(500.0, m, 300.0, ppm, 0.0);
        let c = m * 100.0;
        let bound = ppm * 1e-6 * c + 0.5;
        let off: Vec<f64> = deposits(&[io], &s, &g, 1, 90, Some(&obs_cfg(ppm, 0.0, true)))
            .into_iter().filter(|x| x.1 == 0).map(|x| x.3 as f64 - c).collect();
        assert!(off.iter().all(|o| o.abs() <= bound), "an observation left the uniform bound ±{bound}");
        let (_, sd) = mean_sd(&off);
        assert!(sd > 0.5, "uniform draws did not spread (sd {sd} bins)");
    }

    /// A precursor in group 1's window emits only in g1 MS2 frames (2, 5, 8), not g2 frames — the
    /// diagonal selects exactly the transmitting group.
    #[test]
    fn precursor_emits_only_in_its_transmitting_group() {
        let (s, g) = (schedule(), geom());
        let prod = ms2_render(&[ion(500.0)], &s, &g);
        let frames: std::collections::BTreeSet<u32> = prod.keys().map(|&(f, _, _)| f).collect();
        for f in &frames {
            assert_eq!(s.window_group(*f), Some(1), "emitted in frame {f} which is not group 1");
        }
        assert!(frames.contains(&5), "expected emission in g1 frame 5");
    }
}
