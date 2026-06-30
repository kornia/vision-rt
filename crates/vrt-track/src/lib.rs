//! Multi-object tracking (BoT-SORT-style), generic over the track state so the
//! same engine serves 2D now and 3D later.
//!
//! Tracking-by-detection: feed per-frame detections (e.g. from `vrt-rfdetr`) +
//! the time step, get back stable-id [`TrackOut`]s. The association is light CPU
//! (Kalman predict/update + Hungarian on a small N×M); the heavy lifting is the
//! detector (and, later, ReID), which are the GPU/TensorRT parts.
//!
//! # Generic over state (the 2D→3D swap point)
//! A [`Tracker<M>`] is parameterized by a [`TrackModel`] — the per-track Kalman
//! state. [`Box2DModel`] is the 2D box filter; a future `Box3DModel` implements
//! the same trait and drops in unchanged.
//!
//! # Phase 1 (this module): motion-only BoT-SORT
//! ByteTrack two-stage association (high-confidence detections first, then a
//! second pass with low-confidence ones to recover occluded tracks), per-class
//! gating, Kalman constant-velocity motion, and a tentative→confirmed→lost
//! lifecycle. Appearance/ReID and camera-motion compensation are later phases.

mod assign;
mod kalman;

pub use assign::{iou, iou3d, min_cost_assign};
pub use kalman::{Box2DModel, Box3DModel};

use std::collections::HashMap;

/// A box geometry usable for data association. The overlap metric travels with
/// the geometry, so each dimension defines its own: 2D = axis-aligned IoU
/// (`[f32;4]`), 3D = height-aware BEV IoU ([`Box3D`]).
pub trait Boxlike: Copy {
    /// Association overlap in `[0,1]`; higher = better match.
    fn iou(&self, other: &Self) -> f32;
}

/// 2D axis-aligned box `[x1,y1,x2,y2]`.
impl Boxlike for [f32; 4] {
    fn iou(&self, other: &Self) -> f32 {
        assign::iou(self, other)
    }
}

/// An oriented 3D box: `center` (x,y,z; z up), `size` (l,w,h along heading), and
/// `yaw` (rotation about the vertical axis). The association space for 3D tracking.
#[derive(Debug, Clone, Copy)]
pub struct Box3D {
    pub center: [f32; 3],
    pub size: [f32; 3],
    pub yaw: f32,
}

impl Boxlike for Box3D {
    fn iou(&self, other: &Self) -> f32 {
        assign::iou3d(self, other)
    }
}

/// A detection handed to the tracker.
pub trait Observation {
    /// Geometry of this observation (`[f32;4]` for 2D, [`Box3D`] for 3D).
    type Box: Boxlike;
    fn bbox(&self) -> Self::Box;
    fn score(&self) -> f32;
    fn class_id(&self) -> u32;
}

/// The default 2D observation.
#[derive(Debug, Clone, Copy)]
pub struct Obs2D {
    pub bbox: [f32; 4],
    pub score: f32,
    pub class_id: u32,
}

impl Observation for Obs2D {
    type Box = [f32; 4];
    fn bbox(&self) -> [f32; 4] {
        self.bbox
    }
    fn score(&self) -> f32 {
        self.score
    }
    fn class_id(&self) -> u32 {
        self.class_id
    }
}

/// The default 3D observation.
#[derive(Debug, Clone, Copy)]
pub struct Obs3D {
    pub bbox: Box3D,
    pub score: f32,
    pub class_id: u32,
}

impl Observation for Obs3D {
    type Box = Box3D;
    fn bbox(&self) -> Box3D {
        self.bbox
    }
    fn score(&self) -> f32 {
        self.score
    }
    fn class_id(&self) -> u32 {
        self.class_id
    }
}

/// A per-track motion model (Kalman state). The generic axis: 2D
/// ([`Box2DModel`]) and 3D ([`Box3DModel`]) implement the same trait, so the
/// [`Tracker`] loop is identical — only the box geometry differs.
pub trait TrackModel {
    type Box: Boxlike;
    type Obs: Observation<Box = Self::Box>;
    fn new(obs: &Self::Obs) -> Self;
    fn predict(&mut self, dt: f32);
    fn update(&mut self, obs: &Self::Obs);
    /// Current estimate as a box in the association/output space.
    fn bbox(&self) -> Self::Box;

    /// Association affinity in `[0,1]` between this track's *predicted* state and a
    /// detection — higher = better match. Default: geometric IoU of the predicted
    /// box and the detection (the 2D path). The 3D model overrides this with a
    /// metric **Mahalanobis-distance + size-consistency** score, which is far more
    /// robust than overlap for small, depth-noisy boxes that rarely IoU-overlap.
    fn affinity(&self, obs: &Self::Obs) -> f32 {
        self.bbox().iou(&obs.bbox())
    }

    /// Multiplier applied to `affinity` when the track and detection classes differ.
    /// `0.0` (default) = a **hard** class gate: cross-class pairs never associate (2D).
    /// The 3D model returns a small positive value so strong 3D evidence (near in
    /// metres *and* size-consistent) can bind a momentarily *mislabeled* detection
    /// back to its track instead of spawning a phantom — the sticky class vote then
    /// keeps the established label.
    fn class_mismatch_factor() -> f32 {
        0.0
    }
}

/// Tracker tuning (BoT-SORT / ByteTrack defaults).
#[derive(Debug, Clone, Copy)]
pub struct TrackerConfig {
    /// Detections at or above this score go to the first association pass.
    pub det_high: f32,
    /// Detections in `[det_low, det_high)` go to the second (recovery) pass;
    /// below `det_low` are dropped.
    pub det_low: f32,
    /// Min IoU to accept a first-pass (high-conf) match.
    pub iou_thresh_high: f32,
    /// Min IoU to accept a second-pass (low-conf recovery) match.
    pub iou_thresh_low: f32,
    /// Hits required before a tentative track is confirmed (emitted).
    pub min_hits: u32,
    /// **Motion-coast window**: frames a track keeps participating in motion/IoU
    /// association after a miss. Its predicted box is only trusted this long;
    /// beyond it the position is stale and the track is held appearance-only.
    pub max_age: u32,
    /// **Identity-retention window**: frames a lost track (with an appearance bank)
    /// is kept in the gallery for appearance-only re-acquisition before deletion.
    /// Decouples "remember who this was" from "trust where it is" — set well above
    /// `max_age`. With no appearance, a track is effectively gone after `max_age`.
    pub max_lost: u32,
    /// **Coast emission window**: frames a confirmed track keeps being *emitted* (at its predicted
    /// position) after a missed detection, so a brief gap — a short occlusion or a rejected depth
    /// outlier — doesn't blink the object out of the output. `0` = emit only on frames with a fresh
    /// detection (strict default; preserves 2D behavior). Should be ≤ `max_age`.
    pub coast_emit: u32,
    /// Weight of appearance (cosine) distance added to the IoU cost during
    /// association. `0.0` ⇒ motion-only even when embeddings are supplied.
    /// Appearance only re-ranks IoU-eligible candidates; it never widens the
    /// gate.
    pub appearance_weight: f32,
    /// Deprecated / unused: the appearance bank now keeps several recent views (see [`AppearBank`])
    /// and matches by best cosine, rather than EMA-blending into one vector. Retained for API compat.
    pub appearance_ema: f32,
    /// Re-acquire a *lost* track (coasting, unmatched this frame) by appearance
    /// alone when a high-conf detection's embedding matches its bank — recovers
    /// the original id across a full occlusion with no IoU overlap. Needs
    /// embeddings; `false` disables it.
    pub reacquire: bool,
    /// Min cosine similarity to re-acquire a lost track. Strict (no IoU support
    /// backs this match), so set high to avoid identity bleed.
    pub reacquire_thresh: f32,
    /// **Long-term identity gallery.** When a confirmed track is finally deleted,
    /// its appearance + id are archived here; a newly appearing object is matched
    /// against the gallery (by appearance, cosine ≥ `reacquire_thresh`) *before* a
    /// new id is minted, so someone who leaves the room and returns minutes later
    /// reclaims their original id. `gallery_ttl_secs` is how long (wall-clock, via
    /// `dt`) an identity is remembered; `0` disables the gallery.
    pub gallery_ttl_secs: f32,
    /// Max identities kept in the gallery (LRU by idle time). Caps memory/compute.
    pub gallery_max: usize,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self {
            det_high: 0.5,
            det_low: 0.1,
            iou_thresh_high: 0.2,
            iou_thresh_low: 0.5,
            min_hits: 2,
            max_age: 30,
            max_lost: 90,
            coast_emit: 0,
            appearance_weight: 0.5,
            appearance_ema: 0.9,
            reacquire: true,
            reacquire_thresh: 0.7,
            gallery_ttl_secs: 300.0,
            gallery_max: 64,
        }
    }
}

/// A confirmed track's per-frame output. `B` is the box geometry: `[f32;4]` for
/// 2D (the default), [`Box3D`] for 3D.
#[derive(Debug, Clone)]
pub struct TrackOut<B = [f32; 4]> {
    pub id: u32,
    pub bbox: B,
    pub class_id: u32,
    pub score: f32,
    /// Frames since this track was first created.
    pub age: u32,
    /// The track's L2-normalized EMA appearance bank, when one has been seen
    /// (i.e. `update` was fed embeddings). `None` for motion-only tracking.
    pub embedding: Option<Vec<f32>>,
}

struct Track<M> {
    id: u32,
    model: M,
    /// The track's reported class — the **majority vote** over every detection it
    /// has matched (`class_votes`), not the last frame's guess, so a single
    /// mislabeled detection can't flip a well-established identity.
    class_id: u32,
    /// Per-class hit counts; `class_id` is the arg-max. Cheap (≤ #classes entries).
    class_votes: HashMap<u32, u32>,
    score: f32,
    hits: u32,
    age: u32,
    time_since_update: u32,
    confirmed: bool,
    /// Recent appearance embeddings (a small gallery, not one averaged vector) so the track can be
    /// re-acquired by matching *any* remembered view, not an average of all of them.
    appear: AppearBank,
}

/// A small gallery of recent L2-normalized appearance embeddings for one identity. A single EMA
/// vector can't represent a person across pose/scale/lighting changes, so re-acquisition matches a
/// detection against the BEST (max) cosine over a few stored views (StrongSORT-style). This is what
/// makes re-ID survive someone leaving and returning in a different pose.
#[derive(Clone, Default)]
struct AppearBank {
    feats: Vec<Vec<f32>>,
}

impl AppearBank {
    /// Max remembered views per identity (ring buffer).
    const CAP: usize = 10;
    fn is_empty(&self) -> bool {
        self.feats.is_empty()
    }
    /// Add a view, evicting the oldest past `CAP`.
    fn push(&mut self, f: &[f32]) {
        if f.is_empty() {
            return;
        }
        self.feats.push(f.to_vec());
        if self.feats.len() > Self::CAP {
            self.feats.remove(0);
        }
    }
    /// Best cosine of `q` against any stored view — `q` need match only ONE remembered pose.
    fn max_cosine(&self, q: &[f32]) -> Option<f32> {
        self.feats
            .iter()
            .map(|f| cosine(f, q))
            .fold(None, |m, c| Some(m.map_or(c, |x: f32| x.max(c))))
    }
    /// Fold another bank's views in (used when archiving / refreshing a gallery identity).
    fn merge(&mut self, other: &AppearBank) {
        for f in &other.feats {
            self.push(f);
        }
    }
    /// A single L2-normalized representative (mean view) for downstream consumers (the world store).
    fn repr(&self) -> Option<Vec<f32>> {
        let first = self.feats.first()?;
        let mut m = vec![0.0f32; first.len()];
        for f in &self.feats {
            for (i, &v) in f.iter().enumerate() {
                m[i] += v;
            }
        }
        let n = m.iter().map(|x| x * x).sum::<f32>().sqrt();
        if n > 0.0 {
            for x in &mut m {
                *x /= n;
            }
        }
        Some(m)
    }
}

/// A departed identity remembered for long-term re-acquisition: an id + its appearance bank +
/// class, with no motion model (so it costs nothing to keep).
struct GalleryEntry {
    id: u32,
    class_id: u32,
    appear: AppearBank,
    idle_secs: f32,
}

/// Generic multi-object tracker. `Tracker<Box2DModel>` is the 2D tracker
/// ([`Box2DTracker`]).
pub struct Tracker<M: TrackModel> {
    tracks: Vec<Track<M>>,
    gallery: Vec<GalleryEntry>,
    cfg: TrackerConfig,
    next_id: u32,
    /// Diagnostics: best same-class gallery cosine at the most recent birth (`-1` = no candidate).
    last_gallery_best: f32,
}

/// The 2D box tracker.
pub type Box2DTracker = Tracker<Box2DModel>;

/// The 3D box tracker (same engine, [`Box3DModel`] state + BEV-IoU association).
pub type Box3DTracker = Tracker<Box3DModel>;

impl<M: TrackModel> Tracker<M> {
    pub fn new(cfg: TrackerConfig) -> Self {
        Self {
            tracks: Vec::new(),
            gallery: Vec::new(),
            cfg,
            next_id: 1,
            last_gallery_best: -1.0,
        }
    }

    /// Advance one frame: predict, two-stage associate, update, birth, age out.
    ///
    /// `dt` is the time since the previous `update` in seconds. `embeds`, when
    /// given, holds one appearance embedding per detection (same order/length as
    /// `dets`); an empty inner `Vec` means "no embedding for this detection".
    /// Pass `None` for motion-only tracking.
    pub fn update(
        &mut self,
        dets: &[M::Obs],
        dt: f32,
        embeds: Option<&[Vec<f32>]>,
    ) -> Vec<TrackOut<M::Box>> {
        // 1. Predict every track forward; age the long-term gallery by wall-clock
        //    and forget identities idle past their TTL.
        for t in &mut self.tracks {
            t.model.predict(dt);
            t.age += 1;
            t.time_since_update += 1;
        }
        if self.cfg.gallery_ttl_secs > 0.0 {
            let ttl = self.cfg.gallery_ttl_secs;
            for g in &mut self.gallery {
                g.idle_secs += dt;
            }
            self.gallery.retain(|g| g.idle_secs <= ttl);
        }

        // 2. Split detections by confidence.
        let mut high = Vec::new();
        let mut low = Vec::new();
        for (i, d) in dets.iter().enumerate() {
            let s = d.score();
            if s >= self.cfg.det_high {
                high.push(i);
            } else if s >= self.cfg.det_low {
                low.push(i);
            }
        }

        let mut det_used = vec![false; dets.len()];
        let mut matched = vec![false; self.tracks.len()];

        // Motion/IoU is only trusted within the coast window — beyond `max_age` a
        // track's predicted box is stale, so it's excluded from the IoU passes and
        // can only return via appearance (4b).
        let coast_age = self.cfg.max_age;
        let coast: Vec<usize> = (0..self.tracks.len())
            .filter(|&t| self.tracks[t].time_since_update <= coast_age)
            .collect();

        // 3. Pass 1: coasting tracks ↔ high-conf detections.
        let m1 = self.associate(&coast, &high, dets, embeds, self.cfg.iou_thresh_high);
        for &(ti, dj) in &m1 {
            self.apply_match(ti, dj, dets, embeds);
            det_used[dj] = true;
            matched[ti] = true;
        }

        // 4. Pass 2: remaining coasting tracks ↔ low-conf detections (occlusion recovery).
        let coast2: Vec<usize> = coast.iter().copied().filter(|&t| !matched[t]).collect();
        let low_left: Vec<usize> = low.into_iter().filter(|&d| !det_used[d]).collect();
        let m2 = self.associate(&coast2, &low_left, dets, embeds, self.cfg.iou_thresh_low);
        for &(ti, dj) in &m2 {
            self.apply_match(ti, dj, dets, embeds);
            det_used[dj] = true;
            matched[ti] = true;
        }

        // 4b. Re-acquire by appearance alone (no IoU), over every unmatched
        // confirmed track that still carries an appearance bank — both coasting and
        // lost ones. This recovers a track that reappears DISPLACED (so the IoU
        // passes missed it), whether the displacement came from a short occlusion
        // or a longer absence (still active, up to `max_lost`). The high cosine
        // gate + class gate + best-match assignment keep it from hijacking an id
        // between two visually-similar same-class objects. (Revival after FULL
        // deletion is separate — from the gallery, in step 5.)
        if self.cfg.reacquire {
            let lost: Vec<usize> = (0..self.tracks.len())
                .filter(|&t| {
                    !matched[t] && self.tracks[t].confirmed && !self.tracks[t].appear.is_empty()
                })
                .collect();
            let high_left: Vec<usize> = high.iter().copied().filter(|&d| !det_used[d]).collect();
            let m3 = self.associate_appearance(
                &lost,
                &high_left,
                dets,
                embeds,
                self.cfg.reacquire_thresh,
            );
            for &(ti, dj) in &m3 {
                self.apply_match(ti, dj, dets, embeds);
                det_used[dj] = true;
                matched[ti] = true;
            }
        }

        // 5. Birth tracks from unmatched high-conf detections — but first try to
        //    REVIVE a long-departed identity from the gallery by appearance, so a
        //    person who left and returned keeps their original id instead of a new
        //    one. Only when no gallery match is found is a fresh id minted.
        for &dj in &high {
            if det_used[dj] {
                continue;
            }
            let (class_id, score) = (dets[dj].class_id(), dets[dj].score());
            let da = embed_of(embeds, dj);
            // Returning identity? Match the current embedding against the gallery.
            let revived = da.and_then(|da| self.revive_from_gallery(class_id, da));
            let (id, confirmed) = match revived {
                Some(gid) => (gid, true), // known person back in view → emit immediately
                None => {
                    let id = self.next_id;
                    self.next_id += 1;
                    (id, self.cfg.min_hits <= 1)
                }
            };
            let mut appear = AppearBank::default();
            if let Some(da) = da {
                appear.push(da);
            }
            self.tracks.push(Track {
                id,
                model: M::new(&dets[dj]),
                class_id,
                class_votes: HashMap::from([(class_id, 1u32)]),
                score,
                hits: 1,
                age: 1,
                time_since_update: 0,
                confirmed,
                appear,
            });
        }

        // 6. Delete tracks unseen for too long. A confirmed track with an
        // appearance bank lingers (active) until `max_lost`, then on deletion is
        // ARCHIVED to the long-term gallery so its identity outlives the motion
        // track; anything else is dropped after the `max_age` coast window.
        let (max_age, max_lost) = (self.cfg.max_age, self.cfg.max_lost.max(self.cfg.max_age));
        let gallery_on = self.cfg.gallery_ttl_secs > 0.0;
        let mut archive: Vec<(u32, u32, AppearBank)> = Vec::new();
        self.tracks.retain(|t| {
            let limit = if t.confirmed && !t.appear.is_empty() {
                max_lost
            } else {
                max_age
            };
            let keep = t.time_since_update <= limit;
            if !keep && gallery_on && t.confirmed && !t.appear.is_empty() {
                archive.push((t.id, t.class_id, t.appear.clone()));
            }
            keep
        });
        for (id, class_id, appear) in archive {
            self.archive_to_gallery(id, class_id, appear);
        }

        // 7. Emit confirmed tracks updated this frame, plus confirmed tracks coasting within the
        // `coast_emit` window — emitted at their predicted position so a brief miss (occlusion or a
        // rejected outlier) holds the object steady instead of blinking it out.
        self.tracks
            .iter()
            .filter(|t| t.confirmed && t.time_since_update <= self.cfg.coast_emit)
            .map(|t| TrackOut {
                id: t.id,
                bbox: t.model.bbox(),
                class_id: t.class_id,
                score: t.score,
                age: t.age,
                embedding: t.appear.repr(),
            })
            .collect()
    }

    /// Optimal assignment between a subset of tracks and detections.
    ///
    /// Cost is `(1−IoU)` plus an optional appearance term
    /// `appearance_weight·(1−cosine)` (only when both sides have an embedding);
    /// cross-class pairs are made unselectable. A pair is accepted only if it
    /// clears the per-class IoU `iou_gate` — appearance re-ranks candidates but
    /// never relaxes that gate. Returns matched `(global_track, global_det)`.
    fn associate(
        &self,
        track_sel: &[usize],
        det_sel: &[usize],
        dets: &[M::Obs],
        embeds: Option<&[Vec<f32>]>,
        iou_gate: f32,
    ) -> Vec<(usize, usize)> {
        if track_sel.is_empty() || det_sel.is_empty() {
            return Vec::new();
        }
        let w = self.cfg.appearance_weight;
        let cmf = M::class_mismatch_factor();
        // Affinity of a (track, det) pair: the model's geometric/metric score scaled
        // by the class factor (1 same-class; `cmf` cross-class — 0 = hard gate for 2D).
        let aff = |t: usize, d: usize| -> f32 {
            let cf = if dets[d].class_id() == self.tracks[t].class_id {
                1.0
            } else {
                cmf
            };
            if cf <= 0.0 {
                return 0.0;
            }
            self.tracks[t].model.affinity(&dets[d]) * cf
        };
        let cost: Vec<Vec<f32>> = track_sel
            .iter()
            .map(|&t| {
                let tc = self.tracks[t].class_id;
                det_sel
                    .iter()
                    .map(|&d| {
                        if dets[d].class_id() != tc && cmf <= 0.0 {
                            return 1.0e6;
                        } // hard gate (2D)
                        let mut c = 1.0 - aff(t, d);
                        if w > 0.0 {
                            if let Some(da) = embed_of(embeds, d) {
                                if let Some(cos) = self.tracks[t].appear.max_cosine(da) {
                                    c += w * (1.0 - cos);
                                }
                            }
                        }
                        c
                    })
                    .collect()
            })
            .collect();

        // Accept a pair only if its affinity clears the gate (metres+size for 3D,
        // IoU for 2D); appearance re-ranks candidates but never relaxes the gate.
        min_cost_assign(&cost)
            .into_iter()
            .filter_map(|(a, b)| {
                let (ti, dj) = (track_sel[a], det_sel[b]);
                (aff(ti, dj) >= iou_gate).then_some((ti, dj))
            })
            .collect()
    }

    /// Appearance-only assignment (no IoU) for re-acquiring lost tracks. Cost is
    /// `1−cosine`, class-gated; a pair is accepted only if both sides have an
    /// embedding and cosine ≥ `cos_gate`. Returns matched `(global_track, global_det)`.
    fn associate_appearance(
        &self,
        track_sel: &[usize],
        det_sel: &[usize],
        dets: &[M::Obs],
        embeds: Option<&[Vec<f32>]>,
        cos_gate: f32,
    ) -> Vec<(usize, usize)> {
        if track_sel.is_empty() || det_sel.is_empty() {
            return Vec::new();
        }
        let cos = |t: usize, d: usize| -> Option<f32> {
            let da = embed_of(embeds, d)?;
            if dets[d].class_id() != self.tracks[t].class_id {
                return None;
            }
            self.tracks[t].appear.max_cosine(da)
        };
        let cost: Vec<Vec<f32>> = track_sel
            .iter()
            .map(|&t| {
                det_sel
                    .iter()
                    .map(|&d| cos(t, d).map_or(1.0e6, |c| 1.0 - c))
                    .collect()
            })
            .collect();

        min_cost_assign(&cost)
            .into_iter()
            .filter_map(|(a, b)| {
                let (ti, dj) = (track_sel[a], det_sel[b]);
                (cos(ti, dj).is_some_and(|c| c >= cos_gate)).then_some((ti, dj))
            })
            .collect()
    }

    /// Try to reclaim a departed identity from the long-term gallery by appearance
    /// (class-gated, cosine ≥ `reacquire_thresh`). On a match the gallery entry is
    /// consumed and its id returned, so a returning person keeps their old id.
    fn revive_from_gallery(&mut self, class_id: u32, da: &[f32]) -> Option<u32> {
        if self.cfg.gallery_ttl_secs <= 0.0 {
            return None;
        }
        // Best same-class cosine BEFORE thresholding — recorded for diagnostics so we can see whether
        // a re-entry failed because the match was weak (low-res crop) vs. the gallery had expired.
        let best = self
            .gallery
            .iter()
            .enumerate()
            .filter(|(_, g)| g.class_id == class_id)
            .filter_map(|(i, g)| g.appear.max_cosine(da).map(|c| (i, c)))
            .max_by(|a, b| a.1.total_cmp(&b.1));
        self.last_gallery_best = best.map_or(-1.0, |(_, c)| c);
        match best {
            Some((i, c)) if c >= self.cfg.reacquire_thresh => Some(self.gallery.swap_remove(i).id),
            _ => None,
        }
    }

    /// Archive (or refresh) a departed identity in the gallery, evicting the
    /// most-idle entry if over `gallery_max` (LRU).
    fn archive_to_gallery(&mut self, id: u32, class_id: u32, appear: AppearBank) {
        if let Some(g) = self.gallery.iter_mut().find(|g| g.id == id) {
            g.appear.merge(&appear);
            g.class_id = class_id;
            g.idle_secs = 0.0;
            return;
        }
        self.gallery.push(GalleryEntry {
            id,
            class_id,
            appear,
            idle_secs: 0.0,
        });
        if self.gallery.len() > self.cfg.gallery_max {
            if let Some(i) = self
                .gallery
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.idle_secs.total_cmp(&b.1.idle_secs))
                .map(|(i, _)| i)
            {
                self.gallery.swap_remove(i);
            }
        }
    }

    fn apply_match(&mut self, ti: usize, dj: usize, dets: &[M::Obs], embeds: Option<&[Vec<f32>]>) {
        let d = &dets[dj];
        let new_app = embed_of(embeds, dj);
        let t = &mut self.tracks[ti];
        t.model.update(d);
        t.score = d.score();
        // Sticky class: tally this detection's class and report the majority, so a
        // single mislabeled frame (e.g. a person momentarily called "phone") can't
        // relabel an established track. With the soft 3D class gate, that mislabeled
        // detection is *absorbed* here (its box is person-sized → high size affinity)
        // rather than spawning a phantom — and outvoted, so the label holds.
        *t.class_votes.entry(d.class_id()).or_insert(0) += 1;
        let cur = t.class_votes.get(&t.class_id).copied().unwrap_or(0);
        if let Some((&c, &n)) = t.class_votes.iter().max_by_key(|(_, n)| **n) {
            if n > cur {
                t.class_id = c;
            } // switch only on a strict majority over the current label
        }
        t.hits += 1;
        t.time_since_update = 0;
        if t.hits >= self.cfg.min_hits {
            t.confirmed = true;
        }
        if let Some(da) = new_app {
            t.appear.push(da); // remember this view (ring buffer) — robust re-ID by best match
        }
    }

    /// Number of live tracks (including tentative/lost).
    pub fn len(&self) -> usize {
        self.tracks.len()
    }
    pub fn is_empty(&self) -> bool {
        self.tracks.is_empty()
    }
    /// Diagnostics: archived identities currently held in the long-term re-ID gallery.
    pub fn gallery_len(&self) -> usize {
        self.gallery.len()
    }
    /// Diagnostics: best same-class gallery cosine seen at the most recent birth (`-1` = no
    /// candidate). Below `reacquire_thresh` ⇒ a fresh id was minted instead of reviving.
    pub fn last_gallery_best(&self) -> f32 {
        self.last_gallery_best
    }
}

/// The embedding for detection `i`, or `None` if absent / empty.
fn embed_of(embeds: Option<&[Vec<f32>]>, i: usize) -> Option<&[f32]> {
    embeds
        .and_then(|e| e.get(i))
        .map(Vec::as_slice)
        .filter(|e| !e.is_empty())
}

/// Cosine similarity of two equal-length vectors (a dot product when both are
/// L2-normalized, as ReID embeddings are).
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "embedding dim mismatch");
    if a.len() != b.len() {
        return 0.0;
    }
    let s: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    if s.is_finite() {
        s
    } else {
        0.0
    } // guard NaN/inf from a bad embedding
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obs(b: [f32; 4], s: f32, c: u32) -> Obs2D {
        Obs2D {
            bbox: b,
            score: s,
            class_id: c,
        }
    }

    #[test]
    fn stable_id_across_frames() {
        let mut tr = Box2DTracker::new(TrackerConfig {
            min_hits: 1,
            ..Default::default()
        });
        // A box moving right by 8px/frame.
        let mut id = None;
        for k in 0..5 {
            let x = k as f32 * 8.0;
            let out = tr.update(&[obs([x, 0.0, x + 20.0, 20.0], 0.9, 0)], 1.0, None);
            assert_eq!(out.len(), 1, "frame {k}");
            match id {
                None => id = Some(out[0].id),
                Some(prev) => assert_eq!(out[0].id, prev, "id switched at frame {k}"),
            }
        }
    }

    #[test]
    fn two_objects_keep_distinct_ids() {
        let mut tr = Box2DTracker::new(TrackerConfig {
            min_hits: 1,
            ..Default::default()
        });
        let mut ids = std::collections::HashSet::new();
        for k in 0..4 {
            let x = k as f32 * 5.0;
            let out = tr.update(
                &[
                    obs([x, 0.0, x + 20.0, 20.0], 0.9, 0),
                    obs([x + 200.0, 0.0, x + 220.0, 20.0], 0.9, 0),
                ],
                1.0,
                None,
            );
            assert_eq!(out.len(), 2, "frame {k}");
            for o in &out {
                ids.insert(o.id);
            }
        }
        assert_eq!(
            ids.len(),
            2,
            "expected exactly two distinct ids, got {ids:?}"
        );
    }

    #[test]
    fn different_classes_do_not_match() {
        let mut tr = Box2DTracker::new(TrackerConfig {
            min_hits: 1,
            ..Default::default()
        });
        tr.update(&[obs([0.0, 0.0, 20.0, 20.0], 0.9, 0)], 1.0, None);
        // Same place, different class → must not reuse the track; a new id appears.
        let out = tr.update(&[obs([0.0, 0.0, 20.0, 20.0], 0.9, 7)], 1.0, None);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].class_id, 7);
    }

    #[test]
    fn appearance_prevents_id_switch_when_boxes_overlap() {
        // Two same-class objects swap positions so each ends up nearer the
        // OTHER's previous box — motion (IoU) alone would swap their IDs.
        // Distinct appearance embeddings should keep the IDs pinned to identity.
        let red = vec![1.0, 0.0];
        let blue = vec![0.0, 1.0];
        let cfg = TrackerConfig {
            min_hits: 1,
            appearance_weight: 5.0,
            ..Default::default()
        };
        let mut tr = Box2DTracker::new(cfg);

        // Frame 0: A(red) left, B(blue) right.
        let out0 = tr.update(
            &[
                obs([0.0, 0.0, 30.0, 30.0], 0.9, 0),
                obs([40.0, 0.0, 70.0, 30.0], 0.9, 0),
            ],
            1.0,
            Some(&[red.clone(), blue.clone()]),
        );
        let (mut a_id, mut b_id) = (0, 0);
        for o in &out0 {
            if o.bbox[0] < 20.0 {
                a_id = o.id;
            } else {
                b_id = o.id;
            }
        }
        assert_ne!(a_id, b_id);

        // Frame 1: positions converge (heavy overlap), but appearance is swapped
        // in detection ORDER — det[0] is now blue on the right, det[1] red left.
        let out1 = tr.update(
            &[
                obs([38.0, 0.0, 68.0, 30.0], 0.9, 0),
                obs([2.0, 0.0, 32.0, 30.0], 0.9, 0),
            ],
            1.0,
            Some(&[blue.clone(), red.clone()]),
        );
        // The red detection (left, det[1]) must still carry A's id.
        let red_track = out1.iter().find(|o| o.bbox[0] < 20.0).expect("left track");
        assert_eq!(
            red_track.id, a_id,
            "appearance should keep A's id on the red object"
        );
    }

    #[test]
    fn reacquire_lost_track_by_appearance() {
        let red = vec![1.0, 0.0];
        let mut tr = Box2DTracker::new(TrackerConfig {
            min_hits: 1,
            ..Default::default()
        });
        let id_a = tr.update(
            &[obs([0.0, 0.0, 30.0, 30.0], 0.9, 0)],
            1.0,
            Some(std::slice::from_ref(&red)),
        )[0]
        .id;
        // Fully occluded for a few frames (no detections at all).
        for _ in 0..3 {
            tr.update(&[], 1.0, None);
        }
        // Reappears far away (zero IoU overlap) with the same appearance.
        let out = tr.update(&[obs([300.0, 0.0, 330.0, 30.0], 0.9, 0)], 1.0, Some(&[red]));
        assert_eq!(out.len(), 1);
        assert_eq!(
            out[0].id, id_a,
            "appearance should reclaim A's id across the gap"
        );
    }

    #[test]
    fn reacquire_after_long_gap_beyond_coast_window() {
        // Identity is gated by appearance, not the motion timeout: a track gone
        // longer than `max_age` (coast) but within `max_lost` (gallery) reclaims id.
        let red = vec![1.0, 0.0];
        let mut tr = Box2DTracker::new(TrackerConfig {
            min_hits: 1,
            max_age: 5,
            max_lost: 60,
            ..Default::default()
        });
        let id_a = tr.update(
            &[obs([0.0, 0.0, 30.0, 30.0], 0.9, 0)],
            1.0,
            Some(std::slice::from_ref(&red)),
        )[0]
        .id;
        for _ in 0..40 {
            tr.update(&[], 1.0, None);
        } // gone 40 frames >> max_age=5
        let out = tr.update(&[obs([300.0, 0.0, 330.0, 30.0], 0.9, 0)], 1.0, Some(&[red]));
        assert_eq!(
            out[0].id, id_a,
            "appearance gallery should reclaim id past the coast window"
        );
    }

    #[test]
    fn gallery_revives_id_after_leaving_and_returning() {
        // Home case: person leaves the room (track deleted past max_lost) and
        // returns much later; the long-term gallery reclaims their id by appearance.
        let red = vec![1.0, 0.0];
        let mut tr = Box2DTracker::new(TrackerConfig {
            min_hits: 1,
            max_age: 3,
            max_lost: 5,
            gallery_ttl_secs: 60.0,
            ..Default::default()
        });
        let id_a = tr.update(
            &[obs([0.0, 0.0, 30.0, 30.0], 0.9, 0)],
            0.1,
            Some(std::slice::from_ref(&red)),
        )[0]
        .id;
        // Gone long enough to be deleted AND archived to the gallery (>max_lost).
        for _ in 0..20 {
            tr.update(&[], 0.1, None);
        }
        assert!(
            tr.tracks.is_empty(),
            "track should be deleted (archived to gallery)"
        );
        // Returns ~10 s later, different position — appearance reclaims the id.
        let out = tr.update(
            &[obs([400.0, 200.0, 430.0, 230.0], 0.9, 0)],
            0.1,
            Some(&[red]),
        );
        assert_eq!(
            out[0].id, id_a,
            "gallery should restore the original id on return"
        );
    }

    #[test]
    fn gallery_forgets_after_ttl() {
        let red = vec![1.0, 0.0];
        let mut tr = Box2DTracker::new(TrackerConfig {
            min_hits: 1,
            max_age: 3,
            max_lost: 5,
            gallery_ttl_secs: 2.0,
            ..Default::default()
        });
        let id_a = tr.update(
            &[obs([0.0, 0.0, 30.0, 30.0], 0.9, 0)],
            0.1,
            Some(std::slice::from_ref(&red)),
        )[0]
        .id;
        for _ in 0..40 {
            tr.update(&[], 0.1, None);
        } // 4 s idle > 2 s TTL
        let out = tr.update(
            &[obs([400.0, 200.0, 430.0, 230.0], 0.9, 0)],
            0.1,
            Some(&[red]),
        );
        assert_ne!(
            out[0].id, id_a,
            "past the gallery TTL the identity is forgotten"
        );
    }

    #[test]
    fn track_expires_after_max_lost() {
        // With the long-term gallery OFF, beyond `max_lost` the id is dropped → fresh id.
        let red = vec![1.0, 0.0];
        let mut tr = Box2DTracker::new(TrackerConfig {
            min_hits: 1,
            max_age: 5,
            max_lost: 20,
            gallery_ttl_secs: 0.0,
            ..Default::default()
        });
        let id_a = tr.update(
            &[obs([0.0, 0.0, 30.0, 30.0], 0.9, 0)],
            1.0,
            Some(std::slice::from_ref(&red)),
        )[0]
        .id;
        for _ in 0..25 {
            tr.update(&[], 1.0, None);
        } // gone 25 frames > max_lost=20
        let out = tr.update(&[obs([300.0, 0.0, 330.0, 30.0], 0.9, 0)], 1.0, Some(&[red]));
        assert_ne!(out[0].id, id_a, "past max_lost the id is forgotten");
    }

    #[test]
    fn box3d_two_people_crossing_keep_distinct_ids() {
        // Two people with distinct appearances walk through each other (swap sides, passing within
        // ~0.3 m at the same depth). The tracker must keep exactly TWO ids and keep each id stuck to
        // its person across the crossing — no swap, no proliferation. This is the core guarantee for
        // solid multi-person tracking: 3D separation + appearance resolve the ambiguous crossing.
        let mut tr = Box3DTracker::new(TrackerConfig {
            min_hits: 1,
            iou_thresh_high: 0.05,
            iou_thresh_low: 0.15,
            ..Default::default()
        });
        let red = vec![1.0f32, 0.0];
        let blue = vec![0.0f32, 1.0];
        let person = [0.5, 0.5, 1.7];
        let (mut red_first, mut red_last, mut all) = (None, None, std::collections::HashSet::new());
        for k in 0..7 {
            let xa = -0.9 + k as f32 * 0.3; // red: left → right
            let xb = 0.9 - k as f32 * 0.3; // blue: right → left (they cross around k=3)
            let dets = [
                o3s([xa, 0.0, 3.0], person, 1),
                o3s([xb, 0.0, 3.0], person, 1),
            ];
            let out = tr.update(&dets, 0.1, Some(&[red.clone(), blue.clone()]));
            for o in &out {
                all.insert(o.id);
            }
            // The output whose appearance is reddest = the red person this frame.
            let rid = out
                .iter()
                .max_by(|a, b| {
                    let ca = a.embedding.as_deref().map_or(-1.0, |e| cosine(e, &red));
                    let cb = b.embedding.as_deref().map_or(-1.0, |e| cosine(e, &red));
                    ca.total_cmp(&cb)
                })
                .map(|o| o.id);
            if red_first.is_none() {
                red_first = rid;
            }
            red_last = rid;
        }
        assert_eq!(
            all.len(),
            2,
            "exactly two people must be tracked, got ids {all:?}"
        );
        assert_eq!(
            red_first, red_last,
            "the red person's id must survive the crossing (no swap)"
        );
    }

    #[test]
    fn reid_revives_across_pose_change_via_feature_bank() {
        // A person seen in TWO distinct appearances (poses A and B), then leaves long enough to be
        // archived, then returns looking like B. The multi-feature gallery matches B directly — a
        // single averaged vector (≈midway between A and B) would fall below threshold and mint a new
        // id. Proves the bank, not just a lower threshold, is what fixes exit→return re-ID.
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0]; // orthogonal to A: their mean matches neither well (cos≈0.71)
        let cfg = TrackerConfig {
            min_hits: 1,
            max_age: 3,
            max_lost: 3,
            reacquire_thresh: 0.85,
            gallery_ttl_secs: 100.0,
            ..Default::default()
        };
        let mut tr = Box2DTracker::new(cfg);
        let box_a = [0.0, 0.0, 30.0, 60.0];
        let id0 = tr.update(&[obs(box_a, 0.9, 0)], 1.0, Some(std::slice::from_ref(&a)))[0].id;
        // accumulate a second, very different view of the SAME track
        for _ in 0..3 {
            tr.update(&[obs(box_a, 0.9, 0)], 1.0, Some(std::slice::from_ref(&b)));
        }
        // leave: gone past max_lost → archived to the gallery
        for _ in 0..5 {
            tr.update(&[], 1.0, None);
        }
        // return looking like B (far across the frame so only appearance can revive it)
        let out = tr.update(&[obs([300.0, 0.0, 330.0, 60.0], 0.9, 0)], 1.0, Some(&[b]))[0].id;
        assert_eq!(
            out, id0,
            "feature-bank re-ID must reclaim the original id from a remembered pose"
        );
    }

    #[test]
    fn no_reacquire_when_appearance_differs() {
        let red = vec![1.0, 0.0];
        let blue = vec![0.0, 1.0];
        let mut tr = Box2DTracker::new(TrackerConfig {
            min_hits: 1,
            ..Default::default()
        });
        let id_a = tr.update(&[obs([0.0, 0.0, 30.0, 30.0], 0.9, 0)], 1.0, Some(&[red]))[0].id;
        for _ in 0..3 {
            tr.update(&[], 1.0, None);
        }
        // A different-looking object far away must NOT steal A's id.
        let out = tr.update(
            &[obs([300.0, 0.0, 330.0, 30.0], 0.9, 0)],
            1.0,
            Some(&[blue]),
        );
        assert_eq!(out.len(), 1);
        assert_ne!(out[0].id, id_a, "dissimilar appearance must get a fresh id");
    }

    // ── 3D tracker (Box3DModel + BEV-IoU), validated on synthetic 3D tracks ──

    fn o3(center: [f32; 3], c: u32) -> Obs3D {
        Obs3D {
            bbox: Box3D {
                center,
                size: [4.0, 2.0, 1.5],
                yaw: 0.0,
            },
            score: 0.9,
            class_id: c,
        }
    }
    fn o3s(center: [f32; 3], size: [f32; 3], c: u32) -> Obs3D {
        Obs3D {
            bbox: Box3D {
                center,
                size,
                yaw: 0.0,
            },
            score: 0.9,
            class_id: c,
        }
    }

    #[test]
    fn box3d_stable_id_across_frames() {
        let mut tr = Box3DTracker::new(TrackerConfig {
            min_hits: 1,
            ..Default::default()
        });
        let mut id = None;
        for k in 0..5 {
            let x = k as f32 * 0.5; // 0.5 m/frame in x
            let out = tr.update(&[o3([x, 0.0, 0.0], 0)], 1.0, None);
            assert_eq!(out.len(), 1, "frame {k}");
            match id {
                None => id = Some(out[0].id),
                Some(prev) => assert_eq!(out[0].id, prev, "id switched at frame {k}"),
            }
        }
    }

    #[test]
    fn box3d_two_objects_keep_distinct_ids() {
        let mut tr = Box3DTracker::new(TrackerConfig {
            min_hits: 1,
            ..Default::default()
        });
        let mut ids = std::collections::HashSet::new();
        for k in 0..4 {
            let x = k as f32 * 0.3;
            let out = tr.update(
                &[o3([x, 0.0, 0.0], 0), o3([x + 20.0, 0.0, 0.0], 0)],
                1.0,
                None,
            );
            assert_eq!(out.len(), 2, "frame {k}");
            for o in &out {
                ids.insert(o.id);
            }
        }
        assert_eq!(ids.len(), 2, "expected two distinct ids, got {ids:?}");
    }

    #[test]
    fn box3d_absorbs_mislabeled_detection() {
        // A track seen as class 1 (person) for several frames, then ONE frame the
        // detector mislabels it class 7 at the same place with the same person-sized
        // box. The soft 3D class gate absorbs it (same id, no phantom track) and the
        // sticky vote keeps the established label — this is the misclassification fix.
        let mut tr = Box3DTracker::new(TrackerConfig {
            min_hits: 1,
            ..Default::default()
        });
        let mut id = None;
        for _ in 0..5 {
            id = Some(tr.update(&[o3([0.0, 0.0, 0.0], 1)], 1.0, None)[0].id);
        }
        let out = tr.update(&[o3([0.0, 0.0, 0.0], 7)], 1.0, None);
        assert_eq!(
            out.len(),
            1,
            "a mislabeled frame must not spawn a second track"
        );
        assert_eq!(
            out[0].id,
            id.unwrap(),
            "id must be retained across the mislabel"
        );
        assert_eq!(
            out[0].class_id, 1,
            "sticky vote keeps the established class"
        );
    }

    #[test]
    fn box3d_size_separates_colocated_objects() {
        // A person and a real phone at nearly the same x,y but very different metric
        // size: 3D size consistency keeps them as two distinct tracks (no merge),
        // even though the soft class gate would otherwise let close boxes bind.
        let mut tr = Box3DTracker::new(TrackerConfig {
            min_hits: 1,
            ..Default::default()
        });
        let mut ids = std::collections::HashSet::new();
        for _ in 0..4 {
            let out = tr.update(
                &[
                    o3s([0.0, 0.0, 0.0], [0.6, 0.6, 1.7], 1),     // person
                    o3s([0.1, 0.2, 0.0], [0.10, 0.10, 0.16], 77), // phone in hand
                ],
                1.0,
                None,
            );
            for o in &out {
                ids.insert(o.id);
            }
        }
        assert_eq!(
            ids.len(),
            2,
            "distinct sizes must stay distinct, got {ids:?}"
        );
    }

    #[test]
    fn box3d_coasts_through_brief_miss() {
        // A confirmed track that misses several frames keeps being emitted at its predicted position
        // for `coast_emit` frames (a brief gap must not blink the object out), then stops once the
        // window passes. `coast_emit = 0` (the default) preserves the strict emit-on-detection rule.
        let mut tr = Box3DTracker::new(TrackerConfig {
            min_hits: 1,
            coast_emit: 3,
            ..Default::default()
        });
        let id = tr.update(&[o3s([0.0, 0.0, 3.0], [0.6, 0.6, 1.7], 1)], 0.1, None)[0].id;
        for k in 0..3 {
            let out = tr.update(&[], 0.1, None);
            assert_eq!(
                out.len(),
                1,
                "miss {k}: a coasting track must still be emitted"
            );
            assert_eq!(out[0].id, id, "miss {k}: same id while coasting");
        }
        let out = tr.update(&[], 0.1, None); // 4th miss → past the window
        assert!(
            out.is_empty(),
            "past the coast_emit window the track must stop being emitted"
        );
    }

    #[test]
    fn box3d_far_jitter_is_smoothed_not_lost() {
        // A static object at 5 m with realistic per-frame depth jitter (stereo noise grows with
        // range). The range-tuned filter must (a) keep ONE id every frame — the range-scaled gate
        // associates the jitter instead of rejecting it (which is what dropped the track), and
        // (b) SMOOTH it — the estimate spread must be well under the ±0.4 m input spread.
        let mut tr = Box3DTracker::new(TrackerConfig {
            min_hits: 2,
            max_age: 10,
            iou_thresh_high: 0.05,
            iou_thresh_low: 0.15,
            ..Default::default()
        });
        let mut id = None;
        let mut max_err = 0.0f32;
        for k in 0..20 {
            // deterministic pseudo-jitter in [-0.4, 0.4] m on the measured depth (no rng dependency)
            let j = 0.4 * (k as f32 * 2.3999632).sin(); // deterministic depth jitter, bounded ±0.4 m
            let out = tr.update(&[o3s([0.0, 0.0, 5.0 + j], [0.6, 0.6, 1.7], 1)], 0.1, None);
            if k >= 2 {
                assert_eq!(
                    out.len(),
                    1,
                    "frame {k}: object must stay tracked through jitter (not lost)"
                );
                match id {
                    None => id = Some(out[0].id),
                    Some(p) => assert_eq!(out[0].id, p, "id switched at frame {k}"),
                }
                max_err = max_err.max((out[0].bbox.center[2] - 5.0).abs());
            }
        }
        assert!(
            max_err < 0.3,
            "filter did not smooth far jitter: max err {max_err} m (input ±0.4)"
        );
    }

    #[test]
    fn box3d_oracle_with_noise() {
        // Ground-truth object on a straight path; detections jitter deterministically
        // around it each frame. The track must hold one id and stay near GT.
        let mut tr = Box3DTracker::new(TrackerConfig {
            min_hits: 1,
            ..Default::default()
        });
        let mut id = None;
        for k in 0..10 {
            let gt = k as f32 * 0.5;
            // deterministic pseudo-jitter in [-0.1, 0.1] (no rng dependency)
            let j = ((k as f32 * 12.9898).sin() * 43758.547).fract() * 0.2 - 0.1;
            let out = tr.update(&[o3([gt + j, 0.05 * j, 0.0], 0)], 1.0, None);
            assert_eq!(out.len(), 1, "frame {k}");
            match id {
                None => id = Some(out[0].id),
                Some(prev) => assert_eq!(out[0].id, prev, "id switched at frame {k}"),
            }
            if k >= 2 {
                let est = out[0].bbox.center[0];
                assert!((est - gt).abs() < 0.5, "frame {k}: est {est} vs gt {gt}");
            }
        }
    }
}
