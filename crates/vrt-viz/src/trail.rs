//! Per-track metric-position history for drawing motion trails in the BEV.

use std::collections::HashMap;

use vrt_track::{CameraIntrinsics, Track, TrackState};

const MAX_LEN: usize = 48; // points kept per trail
const TTL: u32 = 30; // frames a trail survives after a track is last seen

struct Trail {
    pts: Vec<[f32; 2]>, // metric (X, Z)
    last: u32,
}

/// Accumulates a short `(X, Z)` history per confirmed track id. Call [`update`] once
/// per frame; pass to [`crate::render_bev`] to draw the trails.
///
/// [`update`]: TrailStore::update
#[derive(Default)]
pub struct TrailStore {
    trails: HashMap<u64, Trail>,
    frame: u32,
}

impl TrailStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append each confirmed track's current metric `(X, Z)` and prune stale trails.
    pub fn update(&mut self, tracks: &[Track], intr: &CameraIntrinsics) {
        self.frame = self.frame.wrapping_add(1);
        let f = self.frame;
        for t in tracks.iter().filter(|t| t.state == TrackState::Confirmed) {
            let [x, _, z] = t.metric_position(intr);
            let e = self.trails.entry(t.id).or_insert_with(|| Trail {
                pts: Vec::new(),
                last: f,
            });
            e.pts.push([x, z]);
            if e.pts.len() > MAX_LEN {
                e.pts.remove(0);
            }
            e.last = f;
        }
        self.trails.retain(|_, e| f.wrapping_sub(e.last) <= TTL);
    }

    /// The metric `(X, Z)` history for a track id (oldest → newest), if any.
    pub fn get(&self, id: u64) -> Option<&[[f32; 2]]> {
        self.trails.get(&id).map(|t| t.pts.as_slice())
    }
}
