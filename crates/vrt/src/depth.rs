//! Metric depth map — the depth-sensor counterpart to [`VrtImage`](crate::VrtImage).
//!
//! A `uint16` **millimetre** depth surface (`0` = no valid measurement), borrowed
//! or owned, mirroring [`VrtImage`]/[`VrtTensor`](crate::VrtTensor) ownership. It
//! is **host-resident**: per-pixel/per-box sampling is on the CPU (cheap, and the
//! data already lands in host RAM from USB depth sensors). A [`borrowed`](Self::borrowed)
//! map is a zero-copy view over a producer's buffer (e.g. the OAK's aligned depth,
//! valid until the next frame); an [`owned`](Self::owned) map carries its `Vec`.
//!
//! Depth is expected **pixel-aligned to the RGB image** it will be sampled against
//! (the producer's job), so `meters_at(u, v)` corresponds to RGB pixel `(u, v)`.

/// A host depth map in millimetres (`u16`, `0` = invalid).
pub struct VrtDepthMap {
    ptr: *const u16,
    width: u32,
    height: u32,
    /// Backing storage kept alive for an owned map (`None` if borrowed).
    _owner: Option<Box<dyn Send>>,
}

// SAFETY: `ptr` addresses host memory stable for this map's lifetime; an owned
// backing buffer is itself `Send`. A borrowed map's validity is the producer's
// documented contract (e.g. "valid until the next frame").
unsafe impl Send for VrtDepthMap {}

impl VrtDepthMap {
    /// Borrow a producer's `width*height` u16 depth buffer (zero-copy).
    ///
    /// # Safety
    /// `ptr` must address at least `width*height` valid `u16`s, alive for as long
    /// as this map is used.
    pub unsafe fn borrowed(ptr: *const u16, width: u32, height: u32) -> Self {
        Self {
            ptr,
            width,
            height,
            _owner: None,
        }
    }

    /// Take ownership of a host depth buffer.
    pub fn owned(data: Vec<u16>, width: u32, height: u32) -> Self {
        assert_eq!(
            data.len(),
            width as usize * height as usize,
            "depth buffer size mismatch"
        );
        let ptr = data.as_ptr();
        Self {
            ptr,
            width,
            height,
            _owner: Some(Box::new(data)),
        }
    }

    pub fn width(&self) -> u32 {
        self.width
    }
    pub fn height(&self) -> u32 {
        self.height
    }

    /// The raw `width*height` millimetre buffer (row-major, `0` = invalid).
    pub fn as_slice(&self) -> &[u16] {
        // SAFETY: `ptr` covers width*height u16 for this map's lifetime (the contract).
        unsafe { std::slice::from_raw_parts(self.ptr, self.width as usize * self.height as usize) }
    }

    /// Fraction of pixels with a valid (non-zero) depth measurement, in `[0, 1]`.
    pub fn valid_fraction(&self) -> f32 {
        let n = self.width as usize * self.height as usize;
        if n == 0 {
            return 0.0;
        }
        self.as_slice().iter().filter(|&&v| v != 0).count() as f32 / n as f32
    }

    /// Raw millimetre value at `(x,y)`, or `None` if out of bounds / invalid (0).
    pub fn mm_at(&self, x: u32, y: u32) -> Option<u16> {
        if x >= self.width || y >= self.height {
            return None;
        }
        // SAFETY: bounds checked above; `ptr` covers width*height u16 (the contract).
        let mm = unsafe { *self.ptr.add(y as usize * self.width as usize + x as usize) };
        (mm != 0).then_some(mm)
    }

    /// Valid depth in **metres** at `(x,y)`, or `None` if out of bounds / invalid.
    pub fn meters_at(&self, x: u32, y: u32) -> Option<f32> {
        self.mm_at(x, y).map(|mm| mm as f32 * 1e-3)
    }

    /// Robust metric depth (metres) for a 2D box `[x1,y1,x2,y2]`: the median of all
    /// valid samples on a grid across the inner `core_frac` of the box. Stereo
    /// depth is holey on textureless regions, so a single center patch frequently
    /// misses; sampling the core and taking the median tolerates holes and rejects
    /// background bleed at the box border. `None` if the whole core is invalid.
    pub fn sample_box(&self, bbox: &[f32; 4], core_frac: f32) -> Option<f32> {
        let cx = (bbox[0] + bbox[2]) * 0.5;
        let cy = (bbox[1] + bbox[3]) * 0.5;
        let hw = (bbox[2] - bbox[0]).abs() * 0.5 * core_frac;
        let hh = (bbox[3] - bbox[1]).abs() * 0.5 * core_frac;
        let x0 = (cx - hw).max(0.0) as u32;
        let x1 = ((cx + hw) as u32).min(self.width.saturating_sub(1));
        let y0 = (cy - hh).max(0.0) as u32;
        let y1 = ((cy + hh) as u32).min(self.height.saturating_sub(1));

        // ~12×12 grid over the core — bounded work regardless of box size.
        const N: u32 = 12;
        let sx = (x1.saturating_sub(x0) / N).max(1);
        let sy = (y1.saturating_sub(y0) / N).max(1);
        let mut vals: Vec<f32> = Vec::with_capacity((N * N) as usize);
        let mut y = y0;
        while y <= y1 {
            let mut x = x0;
            while x <= x1 {
                if let Some(m) = self.meters_at(x, y) {
                    vals.push(m);
                }
                x += sx;
            }
            y += sy;
        }
        if vals.is_empty() {
            return None;
        }
        vals.sort_by(|a, b| a.total_cmp(b));
        Some(vals[vals.len() / 2])
    }

    /// Robust metric depth (metres) at a single point `(u,v)`: the median of valid samples in a
    /// `(2·radius+1)²` window. Stereo depth is holey/noisy at a single pixel (e.g. a keypoint joint),
    /// so a small-window median tolerates holes and rejects spikes. `None` if the whole window is
    /// invalid.
    pub fn sample_point(&self, u: f32, v: f32, radius: i32) -> Option<f32> {
        // Hot path: called once per keypoint (~17/person) + the torso anchor, every frame at
        // 30fps. Collect into a fixed stack buffer instead of a fresh heap `Vec` per call — the
        // window is small and bounded. MAX_RADIUS supports up to a 7×7 window (radius 3, 49
        // samples); typical callers use radius 2 (5×5). A larger radius is clamped to MAX_RADIUS
        // (radius is always small in practice), so the buffer can never overflow.
        const MAX_RADIUS: i32 = 3;
        const CAP: usize = ((2 * MAX_RADIUS + 1) * (2 * MAX_RADIUS + 1)) as usize; // 49
        let radius = radius.min(MAX_RADIUS);
        let (cx, cy) = (u.round() as i32, v.round() as i32);
        let mut buf = [0.0f32; CAP];
        let mut n = 0usize;
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let (x, y) = (cx + dx, cy + dy);
                if x >= 0 && y >= 0 {
                    if let Some(m) = self.meters_at(x as u32, y as u32) {
                        buf[n] = m;
                        n += 1;
                    }
                }
            }
        }
        if n == 0 {
            return None;
        }
        let vals = &mut buf[..n];
        vals.sort_by(|a, b| a.total_cmp(b));
        Some(vals[n / 2])
    }

    /// Like [`sample_point`](Self::sample_point) but **foreground-biased**: returns a low percentile
    /// (the closer ~quarter) of the valid depths in the window instead of the median. A thin limb
    /// (arm/leg) occupies only a few pixels, so the median lands on the BACKGROUND behind it; the limb
    /// is the *closer* cluster, so the lower percentile recovers the limb's depth. Uses the n/4 element
    /// (not the raw minimum) to stay robust against a single noisy sample. `None` if no valid depth.
    pub fn sample_point_foreground(&self, u: f32, v: f32, radius: i32) -> Option<f32> {
        const MAX_RADIUS: i32 = 3;
        const CAP: usize = ((2 * MAX_RADIUS + 1) * (2 * MAX_RADIUS + 1)) as usize; // 49
        let radius = radius.min(MAX_RADIUS);
        let (cx, cy) = (u.round() as i32, v.round() as i32);
        let mut buf = [0.0f32; CAP];
        let mut n = 0usize;
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let (x, y) = (cx + dx, cy + dy);
                if x >= 0 && y >= 0 {
                    if let Some(m) = self.meters_at(x as u32, y as u32) {
                        buf[n] = m;
                        n += 1;
                    }
                }
            }
        }
        if n == 0 {
            return None;
        }
        let vals = &mut buf[..n];
        vals.sort_by(|a, b| a.total_cmp(b));
        Some(vals[n / 4]) // 25th percentile — the closer (foreground) cluster, i.e. the limb
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `meters_at` converts mm → m and maps the 0 sentinel to `None`.
    #[test]
    fn meters_at_converts_mm_and_zero_is_none() {
        // 2×1 map: pixel (0,0)=1500mm valid, pixel (1,0)=0 invalid.
        let depth = VrtDepthMap::owned(vec![1500u16, 0u16], 2, 1);
        // mm→m uses f32 (1e-3 isn't exactly representable) → compare with an epsilon, not ==.
        let m = depth.meters_at(0, 0).expect("1500 mm is valid");
        assert!(
            (m - 1.5).abs() < 1e-4,
            "1500 mm must read as ~1.5 m, got {m}"
        );
        assert_eq!(depth.meters_at(1, 0), None, "0 sentinel must map to None");
        // mm_at agrees on the sentinel.
        assert_eq!(depth.mm_at(1, 0), None, "0 mm is invalid");
        assert_eq!(depth.mm_at(0, 0), Some(1500), "raw mm preserved");
    }

    /// `sample_box` returns the median of valid samples over the box core.
    #[test]
    fn sample_box_median_over_known_grid() {
        // 5×5 map, all 2000 mm except the very center which is 8000 mm. With an odd
        // count of valid samples the median is the dominant 2.0 m, rejecting the spike.
        let w = 5u32;
        let h = 5u32;
        let mut data = vec![2000u16; (w * h) as usize];
        data[(2 * w + 2) as usize] = 8000; // center spike
        let depth = VrtDepthMap::owned(data, w, h);
        let z = depth.sample_box(&[0.0, 0.0, 5.0, 5.0], 1.0).unwrap();
        assert!(
            (z - 2.0).abs() < 1e-6,
            "median must be 2.0 m (spike rejected), got {z}"
        );
    }

    /// `sample_box` skips holes (zeros) and still returns the median of the rest.
    #[test]
    fn sample_box_skips_holes() {
        // 4×4 map of 3000 mm with a scattering of zero holes; median stays 3.0 m.
        let w = 4u32;
        let h = 4u32;
        let mut data = vec![3000u16; (w * h) as usize];
        data[0] = 0;
        data[5] = 0;
        data[10] = 0;
        let depth = VrtDepthMap::owned(data, w, h);
        let z = depth.sample_box(&[0.0, 0.0, 4.0, 4.0], 1.0).unwrap();
        assert!(
            (z - 3.0).abs() < 1e-6,
            "holes skipped, median 3.0 m, got {z}"
        );
    }

    /// An all-invalid (all-zero) box returns `None`.
    #[test]
    fn sample_box_all_invalid_is_none() {
        let depth = VrtDepthMap::owned(vec![0u16; 16], 4, 4);
        assert_eq!(
            depth.sample_box(&[0.0, 0.0, 4.0, 4.0], 1.0),
            None,
            "all-zero core must yield None"
        );
    }

    /// `sample_point` returns the window median, skipping holes.
    #[test]
    fn sample_point_median_skips_holes() {
        // 5×5 map of 1000 mm with the center pixel a hole. The 5×5 window around the
        // center still has 24 valid 1.0 m samples → median 1.0 m, no panic on the hole.
        let w = 5u32;
        let h = 5u32;
        let mut data = vec![1000u16; (w * h) as usize];
        data[(2 * w + 2) as usize] = 0; // hole at the sampled center
        let depth = VrtDepthMap::owned(data, w, h);
        let z = depth.sample_point(2.0, 2.0, 2).unwrap();
        assert!(
            (z - 1.0).abs() < 1e-6,
            "window median must be 1.0 m, got {z}"
        );
    }

    /// An all-invalid window returns `None`.
    #[test]
    fn sample_point_all_invalid_is_none() {
        let depth = VrtDepthMap::owned(vec![0u16; 25], 5, 5);
        assert_eq!(
            depth.sample_point(2.0, 2.0, 2),
            None,
            "all-zero window must yield None"
        );
    }

    /// A radius beyond `MAX_RADIUS` is clamped without panicking and still returns the
    /// median of the (clamped) window — the stack buffer can never overflow.
    #[test]
    fn sample_point_radius_clamped_no_panic() {
        // Uniform 7×7 map at 2500 mm; ask for an absurd radius (100). It clamps to the
        // internal MAX_RADIUS (3 → 7×7), all samples valid, median 2.5 m.
        let depth = VrtDepthMap::owned(vec![2500u16; 49], 7, 7);
        let z = depth.sample_point(3.0, 3.0, 100).unwrap();
        assert!(
            (z - 2.5).abs() < 1e-6,
            "clamped radius must still median to 2.5 m, got {z}"
        );
    }

    /// Out-of-bounds and negative coordinates are handled without panicking.
    #[test]
    fn sample_point_handles_oob_and_negative_coords() {
        let depth = VrtDepthMap::owned(vec![4000u16; 9], 3, 3);
        // Center near a corner so part of the window falls at negative coords; the in-bounds
        // valid samples still drive the median. No panic from the negative dx/dy.
        let z = depth.sample_point(0.0, 0.0, 2).unwrap();
        assert!(
            (z - 4.0).abs() < 1e-6,
            "corner window median 4.0 m, got {z}"
        );
        // A point fully outside the map (and its whole window) yields None, no panic.
        assert_eq!(
            depth.sample_point(50.0, 50.0, 2),
            None,
            "window fully out of bounds must yield None"
        );
        // mm_at / meters_at out-of-bounds are None, not a panic.
        assert_eq!(depth.mm_at(3, 0), None, "x == width is out of bounds");
        assert_eq!(depth.meters_at(0, 3), None, "y == height is out of bounds");
    }
}
