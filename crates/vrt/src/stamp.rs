//! Frame stamping: provenance metadata for results, independent of any
//! orchestration framework.
//!
//! The crates here are pure algorithm libraries — orchestration (microservices,
//! Zenoh, message buses) lives in the application. But *provenance* still needs
//! to travel with the data: which frame, captured when, from which source. These
//! small value types carry that, so an application can stamp a detector's output
//! and publish it without re-inventing a header.

use std::time::Instant;

/// Provenance for one frame: a sequence number, an optional capture timestamp,
/// and an optional source identifier (for multi-camera setups).
#[derive(Debug, Clone, Default)]
pub struct FrameMeta {
    /// Monotonic frame counter (assigned by whoever drives the loop).
    pub seq: u64,
    /// Capture timestamp in nanoseconds (e.g. a camera PTS), if known.
    pub pts_ns: Option<u64>,
    /// Camera / stream identifier for multi-source setups.
    pub source_id: Option<u32>,
}

impl FrameMeta {
    /// Tag `data` with this metadata, producing a [`Stamped`] value.
    pub fn stamp<T>(&self, data: T) -> Stamped<T> {
        Stamped {
            meta: self.clone(),
            data,
        }
    }
}

/// A value tagged with the [`FrameMeta`] of the frame it came from.
///
/// Keeps provenance attached to data as it flows between services — a detector
/// returns `XFeatResult`, the application wraps it as `Stamped<XFeatResult>` and
/// publishes it, and the timestamp/source survive the hop.
#[derive(Debug, Clone)]
pub struct Stamped<T> {
    pub meta: FrameMeta,
    pub data: T,
}

impl<T> Stamped<T> {
    pub fn new(meta: FrameMeta, data: T) -> Self {
        Self { meta, data }
    }

    /// Transform the carried value, keeping the same metadata.
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Stamped<U> {
        Stamped {
            meta: self.meta,
            data: f(self.data),
        }
    }
}

/// A monotonic source of nanosecond timestamps for stamping frames.
///
/// `Send + Sync` so it can be shared across threads.
pub trait Clock: Send + Sync {
    /// Nanoseconds since this clock's epoch. Monotonic non-decreasing.
    fn now_ns(&self) -> u64;
}

/// Default clock: monotonic nanoseconds since construction.
///
/// Backed by [`Instant`] (the platform monotonic clock), so it never goes
/// backwards and is immune to wall-clock adjustments. The epoch is process-local
/// — fine for single-host stamping and latency. For cross-host correlation,
/// supply a [`Clock`] tied to an NTP/PTP-disciplined source instead.
pub struct MonotonicClock {
    base: Instant,
}

impl MonotonicClock {
    pub fn new() -> Self {
        Self {
            base: Instant::now(),
        }
    }
}

impl Default for MonotonicClock {
    fn default() -> Self {
        Self::new()
    }
}

impl Clock for MonotonicClock {
    fn now_ns(&self) -> u64 {
        self.base.elapsed().as_nanos() as u64
    }
}
