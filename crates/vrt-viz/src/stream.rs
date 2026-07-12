//! [`LiveStream`]: a complete two-view H.264 live view. It owns the per-view encoders,
//! the encode worker thread, and the WebSocket [`StreamServer`]; the caller just renders
//! RGB frame pairs and [`submit`](LiveStream::submit)s them. Encoding runs off the
//! caller's thread (a latest-only handoff slot, drop-stale) and access units are
//! broadcast to browser viewers (WebCodecs). Behind the `h264` feature.

use std::sync::{Arc, Condvar, Mutex};

use crate::h264::H264Encoder;
use crate::serve::{Stream, StreamServer};
use crate::VizError;

/// A rendered `(main_rgb, bev_rgb)` pair awaiting encode.
type FramePair = (Vec<u8>, Vec<u8>);
/// Latest-only handoff slot to the encode worker (drop-stale).
type FrameSlot = Arc<(Mutex<Option<FramePair>>, Condvar)>;

/// A running two-view (main + BEV) H.264 live stream. Construct with [`spawn`], feed
/// frames with [`submit`].
///
/// [`spawn`]: LiveStream::spawn
/// [`submit`]: LiveStream::submit
pub struct LiveStream {
    slot: FrameSlot,
}

impl LiveStream {
    /// Bind the WebSocket server on `port` and spawn the encode worker. `main`/`bev` are
    /// each `(width, height)`; `main_kbps`/`bev_kbps` the target bitrates; `fps` the
    /// nominal rate (also the keyframe interval, so a viewer joins/re-syncs within ~1 s).
    pub fn spawn(
        port: u16,
        main: (usize, usize),
        bev: (usize, usize),
        main_kbps: u32,
        bev_kbps: u32,
        fps: i32,
    ) -> Result<Self, VizError> {
        let server = StreamServer::spawn(port)?;
        let slot: FrameSlot = Arc::new((Mutex::new(None), Condvar::new()));
        let wslot = slot.clone();
        // Build the encoders up front so a bad config surfaces here as an error, not as
        // a panic in the detached worker thread. Each tuple is (encoder, which view,
        // whether its codec-config has been published yet).
        let mut encs: Vec<(H264Encoder, Stream, bool)> = [
            (main, main_kbps, Stream::Main),
            (bev, bev_kbps, Stream::Bev),
        ]
        .into_iter()
        .map(|((w, h), kbps, stream)| Ok((H264Encoder::new(w, h, fps, kbps, fps)?, stream, false)))
        .collect::<Result<_, VizError>>()?;
        std::thread::spawn(move || {
            let (lock, cv) = &*wslot;
            loop {
                let (main_rgb, bev_rgb): FramePair = {
                    let mut g = lock.lock().unwrap_or_else(|e| e.into_inner());
                    while g.is_none() {
                        g = cv.wait(g).unwrap_or_else(|e| e.into_inner());
                    }
                    g.take().unwrap()
                };
                for ((enc, stream, cfg_sent), rgb) in encs.iter_mut().zip([main_rgb, bev_rgb]) {
                    let Ok(aus) = enc.encode(rgb) else { continue };
                    // Publish the avcC config once, on the first frame it's available.
                    if !*cfg_sent {
                        if let Some(cd) = enc.codec_data() {
                            server.publish_h264_config(*stream, cd);
                            *cfg_sent = true;
                        }
                    }
                    for au in aus {
                        server.publish_h264_frame(*stream, au.key, au.data);
                    }
                }
            }
        });
        Ok(Self { slot })
    }

    /// Hand the latest rendered pair to the encode worker, overwriting any pair not yet
    /// encoded (drop-stale → viewers always get the freshest frame).
    pub fn submit(&self, main: Vec<u8>, bev: Vec<u8>) {
        let (lock, cv) = &*self.slot;
        *lock.lock().unwrap_or_else(|e| e.into_inner()) = Some((main, bev));
        cv.notify_one();
    }
}
