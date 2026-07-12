//! Streaming H.264 encoder over gstreamer (software `x264enc`): push host RGB frames,
//! pull **AVC** (length-prefixed) access units plus an `avcC` codec-config record —
//! exactly the input the browser's WebCodecs `VideoDecoder` wants (portable across
//! Chrome and iOS Safari, unlike raw Annex-B). Behind the `h264` feature.
//!
//! Pipeline: `appsrc → videoconvert → x264enc(zerolatency) → h264parse →
//! video/x-h264,stream-format=avc,alignment=au → appsink`. `zerolatency` + a small
//! `key-int-max` (GOP) means no frame reordering and a bounded keyframe interval, so a
//! new viewer can join within one GOP and every access unit is decodable in order.

use gstreamer::prelude::*;
use gstreamer_app::{AppSink, AppSrc};

use crate::VizError;

/// One encoded access unit (a single frame's worth of AVC NAL units, length-prefixed).
pub struct EncodedAu {
    /// AVCC (length-prefixed) NAL bytes for one frame.
    pub data: Vec<u8>,
    /// Whether this is a keyframe (IDR) — a viewer must start on one.
    pub key: bool,
}

/// A live RGB→H.264 encoder. Build once per stream, `encode` each frame.
pub struct H264Encoder {
    pipeline: gstreamer::Pipeline,
    appsrc: AppSrc,
    appsink: AppSink,
    fps: i32,
    counter: u64,
    codec_data: Option<Vec<u8>>,
}

impl H264Encoder {
    /// Build an encoder for `width`×`height` RGB at `fps`, targeting `bitrate_kbps`
    /// with a keyframe at least every `gop` frames.
    ///
    /// Uses software **`x264enc`** (`tune=zerolatency`, `ultrafast`). The Jetson Orin
    /// Nano has **no hardware video encoder** (NVDEC decode only; NVENC was removed —
    /// encode is CPU-only per NVIDIA's spec). On an NVENC-equipped Jetson (Orin NX/AGX)
    /// swap `x264enc` → `nvv4l2h264enc` (NVMM path) — that's the only line that changes.
    pub fn new(
        width: usize,
        height: usize,
        fps: i32,
        bitrate_kbps: u32,
        gop: i32,
    ) -> Result<Self, VizError> {
        if !gstreamer::INITIALIZED.load(std::sync::atomic::Ordering::Relaxed) {
            gstreamer::init().map_err(|e| VizError::Encode(e.to_string()))?;
        }
        // Force I420 (4:2:0) + main profile so the stream is browser-decodable:
        // feeding RGB straight to x264enc yields High 4:4:4, which WebCodecs can't decode.
        let pipeline_str = format!(
            "appsrc name=src ! videoconvert ! video/x-raw,format=I420 ! \
             x264enc tune=zerolatency speed-preset=ultrafast bitrate={bitrate_kbps} \
             key-int-max={gop} ! video/x-h264,profile=main ! h264parse ! \
             video/x-h264,stream-format=avc,alignment=au ! appsink name=sink sync=false"
        );
        let pipeline = gstreamer::parse::launch(&pipeline_str)
            .map_err(|e| VizError::Encode(e.to_string()))?
            .dynamic_cast::<gstreamer::Pipeline>()
            .map_err(|_| VizError::Encode("pipeline downcast".into()))?;
        let appsrc = pipeline
            .by_name("src")
            .ok_or_else(|| VizError::Encode("no appsrc".into()))?
            .dynamic_cast::<AppSrc>()
            .map_err(|_| VizError::Encode("appsrc downcast".into()))?;
        let appsink = pipeline
            .by_name("sink")
            .ok_or_else(|| VizError::Encode("no appsink".into()))?
            .dynamic_cast::<AppSink>()
            .map_err(|_| VizError::Encode("appsink downcast".into()))?;

        appsrc.set_format(gstreamer::Format::Time);
        appsrc.set_caps(Some(
            &gstreamer::Caps::builder("video/x-raw")
                .field("format", "RGB")
                .field("width", width as i32)
                .field("height", height as i32)
                .field("framerate", gstreamer::Fraction::new(fps, 1))
                .build(),
        ));
        appsrc.set_is_live(true);
        appsrc.set_property("block", false);

        pipeline
            .set_state(gstreamer::State::Playing)
            .map_err(|e| VizError::Encode(e.to_string()))?;

        Ok(Self {
            pipeline,
            appsrc,
            appsink,
            fps,
            counter: 0,
            codec_data: None,
        })
    }

    /// The `avcC` codec-configuration record, available after the first encoded frame.
    /// A WebCodecs `VideoDecoder` needs this as its `description`.
    pub fn codec_data(&self) -> Option<&[u8]> {
        self.codec_data.as_deref()
    }

    /// Encode one RGB frame, returning any access units produced (usually one). Takes
    /// the buffer by value and wraps it zero-copy (`Buffer::from_slice`) — no per-frame
    /// copy of the ~2.7 MB frame.
    pub fn encode(&mut self, rgb: Vec<u8>) -> Result<Vec<EncodedAu>, VizError> {
        let mut buffer = gstreamer::Buffer::from_slice(rgb);
        {
            let bref = buffer
                .get_mut()
                .ok_or_else(|| VizError::Encode("buffer get_mut".into()))?;
            let ns_per = 1_000_000_000 / self.fps as u64;
            bref.set_pts(Some(gstreamer::ClockTime::from_nseconds(
                self.counter * ns_per,
            )));
            bref.set_duration(Some(gstreamer::ClockTime::from_nseconds(ns_per)));
        }
        self.counter += 1;
        self.appsrc
            .push_buffer(buffer)
            .map_err(|e| VizError::Encode(e.to_string()))?;

        let mut out = Vec::new();
        // The first pull may block briefly until the encoder emits; then drain the rest
        // non-blocking. `zerolatency` keeps this ~1 access unit per frame.
        let mut timeout = gstreamer::ClockTime::from_mseconds(500);
        while let Some(sample) = self.appsink.try_pull_sample(timeout) {
            timeout = gstreamer::ClockTime::ZERO;
            if self.codec_data.is_none() {
                if let Some(cd) = sample
                    .caps()
                    .and_then(|c| c.structure(0).map(|s| s.to_owned()))
                    .and_then(|s| s.get::<gstreamer::Buffer>("codec_data").ok())
                    .and_then(|b| b.map_readable().ok().map(|m| m.as_slice().to_vec()))
                {
                    self.codec_data = Some(cd);
                }
            }
            if let Some(buf) = sample.buffer() {
                let key = !buf.flags().contains(gstreamer::BufferFlags::DELTA_UNIT);
                if let Ok(m) = buf.map_readable() {
                    out.push(EncodedAu {
                        data: m.as_slice().to_vec(),
                        key,
                    });
                }
            }
        }
        Ok(out)
    }
}

impl Drop for H264Encoder {
    fn drop(&mut self) {
        let _ = self.pipeline.set_state(gstreamer::State::Null);
    }
}
