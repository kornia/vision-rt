use std::sync::{Arc, Mutex};

use anyhow::Context as _;
use bubbaloop_node::{CborPublisher, Node, NodeContext};
use serde::{Deserialize, Serialize};
use trt::{Engine, Logger, Runtime, Session};
use trt::logger::Severity;

// ── Config ───────────────────────────────────────────────────────────────────

#[derive(Deserialize, Debug)]
pub struct DetectorConfig {
    /// Per-instance name (e.g. "entrance_detector").
    pub name: String,
    /// Camera instance name to subscribe to (e.g. "tapo_entrance").
    pub camera: String,
    /// Path to the TensorRT .engine file.
    pub engine_path: String,
    /// Model input width/height (square, default 640).
    #[serde(default = "default_model_size")]
    pub model_size: u32,
    /// Minimum class confidence to keep (default 0.25).
    #[serde(default = "default_conf")]
    pub conf_threshold: f32,
    /// IoU threshold for NMS (default 0.45).
    #[serde(default = "default_iou")]
    pub iou_threshold: f32,
}

fn default_model_size() -> u32 { 640 }
fn default_conf() -> f32 { 0.25 }
fn default_iou() -> f32 { 0.45 }

// ── Wire types ───────────────────────────────────────────────────────────────

/// CBOR body of the RawImage published by the camera node.
#[derive(Deserialize, Debug)]
struct RawImageBody {
    pub width: u32,
    pub height: u32,
    #[allow(dead_code)]
    pub encoding: String,
    #[allow(dead_code)]
    pub step: u32,
    #[serde(with = "serde_bytes")]
    pub data: Vec<u8>,
}

/// A single detection result in original image coordinates.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DetectionResult {
    pub class_id: u32,
    pub score: f32,
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

/// All detections for one frame, plus inference timing.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FrameDetections {
    pub detections: Vec<DetectionResult>,
    /// Inference wall-clock time in milliseconds.
    pub frame_ms: u64,
}

// ── Node ─────────────────────────────────────────────────────────────────────

/// The TRT detector node.
///
/// `Session` is `Send` but `!Sync` (IExecutionContext is not thread-safe).
/// The `Node` trait bound requires `Sync`, so we wrap `Session` in a `Mutex`.
/// A single async task drives the inference loop — the Mutex is never
/// actually contended, but it satisfies the `Sync` requirement.
pub struct TrtDetectorNode {
    config: DetectorConfig,
    session: Mutex<Session>,
}

#[bubbaloop_node::async_trait::async_trait]
impl Node for TrtDetectorNode {
    type Config = DetectorConfig;

    fn name() -> &'static str {
        "trt-detector"
    }

    async fn init(
        _ctx: &NodeContext,
        config: &DetectorConfig,
    ) -> anyhow::Result<Self> {
        log::info!(
            "trt-detector: loading engine from '{}'",
            config.engine_path
        );

        let logger = Logger::new(Severity::Warning)
            .context("failed to create TRT logger")?;
        let runtime = Runtime::new(logger)
            .context("failed to create TRT runtime")?;
        let engine: Arc<Engine> = Engine::from_file(runtime, &config.engine_path)
            .context("failed to load TRT engine")?;
        let session = Session::new(engine)
            .context("failed to create TRT session")?;

        log::info!(
            "trt-detector: engine loaded (model_size={}, conf={}, iou={})",
            config.model_size,
            config.conf_threshold,
            config.iou_threshold,
        );

        Ok(Self {
            config: DetectorConfig {
                name: config.name.clone(),
                camera: config.camera.clone(),
                engine_path: config.engine_path.clone(),
                model_size: config.model_size,
                conf_threshold: config.conf_threshold,
                iou_threshold: config.iou_threshold,
            },
            session: Mutex::new(session),
        })
    }

    async fn run(self, ctx: NodeContext) -> anyhow::Result<()> {
        // Subscribe to camera raw frames over SHM.
        // The camera publishes on:
        //   bubbaloop/local/{machine}/{camera_name}/raw
        // subscriber_raw with local=true resolves the absolute suffix
        //   {camera_name}/raw  →  bubbaloop/local/{machine}/{camera_name}/raw
        let camera_suffix = format!("{}/raw", self.config.camera);
        let raw_sub = ctx
            .subscriber_raw(&camera_suffix, true)
            .await
            .context("failed to declare camera subscriber")?;

        // Publisher for detections (global, auto-scoped under instance_name).
        //   bubbaloop/global/{machine}/{instance_name}/detections
        let det_pub: CborPublisher = ctx
            .publisher_cbor("detections")
            .await
            .context("failed to declare detections publisher")?;

        log::info!(
            "trt-detector: watching camera '{}' on local topic, engine '{}'",
            self.config.camera,
            self.config.engine_path,
        );

        let mut shutdown_rx = ctx.shutdown_rx.clone();

        loop {
            tokio::select! {
                biased;

                _ = shutdown_rx.changed() => {
                    log::info!("trt-detector: shutdown signal received");
                    break;
                }

                sample = raw_sub.recv() => {
                    let Some(zbytes) = sample else { break; };
                    let raw_bytes = zbytes.to_bytes().to_vec();
                    // Run the synchronous GPU inference (non-async) while holding
                    // the mutex, then drop the guard before any await point.
                    let inference_result = {
                        let mut session = self.session.lock().expect("session mutex poisoned");
                        run_inference(&raw_bytes, &mut session, &self.config)
                    };
                    match inference_result {
                        Ok(frame_dets) => {
                            if let Err(e) = det_pub.put(&frame_dets).await {
                                log::warn!("trt-detector: publish error: {e:#}");
                            }
                        }
                        Err(e) => {
                            log::warn!("trt-detector: frame error: {e:#}");
                        }
                    }
                }
            }
        }

        log::info!("trt-detector: shut down cleanly");
        Ok(())
    }
}

// ── Frame processing ─────────────────────────────────────────────────────────

/// Synchronous (non-async) inference pipeline.
///
/// Decodes the CBOR frame, runs TRT inference, and returns the detections.
/// Must NOT hold any locks across await points — callers must drop the
/// `MutexGuard<Session>` before calling `.await` on the publisher.
fn run_inference(
    raw_bytes: &[u8],
    session: &mut Session,
    config: &DetectorConfig,
) -> anyhow::Result<FrameDetections> {
    // Decode CBOR bytes.  The camera wraps the payload in the SDK provenance
    // envelope: {header: {...}, body: {width, height, encoding, step, data}}.
    // decode_raw_image handles both enveloped and bare CBOR.
    let image: RawImageBody = decode_raw_image(raw_bytes)
        .context("failed to decode RawImage CBOR")?;

    let model_size = config.model_size;

    // Pre-process: letterbox RGBA → CHW f32 tensor.
    let (chw_input, lb_info) = trt::models::yolo::letterbox_rgba_to_chw(
        &image.data,
        image.width,
        image.height,
        model_size,
        model_size,
    );

    // Run TRT inference (blocking GPU call — fast on Jetson Orin, ~5-15ms).
    let t0 = std::time::Instant::now();
    let outputs = session
        .run(&[("images", &chw_input)])
        .context("TRT inference failed")?;
    let frame_ms = t0.elapsed().as_millis() as u64;

    // Pick the first output tensor.
    let out_tensor = outputs
        .into_values()
        .next()
        .ok_or_else(|| anyhow::anyhow!("no output tensors from model"))?;

    // Decode the raw tensor data — handle both Float32 and Float16 engines.
    let raw_f32_owned: Vec<f32>;
    let (raw_output, output_shape) = if out_tensor.dtype == trt::DataType::Float16 {
        raw_f32_owned = out_tensor.as_f32_from_f16();
        (raw_f32_owned.as_slice(), out_tensor.shape.as_slice())
    } else {
        (out_tensor.as_f32(), out_tensor.shape.as_slice())
    };

    // Full YOLO post-process: decode → NMS → un-letterbox.
    let detections = trt::models::yolo::postprocess(
        raw_output,
        output_shape,
        &lb_info,
        config.conf_threshold,
        config.iou_threshold,
        None, // no label map
    );

    // Convert to wire format.
    let results: Vec<DetectionResult> = detections
        .into_iter()
        .map(|d| DetectionResult {
            class_id: d.class_id,
            score: d.score,
            x1: d.bbox[0],
            y1: d.bbox[1],
            x2: d.bbox[2],
            y2: d.bbox[3],
        })
        .collect();

    log::debug!(
        "trt-detector: {} detections in {}ms",
        results.len(),
        frame_ms
    );

    Ok(FrameDetections {
        detections: results,
        frame_ms,
    })
}

/// Decode a CBOR-encoded RawImage body from the raw byte payload.
///
/// The camera publishes using the SDK provenance envelope
/// `{header: {...}, body: <RawImageBody>}`. We try the full envelope first;
/// on failure fall back to bare CBOR (no envelope wrapper).
fn decode_raw_image(bytes: &[u8]) -> anyhow::Result<RawImageBody> {
    if let Ok(env) =
        ciborium::from_reader::<bubbaloop_node::Envelope<RawImageBody>, _>(bytes)
    {
        return Ok(env.body);
    }
    ciborium::from_reader::<RawImageBody, _>(bytes)
        .map_err(|e| anyhow::anyhow!("CBOR decode failed: {e}"))
}
