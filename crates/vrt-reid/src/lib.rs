//! Appearance re-identification embeddings: **GPU per-box crop+resize → OSNet
//! (TensorRT) → L2-normalized feature vectors**, for the tracker's appearance
//! association.
//!
//! [`ReId`] turns a frame + a list of detection boxes into one embedding per
//! box. The crop is zero-copy and all-GPU: the NVMM camera frame stays on the
//! device while a single kernel ([`crop_resize_norm`]) extracts each box,
//! resizes it to the engine's input (256×128), and applies OSNet's
//! preprocessing (RGB, `/255`, ImageNet mean/std) directly into the batched
//! `[B,3,H,W]` input tensor. Only the surviving embeddings are copied to host.
//!
//! Model: `osnet-reid` in `vrt-hub` (OSNet x0_25, MSMT17) — a **fixed batch**
//! engine (`[16,3,256,128] → [16,512]`). So [`embed`](ReId::embed) processes up
//! to `batch` boxes per call at constant cost; the caller passes the top-`batch`
//! detections (by score) when there are more.
//!
//! # ReID is class-specialized
//! OSNet is a *person* embedding net — it discriminates people (and, with a
//! vehicle model, vehicles). Embeddings of generic objects are still stable
//! frame-to-frame (so they don't hurt single-object identity) but are not
//! meaningfully discriminative across different generic objects. Appearance is
//! therefore an *optional* aid to the motion tracker, not a replacement.

use std::sync::Arc;

use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{CudaSlice, CudaStream};
use kornia_image::Image;
use kornia_tensor::{zeros_cuda, CudaKernel, Tensor};
use vrt::{BoxError, Engine, ModelSession};

// One thread per output pixel of one box: map the output pixel into the box's
// source region (align_corners=False), bilinear-sample the RGBA8 frame, and
// write the normalized RGB value into the CHW batch slot. Degenerate boxes
// (sub-pixel) write zeros. RGBA byte order is R,G,B,A (channel 0 = red).
const KERNEL_SRC: &str = r#"
extern "C" __global__ void crop_resize_norm(
    const unsigned char* __restrict__ src,   // RGB8/RGBA8 interleaved, pitch-linear
    int src_pitch, int src_w, int src_h, int src_bpp,
    const float* __restrict__ boxes,         // [n*4] x1,y1,x2,y2 in src pixels
    int n, int out_h, int out_w,
    float* __restrict__ dst                  // [B,3,out_h,out_w]
){
    int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * out_h * out_w;
    if (idx >= total) return;

    int ox = idx % out_w;
    int oy = (idx / out_w) % out_h;
    int b  = idx / (out_w * out_h);

    const float* bx = boxes + (long)b * 4;
    float x1 = bx[0], y1 = bx[1];
    float bw = bx[2] - x1, bh = bx[3] - y1;
    int  plane = out_h * out_w;
    long base  = (long)b * 3 * plane + (long)oy * out_w + ox;

    if (bw < 1.0f || bh < 1.0f) {                       // degenerate -> zeros
        dst[base] = 0.0f; dst[base + plane] = 0.0f; dst[base + 2*plane] = 0.0f;
        return;
    }

    // align_corners=False mapping into the box region, then clamp to the frame.
    float sx = x1 + ((float)ox + 0.5f) * bw / (float)out_w - 0.5f;
    float sy = y1 + ((float)oy + 0.5f) * bh / (float)out_h - 0.5f;
    sx = fminf(fmaxf(sx, 0.0f), (float)(src_w - 1));
    sy = fminf(fmaxf(sy, 0.0f), (float)(src_h - 1));
    int x0 = (int)floorf(sx), y0 = (int)floorf(sy);
    int x1i = min(x0 + 1, src_w - 1), y1i = min(y0 + 1, src_h - 1);
    float ax = sx - (float)x0, ay = sy - (float)y0;

    float w00 = (1.0f-ax)*(1.0f-ay), w10 = ax*(1.0f-ay);
    float w01 = (1.0f-ax)*ay,        w11 = ax*ay;
    #define P(xx,yy,c) ((float)__ldg(&src[(long)(yy)*src_pitch + (long)(xx)*src_bpp + (c)]))
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float istd[3] = {1.0f/0.229f, 1.0f/0.224f, 1.0f/0.225f};
    #pragma unroll
    for (int c = 0; c < 3; ++c) {
        float v = w00*P(x0,y0,c) + w10*P(x1i,y0,c) + w01*P(x0,y1i,c) + w11*P(x1i,y1i,c);
        dst[base + (long)c*plane] = (v / 255.0f - mean[c]) * istd[c];
    }
    #undef P
}
"#;

/// OSNet appearance embedder: `(Image<u8, 3>, &[box]) → Vec<embedding>`.
///
/// Holds the TRT session, the reused batched input tensor, and the JIT crop
/// kernel — all on one shared CUDA stream (one sync per [`embed`](Self::embed)).
pub struct ReId {
    model: ModelSession,
    input: Tensor<f32, 4>, // [B,3,in_h,in_w] CHW f32 device, reused
    crop: CudaKernel,
    stream: Arc<CudaStream>,
    batch: usize,
    in_h: usize,
    in_w: usize,
    embed_dim: usize,
    out_name: String,
}

impl ReId {
    /// Build an embedder sharing `cuda_stream` with the rest of the application.
    ///
    /// Dimensions are read from the engine: a static `[B,3,H,W]` input and a
    /// `[B,E]` output (OSNet: `[16,3,256,128] → [16,512]`).
    pub fn new(engine: Arc<Engine>, cuda_stream: Arc<CudaStream>) -> Result<Self, BoxError> {
        let ospec = engine
            .outputs()
            .next()
            .ok_or("reid engine has no output")?
            .clone();
        let [batch, _, in_h, in_w] = engine.static_input_nchw()?;
        let embed_dim = *ospec
            .dims
            .last()
            .filter(|&&e| e > 0)
            .ok_or("reid output must end in a static embedding dim")?
            as usize;

        let input = zeros_cuda::<f32, 4>([batch, 3, in_h, in_w], &cuda_stream)?;
        let crop = CudaKernel::compile(cuda_stream.context(), KERNEL_SRC, "crop_resize_norm")?;
        let model = ModelSession::new(engine, cuda_stream.clone())?;

        Ok(Self {
            model,
            input,
            crop,
            stream: cuda_stream,
            batch,
            in_h,
            in_w,
            embed_dim,
            out_name: ospec.name,
        })
    }

    /// Max boxes embedded per call (the engine's fixed batch).
    pub fn batch(&self) -> usize {
        self.batch
    }
    /// Embedding dimension.
    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    /// Embed up to [`batch`](Self::batch) boxes — GPU crop → TRT → L2-normalize.
    ///
    /// Returns one unit-length embedding per input box, in the same order.
    /// `boxes` beyond `batch` are ignored, so pass the top-`batch` by score.
    /// Synchronous (syncs the stream internally); `img` is a device-resident
    /// RGBA surface, boxes are `[x1,y1,x2,y2]` in its pixel coordinates.
    pub fn embed(
        &mut self,
        img: &Image<u8, 3>,
        boxes: &[[f32; 4]],
    ) -> Result<Vec<Vec<f32>>, BoxError> {
        let n = boxes.len().min(self.batch);
        if n == 0 {
            return Ok(Vec::new());
        }

        // Boxes → device (n*4 floats).
        let flat: Vec<f32> = boxes[..n].concat();
        let boxes_dev: CudaSlice<f32> = self.stream.clone_htod(&flat)?;

        // GPU crop+resize+normalize into self.input[0..n]. The kornia image is a
        // contiguous RGB8 device surface: 3 bytes/pixel, pitch = width*3.
        let src_slice = img
            .as_cudaslice()
            .ok_or("reid: image is not device-resident")?;
        let dst_slice = self
            .input
            .as_cudaslice_mut()
            .ok_or("reid: input tensor not device-resident")?;
        let (sw, sh) = (img.width() as i32, img.height() as i32);
        let (sp, bpp) = (sw * 3, 3i32);
        let (ni, oh, ow) = (n as i32, self.in_h as i32, self.in_w as i32);
        self.crop
            .launch_builder(&self.stream)
            .arg(src_slice)
            .arg(&sp)
            .arg(&sw)
            .arg(&sh)
            .arg(&bpp)
            .arg(&boxes_dev)
            .arg(&ni)
            .arg(&oh)
            .arg(&ow)
            .arg(dst_slice)
            .launch_1d((n * self.in_h * self.in_w) as u32)?;

        // Inference; outputs stay on device until the sync below.
        let out = self.model.run(&self.input)?;
        let ov = out
            .get(&self.out_name)
            .ok_or_else(|| format!("no output '{}'", self.out_name))?;
        let oraw = ov.f32_ptr()? as usize as CUdeviceptr;

        // D2H only the first n rows (contiguous in the [B,E] output).
        let mut host = vec![0f32; n * self.embed_dim];
        unsafe {
            cudarc::driver::result::memcpy_dtoh_async(&mut host, oraw, self.stream.cu_stream())?;
        }
        self.stream.synchronize()?;

        // L2-normalize each embedding (cosine similarity becomes a dot product).
        Ok(host
            .chunks_exact(self.embed_dim)
            .map(|e| {
                let inv = 1.0 / e.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
                e.iter().map(|v| v * inv).collect()
            })
            .collect())
    }
}
