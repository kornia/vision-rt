# vrt

Safe Rust core for real-time TensorRT inference on Jetson. The base of the
[`vision-rt`](https://github.com/kornia/vision-rt) workspace.

- `Logger → Runtime → Engine → Session` as an `Arc` chain; `ModelSession` takes a
  kornia `Tensor<f32,4>` device input and runs one async enqueue + a single
  `cudaStreamSynchronize` per call.
- `cuda` launch-config helpers (`cfg_1d`/`cfg_2d`/`cfg_per_item`), `buffer`
  (`PinnedBuffer`/`Stream`), and typed `engine`/`session` access.
- `builder` feature: in-process ONNX→engine builder via `trt-sys`'s nvonnxparser
  shim.

Model crates (e.g. [`vrt-xfeat`](../vrt-xfeat)) build on this core; weight
distribution + the on-device engine cache live in [`vrt-hub`](../vrt-hub).

License: Apache-2.0
