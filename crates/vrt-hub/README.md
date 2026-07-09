# vrt-hub

Model-weight distribution + on-device TensorRT engine cache for the
[`vision-rt`](https://github.com/kornia/vision-rt) workspace. Portable ONNX comes
from Hugging Face; the machine-locked `.engine` is built and cached **on the
Jetson itself** (keyed by TRT version + GPU arch), so nothing engine-shaped is
ever copied between boards.

- **`ModelHub`** (feature `hub`): downloads pinned ONNX weights from Hugging Face
  Hub into the standard HF cache and verifies every file against its sha256 pin.
  A static `REGISTRY` maps model names → HF repo + files (e.g. `xfeat-backbone` →
  `kornia/xfeat`). For a private/gated repo, export `HF_TOKEN`.
- **`EngineCache`**: resolves an `.onnx` to a built `.engine`, keyed by ONNX
  content + build profile + TensorRT version + GPU compute capability, under
  `~/.cache/vision-rt/engines/`. Writes are atomic (tmp + rename). `.engine`
  inputs pass through unchanged.
- Build backend: in-process `nvonnxparser` with feature `builder`; otherwise a
  `trtexec` subprocess.

License: Apache-2.0
