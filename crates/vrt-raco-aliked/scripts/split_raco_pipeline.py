#!/usr/bin/env python3
"""
Split the fused **RaCo-ALIKED-LightGlue+** ONNX pipeline into two TRT-friendly halves.

Model source — three separate upstreams, all credit to the original authors:
  * RaCo detector + ranker  https://github.com/cvg/RaCo            Apache-2.0
    Shenoi, Lindenberger, Sarlin, Pollefeys, "RaCo: Ranking and Covariance for
    Practical Learned Keypoints", 3DV 2026, arXiv:2602.15755
  * ALIKED descriptors      https://github.com/Shiaoming/ALIKED     BSD-3-Clause
  * LightGlue+ matcher      https://github.com/cvg/LightGlue        Apache-2.0
The fused ONNX we cut up is produced by fabio-sim/LightGlue-ONNX (Apache-2.0), whose
TensorRT graph optimizations (hierarchical chunked TopK, BatchNorm folding, native
SELU + logit-space NMS, ranker boundary-reranking, the DeformConv->GridSample rewrite,
and the integer-floor-div fix for the FP16 index overflow above 65504) are baked into
the released graph. This script preserves all of them — it only re-cuts the graph.

WHY SPLIT AT ALL
----------------
Upstream only publishes the fused extractor+matcher graph:

    images (2B,3,H,W) -> keypoints (2B,K,2) f32
                      -> matches   (M,3)    int64   <- data-dependent shape
                      -> mscores   (M,)     f32     <- data-dependent shape

That graph cannot run under vrt: int64 is rejected at engine load
(crates/vrt/src/engine.rs), `M` is data-dependent and vrt has no IOutputAllocator,
and it never exposes descriptors — so it cannot feed a descriptor bank or a
relocalization database. A single fused forward can only ever match the two images
you hand it.

Both problems have the same root: LightGlue computes per-query `matches0`/`mscores0`
and *then* compacts them with a single `NonZero`. That NonZero is the ONLY
TRT-hostile op in the entire graph. Cutting upstream of it yields fully static
shapes, and cutting again at the extractor/matcher seam yields two reusable engines.

    extractor: images (B,3,H,W) -> keypoints            (B,K,2)   f32
                                -> normalized_keypoints (B,K,2)   f32
                                -> descriptors          (B,K,128) f32   L2-normalised

    matcher:   normalized_keypoints (2P,1,K,2)   f32
               descriptors          (2P,1,K,128) f32
                                -> matches0 (P,K) int32   (-1 = unmatched)
                                -> mscores0 (P,K) f32

The matcher's rank-4 inputs are a deliberate padding: vrt's `ModelSession::run_inputs`
binds `&Tensor<f32,4>` only, so we splice Reshape nodes at the graph boundary rather
than making vrt rank-generic. Images are stacked interleaved [L0,R0,L1,R1,...].

The matcher is resolution-independent: this script asserts that the image H/W scalars
do not reach the match head (only the batch scalar does, purely as shape plumbing),
which is what makes cutting at *normalized* keypoints mathematically exact.

USAGE
-----
    curl -sSLO https://github.com/fabio-sim/LightGlue-ONNX/releases/download/v3.0/\
raco_aliked_lightglue_pipeline_k1024.onnx

    python3 crates/vrt-raco-aliked/scripts/split_raco_pipeline.py \
        --input  models/onnx/raco/raco_aliked_lightglue_pipeline_k1024.onnx \
        --outdir models/onnx/raco

Needs only `onnx` (no torch, no onnxruntime, no Python 3.12) — it runs on the Jetson's
stock python3. Then build engines with build_engine.sh.

Tensor names inside the released graph are export-run-specific (`div_87`, `getitem_7`,
...) and differ between the k512/k1024/.../k3584 assets, so every boundary tensor here
is discovered STRUCTURALLY — via the stable torch module names on the weights
(`matcher.input_proj.weight`, `matcher.posenc.Wr.weight`) and via op topology around
the unique NonZero. Nothing is hardcoded.
"""

from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

import onnx
from onnx import TensorProto, helper, shape_inference

# The one op that makes the released graph data-dependent, plus the usual suspects
# TensorRT's ONNX parser either rejects or lowers badly. Both halves must be free of
# all of them (same guard style as crates/vrt-dinov3/scripts/export_dinov3.py).
TRT_HOSTILE = {
    "NonZero",
    "If",
    "Loop",
    "Scan",
    "SequenceAt",
    "NonMaxSuppression",
    "DeformConv",
}

# Stable torch module paths on the matcher's first two projections. These survive the
# ONNX export because torch names the initializers after the module tree, so they are
# the reliable anchors for locating the extractor/matcher seam.
W_DESC_PROJ = "matcher.input_proj.weight"
W_POSENC = "matcher.posenc.Wr.weight"


class SplitError(RuntimeError):
    """Raised when the graph does not have the structure this script relies on."""


# ── graph helpers ─────────────────────────────────────────────────────────────


class Graph:
    """Indexed view over an ONNX graph: producers, consumers, shapes, dtypes."""

    def __init__(self, model: onnx.ModelProto) -> None:
        self.model = model
        self.g = model.graph
        self.init = {i.name for i in self.g.initializer}
        self.producer = {o: n for n in self.g.node for o in n.output}
        self.consumers: dict[str, list[onnx.NodeProto]] = {}
        for n in self.g.node:
            for i in n.input:
                self.consumers.setdefault(i, []).append(n)
        self.info: dict[str, tuple[list, int]] = {}
        for vi in list(self.g.value_info) + list(self.g.input) + list(self.g.output):
            t = vi.type.tensor_type
            dims = [
                d.dim_value if d.HasField("dim_value") else (d.dim_param or "?")
                for d in t.shape.dim
            ]
            self.info[vi.name] = (dims, t.elem_type)

    def shape(self, name: str) -> list:
        return self.info.get(name, ([], 0))[0]

    def dtype(self, name: str) -> int:
        return self.info.get(name, ([], 0))[1]

    def describe(self, name: str) -> str:
        dims, dt = self.info.get(name, (None, 0))
        tname = TensorProto.DataType.Name(dt) if dt else "?"
        return f"{name}{dims} {tname}"

    def only_node(self, op_type: str) -> onnx.NodeProto:
        found = [n for n in self.g.node if n.op_type == op_type]
        if len(found) != 1:
            raise SplitError(f"expected exactly one {op_type} node, found {len(found)}")
        return found[0]

    def attr(self, node: onnx.NodeProto, name: str, default=None):
        for a in node.attribute:
            if a.name == name:
                if a.type == a.INT:
                    return a.i
                if a.type == a.INTS:
                    return list(a.ints)
                if a.type == a.STRING:
                    return a.s
        return default

    def backward_cone(self, roots: list[str], stop: set[str]) -> set[str]:
        """All tensors the `roots` depend on, not descending past `stop`."""
        cone: set[str] = set()
        q = deque(roots)
        while q:
            t = q.popleft()
            if t in cone or t in stop:
                continue
            cone.add(t)
            n = self.producer.get(t)
            if n is not None:
                q.extend(n.input)
        return cone


def matmul_data_input(gr: Graph, weight_name: str) -> str:
    """Follow `weight_name` through its Transpose into the MatMul, return the *data* input.

    torch exports `nn.Linear` as Transpose(weight) -> MatMul(x, wT), so the tensor we
    want is the MatMul input that is not the transposed weight.
    """
    if weight_name not in gr.init:
        raise SplitError(f"{weight_name!r} is not an initializer — graph layout changed")
    transposes = [n for n in gr.consumers.get(weight_name, []) if n.op_type == "Transpose"]
    if len(transposes) != 1:
        raise SplitError(
            f"expected {weight_name!r} to feed exactly one Transpose, got {len(transposes)}"
        )
    wt = transposes[0].output[0]
    matmuls = [n for n in gr.consumers.get(wt, []) if n.op_type == "MatMul"]
    if len(matmuls) != 1:
        raise SplitError(f"expected {wt!r} to feed exactly one MatMul, got {len(matmuls)}")
    data = [i for i in matmuls[0].input if i != wt]
    if len(data) != 1:
        raise SplitError(f"MatMul {matmuls[0].name!r} has unexpected inputs {list(matmuls[0].input)}")
    return data[0]


# ── structural discovery ──────────────────────────────────────────────────────


class Boundary:
    """The tensor names this split hinges on, all discovered structurally."""

    def __init__(self, gr: Graph) -> None:
        # -- extractor/matcher seam: the matcher's first two projections.
        self.descriptors = matmul_data_input(gr, W_DESC_PROJ)
        self.norm_kpts = matmul_data_input(gr, W_POSENC)

        # -- keypoints: the graph output that is not one of the two DDS outputs.
        self.keypoints = self._find_keypoints(gr)

        # -- match head: everything hangs off the single NonZero.
        nonzero = gr.only_node("NonZero")
        self.valid = nonzero.input[0]  # And(score>thr, mutual-NN)
        self.matches0, self.mscores0 = self._find_match_head(gr, self.valid)

        # -- the batch scalar that leaks from `images` into the matcher.
        self.batch_shape_node = self._find_batch_shape_node(gr)

        self._validate(gr)

    @staticmethod
    def _find_keypoints(gr: Graph) -> str:
        cands = [
            o.name
            for o in gr.g.output
            if len(gr.shape(o.name)) == 3
            and gr.shape(o.name)[-1] == 2
            and gr.dtype(o.name) == TensorProto.FLOAT
        ]
        if len(cands) != 1:
            raise SplitError(f"expected exactly one (B,K,2) float graph output, got {cands}")
        return cands[0]

    @staticmethod
    def _find_match_head(gr: Graph, valid: str) -> tuple[str, str]:
        """From the NonZero mask, recover per-query matches0 and mscores0.

        Structure (LightGlue's `filter_matches`, pre-compaction):
            scores = <log-assignment>              (P,K,K)
            m0     = ArgMax(scores, axis=2)        (P,K)   <- matches0
            m1     = ArgMax(scores, axis=1)        (P,K)
            mx     = ReduceMax(scores, axis=2)     (P,K)
            exp    = Exp(mx)                       (P,K)   <- mscores0
            valid  = Greater(exp, thr) AND Equal(arange, GatherElements(m1, m0))
        """
        and_node = gr.producer.get(valid)
        if and_node is None or and_node.op_type != "And":
            raise SplitError(f"NonZero input {valid!r} is not produced by And")

        greater = [
            gr.producer[i]
            for i in and_node.input
            if gr.producer.get(i) is not None and gr.producer[i].op_type == "Greater"
        ]
        if len(greater) != 1:
            raise SplitError("expected exactly one Greater feeding the validity And")
        exp_t = [i for i in greater[0].input if gr.producer.get(i) is not None
                 and gr.producer[i].op_type == "Exp"]
        if len(exp_t) != 1:
            raise SplitError("expected the Greater to compare an Exp output against a threshold")
        mscores0 = exp_t[0]

        # matches0 = the ArgMax over the last axis of the same scores tensor the
        # ReduceMax behind `exp` reduces.
        reduce_max = gr.producer[gr.producer[mscores0].input[0]]
        if reduce_max.op_type != "ReduceMax":
            raise SplitError(f"expected ReduceMax behind Exp, got {reduce_max.op_type}")
        scores = reduce_max.input[0]
        argmaxes = [n for n in gr.consumers.get(scores, []) if n.op_type == "ArgMax"]
        last_axis = len(gr.shape(scores)) - 1
        m0 = [n for n in argmaxes if gr.attr(n, "axis") == last_axis]
        if len(m0) != 1:
            raise SplitError(
                f"expected one ArgMax(axis={last_axis}) on the scores tensor, got {len(m0)}"
            )
        return m0[0].output[0], mscores0

    @staticmethod
    def _find_batch_shape_node(gr: Graph) -> onnx.NodeProto:
        """The `Shape(images, start=0, end=1)` node — the batch scalar's source."""
        img = gr.g.input[0].name
        cands = [
            n
            for n in gr.consumers.get(img, [])
            if n.op_type == "Shape" and gr.attr(n, "start", 0) == 0 and gr.attr(n, "end") == 1
        ]
        if len(cands) != 1:
            raise SplitError(f"expected one Shape(images, 0:1) node, got {len(cands)}")
        return cands[0]

    def _validate(self, gr: Graph) -> None:
        """Assert the cut is exact: the match head must not depend on image H/W."""
        d = gr.shape(self.descriptors)
        if len(d) != 3 or d[-1] != 128:
            raise SplitError(f"descriptors {self.descriptors!r} has shape {d}, expected (B,K,128)")
        k = gr.shape(self.norm_kpts)
        if len(k) != 3 or k[-1] != 2:
            raise SplitError(f"norm_kpts {self.norm_kpts!r} has shape {k}, expected (B,K,2)")

        img = gr.g.input[0].name
        cone = gr.backward_cone(
            [self.matches0, self.mscores0, self.valid],
            stop={self.descriptors, self.norm_kpts},
        )
        # Every Shape(images, ...) other than the batch one must stay out of the cone,
        # otherwise the matcher is resolution-dependent and this cut is unsound.
        for n in gr.consumers.get(img, []):
            if n.op_type != "Shape" or n is self.batch_shape_node:
                continue
            leaked = [o for o in n.output if o in cone]
            if leaked:
                raise SplitError(
                    f"match head depends on image geometry via {n.name!r} ({leaked}) — "
                    "cutting at normalized keypoints would not be exact"
                )
        if self.batch_shape_node.output[0] not in cone:
            print("  note: matcher does not reference the batch scalar; rewire is a no-op")

    def report(self, gr: Graph) -> None:
        print("== discovered boundary tensors")
        for label, name in (
            ("keypoints", self.keypoints),
            ("normalized_keypoints", self.norm_kpts),
            ("descriptors", self.descriptors),
            ("matches0 (pre-compaction)", self.matches0),
            ("mscores0 (pre-compaction)", self.mscores0),
            ("valid mask", self.valid),
        ):
            print(f"   {label:28s} {gr.describe(name)}")
        print(f"   {'batch Shape node':28s} {self.batch_shape_node.name!r}")


# ── graph edits ───────────────────────────────────────────────────────────────


def rename_tensor(graph: onnx.GraphProto, old: str, new: str) -> None:
    """Rename a tensor everywhere it appears (I/O, node inputs/outputs, value_info)."""
    if old == new:
        return
    for coll in (graph.input, graph.output, graph.value_info):
        for vi in coll:
            if vi.name == old:
                vi.name = new
    for n in graph.node:
        n.input[:] = [new if i == old else i for i in n.input]
        n.output[:] = [new if o == old else o for o in n.output]


def drop_input(graph: onnx.GraphProto, name: str) -> None:
    keep = [vi for vi in graph.input if vi.name != name]
    del graph.input[:]
    graph.input.extend(keep)


def prune_unreachable(graph: onnx.GraphProto) -> int:
    """Drop nodes and initializers that no graph output depends on."""
    producer = {o: n for n in graph.node for o in n.output}
    live: set[str] = set()
    q = deque(o.name for o in graph.output)
    while q:
        t = q.popleft()
        if t in live:
            continue
        live.add(t)
        n = producer.get(t)
        if n is not None:
            q.extend(n.input)
    keep_nodes = [n for n in graph.node if any(o in live for o in n.output)]
    removed = len(graph.node) - len(keep_nodes)
    del graph.node[:]
    graph.node.extend(keep_nodes)
    keep_init = [i for i in graph.initializer if i.name in live]
    del graph.initializer[:]
    graph.initializer.extend(keep_init)
    return removed


def add_rank4_input(graph: onnx.GraphProto, target: str, new_name: str, k_dim, last: int) -> None:
    """Replace rank-3 graph input `target` with a rank-4 input + Reshape.

    vrt's ModelSession::run_inputs binds &Tensor<f32,4>, so the engine must take
    (B,1,K,D) and reshape to the (B,K,D) the original graph expects.
    """
    old = next((vi for vi in graph.input if vi.name == target), None)
    if old is None:
        raise SplitError(f"{target!r} is not a graph input")
    batch = old.type.tensor_type.shape.dim[0]
    batch_dim = batch.dim_value if batch.HasField("dim_value") else (batch.dim_param or "batch")

    padded = helper.make_tensor_value_info(
        new_name, TensorProto.FLOAT, [batch_dim, 1, k_dim, last]
    )
    shape_init = helper.make_tensor(
        f"{new_name}_shape", TensorProto.INT64, [3], [0, k_dim if isinstance(k_dim, int) else -1, last]
    )
    graph.initializer.append(shape_init)
    graph.node.insert(
        0, helper.make_node("Reshape", [new_name, f"{new_name}_shape"], [target],
                            name=f"node_unpad_{new_name}")
    )
    drop_input(graph, target)
    graph.input.append(padded)


def finalize_matches(graph: onnx.GraphProto, matches0: str, valid: str, k_dim) -> None:
    """int64 matches -> int32, with the validity mask folded in as -1 (LightGlue convention).

    vrt rejects int64 at engine load (crates/vrt/src/engine.rs), and carrying a
    separate BOOL mask output would need another dtype accessor. Folding both into one
    int32 tensor keeps the Rust side to a single `i32_ptr()` read.
    """
    cast_out = f"{matches0}_i32"
    graph.node.append(
        helper.make_node("Cast", [matches0], [cast_out], to=TensorProto.INT32, name="node_matches_i32")
    )
    neg1 = helper.make_tensor("const_unmatched", TensorProto.INT32, [1], [-1])
    graph.initializer.append(neg1)
    graph.node.append(
        helper.make_node("Where", [valid, cast_out, "const_unmatched"], ["matches0"],
                         name="node_matches_masked")
    )
    keep = [o for o in graph.output if o.name not in (matches0, valid)]
    del graph.output[:]
    graph.output.extend(keep)
    batch = "pair_count"
    graph.output.append(
        helper.make_tensor_value_info("matches0", TensorProto.INT32, [batch, k_dim])
    )


# ── TensorRT 10.3 compatibility ───────────────────────────────────────────────
#
# The released graph targets TensorRT 10.16 / CUDA 13 (see upstream pyproject), which
# constant-folds aggressively inside its ONNX parser. TensorRT 10.3 — the Jetson
# JetPack 6 version — does not, and rejects two constructs that torch's dynamo exporter
# emits. Both appear in the *fused* model too, so this is an upstream/TRT-version gap,
# not something the split introduces:
#
#   1. Conv with a computed bias. `conv(x, w)` with no bias is exported as an explicit
#      zero bias `Expand(CastLike(0.0, ...), Shape(w)[0:1])`. Because that third input
#      is not an initializer, TRT takes its `convMultiInput` path and dies with
#      "checkSpatialDims(kernel_weights.shape): input tensor shape misaligns with the
#      input kernel shape".
#   2. Reduce* with a computed `axes` input, which TRT requires to be an initializer:
#      "inputAxes.is_weights(): Axis input must be an initializer!".
#
# Both operands are constant-derived, so we evaluate them here and bake them into
# initializers. This is a no-op on TRT versions that would have folded them anyway.

_NUMPY_ATTR = ("value", "value_ints", "value_int", "value_floats", "value_float")


class ConstFolder:
    """Evaluates constant-derived tensors in a graph to numpy arrays."""

    def __init__(self, graph: onnx.GraphProto) -> None:
        from onnx import numpy_helper

        self._np = numpy_helper
        self.graph = graph
        self.init = {i.name: i for i in graph.initializer}
        self.producer = {o: n for n in graph.node for o in n.output}

    def _constant_node(self, node: onnx.NodeProto):
        import numpy as np

        for a in node.attribute:
            if a.name not in _NUMPY_ATTR:
                continue
            if a.name == "value":
                return self._np.to_array(a.t)
            if a.name == "value_ints":
                return np.asarray(list(a.ints), dtype=np.int64)
            if a.name == "value_int":
                return np.asarray(a.i, dtype=np.int64)
            if a.name == "value_floats":
                return np.asarray(list(a.floats), dtype=np.float32)
            if a.name == "value_float":
                return np.asarray(a.f, dtype=np.float32)
        return None

    def value(self, name: str, depth: int = 0):
        """Return a numpy array for `name`, or None if it is not constant-derived."""
        import numpy as np

        if depth > 24:
            return None
        if name in self.init:
            return self._np.to_array(self.init[name])
        node = self.producer.get(name)
        if node is None:
            return None
        op = node.op_type

        if op == "Constant":
            return self._constant_node(node)

        # CastLike's second input supplies only a dtype, never a value, so it must not
        # be evaluated — it is usually a live activation. The caller casts to the dtype
        # it actually needs when materialising the initializer.
        if op == "CastLike":
            return self.value(node.input[0], depth + 1)

        if op == "Shape":
            src = node.input[0]
            if src in self.init:
                dims = np.asarray(list(self.init[src].dims), dtype=np.int64)
            else:
                return None  # a dynamic tensor's shape is not a compile-time constant
            start = _int_attr(node, "start", 0)
            end = _int_attr(node, "end", len(dims))
            return dims[start:end]

        vals = [self.value(i, depth + 1) for i in node.input]
        if any(v is None for v in vals):
            return None
        try:
            if op == "Cast":
                return vals[0]
            if op == "Expand":
                return np.broadcast_to(vals[0], tuple(np.asarray(vals[1]).ravel().astype(int))).copy()
            if op == "Reshape":
                return vals[0].reshape(tuple(np.asarray(vals[1]).ravel().astype(int)))
            if op == "Concat":
                return np.concatenate([np.atleast_1d(v) for v in vals])
            if op == "Unsqueeze":
                return np.expand_dims(vals[0], tuple(np.asarray(vals[1]).ravel().astype(int)))
            if op == "Squeeze":
                return np.squeeze(vals[0]) if len(vals) == 1 else np.squeeze(
                    vals[0], tuple(np.asarray(vals[1]).ravel().astype(int))
                )
            if op == "Gather":
                axis = _int_attr(node, "axis", 0)
                return np.take(vals[0], np.asarray(vals[1]).astype(int), axis=axis)
            if op == "Slice":
                starts, ends = np.asarray(vals[1]).ravel(), np.asarray(vals[2]).ravel()
                axes = np.asarray(vals[3]).ravel() if len(vals) > 3 else np.arange(len(starts))
                steps = np.asarray(vals[4]).ravel() if len(vals) > 4 else np.ones_like(starts)
                out = vals[0]
                for s, e, ax, st in zip(starts, ends, axes, steps):
                    idx = [slice(None)] * out.ndim
                    idx[int(ax)] = slice(int(s), int(e), int(st))
                    out = out[tuple(idx)]
                return out
            if op == "Range":
                return np.arange(vals[0], vals[1], vals[2])
            if op == "Mul":
                return vals[0] * vals[1]
            if op == "Add":
                return vals[0] + vals[1]
            if op == "Sub":
                return vals[0] - vals[1]
            if op == "Div":
                return vals[0] / vals[1]
        except Exception:  # noqa: BLE001 - any eval failure just means "not constant"
            return None
        return None


def _int_attr(node: onnx.NodeProto, name: str, default: int) -> int:
    for a in node.attribute:
        if a.name == name and a.type == a.INT:
            return a.i
    return default


def fix_trt103_compat(graph: onnx.GraphProto, label: str) -> None:
    """Bake computed Conv biases and Reduce axes into initializers (see note above)."""
    import numpy as np
    from onnx import numpy_helper

    folder = ConstFolder(graph)
    init_names = set(folder.init)

    targets: list[tuple[onnx.NodeProto, int, type]] = []
    for n in graph.node:
        if n.op_type == "Conv" and len(n.input) > 2 and n.input[2] and n.input[2] not in init_names:
            targets.append((n, 2, np.float32))
        elif n.op_type.startswith("Reduce") and len(n.input) > 1 and n.input[1] not in init_names:
            targets.append((n, 1, np.int64))

    if not targets:
        print(f"   {label}: no TRT-10.3 compat fixes needed")
        return

    fixed, failed = 0, []
    for node, slot, dtype in targets:
        src = node.input[slot]
        val = folder.value(src)
        if val is None:
            failed.append((node.name, src))
            continue
        arr = np.ascontiguousarray(np.asarray(val, dtype=dtype))
        name = f"{node.name}_folded_{'bias' if slot == 2 else 'axes'}"
        graph.initializer.append(numpy_helper.from_array(arr, name=name))
        node.input[slot] = name
        fixed += 1

    if failed:
        raise SplitError(
            f"{label}: could not constant-fold {len(failed)} operand(s) TRT 10.3 requires "
            f"as initializers: {failed[:5]}"
        )
    removed = prune_unreachable(graph)
    print(f"   {label}: folded {fixed} Conv-bias/Reduce-axes operand(s), pruned {removed} dead node(s)")


def assert_trt_clean(model: onnx.ModelProto, label: str) -> None:
    present = {n.op_type for n in model.graph.node} & TRT_HOSTILE
    if present:
        offenders = [
            f"{n.op_type}({n.name})" for n in model.graph.node if n.op_type in present
        ]
        raise SplitError(f"{label}: TRT-hostile ops survived: {offenders}")
    print(f"   {label}: no TRT-hostile ops ({len(model.graph.node)} nodes)")


def describe_io(model: onnx.ModelProto, label: str) -> None:
    print(f"== {label}")
    for tag, coll in (("IN ", model.graph.input), ("OUT", model.graph.output)):
        for vi in coll:
            t = vi.type.tensor_type
            dims = [
                d.dim_value if d.HasField("dim_value") else (d.dim_param or "?")
                for d in t.shape.dim
            ]
            print(f"   {tag} {vi.name}: {dims} {TensorProto.DataType.Name(t.elem_type)}")


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path, help="fused raco_aliked_lightglue_pipeline_kN.onnx")
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--tag", default=None, help="name suffix; inferred from the input filename")
    args = ap.parse_args()

    tag = args.tag or args.input.stem.replace("raco_aliked_lightglue_pipeline_", "")
    args.outdir.mkdir(parents=True, exist_ok=True)

    print(f"== loading {args.input}")
    model = onnx.load(str(args.input))
    opset = {o.domain: o.version for o in model.opset_import}
    print(f"   ir_version={model.ir_version} opset={opset}")

    print("== shape inference")
    inferred = shape_inference.infer_shapes(model, strict_mode=False, data_prop=True)
    # extract_model needs value_info for the internal tensors we cut at, so work from
    # the inferred model on disk.
    inferred_path = args.outdir / f"_inferred_{tag}.onnx"
    onnx.save(inferred, str(inferred_path))

    gr = Graph(inferred)
    b = Boundary(gr)
    b.report(gr)

    k_dim = gr.shape(b.descriptors)[1]
    img = gr.g.input[0].name

    extractor_path = args.outdir / f"raco_aliked_extractor_{tag}.onnx"
    matcher_path = args.outdir / f"lightglue_matcher_{tag}.onnx"

    # ── extractor half ────────────────────────────────────────────────────────
    print(f"\n== extracting extractor half -> {extractor_path.name}")
    onnx.utils.extract_model(
        str(inferred_path),
        str(extractor_path),
        input_names=[img],
        output_names=[b.keypoints, b.norm_kpts, b.descriptors],
        check_model=False,
    )
    ex = onnx.load(str(extractor_path))
    rename_tensor(ex.graph, b.norm_kpts, "normalized_keypoints")
    rename_tensor(ex.graph, b.descriptors, "descriptors")
    rename_tensor(ex.graph, b.keypoints, "keypoints")
    fix_trt103_compat(ex.graph, "extractor")
    onnx.checker.check_model(ex)
    assert_trt_clean(ex, "extractor")
    onnx.save(ex, str(extractor_path))
    describe_io(ex, f"extractor ({extractor_path.name})")

    # ── matcher half ──────────────────────────────────────────────────────────
    # `images` has to come along initially: the match head reads the batch size via
    # Shape(images, 0:1). We rewire that to Shape(descriptors, 0:1) — identical value,
    # since both tensors share dim 0 — and then drop `images` entirely. Doing the
    # rewire on the *extracted* graph avoids the cycle it would create in the full one
    # (descriptors is computed long after the batch scalar is first consumed).
    print(f"\n== extracting matcher half -> {matcher_path.name}")
    onnx.utils.extract_model(
        str(inferred_path),
        str(matcher_path),
        input_names=[b.norm_kpts, b.descriptors, img],
        output_names=[b.matches0, b.mscores0, b.valid],
        check_model=False,
    )
    mt = onnx.load(str(matcher_path))
    rewired = 0
    for n in mt.graph.node:
        if n.op_type == "Shape" and n.input[0] == img:
            n.input[0] = b.descriptors
            rewired += 1
    print(f"   rewired {rewired} Shape(images) node(s) onto descriptors")
    drop_input(mt.graph, img)
    pruned = prune_unreachable(mt.graph)
    print(f"   pruned {pruned} node(s) left dead by the rewire")
    fix_trt103_compat(mt.graph, "matcher")

    finalize_matches(mt.graph, b.matches0, b.valid, k_dim)
    rename_tensor(mt.graph, b.mscores0, "mscores0")
    add_rank4_input(mt.graph, b.norm_kpts, "normalized_keypoints", k_dim, 2)
    add_rank4_input(mt.graph, b.descriptors, "descriptors", k_dim, 128)

    mt = shape_inference.infer_shapes(mt, strict_mode=False, data_prop=True)
    onnx.checker.check_model(mt)
    assert_trt_clean(mt, "matcher")
    onnx.save(mt, str(matcher_path))
    describe_io(mt, f"matcher ({matcher_path.name})")

    inferred_path.unlink(missing_ok=True)
    print("\n== done")
    print(f"   {extractor_path}")
    print(f"   {matcher_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SplitError as e:
        print(f"\nSPLIT FAILED: {e}", file=sys.stderr)
        sys.exit(1)
