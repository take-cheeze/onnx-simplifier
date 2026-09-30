#!/usr/bin/env python3
"""Turn any CNN / transformer ONNX model into the power-of-two QDQ graph the layer engine compiles.

The engine needs activations as uint8 (zero point 128) with power-of-two scales and int8 power-of-two weights, so a
requantization shift is exact. Models arrive in three forms:

- float (fp32 / fp16 exports): calibrated and quantized directly by ``quantize_pow2_graph``;
- QDQ with arbitrary scales (what Vitis AI's quantizer, onnxruntime's ``quantize_static`` or AIMET produce): the
  activation Q/DQ pairs are removed, the int8 weights are dequantized back to float, and the graph is then
  requantized with power-of-two scales (this is *not* bit-exact to the original QDQ model; ``--report`` prints the
  output cosine against it);
- QDQ that is already power-of-two / zero point 128: copied unchanged.

    python onnx_to_engine.py MODEL.onnx OUT.onnx [--report]
"""

from __future__ import annotations

import argparse
import math
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def _is_pow2(x: float) -> bool:
    return x > 0 and math.log2(x) == int(math.log2(x))


def classify(model: onnx.ModelProto) -> str:
    """'float', 'engine' (QDQ the engine takes as is) or 'qdq' (needs requantization)."""
    init = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    qdq = [n for n in model.graph.node if n.op_type in ("QuantizeLinear", "DequantizeLinear")]
    if not qdq:
        return "float"
    for n in qdq:
        scale = init.get(n.input[1])
        zero = init.get(n.input[2]) if len(n.input) > 2 else None
        if scale is None or scale.size != 1 or not _is_pow2(float(scale.reshape(-1)[0])):
            return "qdq"
        if n.op_type == "QuantizeLinear" or n.input[0] not in init:  # activation pairs: uint8, zero point 128
            if zero is None or zero.dtype != np.uint8 or int(zero.reshape(-1)[0]) != 128:
                return "qdq"
    return "engine"


def strip_qdq(model: onnx.ModelProto) -> onnx.ModelProto:
    """The float model behind a QDQ graph: weights dequantized, activation Q/DQ pairs removed."""
    init = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    model_inputs = {i.name for i in model.graph.input}
    alias: dict[str, str] = {}
    quantized_from: dict[str, str] = {}
    folded: list = []
    nodes: list = []
    for node in model.graph.node:
        if node.op_type == "QuantizeLinear" and node.input[0] not in init:
            source = alias.get(node.input[0], node.input[0])
            zero = init.get(node.input[2]) if len(node.input) > 2 else None
            if zero is not None and zero.dtype == np.uint8 and int(zero.reshape(-1)[0]) == 0 and source not in model_inputs:
                # an unsigned zero-point-0 range is how QDQ quantizers fold a following Relu/Clip into the Q: put it back
                relu = helper.make_node("Relu", [source], [f"{source}_relu"], name=f"{source}_relu")
                nodes.append(relu)
                source = relu.output[0]
            quantized_from[node.output[0]] = source
        elif node.op_type == "DequantizeLinear":
            x = node.input[0]
            if x in init:  # weights / bias
                scale = init[node.input[1]].astype(np.float32)
                zero = init[node.input[2]].astype(np.float32) if len(node.input) > 2 else np.float32(0)
                axis = next((a.i for a in node.attribute if a.name == "axis"), 1)
                if scale.ndim == 1 and scale.size > 1:  # per-channel
                    shape = [1] * init[x].ndim
                    shape[axis] = -1
                    scale, zero = scale.reshape(shape), np.asarray(zero).reshape(shape) if np.ndim(zero) else zero
                folded.append(numpy_helper.from_array(((init[x].astype(np.float32) - zero) * scale).astype(np.float32), node.output[0]))
            elif x in quantized_from:
                alias[node.output[0]] = quantized_from[x]
            else:
                nodes.append(node)
        else:
            nodes.append(node)
    out = onnx.ModelProto()
    out.CopyFrom(model)
    del out.graph.node[:]
    del out.graph.initializer[:]
    del out.graph.value_info[:]
    for node in nodes:
        for i, name in enumerate(node.input):
            if name in alias:
                node.input[i] = alias[name]
    out.graph.node.extend(nodes)
    for o in out.graph.output:  # a network output that was a DequantizeLinear result
        if o.name in alias:
            out.graph.node.append(helper.make_node("Identity", [alias[o.name]], [o.name], name=f"{o.name}_identity"))
    dropped = {i for n in model.graph.node if n.op_type in ("QuantizeLinear", "DequantizeLinear") for i in n.input[1:]}
    dropped |= {n.input[0] for n in model.graph.node if n.op_type == "DequantizeLinear"}
    out.graph.initializer.extend([i for i in model.graph.initializer if i.name not in dropped] + folded)
    return onnx.shape_inference.infer_shapes(out)


def to_engine_model(src: Path, dst: Path, report: bool = False) -> str:
    from quantize_pow2_graph import quantize

    model = onnx.load(str(src))
    kind = classify(model)
    if kind == "engine":
        onnx.save(model, str(dst))
        return kind
    with tempfile.TemporaryDirectory() as tmp:
        float_path = Path(tmp) / "float.onnx"
        onnx.save(strip_qdq(model) if kind == "qdq" else model, str(float_path))
        quantize(float_path, dst)
    if report and kind == "qdq":
        import onnxruntime as ort

        a = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"])
        b = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in a.get_inputs()[0].shape]
        x = np.random.default_rng(3).random(shape, dtype=np.float32)
        ya, yb = a.run(None, {a.get_inputs()[0].name: x})[0].reshape(-1), b.run(None, {b.get_inputs()[0].name: x})[0].reshape(-1)
        print(f"output cosine vs the original QDQ model: {float(ya @ yb / (np.linalg.norm(ya) * np.linalg.norm(yb))):.5f}")
    return kind


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    print(f"{args.model}: {to_engine_model(args.model, args.out, args.report)} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
