#!/usr/bin/env python3
"""Power-of-two QDQ quantizer for general CNN graphs (YOLO, MobileNet-style, ...).

Like ``quantize_pow2_resnet.py`` but driven by the graph instead of a ResNet pattern: every activation
tensor produced by Conv / Relu / Clip / unary op / Add / Concat / Split / Resize / pooling becomes uint8 with
zero point 128 and a power-of-two scale (calibrated absmax), weights int8 per tensor, biases int8. Once an
operator has a float input (e.g. the YOLO detection head after the last Conv) it and everything after it is
copied unquantized, fed from the DequantizeLinear of its quantized inputs.

``Sigmoid`` feeding only a ``Mul`` with its own input (SiLU) is left unquantized in between, so the pair
between two Q/DQ pairs is one pointwise function the engine turns into a single lookup table.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

UNARY = {
    "HardSwish",
    "HardSigmoid",
    "Sigmoid",
    "Tanh",
    "Relu",
    "Clip",
    "Erf",
    "Softplus",
    "Gelu",
    "Mish",
    "LeakyRelu",
}


def _pow2_up(value: float) -> float:
    return 2.0 ** math.ceil(math.log2(max(value, 1e-12)))


def quantize(fp32_path: Path, out_path: Path, seed: int = 0, samples: int = 4) -> None:
    model = onnx.shape_inference.infer_shapes(onnx.load(str(fp32_path)))
    graph = model.graph
    init = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
    for node in graph.node:
        if node.op_type == "Constant":
            init[node.output[0]] = numpy_helper.to_array(node.attribute[0].t)
        elif node.op_type == "Identity" and node.input[0] in init:
            init[node.output[0]] = init[
                node.input[0]
            ]  # exporters alias weights/biases through Identity
    consumers: dict[str, list] = {}
    for node in graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
    float_names = {
        v.name
        for v in list(graph.value_info) + list(graph.output)
        if v.type.tensor_type.elem_type == TensorProto.FLOAT
    }
    exposed = [
        o
        for n in graph.node
        if n.op_type != "Constant"
        for o in n.output
        if o in float_names
    ]
    probe = onnx.ModelProto()
    probe.CopyFrom(model)
    for name in exposed:
        probe.graph.output.append(
            helper.make_tensor_value_info(name, TensorProto.FLOAT, None)
        )
    session = ort.InferenceSession(
        probe.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    shape = [d.dim_value for d in graph.input[0].type.tensor_type.shape.dim]
    rng = np.random.default_rng(seed)
    absmax: dict[str, float] = {}
    for _ in range(samples):
        values = session.run(
            exposed, {graph.input[0].name: rng.random(shape, dtype=np.float32)}
        )
        for name, value in zip(exposed, values):
            absmax[name] = max(absmax.get(name, 0.0), float(np.abs(value).max()))
    absmax[graph.input[0].name] = 1.0

    nodes, inits = [], []
    dq_of: dict[str, str] = {}
    scale_of: dict[str, float] = {}
    silu_sigmoid: set[str] = set()
    raw_conv: set[str] = set()  # Conv outputs left unquantized because a Relu/Clip follows

    def const(name, array):
        inits.append(numpy_helper.from_array(np.asarray(array), name))
        return name

    def qdq(tensor: str, scale: float | None = None) -> str:
        scale = scale if scale is not None else _pow2_up(absmax[tensor] / 127.0)
        scale_of[tensor] = scale
        s, z = (
            const(f"{tensor}_scale", np.float32(scale)),
            const(f"{tensor}_zero_point", np.uint8(128)),
        )
        nodes.append(
            helper.make_node(
                "QuantizeLinear", [tensor, s, z], [f"{tensor}_q"], name=f"{tensor}_Q"
            )
        )
        nodes.append(
            helper.make_node(
                "DequantizeLinear",
                [f"{tensor}_q", s, z],
                [f"{tensor}_dq"],
                name=f"{tensor}_DQ",
            )
        )
        dq_of[tensor] = f"{tensor}_dq"
        return dq_of[tensor]

    def emit(node, inputs, outputs=None):
        copy = helper.make_node(
            node.op_type, inputs, list(outputs or node.output), name=node.name
        )
        copy.attribute.extend(node.attribute)
        nodes.append(copy)

    def q_in(node):
        return [dq_of.get(i, i) for i in node.input]

    def all_quantized(node, indices=None):
        names = [
            node.input[i]
            for i in (indices or range(len(node.input)))
            if node.input[i] and node.input[i] not in init
        ]
        return bool(names) and all(n in dq_of for n in names)

    def ensure_q(name: str) -> None:
        """Re-enter the quantized domain: a float activation feeding a quantizable op gets its own Q/DQ."""
        if name and name not in dq_of and name not in init and name in absmax:
            qdq(name)

    def qweights(n: str, x: str, w: np.ndarray, b: np.ndarray) -> tuple[str, str]:
        """int8 power-of-two weight and bias DequantizeLinear nodes for ``n``; returns their output names."""
        w_scale = _pow2_up(float(np.abs(w).max()) / 127.0)
        bias_scale = max(_pow2_up(float(np.abs(b).max()) / 127.0), scale_of[x] * w_scale)
        for kind, arr, sc in (("w", w, w_scale), ("b", b, bias_scale)):
            q = np.clip(np.rint(arr / sc), -128, 127).astype(np.int8)
            nodes.append(
                helper.make_node(
                    "DequantizeLinear",
                    [const(f"{n}_{kind}q", q), const(f"{n}_{kind}s", np.float32(sc)), const(f"{n}_{kind}z", np.int8(0))],
                    [f"{n}_{kind}"],
                    name=f"{n}_{kind}DQ",
                )
            )
        return f"{n}_w", f"{n}_b"

    skip: set[str] = set()
    after_shuffle: dict[str, float] = {}

    def shuffle_of(node):
        """(Transpose, final Reshape) when ``node`` starts Reshape -> Transpose [0, 2, 1, 3, 4] -> Reshape."""
        after = consumers.get(node.output[0], [])
        if len(after) != 1 or after[0].op_type != "Transpose":
            return None
        if list(next((a.ints for a in after[0].attribute if a.name == "perm"), [])) != [0, 2, 1, 3, 4]:
            return None
        last = consumers.get(after[0].output[0], [])
        return (after[0], last[0]) if len(last) == 1 and last[0].op_type == "Reshape" else None

    graph_outputs = {o.name for o in graph.output}
    renamed: dict[str, str] = {}

    def qdq_output(tensor: str, scale: float | None = None) -> None:
        """Q/DQ after ``tensor``; a network output keeps its name as the DequantizeLinear's alias."""
        qdq(tensor, scale)
        if tensor in graph_outputs:
            nodes.append(helper.make_node("Identity", [dq_of[tensor]], [f"{tensor}_final"], name=f"{tensor}_final"))
            renamed[tensor] = f"{tensor}_final"

    qdq(graph.input[0].name, 2.0**-7)
    for node in graph.node:
        op = node.op_type
        if node.name in skip:
            continue
        if node.name in after_shuffle:
            emit(node, list(node.input))
            qdq(node.output[0], after_shuffle[node.name])
            continue
        if op in ("Conv", "ConvTranspose") and node.input[0] not in dq_of:
            ensure_q(node.input[0])
        elif op in ("Add", "Concat") and any(i in dq_of for i in node.input if i):
            for i in node.input:
                ensure_q(i)
        if op == "Constant":
            nodes.append(node)
        elif op == "Identity" and node.input[0] in init:
            const(node.output[0], init[node.input[0]])  # float-region consumers still refer to the alias name
            continue
        elif op == "Identity" and node.input[0] in dq_of:
            dq_of[node.output[0]] = dq_of[node.input[0]]
            scale_of[node.output[0]] = scale_of[node.input[0]]
            if node.output[0] in graph_outputs:  # the network output is the DequantizeLinear result itself
                nodes.append(helper.make_node("Identity", [dq_of[node.input[0]]], list(node.output), name=node.name))
        elif op == "ConvTranspose" and node.input[0] in dq_of:
            x, w = node.input[0], init[node.input[1]]  # weights [ic][oc][kh][kw]
            b = init[node.input[2]] if len(node.input) > 2 else np.zeros(w.shape[1], dtype=np.float32)
            w_scale = _pow2_up(float(np.abs(w).max()) / 127.0)
            bias_scale = max(_pow2_up(float(np.abs(b).max()) / 127.0), scale_of[x] * w_scale)
            n = node.name
            nodes.append(helper.make_node("DequantizeLinear", [const(f"{n}_wq", np.clip(np.rint(w / w_scale), -128, 127).astype(np.int8)), const(f"{n}_ws", np.float32(w_scale)), const(f"{n}_wz", np.int8(0))], [f"{n}_w"], name=f"{n}_wDQ"))
            nodes.append(helper.make_node("DequantizeLinear", [const(f"{n}_bq", np.clip(np.rint(b / bias_scale), -128, 127).astype(np.int8)), const(f"{n}_bs", np.float32(bias_scale)), const(f"{n}_bz", np.int8(0))], [f"{n}_b"], name=f"{n}_bDQ"))
            emit(node, [dq_of[x], f"{n}_w", f"{n}_b"])
            qdq(node.output[0])
        elif op == "Conv" and node.input[0] in dq_of:
            x, w, b = (
                node.input[0],
                init[node.input[1]],
                (init[node.input[2]] if len(node.input) > 2 else None),
            )
            b = b if b is not None else np.zeros(w.shape[0], dtype=np.float32)
            w_scale = _pow2_up(float(np.abs(w).max()) / 127.0)
            bias_scale = max(
                _pow2_up(float(np.abs(b).max()) / 127.0), scale_of[x] * w_scale
            )
            n = node.name
            nodes.append(
                helper.make_node(
                    "DequantizeLinear",
                    [
                        const(
                            f"{n}_wq",
                            np.clip(np.rint(w / w_scale), -128, 127).astype(np.int8),
                        ),
                        const(f"{n}_ws", np.float32(w_scale)),
                        const(f"{n}_wz", np.int8(0)),
                    ],
                    [f"{n}_w"],
                    name=f"{n}_wDQ",
                )
            )
            nodes.append(
                helper.make_node(
                    "DequantizeLinear",
                    [
                        const(
                            f"{n}_bq",
                            np.clip(np.rint(b / bias_scale), -128, 127).astype(np.int8),
                        ),
                        const(f"{n}_bs", np.float32(bias_scale)),
                        const(f"{n}_bz", np.int8(0)),
                    ],
                    [f"{n}_b"],
                    name=f"{n}_bDQ",
                )
            )
            emit(node, [dq_of[x], f"{n}_w", f"{n}_b"])
            after = consumers.get(node.output[0], [])
            if not (len(after) == 1 and after[0].op_type in ("Relu", "Clip")):
                qdq(node.output[0])
            else:
                raw_conv.add(node.output[0])
        elif op in ("Relu", "Clip") and node.input[0] in raw_conv:
            nodes.append(node)  # the conv/add output stays float into this activation, then one Q/DQ
            qdq(node.output[0])
        elif (
            op == "Sigmoid"
            and node.input[0] in dq_of
            and len(consumers.get(node.output[0], [])) == 1
            and consumers[node.output[0]][0].op_type == "Mul"
            and node.input[0] in consumers[node.output[0]][0].input
        ):
            emit(
                node, [dq_of[node.input[0]]]
            )  # SiLU: the sigmoid stays float until after the Mul
            silu_sigmoid.add(node.output[0])
        elif op == "Mul" and any(i in silu_sigmoid for i in node.input):
            emit(node, [dq_of.get(i, i) for i in node.input])
            qdq(node.output[0])
        elif op == "Mul" and all_quantized(node):
            emit(node, q_in(node))  # squeeze-excite: activation x per-channel gate
            qdq(node.output[0])
        elif op in UNARY and node.input[0] in dq_of:
            emit(node, [dq_of[node.input[0]]] + list(node.input[1:]))
            qdq(node.output[0])
        elif op == "Add" and all_quantized(node):
            emit(node, q_in(node))
            after = consumers.get(node.output[0], [])
            if not (len(after) == 1 and after[0].op_type in ("Relu", "Clip")):
                qdq(node.output[0])
            else:
                raw_conv.add(node.output[0])
        elif op == "Concat" and all_quantized(node):
            emit(node, q_in(node))
            qdq(node.output[0])
        elif op == "Slice" and all_quantized(node, [0]):
            emit(node, [dq_of[node.input[0]]] + list(node.input[1:]))
            qdq(node.output[0])
        elif op == "Split" and all_quantized(node, [0]):
            emit(node, [dq_of[node.input[0]]] + list(node.input[1:]))
            for out in node.output:
                qdq(out)
        elif op == "Resize" and all_quantized(node, [0]):
            emit(node, [dq_of[node.input[0]]] + list(node.input[1:]))
            qdq(node.output[0])
        elif op == "Reshape" and node.input[0] in dq_of and shuffle_of(node) is not None:
            transpose, back = shuffle_of(node)  # channel shuffle: one Q/DQ after the Reshape/Transpose/Reshape, same scale
            emit(node, [dq_of[node.input[0]]] + list(node.input[1:]))
            nodes.append(transpose)
            skip.add(transpose.name)
            after_shuffle[back.name] = scale_of[node.input[0]]  # emitted when the loop reaches it (its shape constant may come later)
        elif op == "BatchNormalization" and node.input[0] in dq_of and node.input[1] in init:
            # standalone (pre-activation) BatchNorm = a per-channel affine: a depthwise 1x1 conv with int8 weights
            x, (gamma, beta, mean, var) = node.input[0], [init[i] for i in node.input[1:5]]
            eps = next((a.f for a in node.attribute if a.name == "epsilon"), 1e-5)
            w = (gamma / np.sqrt(var + eps)).astype(np.float32)
            wn, bn = qweights(node.name, x, w.reshape(-1, 1, 1, 1), (beta - mean * w).astype(np.float32))
            conv = helper.make_node("Conv", [dq_of[x], wn, bn], list(node.output), name=node.name, kernel_shape=[1, 1], group=int(w.size))
            nodes.append(conv)
            after = consumers.get(node.output[0], [])
            if not (len(after) == 1 and after[0].op_type in ("Relu", "Clip")):
                qdq(node.output[0])
            else:
                raw_conv.add(node.output[0])
        elif op == "Gemm" and node.input[0] in dq_of and node.input[1] in init:
            w = init[node.input[1]]
            w = w if next((a.i for a in node.attribute if a.name == "transB"), 0) else w.T  # [out][in]
            b = init[node.input[2]] if len(node.input) > 2 else np.zeros(w.shape[0], dtype=np.float32)
            wn, bn = qweights(node.name, node.input[0], w, b)
            nodes.append(helper.make_node("Gemm", [dq_of[node.input[0]], wn, bn], list(node.output), name=node.name, transB=1))
            qdq_output(node.output[0])
        elif op == "Flatten" and node.input[0] in dq_of:
            emit(node, [dq_of[node.input[0]]])
            qdq(node.output[0], scale_of[node.input[0]])
        elif op == "ReduceMean" and node.input[0] in dq_of:
            emit(node, [dq_of[node.input[0]]])
            qdq(node.output[0])
        elif (
            op in ("MaxPool", "AveragePool", "GlobalAveragePool")
            and node.input[0] in dq_of
        ):
            emit(node, [dq_of[node.input[0]]])
            qdq(node.output[0], scale_of[node.input[0]] if op == "MaxPool" else None)
        else:
            emit(node, q_in(node))  # float region: fed from DequantizeLinear outputs
    for tensor, final in renamed.items():  # the quantized value takes over the network output's name
        for node in nodes:
            for i, o in enumerate(node.output):
                if o == tensor:
                    node.output[i] = f"{tensor}_float"
            for i, name in enumerate(node.input):
                if name == tensor:
                    node.input[i] = f"{tensor}_float"
        for node in nodes:
            if node.name == f"{tensor}_final":
                node.output[0] = tensor
    out = onnx.ModelProto()
    out.CopyFrom(model)
    del out.graph.node[:]
    del out.graph.initializer[:]
    del out.graph.value_info[:]
    out.graph.node.extend(nodes)
    consumed = {
        i
        for n in graph.node
        if n.op_type in ("Conv", "Gemm", "BatchNormalization")
        for i in n.input[1:]
        if n.input[0] in dq_of
    }
    out.graph.initializer.extend(
        inits + [i for i in graph.initializer if i.name not in consumed]
    )
    out = onnx.shape_inference.infer_shapes(out)
    onnx.checker.check_model(out)
    onnx.save(out, str(out_path))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    quantize(args.model, args.out, args.seed)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
