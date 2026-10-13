#!/usr/bin/env python3
"""Convert official Darknet YOLOv2 608 COCO weights to a Keras 3 model.

Usage:
  python convert_yolov2_darknet_to_keras3.py yolov2.cfg yolov2.weights output.keras

Requirements: tensorflow, keras >=3, numpy.
The resulting model outputs raw (batch, 19, 19, 425) logits and is compatible
with the YOLOv2 postprocessing equations used in the Coursera C4W3A1 notebook.
"""
from __future__ import annotations

import argparse
import configparser
import io
from collections import defaultdict
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf


@keras.saving.register_keras_serializable(package="YOLOv2")
class SpaceToDepth(keras.layers.Layer):
    """Darknet [reorg] layer with stride=2, retaining serializability."""

    def call(self, inputs):
        return tf.nn.space_to_depth(inputs, block_size=2)

    def compute_output_shape(self, input_shape):
        batch, h, w, c = input_shape
        return (batch, None if h is None else h // 2,
                None if w is None else w // 2,
                None if c is None else c * 4)


def read_cfg(path: Path) -> configparser.ConfigParser:
    # Darknet uses repeated section headers; ConfigParser requires unique ones.
    counts: defaultdict[str, int] = defaultdict(int)
    stream = io.StringIO()
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.lstrip().startswith("["):
                head, rest = line.split("]", 1)
                name = head.strip()[1:]
                line = f"[{name}_{counts[name]}]{rest}"
                counts[name] += 1
            stream.write(line)
    stream.seek(0)
    cp = configparser.ConfigParser(interpolation=None)
    cp.read_file(stream)
    return cp


def read_float32(f, n: int, label: str) -> np.ndarray:
    buf = f.read(4 * n)
    if len(buf) != 4 * n:
        raise ValueError(f"Incomplete weights while reading {label}: "
                         f"expected {4*n} bytes, got {len(buf)}")
    return np.frombuffer(buf, dtype="<f4").copy()


def convert(cfg_path: Path, weights_path: Path, output_path: Path) -> None:
    cfg = read_cfg(cfg_path)
    image_h = cfg.getint("net_0", "height")
    image_w = cfg.getint("net_0", "width")
    if (image_h, image_w) != (608, 608):
        raise ValueError(f"Expected official YOLOv2 608 cfg, got {image_h}x{image_w}")

    input_layer = keras.Input(shape=(image_h, image_w, 3), name="yolo_input")
    x = input_layer
    layers = []  # Darknet layer indices, not individual Keras layers

    with weights_path.open("rb") as f:
        header = np.frombuffer(f.read(12), dtype="<i4")
        if header.size != 3:
            raise ValueError("Invalid Darknet weights header")
        major, minor, revision = map(int, header)
        is_64 = (major * 10 + minor >= 2) and major < 1000 and minor < 1000
        seen = f.read(8 if is_64 else 4)
        if len(seen) != (8 if is_64 else 4):
            raise ValueError("Invalid Darknet weights header: missing seen")
        print(f"Darknet version: {major}.{minor}.{revision}, seen={int.from_bytes(seen, 'little')}")

        for sec in list(cfg.sections())[1:]:
            opts = cfg[sec]
            kind = sec.rsplit("_", 1)[0]
            idx = len(layers)
            if kind == "convolutional":
                filters = int(opts["filters"])
                size = int(opts["size"])
                stride = int(opts["stride"])
                has_bn = "batch_normalize" in opts
                activation = opts["activation"]
                in_channels = int(x.shape[-1])
                kernel_n = filters * in_channels * size * size
                beta_or_bias = read_float32(f, filters, sec + ": bias/beta")
                if has_bn:
                    other_bn = read_float32(f, 3 * filters, sec + ": BN").reshape(3, filters)
                darknet_kernel = read_float32(f, kernel_n, sec + ": kernel")
                kernel = darknet_kernel.reshape(filters, in_channels, size, size)
                kernel = kernel.transpose(2, 3, 1, 0)

                conv = keras.layers.Conv2D(
                    filters, size, strides=stride,
                    padding="same" if int(opts["pad"]) else "valid",
                    use_bias=not has_bn, name=f"darknet_conv_{idx}"
                )
                x = conv(x)
                conv.set_weights([kernel] if has_bn else [kernel, beta_or_bias])
                if has_bn:
                    # Darknet stores BN: beta (already read), gamma, mean, variance.
                    bn = keras.layers.BatchNormalization(epsilon=1e-3,
                                                         name=f"darknet_bn_{idx}")
                    x = bn(x)
                    bn.set_weights([other_bn[0], beta_or_bias, other_bn[1], other_bn[2]])
                if activation == "leaky":
                    x = keras.layers.LeakyReLU(negative_slope=0.1)(x)
                elif activation != "linear":
                    raise ValueError(f"Unsupported activation: {activation}")
            elif kind == "maxpool":
                x = keras.layers.MaxPooling2D(
                    pool_size=int(opts["size"]), strides=int(opts["stride"]),
                    padding="same"
                )(x)
            elif kind == "route":
                ids = [int(t.strip()) for t in opts["layers"].split(",")]
                inputs = [layers[j if j >= 0 else idx + j] for j in ids]
                x = inputs[0] if len(inputs) == 1 else keras.layers.Concatenate()(inputs)
            elif kind == "reorg":
                if int(opts["stride"]) != 2:
                    raise ValueError("Only reorg stride=2 supported")
                x = SpaceToDepth(name="space_to_depth")(x)
            elif kind == "region":
                if int(opts["classes"]) != 80 or int(opts["num"]) != 5:
                    raise ValueError("Expected COCO 80 classes and 5 anchors")
                continue
            else:
                raise ValueError(f"Unsupported Darknet section: {sec}")
            layers.append(x)
            print(f"[{idx:02d}] {kind:15s} -> {tuple(x.shape)}")

        extra = f.read(1)
        if extra:
            raise ValueError("Unread bytes in weight file; cfg and weights may not match")

    model = keras.Model(input_layer, x, name="yolov2_608_coco")
    if tuple(model.output_shape[1:]) != (19, 19, 425):
        raise AssertionError(f"Unexpected model output: {model.output_shape}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    print(f"Saved {output_path} | input={model.input_shape} output={model.output_shape}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cfg", type=Path)
    ap.add_argument("weights", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    if args.output.suffix != ".keras":
        ap.error("output must end in .keras")
    convert(args.cfg, args.weights, args.output)
