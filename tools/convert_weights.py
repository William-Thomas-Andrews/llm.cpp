#!/usr/bin/env python3
import sys
import os
import glob
import struct
import numpy as np
import torch
from safetensors.torch import load_file

def convert(model_dir: str, output_path: str):
    print("loading weights...")
    # Support both single-file and multi-shard models
    shards = sorted(glob.glob(f"{model_dir}/model-*-of-*.safetensors"))
    if shards:
        tensors = {}
        for shard in shards:
            print(f"  loading shard {os.path.basename(shard)}...")
            tensors.update(load_file(shard))
    else:
        tensors = load_file(f"{model_dir}/model.safetensors")

    with open(output_path, "wb") as f:
        # write number of tensors
        f.write(struct.pack("i", len(tensors)))

        for name, data in tensors.items():
            # per-tensor symmetric quantization to int8
            data_f = data.to(torch.float32)
            max_val = data_f.abs().max().item()
            scale = max_val / 127.0 if max_val > 0 else 1.0
            data_q = (data_f / scale).round().clamp(-127, 127).to(torch.int8)
            print(f"  {name}: {list(data_q.shape)}  scale={scale:.6f}")

            # write name (64 bytes, null padded)
            name_bytes = name.encode("utf-8")[:63]
            f.write(name_bytes + b"\x00" * (64 - len(name_bytes)))

            # write ndim
            ndim = len(data_q.shape)
            f.write(struct.pack("i", ndim))

            # write shape (padded to MAX_DIMS=8)
            for s in data_q.shape:
                f.write(struct.pack("i", s))
            for _ in range(8 - ndim):
                f.write(struct.pack("i", 0))

            # write scale factor (float32)
            f.write(struct.pack("f", scale))

            # write quantized int8 data
            f.write(data_q.numpy().tobytes())

    size_gb = os.path.getsize(output_path) / 1e9
    print(f"\nwrote {len(tensors)} tensors to {output_path}")
    print(f"file size: {size_gb:.2f} GB")

if __name__ == "__main__":
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "models/mistral-7b"
    output    = sys.argv[2] if len(sys.argv) > 2 else "models/mistral-7b/model.bin"
    convert(model_dir, output)