#!/usr/bin/env python3
#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Minimal encode_batch_native loop for NCU/Nsight profiling."""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch


def _repo_root() -> Path:
    env = os.environ.get("MAHOUT_ROOT")
    if env:
        return Path(env)
    for parent in Path(__file__).resolve().parents:
        if (parent / "qdp" / "qdp-python" / "benchmark").is_dir():
            return parent
    raise SystemExit("Cannot locate repo root (set MAHOUT_ROOT)")


BENCH_DIR = _repo_root() / "qdp" / "qdp-python" / "benchmark"
sys.path.insert(0, str(BENCH_DIR))

from qumat_qdp import QdpEngine
from utils import generate_batch_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qubits", type=int, default=14)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--encoding-method", default="iqp-z")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    engine = QdpEngine(device_id=0, precision="float64")
    if not hasattr(engine, "encode_batch_native"):
        raise SystemExit("encode_batch_native not available")

    dim = args.qubits if args.encoding_method == "iqp-z" else args.qubits
    data = generate_batch_data(args.samples, dim, args.encoding_method, seed=42).astype(
        np.float64
    )

    for _ in range(3):
        engine.encode_batch_native(data, args.qubits, args.encoding_method)
    torch.cuda.synchronize()

    for _ in range(args.rounds):
        engine.encode_batch_native(data, args.qubits, args.encoding_method)
    torch.cuda.synchronize()

    print(
        f"PROFILE_DONE N={args.qubits} samples={args.samples} "
        f"rounds={args.rounds} encoding={args.encoding_method}"
    )


if __name__ == "__main__":
    main()
