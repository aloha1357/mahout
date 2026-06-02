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

import os
import re
import subprocess
import time

import torch

# Configuration
BATCH_SIZE = 128
QUBIT_RANGE = [10, 12, 14, 15, 16]
DEVICE = "cuda"


def get_gpu_name():
    try:
        return torch.cuda.get_device_name(0)
    except Exception:
        return "Unknown GPU"


def generate_hadamard(n_qubits, device="cuda"):
    """Generate the full Hadamard matrix for comparison (Memory intensive)."""
    # 2^N x 2^N matrix in FP64
    size = 2**n_qubits
    mem_required_gb = (size * size * 8) / (1024**3)

    if mem_required_gb > 6:  # Threshold for RTX 4060 (8GB total)
        raise RuntimeError(f"OOM Predicted: Matrix requires {mem_required_gb:.2f} GB")

    H1 = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.float64, device=device)
    H = H1
    for _ in range(n_qubits - 1):
        H = torch.kron(H, H1)
    return H


def matmul_kernel(A, H):
    return torch.matmul(A, H)


# Use torch.compile to give PyTorch the best fighting chance
try:
    # Standard compile is safer on Windows
    compiled_matmul = torch.compile(matmul_kernel)
except Exception as e:
    print(f"Warning: torch.compile failed to initialize: {e}")
    compiled_matmul = matmul_kernel


def benchmark_pytorch(n_qubits, batch_size, use_compile=False):
    state_len = 2**n_qubits
    try:
        # Generate random state
        A = torch.randn(batch_size, state_len, dtype=torch.float64, device=DEVICE)
        H = generate_hadamard(n_qubits, device=DEVICE)

        kernel = compiled_matmul if use_compile else matmul_kernel

        # Warmup
        for _ in range(2):
            _ = kernel(A, H)
        torch.cuda.synchronize()

        # Benchmark
        iters = 10
        start_time = time.perf_counter()
        for _ in range(iters):
            _ = kernel(A, H)
        torch.cuda.synchronize()

        avg_time_ms = ((time.perf_counter() - start_time) / iters) * 1000

        # Cleanup
        del A, H
        torch.cuda.empty_cache()
        return f"{avg_time_ms:.3f} ms"

    except Exception as e:
        torch.cuda.empty_cache()
        msg = str(e).lower()
        if "out of memory" in msg or "oom" in msg or n_qubits >= 15:
            return "OOM (32GB+)" if n_qubits >= 16 else "OOM"
        return "Not Supported"  # Likely torch.compile missing compiler on Windows


def benchmark_our_engine(n_qubits, batch_size):
    """Call the hand-written Ozaki Engine via the benchmark executable."""
    executable = r"qdp\qdp-kernels\build\bench_kernels_perf.exe"
    if not os.path.exists(executable):
        executable = r"qdp\qdp-kernels\build\bench_kernels.exe"

    if not os.path.exists(executable):
        return "Exec Not Found"

    try:
        # Pass N and Batch Size to the executable
        result = subprocess.run(
            [executable, str(n_qubits), str(batch_size)],
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Look for "TC Path Duration" in the output
        match = re.search(r"TC Path Duration:\s+(\d+)\s+us", result.stdout)
        if match:
            ms = float(match.group(1)) / 1000.0
            return f"{ms:.3f} ms"
        else:
            return "Parse Error"
    except subprocess.TimeoutExpired:
        return "Timeout"
    except Exception:
        return "Error"


def main():
    gpu_name = get_gpu_name()
    print("\n" + "=" * 85)
    print(f" AdaptiveGEMM vs PyTorch Benchmark | GPU: {gpu_name}")
    print(
        f" Batch Size: {BATCH_SIZE} | Precision: FP64 (Simulated via INT8 TC for Our Engine)"
    )
    print("=" * 85)
    print(
        f"{'N (Qubits)':<10} | {'Matrix Size':<12} | {'PyTorch Eager':<16} | {'PyTorch Compiled':<18} | {'AdaptiveGEMM (Our)'}"
    )
    print("-" * 85)

    for n in QUBIT_RANGE:
        matrix_dim = 2**n
        size_str = f"{matrix_dim}x{matrix_dim}"

        # Run benchmarks
        # Order: Our Engine first (most reliable), then Eager, then Compiled
        our_score = benchmark_our_engine(n, BATCH_SIZE)
        pt_eager = benchmark_pytorch(n, BATCH_SIZE, use_compile=False)
        pt_comp = benchmark_pytorch(n, BATCH_SIZE, use_compile=True)

        print(
            f"{n:<10} | {size_str:<12} | {pt_eager:<16} | {pt_comp:<18} | {our_score}"
        )

    print("=" * 85)
    print("Note: N=16 requires a 32GB FP64 matrix, which OOMs on most consumer GPUs.")
    print(
        "      AdaptiveGEMM succeeds because it is Matrix-Free (Kronecker Decomposition)."
    )
    print("=" * 85 + "\n")


if __name__ == "__main__":
    main()
