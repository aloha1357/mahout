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

"""
E2E smoke tests for PR8 Native path (encode + DLPack + downstream consume).

Mirrors the core steps of benchmark_e2e.py without disk IO competitors.
"""

import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from qumat_qdp import QdpEngine

from .qdp_test_utils import requires_qdp

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPO_ROOT / "qdp" / "qdp-python" / "benchmark"


def _iqp_param_count(num_qubits: int, encoding_method: str) -> int:
    if encoding_method == "iqp-z":
        return num_qubits
    return num_qubits + num_qubits * (num_qubits - 1) // 2


class _DummyQNN(nn.Module):
    def __init__(self, n_qubits: int) -> None:
        super().__init__()
        self.fc = nn.Linear(1 << n_qubits, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


@pytest.fixture(scope="module")
def engine():
    pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    try:
        eng = QdpEngine(device_id=0, precision="float64")
    except Exception as exc:
        pytest.skip(f"Could not initialize QdpEngine: {exc}")
    if not hasattr(eng, "encode_batch_native"):
        pytest.skip("encode_batch_native not available in this build")
    return eng


@requires_qdp
@pytest.mark.gpu
@pytest.mark.parametrize("num_qubits", [12, 14, 16])
@pytest.mark.parametrize("encoding_method", ["iqp-z"])
def test_native_e2e_encode_dlpack_forward(engine, num_qubits, encoding_method):
    """Native path: batch encode -> DLPack clone -> float32 consume (E2E core)."""
    batch_size = 32
    data_len = _iqp_param_count(num_qubits, encoding_method)
    data = torch.randn(batch_size, data_len, dtype=torch.float64).numpy()

    model = _DummyQNN(num_qubits).cuda()
    torch.cuda.synchronize()

    qtensor = engine.encode_batch_native(data, num_qubits, encoding_method)
    gpu_batched = torch.from_dlpack(qtensor).clone()

    state_len = 1 << num_qubits
    assert gpu_batched.shape == (batch_size, state_len)

    gpu_all_data = gpu_batched.abs().to(torch.float32)
    for i in range(0, batch_size, 8):
        _ = model(gpu_all_data[i : i + 8])

    torch.cuda.synchronize()
    assert torch.isfinite(gpu_batched).all()


@requires_qdp
@pytest.mark.gpu
@pytest.mark.parametrize("num_qubits", [12, 14, 16])
def test_native_e2e_matches_fwt_and_tc(engine, num_qubits):
    """E2E tensor outputs: Native vs FWT and Native vs TC must agree."""
    batch_size = 32
    encoding_method = "iqp-z"
    data_len = _iqp_param_count(num_qubits, encoding_method)
    data = torch.randn(batch_size, data_len, dtype=torch.float64).numpy()

    native = torch.from_dlpack(
        engine.encode_batch_native(data, num_qubits, encoding_method)
    ).clone()
    fwt = torch.from_dlpack(
        engine.encode(data, num_qubits, encoding_method)
    ).clone()

    native_fwt_err = (native - fwt).abs().max().item()
    assert native_fwt_err < 1e-5, f"Native vs FWT E2E err {native_fwt_err} at N={num_qubits}"

    if hasattr(engine, "encode_batch_tc"):
        tc = torch.from_dlpack(
            engine.encode_batch_tc(data, num_qubits, encoding_method)
        ).clone()
        native_tc_err = (native - tc).abs().max().item()
        assert native_tc_err < 1e-5, f"Native vs TC E2E err {native_tc_err} at N={num_qubits}"


@requires_qdp
@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("num_qubits", [12, 16])
def test_benchmark_e2e_script_native_path(num_qubits):
    """Run benchmark_e2e.py mahout-native path (full disk->GPU pipeline)."""
    script = BENCHMARK_DIR / "benchmark_e2e.py"
    if not script.is_file():
        pytest.skip(f"benchmark script not found: {script}")

    cmd = [
        sys.executable,
        str(script),
        "--qubits",
        str(num_qubits),
        "--samples",
        "32",
        "--encoding-method",
        "iqp-z",
        "--frameworks",
        "mahout-native",
        "mahout-tc",
        "mahout-arrow",
    ]
    result = subprocess.run(
        cmd,
        cwd=str(BENCHMARK_DIR),
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, (
        f"benchmark_e2e failed (exit {result.returncode})\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "Mahout-Native (PR8)" in result.stdout
    assert "SUCCESS: Quantum States Match!" in result.stdout