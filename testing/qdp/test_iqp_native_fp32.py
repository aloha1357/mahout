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

"""PR9c/PR9d: FP32 native IQP encode tests (fused N<=12, Kronecker N>12)."""

import pytest
import torch
from qumat_qdp import QdpEngine

from .qdp_test_utils import requires_qdp


def _iqp_param_count(num_qubits: int, encoding_method: str) -> int:
    if encoding_method == "iqp-z":
        return num_qubits
    return num_qubits + num_qubits * (num_qubits - 1) // 2


@pytest.fixture(scope="module")
def engine_f32():
    pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    try:
        eng = QdpEngine(device_id=0, precision="float32")
    except Exception as exc:
        pytest.skip(f"Could not initialize QdpEngine: {exc}")
    if not hasattr(eng, "encode_batch_native"):
        pytest.skip("encode_batch_native not available in this build")
    return eng


@pytest.fixture(scope="module")
def engine_f64():
    pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return QdpEngine(device_id=0, precision="float64")


@requires_qdp
@pytest.mark.gpu
@pytest.mark.parametrize("num_qubits", [8, 12])
@pytest.mark.parametrize("encoding_method", ["iqp-z"])
def test_native_fp32_matches_fp64_fused(
    engine_f32, engine_f64, num_qubits, encoding_method
):
    batch_size = 16
    data_len = _iqp_param_count(num_qubits, encoding_method)
    data_f32 = torch.randn(batch_size, data_len, dtype=torch.float32).numpy()
    data_f64 = data_f32.astype("float64")

    out_f32 = torch.from_dlpack(
        engine_f32.encode_batch_native(data_f32, num_qubits, encoding_method)
    ).clone()
    out_f64 = torch.from_dlpack(
        engine_f64.encode_batch_native(data_f64, num_qubits, encoding_method)
    ).clone()

    assert out_f32.shape == (batch_size, 1 << num_qubits)
    assert torch.isfinite(out_f32).all()

    err = (
        (out_f32.to(torch.complex128) - out_f64.to(torch.complex128)).abs().max().item()
    )
    assert err < 1e-4, f"FP32 vs FP64 native err {err} at N={num_qubits}"


@requires_qdp
@pytest.mark.gpu
@pytest.mark.parametrize("num_qubits", [14, 16])
def test_native_fp32_kronecker_matches_fp64(engine_f32, engine_f64, num_qubits):
    """N>12 uses fused-transpose Kronecker (PR9d); compare against fp64 reference."""
    batch_size = 8
    data_f32 = torch.randn(batch_size, num_qubits, dtype=torch.float32).numpy()
    data_f64 = data_f32.astype("float64")

    out_f32 = torch.from_dlpack(
        engine_f32.encode_batch_native(data_f32, num_qubits, "iqp-z")
    ).clone()
    out_f64 = torch.from_dlpack(
        engine_f64.encode_batch_native(data_f64, num_qubits, "iqp-z")
    ).clone()

    assert out_f32.shape == (batch_size, 1 << num_qubits)
    assert torch.isfinite(out_f32).all()

    err = (
        (out_f32.to(torch.complex128) - out_f64.to(torch.complex128)).abs().max().item()
    )
    assert err < 1e-4, f"FP32 vs FP64 Kronecker err {err} at N={num_qubits}"
