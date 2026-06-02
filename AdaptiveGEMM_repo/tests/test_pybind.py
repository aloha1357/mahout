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

import numpy as np
import pytest
import adaptive_gemm_py

def test_gpu_gemm_accuracy():
    np.random.seed(42)
    n = 256
    
    # Generate random double precision matrices
    A = np.random.uniform(-1.0, 1.0, (n, n)).astype(np.float64)
    B = np.random.uniform(-1.0, 1.0, (n, n)).astype(np.float64)

    # Reference computation using NumPy
    C_ref = A @ B

    # Computation using our custom AdaptiveGEMM GPU Engine
    engine = adaptive_gemm_py.AdaptiveGEMM()
    C_gpu = engine.gemm(A, B)

    # Verify maximum error is within expected tolerance
    np.testing.assert_allclose(C_gpu, C_ref, rtol=1e-5, atol=1e-8, err_msg="GPU GEMM result differs from NumPy reference")

if __name__ == "__main__":
    pytest.main([__file__])
