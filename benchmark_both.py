import torch
import time
import subprocess
import re

def generate_hadamard(n_qubits, device='cuda'):
    H1 = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.float64, device=device)
    H = H1
    for _ in range(n_qubits - 1):
        H = torch.kron(H, H1)
    return H

def matmul_kernel(A, H):
    return torch.matmul(A, H)

compiled_matmul = torch.compile(matmul_kernel)

def benchmark_pytorch(n_qubits, batch_size, use_compile=False):
    state_len = 2**n_qubits
    try:
        A = torch.randn(batch_size, state_len, dtype=torch.float64, device='cuda')
        H = generate_hadamard(n_qubits, device='cuda')
    except RuntimeError as e:
        return "OOM"

    kernel = compiled_matmul if use_compile else matmul_kernel

    try:
        for _ in range(3):
            _ = kernel(A, H)
        torch.cuda.synchronize()
    except Exception:
        return "Failed/OOM"

    start_time = time.perf_counter()
    iters = 10
    try:
        for _ in range(iters):
            _ = kernel(A, H)
        torch.cuda.synchronize()
        avg_time_ms = ((time.perf_counter() - start_time) / iters) * 1000
        del A, H
        torch.cuda.empty_cache()
        return f"{avg_time_ms:.2f} ms"
    except Exception:
        return "OOM"

def benchmark_qdp(n_qubits):
    try:
        result = subprocess.run([r"qdp\qdp-kernels\build\bench_kernels.exe", str(n_qubits)], capture_output=True, text=True)
        match = re.search(r"Duration:\s+(\d+)\s+us", result.stdout)
        if match:
            ms = float(match.group(1)) / 1000.0
            return f"{ms:.2f} ms"
        else:
            return "Error parsing"
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    batch = 128
    print("=" * 75)
    print(f"{'Qubits':<8} | {'PyTorch Eager':<18} | {'PyTorch Compiled':<18} | {'Our QDP TC'}")
    print("-" * 75)
    
    for n in [10, 12, 14, 15, 16]:
        qdp_tc = benchmark_qdp(n)
        pt_eager = benchmark_pytorch(n, batch, use_compile=False)
        pt_comp = benchmark_pytorch(n, batch, use_compile=True)
        
        print(f"{n:<8} | {pt_eager:<18} | {pt_comp:<18} | {qdp_tc}")
    print("=" * 75)
