# Report: Final Benchmark and Architectural Summary

## Goal
Summarize all findings across the AdaptiveGEMM integration.

## Hypothesis
We successfully transitioned from memory-bound dense operations to compute-bound, scalable tensor operations.

## Final Implementation Metrics
- **Performance**:
  - N=14: 65us (47.2x speedup vs dense baseline).
  - N=16: ~74us to ~103us (up to 375.9x speedup vs optimized SIMT baseline).
  - PyTorch Eager/Compiled baseline OOMs at N=16 on 32GB systems.
- **Precision**: Exact match via Ozaki decomposition, overcoming standard FP16/TF32 truncation limits.

## NCU Analysis
- **SM Throughput Final**: 67.54%
- **DRAM Throughput Final**: 0.45%
- *Conclusion: A textbook shift from Memory-Bound to Compute-Bound execution.*

## Known Limitations
- Full Ozaki Tensor Core profiling (`sm__pipe_tensor_active`) fails under Windows WDDM (requires Linux TCC mode).
- Hardware targeting requires SM80+ for TF32; fallback logic is mandatory (currently throws compiler errors on SM75).

## Multi-GPU Roadmap
Future work must focus on multi-GPU synchronization and scaling to bypass single-node VRAM limits for $N > 32$.