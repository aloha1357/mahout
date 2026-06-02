// Stub for GPUs with CC < 8.0 (e.g. Colab T4): Ozaki m16n8k32 MMA is not used on the SIMT path (N<=12).
// If launch_iqp_encode_tc is called with N>12, we fail with a clear message instead of link errors.
#include "ImplicitHadamardOzaki.h"
#include <cstdio>
#include <cstdlib>

namespace ozaki {

void ImplicitHadamardOzakiEngine::execute_implicit_hadamard(
    const double*, double*, int, int, int, double, cudaStream_t) {
    std::fprintf(stderr,
        "ImplicitHadamardOzaki (Tensor Core m16n8k32) requires GPU compute capability >= 8.0.\n"
        "This build targets SIMT path (N<=12). For PR007 N=14/16 use A100/L4 or a CC>=8 GPU.\n");
    std::exit(1);
}

} // namespace ozaki