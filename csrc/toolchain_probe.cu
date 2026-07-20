// Minimal CUTLASS+CUDA toolchain probe for GB10 sm_121 (sm_120 family).
// Confirms: nvcc builds for the target arch, CUTLASS 4.x sub-byte FP4 (E2M1)
// types + numeric headers compile, and a trivial device kernel links.
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/float_subbyte.h>
#include <cutlass/version.h>
#include <cuda_runtime.h>
#include <cstdio>

using fp4_t = cutlass::float_e2m1_t;   // NVFP4 code type

__global__ void probe_kernel(int* out) {
  // exercise the sub-byte type on-device (compile + a real store)
  fp4_t a = fp4_t(1.5f);
  float f = float(a);
  out[threadIdx.x] = (f == 1.5f) ? 1 : 0;
}

int main() {
  int* d; cudaMalloc(&d, 32 * sizeof(int));
  probe_kernel<<<1, 32>>>(d);
  cudaError_t e = cudaDeviceSynchronize();
  int h[32]; cudaMemcpy(h, d, 32 * sizeof(int), cudaMemcpyDeviceToHost);
  printf("CUTLASS_VERSION=%d.%d.%d  fp4(1.5)->%s  sync=%s\n",
         CUTLASS_MAJOR, CUTLASS_MINOR, CUTLASS_PATCH,
         h[0] ? "ok" : "MISMATCH", cudaGetErrorString(e));
  cudaFree(d);
  return e == cudaSuccess ? 0 : 1;
}
