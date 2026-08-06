// Minimal HIP GEMM using rocWMMA - probes FP8 WMMA on gfx1201 (RDNA4)
// Compiles with: hipcc --offload-arch=gfx1201 gemm_wmma_hip.cpp -o gemm_wmma --std=c++17 -lrocwmma ?
// Actually rocwmma is header-only.
// Validates against CPU reference.

#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <hip/hip_fp8.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cstdlib>

using namespace rocwmma;

// FP8 OCP E4M3 -> float conversion uses HIP API on host
// For CPU reference we use the same HIP conversion but also a manual table for cross-check
float fp8_e4m3_to_float_host(uint8_t bits){
    // Use HIP's conversion: construct hip_fp8_e4m3 from bits via reinterpret
    hip_fp8_e4m3 v;
    // HIP stores as __x; we can set via union
    union { uint8_t u; hip_fp8_e4m3 f; } u;
    u.u = bits;
    // But hip_fp8_e4m3 doesn't have from_bits constructor; use storage directly
    // Hack: copy bits into __x member (public)
    // Check struct layout: __hip_fp8_storage_t __x is first member, so memcpy works
    // Safer: use reinterpret
    hip_fp8_e4m3 fp8;
    fp8.__x = bits;
    return (float)fp8;
}
uint8_t float_to_fp8_e4m3_host(float f){
    hip_fp8_e4m3 fp8(f);
    return fp8.__x;
}

// WMMA kernel for FP8 E4M3, 16x16x16, row_major A, row_major B
// A: MxK row_major, B: KxN row_major, C: MxN row_major float
template <bool B_COL_MAJOR>
__global__ void wmma_fp8_kernel(const hip_fp8_e4m3* A, const hip_fp8_e4m3* B, float* C, int M, int N, int K){
    // 2D grid: blockIdx.x = tileN, blockIdx.y = tileM
    int tileM = blockIdx.y;
    int tileN = blockIdx.x;
    // Each block has one wave (32 threads) on RDNA4
    // Use 16x16x16 WMMA fragments
    fragment<matrix_a, 16,16,16, hip_fp8_e4m3, row_major> fragA;
    // B layout selectable
    using FragBType = fragment<matrix_b, 16,16,16, hip_fp8_e4m3, row_major>;
    using FragBColType = fragment<matrix_b, 16,16,16, hip_fp8_e4m3, col_major>;
    fragment<accumulator, 16,16,16, float> fragC;
    fill_fragment(fragC, 0.0f);

    // Loop over K tiles
    for(int k0=0;k0<K;k0+=16){
        // Load A tile: (tileM*16, k0) -> pointer A + tileM*16*K + k0
        const hip_fp8_e4m3* a_ptr = A + tileM*16*K + k0;
        load_matrix_sync(fragA, a_ptr, K);
        if constexpr (B_COL_MAJOR) {
            // For col_major B, we need B stored col_major.
            // If B is KxN row_major original, its transpose is NxK col_major equivalent?
            // To test, we store B row_major normally but interpret as col_major with ldm=K
            // This is confusing - we try both interpretations and let host harness decide.
            // Here we assume B is stored col_major (N*K) where element (k,n) at n*K + k
            const hip_fp8_e4m3* b_ptr = B + tileN*16*K + k0; // if B col_major KxN, stride K
            FragBColType fragB;
            load_matrix_sync(fragB, b_ptr, K);
            mma_sync(fragC, fragA, fragB, fragC);
        } else {
            const hip_fp8_e4m3* b_ptr = B + k0*N + tileN*16;
            FragBType fragB;
            load_matrix_sync(fragB, b_ptr, N);
            mma_sync(fragC, fragA, fragB, fragC);
        }
    }
    float* c_ptr = C + tileM*16*N + tileN*16;
    store_matrix_sync(c_ptr, fragC, N, mem_row_major);
}

// FP16 WMMA kernel (fallback)
__global__ void wmma_fp16_kernel(const half* A, const half* B, float* C, int M, int N, int K){
    int tileM = blockIdx.y;
    int tileN = blockIdx.x;
    fragment<matrix_a, 16,16,16, half, row_major> fragA;
    fragment<matrix_b, 16,16,16, half, row_major> fragB;
    fragment<accumulator, 16,16,16, float> fragC;
    fill_fragment(fragC, 0.0f);
    for(int k0=0;k0<K;k0+=16){
        const half* a_ptr = A + tileM*16*K + k0;
        const half* b_ptr = B + k0*N + tileN*16;
        load_matrix_sync(fragA, a_ptr, K);
        load_matrix_sync(fragB, b_ptr, N);
        mma_sync(fragC, fragA, fragB, fragC);
    }
    float* c_ptr = C + tileM*16*N + tileN*16;
    store_matrix_sync(c_ptr, fragC, N, mem_row_major);
}

// Naive FP32 for reference after quantizing to FP8
void cpu_gemm_fp8_reference(const std::vector<uint8_t>& A_bits, const std::vector<uint8_t>& B_bits, std::vector<float>& C, int M,int N,int K){
    // Convert bits to float
    std::vector<float> A_f(M*K), B_f(K*N);
    for(int i=0;i<M*K;i++) A_f[i]=fp8_e4m3_to_float_host(A_bits[i]);
    for(int i=0;i<K*N;i++) B_f[i]=fp8_e4m3_to_float_host(B_bits[i]);
    for(int m=0;m<M;m++){
        for(int n=0;n<N;n++){
            float acc=0;
            for(int k=0;k<K;k++) acc += A_f[m*K+k] * B_f[k*N+n];
            C[m*N+n]=acc;
        }
    }
}

bool test_fp8(int M,int N,int K, bool col_major){
    std::cout << "=== test_fp8 M="<<M<<" N="<<N<<" K="<<K<<" col_major_B="<<col_major<<" ==="<<std::endl;
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> val_dist(0,8);
    // Values that are exact E4M3: -2,-1.5,-1,-0.5,0,0.5,1,1.5,2
    float vals[9]={-2.f,-1.5f,-1.f,-0.5f,0.f,0.5f,1.f,1.5f,2.f};
    std::vector<float> A_f(M*K), B_f(K*N);
    for(int i=0;i<M*K;i++) A_f[i]=vals[val_dist(rng)%9];
    for(int i=0;i<K*N;i++) B_f[i]=vals[val_dist(rng)%9];
    // Quantize to FP8 bits
    std::vector<uint8_t> A_bits(M*K), B_bits(K*N);
    for(int i=0;i<M*K;i++) A_bits[i]=float_to_fp8_e4m3_host(A_f[i]);
    for(int i=0;i<K*N;i++) B_bits[i]=float_to_fp8_e4m3_host(B_f[i]);
    // Also create device buffers as hip_fp8_e4m3
    std::vector<hip_fp8_e4m3> A_fp8(M*K), B_fp8(K*N);
    for(int i=0;i<M*K;i++) A_fp8[i].__x = A_bits[i];
    for(int i=0;i<K*N;i++) {
        // For col_major test, we need to store B transposed? Let's handle:
        // If testing col_major, we will store B in col_major layout for kernel.
        // But cpu reference always uses row_major. So we need to create col_major device copy if needed.
        // For now host B_fp8 row_major. Device will reinterpret if col_major kernel uses col_major stride.
        // To make both comparable, we prepare two device layouts and test each kernel separately.
        B_fp8[i].__x = B_bits[i];
    }
    std::vector<hip_fp8_e4m3> B_fp8_col(K*N);
    if(col_major){
        // Convert row_major B to col_major storage: B_col[n*K + k] = B_row[k*N + n]
        for(int k=0;k<K;k++) for(int n=0;n<N;n++) B_fp8_col[n*K+k].__x = B_bits[k*N+n];
    }

    std::vector<float> C_ref(M*N);
    cpu_gemm_fp8_reference(A_bits, B_bits, C_ref, M,N,K);

    // Device allocation
    hip_fp8_e4m3 *dA,*dB;
    float *dC;
    hipMalloc(&dA, M*K*sizeof(hip_fp8_e4m3));
    hipMalloc(&dB, K*N*sizeof(hip_fp8_e4m3));
    hipMalloc(&dC, M*N*sizeof(float));
    hipMemcpy(dA, A_fp8.data(), M*K*sizeof(hip_fp8_e4m3), hipMemcpyHostToDevice);
    if(col_major) hipMemcpy(dB, B_fp8_col.data(), K*N*sizeof(hip_fp8_e4m3), hipMemcpyHostToDevice);
    else hipMemcpy(dB, B_fp8.data(), K*N*sizeof(hip_fp8_e4m3), hipMemcpyHostToDevice);
    hipMemset(dC,0,M*N*sizeof(float));

    dim3 grid(N/16, M/16);
    dim3 block(32); // one wave
    hipLaunchKernelGGL((col_major? wmma_fp8_kernel<true> : wmma_fp8_kernel<false>), grid, block, 0,0, dA, dB, dC, M,N,K);
    hipError_t err = hipGetLastError();
    if(err!=hipSuccess){ std::cout<<"launch error: "<<hipGetErrorString(err)<<std::endl; return false; }
    hipDeviceSynchronize();
    err = hipGetLastError();
    if(err!=hipSuccess){ std::cout<<"sync error: "<<hipGetErrorString(err)<<std::endl; return false; }

    std::vector<float> C_dev(M*N);
    hipMemcpy(C_dev.data(), dC, M*N*sizeof(float), hipMemcpyDeviceToHost);
    hipFree(dA); hipFree(dB); hipFree(dC);

    // Compare
    float max_abs=0, max_rel=0, sum_sq=0, ref_sum_sq=0;
    int mismatches=0;
    for(int i=0;i<M*N;i++){
        float diff = std::abs(C_dev[i]-C_ref[i]);
        max_abs = std::max(max_abs, diff);
        float denom = std::max(std::abs(C_ref[i]), 1e-6f);
        max_rel = std::max(max_rel, diff/denom);
        sum_sq += diff*diff;
        ref_sum_sq += C_ref[i]*C_ref[i];
        if(diff>1e-3) mismatches++;
    }
    float rms = std::sqrt(sum_sq/(M*N));
    float rel_rms = rms / std::sqrt(ref_sum_sq/(M*N) + 1e-12);
    std::cout<<" max_abs="<<max_abs<<" max_rel="<<max_rel<<" rms="<<rms<<" rel_rms="<<rel_rms<<" mismatches>1e-3: "<<mismatches<<"/"<<M*N<<std::endl;
    // Print first few
    for(int i=0;i<std::min(4,M*N);i++) std::cout<<"  C["<<i<<"] dev="<<C_dev[i]<<" ref="<<C_ref[i]<<" diff="<<C_dev[i]-C_ref[i]<<std::endl;
    bool pass = max_abs < 1e-2 && rms < 1e-3;
    std::cout<<(pass?"PASS":"FAIL")<<std::endl;
    return pass;
}

bool test_fp16(int M,int N,int K){
    std::cout << "=== test_fp16 M="<<M<<" N="<<N<<" K="<<K<<" ==="<<std::endl;
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1,1);
    std::vector<half> A(M*K), B(K*N);
    std::vector<float> A_f(M*K), B_f(K*N);
    for(int i=0;i<M*K;i++){ float v=dist(rng); A_f[i]=v; A[i]=__float2half(v); }
    for(int i=0;i<K*N;i++){ float v=dist(rng); B_f[i]=v; B[i]=__float2half(v); }
    std::vector<float> C_ref(M*N,0);
    for(int m=0;m<M;m++) for(int n=0;n<N;n++){ float acc=0; for(int k=0;k<K;k++) acc+=(float)A[m*K+k]*(float)B[k*N+n]; C_ref[m*N+n]=acc; }

    half *dA,*dB; float *dC;
    hipMalloc(&dA,M*K*sizeof(half));
    hipMalloc(&dB,K*N*sizeof(half));
    hipMalloc(&dC,M*N*sizeof(float));
    hipMemcpy(dA,A.data(),M*K*sizeof(half),hipMemcpyHostToDevice);
    hipMemcpy(dB,B.data(),K*N*sizeof(half),hipMemcpyHostToDevice);
    hipMemset(dC,0,M*N*sizeof(float));
    dim3 grid(N/16,M/16);
    dim3 block(32);
    hipLaunchKernelGGL(wmma_fp16_kernel, grid, block,0,0,dA,dB,dC,M,N,K);
    hipDeviceSynchronize();
    std::vector<float> C_dev(M*N);
    hipMemcpy(C_dev.data(),dC,M*N*sizeof(float),hipMemcpyDeviceToHost);
    hipFree(dA);hipFree(dB);hipFree(dC);
    float max_abs=0;
    for(int i=0;i<M*N;i++) max_abs=std::max(max_abs,std::abs(C_dev[i]-C_ref[i]));
    std::cout<<" max_abs="<<max_abs<<std::endl;
    bool pass = max_abs < 0.05;
    std::cout<<(pass?"PASS":"FAIL")<<std::endl;
    return pass;
}

int main(){
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop,0);
    std::cout<<"Device: "<<prop.name<<" gfx"<<prop.gcnArchName<<" warpSize "<<prop.warpSize<<std::endl;

    bool ok_row = test_fp8(16,16,16,false);
    bool ok_col = test_fp8(16,16,16,true);
    bool ok32_row = test_fp8(32,32,32,false);
    bool ok32_col = test_fp8(32,32,32,true);
    bool ok_fp16 = test_fp16(16,16,16);

    std::cout<<"\nSummary:\n";
    std::cout<<" FP8 row_major 16: "<<(ok_row?"PASS":"FAIL")<<"\n";
    std::cout<<" FP8 col_major 16: "<<(ok_col?"PASS":"FAIL")<<"\n";
    std::cout<<" FP8 row_major 32: "<<(ok32_row?"PASS":"FAIL")<<"\n";
    std::cout<<" FP8 col_major 32: "<<(ok32_col?"PASS":"FAIL")<<"\n";
    std::cout<<" FP16 16: "<<(ok_fp16?"PASS":"FAIL")<<"\n";

    // Choose best path
    if(ok_row||ok_col) std::cout<<"FP8 WMMA is functional (best path=FP8 WMMA)\n";
    else if(ok_fp16) std::cout<<"Fallback to FP16 WMMA with FP8 storage\n";
    else std::cout<<"No WMMA path validated - need scalar fallback\n";

    return 0;
}
