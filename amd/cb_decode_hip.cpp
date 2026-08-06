// FP8-CB decode + GEMM lane for gfx1201 (RDNA4 WMMA)
// Validates decode bit-exact against Python reference (tests/cb_torch_reference.py)
// Dense lane only (N,K multiples of 256). Supports product n_sub=4, generic k.
// Compile: hipcc --offload-arch=gfx1201 cb_decode_hip.cpp -o cb_decode_hip --std=c++17 -O2

#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <hip/hip_fp8.h>
#include <iostream>
#include <vector>
#include <random>
#include <cstdint>
#include <cmath>
#include <cassert>

using namespace rocwmma;

// ---- CPU reference helpers mirroring tests/cb_torch_reference.py ----
std::vector<int> split_widths(int k_bits, int n_sub){
    if(n_sub!=1 && n_sub!=2 && n_sub!=4) throw std::runtime_error("n_sub");
    int base = k_bits / n_sub;
    int extra = k_bits % n_sub;
    std::vector<int> w(n_sub);
    for(int i=0;i<n_sub;i++) w[i]= base + (i<extra?1:0);
    return w;
}

// Extract codewords: [N, n_sb, 32]  k-bit ints, LSB-first
// qw_padded: [N, row_bytes] uint8, row_bytes = n_sb*type_size (4*k)
std::vector<int> cpu_extract_codewords(const std::vector<uint8_t>& qw, int N, int K, int k_bits, int type_size){
    int n_sb = K/256;
    int row_bytes = n_sb*type_size;
    // Pad 8 zeros per row for window
    std::vector<int> codes(N * n_sb * 32, 0);
    for(int n=0;n<N;n++){
        for(int sb=0;sb<n_sb;sb++){
            for(int v=0;v<32;v++){
                int byte_base = (v*k_bits)/8;
                int bit_shift = (v*k_bits)%8;
                int sb_base = sb*type_size;
                int start = sb_base + byte_base;
                uint64_t window=0;
                for(int b=0;b<8;b++){
                    int idx = start + b;
                    uint8_t byte = 0;
                    if(idx < type_size) byte = qw[n*row_bytes + sb*type_size + byte_base + b]; // actually need row_bytes indexing
                    // Wait qw is flat [N*row_bytes] but we wrote indexing wrong: qw[n*row_bytes + sb*type_size + byte_offset]
                    // For window we need to handle cross-superblock? No, per superblock window only within that superblock's 4*k bytes.
                    // But the type_size includes only index bytes for FP8 (no scale), so 4*k.
                    // So for FP8, byte_base < 4*k, and window never crosses superblock boundary (since 4*k +7 < type_size for k<=48? 4*48=192, plus 7=199 <192? Actually 199>192, so could cross? But we have pad zeros for last bytes.
                    // Simpler: read from qw_padded with 8 zeros pad per row, but per superblock we pad individually.
                    // We'll just use the earlier logic: padded row has 8 extra zeros, and starts = sb_base + byte_base.
                    // So implement with padded buffer.
                }
                // placeholder
            }
        }
    }
    return codes;
}

// Full CPU reference using explicit padded row (like python)
struct CodewordMatrix {
    int N, n_sb;
    std::vector<uint64_t> data; // [N][n_sb][32]
    uint64_t at(int n,int sb,int v) const { return data[(n*n_sb+sb)*32+v]; }
};

CodewordMatrix extract_codewords_cpu(const std::vector<uint8_t>& qw_padded, int N,int K,int k_bits,int type_size){
    int n_sb = K/256;
    int row_bytes = n_sb*type_size;
    CodewordMatrix m{N,n_sb, std::vector<uint64_t>(N*n_sb*32,0)};
    // Build padded rows: raw + 8 zeros
    std::vector<uint8_t> padded(N*(row_bytes+8),0);
    for(int n=0;n<N;n++) for(int i=0;i<row_bytes;i++) padded[n*(row_bytes+8)+i]= qw_padded[n*row_bytes + i];
    for(int n=0;n<N;n++){
        for(int sb=0;sb<n_sb;sb++){
            int sb_base = sb*type_size;
            for(int v=0;v<32;v++){
                int byte_base = (v*k_bits)/8;
                int bit_shift = (v*k_bits)%8;
                int start = sb_base + byte_base;
                uint64_t window=0;
                for(int b=0;b<8;b++){
                    uint8_t byte = padded[n*(row_bytes+8) + start + b];
                    window |= (uint64_t)byte << (8*b);
                }
                uint64_t mask = (k_bits==64? ~0ULL : ((1ULL<<k_bits)-1));
                uint64_t code = (window >> bit_shift) & mask;
                m.data[(n*n_sb+sb)*32 + v] = code;
            }
        }
    }
    return m;
}

// CPU decode values (unscaled) -> [N,K] E4M3 bytes
std::vector<uint8_t> cpu_decode_values(const std::vector<uint8_t>& qw_padded,
                                        const std::vector<uint8_t>& cb_flat_fp8,
                                        const std::vector<int32_t>& row_offset,
                                        int N,int K,int k_bits,int n_sub,int type_size){
    auto codes = extract_codewords_cpu(qw_padded,N,K,k_bits,type_size);
    int n_sb = K/256;
    auto widths = split_widths(k_bits,n_sub);
    int sub_dim = 8 / n_sub;
    // table bases
    std::vector<int> table_base(n_sub,0);
    int cum=0;
    for(int i=0;i<n_sub;i++){ table_base[i]=cum; cum += (1<<widths[i])*sub_dim; }
    std::vector<uint8_t> out(N*K);
    for(int n=0;n<N;n++){
        int base = row_offset[n];
        for(int sb=0;sb<n_sb;sb++){
            for(int v=0;v<32;v++){
                uint64_t code = codes.at(n,sb,v);
                int bit_off=0;
                // For each coord j 0..7, determine sub and gather
                for(int j=0;j<8;j++){
                    int sub = j / sub_dim;
                    int local = j % sub_dim;
                    int width = widths[sub];
                    uint64_t sub_idx = (code >> bit_off + [&](){int s=0;for(int t=0;t<sub;t++) s+=widths[t]; return s;}()) & ((1ULL<<width)-1);
                    // compute bit offset correctly: sum widths[0..sub-1]
                    int off=0; for(int t=0;t<sub;t++) off+=widths[t];
                    sub_idx = (code >> off) & ((1ULL<<width)-1);
                    int idx = base + table_base[sub] + sub_idx*sub_dim + local;
                    uint8_t val = cb_flat_fp8[idx];
                    int col = sb*256 + v*8 + j;
                    out[n*K + col] = val;
                }
            }
        }
    }
    return out;
}

// HIP device helpers
__device__ inline void split_widths_device(int k_bits,int n_sub,int* out){
    int base = k_bits / n_sub;
    int extra = k_bits % n_sub;
    for(int i=0;i<n_sub;i++) out[i]= base + (i<extra?1:0);
}

// Decode kernel: unscaled FP8 bytes
__global__ void cb_decode_fp8_kernel(
    const uint8_t* __restrict__ qw, // [N, row_bytes]
    const uint8_t* __restrict__ cb_flat, // [flat_size]
    const int32_t* __restrict__ row_offset, // [N]
    uint8_t* __restrict__ out, // [N,K]
    int N,int K,int k_bits,int n_sub,int type_size,int row_bytes)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int total = N*K;
    if(idx>=total) return;
    int row = idx / K;
    int col = idx % K;
    int sb = col / 256;
    int col_in_sb = col % 256;
    int vec = col_in_sb / 8;
    int coord = col_in_sb % 8;

    int sub_dim = 8 / n_sub;
    int sub = coord / sub_dim;
    int local = coord % sub_dim;

    // widths
    int widths[4];
    split_widths_device(k_bits,n_sub,widths);
    int bit_off=0;
    for(int i=0;i<sub;i++) bit_off+=widths[i];
    int mask = (widths[sub]==32? 0xFFFFFFFF : ((1<<widths[sub])-1));

    // Extract codeword via 8-byte window LSB-first
    int byte_base = (vec * k_bits) / 8;
    int bit_shift = (vec * k_bits) % 8;
    uint64_t window=0;
    for(int b=0;b<8;b++){
        uint8_t byte = 0;
        if(byte_base + b < type_size) byte = qw[row*row_bytes + sb*type_size + byte_base + b];
        window |= (uint64_t)byte << (8*b);
    }
    uint64_t code = (window >> bit_shift) & (k_bits==64? ~0ULL : ((k_bits>=32? ((1ULL<<k_bits)-1) : ((1U<<k_bits)-1))));
    uint64_t sub_idx = (code >> bit_off) & mask;

    // table base
    int table_base=0;
    for(int i=0;i<sub;i++) table_base += (1<<widths[i])*sub_dim;
    int base = row_offset[row];
    int flat_idx = base + table_base + sub_idx*sub_dim + local;
    uint8_t val = cb_flat[flat_idx];
    out[idx]=val;
}

// Scaled decode to BF16 (for fallback GEMM)
__global__ void cb_decode_scaled_bf16_kernel(
    const uint8_t* __restrict__ qw,
    const uint8_t* __restrict__ cb_flat,
    const int32_t* __restrict__ row_offset,
    const float* __restrict__ weight_scale, // [N]
    hip_bfloat16* __restrict__ out, // [N,K] bf16 scaled
    int N,int K,int k_bits,int n_sub,int type_size,int row_bytes)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int total = N*K;
    if(idx>=total) return;
    int row = idx / K;
    int col = idx % K;
    int sb = col / 256;
    int col_in_sb = col % 256;
    int vec = col_in_sb / 8;
    int coord = col_in_sb % 8;
    int sub_dim = 8 / n_sub;
    int sub = coord / sub_dim;
    int local = coord % sub_dim;
    int widths[4];
    split_widths_device(k_bits,n_sub,widths);
    int bit_off=0;
    for(int i=0;i<sub;i++) bit_off+=widths[i];
    int mask = ((1<<widths[sub])-1);
    int byte_base = (vec * k_bits) / 8;
    int bit_shift = (vec * k_bits) % 8;
    uint64_t window=0;
    for(int b=0;b<8;b++){
        uint8_t byte = 0;
        if(byte_base + b < type_size) byte = qw[row*row_bytes + sb*type_size + byte_base + b];
        window |= (uint64_t)byte << (8*b);
    }
    uint64_t code = (window >> bit_shift) & ((1ULL<<k_bits)-1);
    uint64_t sub_idx = (code >> bit_off) & mask;
    int table_base=0;
    for(int i=0;i<sub;i++) table_base += (1<<widths[i])*sub_dim;
    int flat_idx = row_offset[row] + table_base + sub_idx*sub_dim + local;
    uint8_t cb_byte = cb_flat[flat_idx];
    hip_fp8_e4m3 fp8; fp8.__x = cb_byte;
    float f = (float)fp8;
    float scaled = f * weight_scale[row];
    out[idx] = hip_bfloat16(scaled);
}

// GEMM kernel for CB weight: A MxK (half/bf16) * W NxK (FP8 decoded) -> C MxN (float), with per-row scale after GEMM
// W is NxK row_major, reinterpreted as B KxN col_major with ld=K
// Handles arbitrary M,N (including small N like 2,4,8) via padded tile + masked store.
// K is assumed multiple of 16 (256-aligned superblocks guarantee it).
template<typename A_T>
__global__ void wmma_gemm_cb_fp8(
    const A_T* __restrict__ A, // MxK row_major
    const hip_fp8_e4m3* __restrict__ W, // NxK row_major (decoded FP8)
    const float* __restrict__ weight_scale, // [N] (unused in tile; separate scale kernel)
    float* __restrict__ C, // MxN row_major float
    int M,int N,int K)
{
    int tileM = blockIdx.y;
    int tileN = blockIdx.x;
    int row_base = tileM*16;
    int col_base = tileN*16;
    // Shared padded tiles (LDS) for masked loads: 16*16 each
    __shared__ A_T tileA[256];
    __shared__ hip_fp8_e4m3 tileB[256]; // as col_major K_tile x N_tile? We'll store col_major 16x16 contiguous col_major
    // For B, rocWMMA col_major layout expects column-major contiguous with ldm = 16 (since we point to 16x16 tile)
    // So we pack tileB col_major: element (k_local, n_local) at n_local*16 + k_local
    fragment<matrix_a,16,16,16, A_T, row_major> fragA;
    fragment<matrix_b,16,16,16, hip_fp8_e4m3, col_major> fragB;
    fragment<accumulator,16,16,16, float> fragC;
    fill_fragment(fragC,0.0f);
    for(int k0=0;k0<K;k0+=16){
        // Fill tileA row_major 16x16 with zero pad
        for(int i=threadIdx.x; i<256; i+=32){
            int r = i / 16;
            int c = i % 16;
            int gr = row_base + r;
            int gk = k0 + c;
            A_T val{};
            if(gr < M && gk < K) val = A[gr*K + gk];
            // tileA stored row_major 16 stride 16
            tileA[r*16 + c] = val;
        }
        for(int i=threadIdx.x; i<256; i+=32){
            int k_local = i % 16;
            int n_local = i / 16;
            int gk = k0 + k_local;
            int gn = col_base + n_local;
            hip_fp8_e4m3 val; val.__x = 0; // zero
            if(gk < K && gn < N) val = W[gn*K + gk];
            // store col_major: column n_local is contiguous K=16
            tileB[n_local*16 + k_local] = val;
        }
        __syncthreads();
        load_matrix_sync(fragA, tileA, 16);
        load_matrix_sync(fragB, tileB, 16);
        mma_sync(fragC, fragA, fragB, fragC);
        __syncthreads();
    }
    // Masked store: we cannot store whole 16x16 if edge tile exceeds M/N.
    // Use scalar store after extracting via store_matrix_sync to temp then scatter? Simpler: store to shared then scatter.
    __shared__ float tileC[256];
    store_matrix_sync(tileC, fragC, 16, mem_row_major);
    __syncthreads();
    for(int i=threadIdx.x; i<256; i+=32){
        int r = i / 16;
        int c = i % 16;
        int gr = row_base + r;
        int gc = col_base + c;
        if(gr < M && gc < N){
            C[gr*N + gc] = tileC[r*16 + c];
        }
    }
}

__global__ void scale_rows_kernel(float* C, const float* scale, int M,int N){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int total = M*N;
    if(idx>=total) return;
    int n = idx % N;
    C[idx] *= scale[n];
}

// Scalar fallback for arbitrary sizes (also used to validate small-N correctness without WMMA masking issues)
template<typename A_T>
__global__ void gemm_scalar_fp8(
    const A_T* __restrict__ A, // MxK
    const hip_fp8_e4m3* __restrict__ W, // NxK
    const float* __restrict__ scale,
    float* __restrict__ C,
    int M,int N,int K)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int total = M*N;
    if(idx>=total) return;
    int m = idx / N;
    int n = idx % N;
    float acc=0;
    for(int k=0;k<K;k++){
        float a = (float)A[m*K + k];
        hip_fp8_e4m3 w = W[n*K + k];
        float wf = (float)w;
        acc += a * wf;
    }
    C[idx] = acc * scale[n];
}

__global__ void gemm_scalar_bf16(
    const hip_bfloat16* __restrict__ A,
    const hip_bfloat16* __restrict__ W,
    float* __restrict__ C,
    int M,int N,int K)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx>=M*N) return;
    int m = idx / N;
    int n = idx % N;
    float acc=0;
    for(int k=0;k<K;k++) acc += (float)A[m*K+k] * (float)W[n*K+k];
    C[idx]=acc;
}

// BF16 decode + FP16 WMMA fallback: decode produces BF16 scaled weight, then GEMM via BF16 WMMA
// Masked version similar to FP8
__global__ void wmma_gemm_bf16_cb(
    const hip_bfloat16* __restrict__ A, // MxK
    const hip_bfloat16* __restrict__ W, // NxK decoded BF16 scaled
    float* __restrict__ C, // MxN
    int M,int N,int K)
{
    int tileM = blockIdx.y;
    int tileN = blockIdx.x;
    int row_base = tileM*16;
    int col_base = tileN*16;
    __shared__ bfloat16_t tileA[256];
    __shared__ bfloat16_t tileB[256];
    __shared__ float tileC[256];
    fragment<matrix_a,16,16,16, bfloat16_t, row_major> fragA;
    fragment<matrix_b,16,16,16, bfloat16_t, col_major> fragB;
    fragment<accumulator,16,16,16, float> fragC;
    fill_fragment(fragC,0.0f);
    for(int k0=0;k0<K;k0+=16){
        for(int i=threadIdx.x; i<256; i+=32){
            int r=i/16, c=i%16;
            int gr=row_base+r, gk=k0+c;
            bfloat16_t val; val.data=0;
            if(gr<M && gk<K) val = reinterpret_cast<const bfloat16_t*>(A)[gr*K+gk];
            tileA[r*16+c]=val;
        }
        for(int i=threadIdx.x; i<256; i+=32){
            int k_local=i%16, n_local=i/16;
            int gk=k0+k_local, gn=col_base+n_local;
            bfloat16_t val; val.data=0;
            if(gk<K && gn<N) val = reinterpret_cast<const bfloat16_t*>(W)[gn*K+gk];
            tileB[n_local*16+k_local]=val;
        }
        __syncthreads();
        load_matrix_sync(fragA, tileA, 16);
        load_matrix_sync(fragB, tileB, 16);
        mma_sync(fragC, fragA, fragB, fragC);
        __syncthreads();
    }
    store_matrix_sync(tileC, fragC, 16, mem_row_major);
    __syncthreads();
    for(int i=threadIdx.x;i<256;i+=32){
        int r=i/16,c=i%16;
        int gr=row_base+r, gc=col_base+c;
        if(gr<M && gc<N) C[gr*N+gc]=tileC[r*16+c];
    }
}

// Plain WMMA wrapper for bench (non-templated to avoid launch macro issues)
__global__ void wmma_plain_bench(
    const hip_fp8_e4m3* A,
    const hip_fp8_e4m3* W,
    const float* scale,
    float* C,
    int M,int N,int K)
{
    int tileM = blockIdx.y;
    int tileN = blockIdx.x;
    int row_base = tileM*16;
    int col_base = tileN*16;
    __shared__ hip_fp8_e4m3 tileA[256];
    __shared__ hip_fp8_e4m3 tileB[256];
    __shared__ float tileC[256];
    fragment<matrix_a,16,16,16, hip_fp8_e4m3, row_major> fragA;
    fragment<matrix_b,16,16,16, hip_fp8_e4m3, col_major> fragB;
    fragment<accumulator,16,16,16, float> fragC;
    fill_fragment(fragC,0.0f);
    for(int k0=0;k0<K;k0+=16){
        for(int i=threadIdx.x;i<256;i+=32){ int r=i/16,c=i%16; int gr=row_base+r,gk=k0+c; hip_fp8_e4m3 v; v.__x=0; if(gr<M&&gk<K) v=A[gr*K+gk]; tileA[r*16+c]=v; }
        for(int i=threadIdx.x;i<256;i+=32){ int kl=i%16,nl=i/16; int gk=k0+kl,gn=col_base+nl; hip_fp8_e4m3 v; v.__x=0; if(gk<K&&gn<N) v=W[gn*K+gk]; tileB[nl*16+kl]=v; }
        __syncthreads();
        load_matrix_sync(fragA,tileA,16);
        load_matrix_sync(fragB,tileB,16);
        mma_sync(fragC,fragA,fragB,fragC);
        __syncthreads();
    }
    store_matrix_sync(tileC,fragC,16,mem_row_major);
    __syncthreads();
    for(int i=threadIdx.x;i<256;i+=32){ int r=i/16,c=i%16; int gr=row_base+r,gc=col_base+c; if(gr<M&&gc<N) C[gr*N+gc]=tileC[r*16+c]; }
}

// Fused decode+GEMM: A MxK * decoded(W via cb) -> C MxN ; decodes FP8 tiles on the fly, no global W buffer.
// Dense lane: N,K multiples of 256 for row_bytes but tile masking still supported.
__global__ void wmma_gemm_cb_fused(
    const hip_fp8_e4m3* __restrict__ A, // MxK
    const uint8_t* __restrict__ qw, // [N, row_bytes]
    const uint8_t* __restrict__ cb_flat,
    const int32_t* __restrict__ row_offset,
    float* __restrict__ C,
    int M,int N,int K, int k_bits,int n_sub,int type_size,int row_bytes)
{
    int tileM = blockIdx.y;
    int tileN = blockIdx.x;
    int row_base = tileM*16;
    int col_base = tileN*16;
    __shared__ hip_fp8_e4m3 tileA_fp8[256];
    __shared__ hip_fp8_e4m3 tileB_fp8[256];
    __shared__ float tileC[256];
    fragment<matrix_a,16,16,16, hip_fp8_e4m3, row_major> fragA;
    fragment<matrix_b,16,16,16, hip_fp8_e4m3, col_major> fragB;
    fragment<accumulator,16,16,16, float> fragC;
    fill_fragment(fragC,0.0f);
    int sub_dim = 8 / n_sub;
    int widths[4];
    split_widths_device(k_bits,n_sub,widths);
    int table_base_arr[4];
    table_base_arr[0]=0;
    for(int i=1;i<n_sub;i++) table_base_arr[i]= table_base_arr[i-1] + (1<<widths[i-1])*sub_dim;
    for(int k0=0;k0<K;k0+=16){
        // A tile
        for(int i=threadIdx.x;i<256;i+=32){
            int r=i/16,c=i%16;
            int gr=row_base+r, gk=k0+c;
            hip_fp8_e4m3 val; val.__x=0;
            if(gr<M && gk<K) val = A[gr*K+gk];
            tileA_fp8[r*16+c]=val;
        }
        // B tile via decode on the fly
        for(int i=threadIdx.x;i<256;i+=32){
            int k_local=i%16;
            int n_local=i/16;
            int gk=k0+k_local;
            int gn=col_base+n_local;
            hip_fp8_e4m3 out; out.__x=0;
            if(gk<K && gn<N){
                int sb = gk / 256;
                int col_in_sb = gk % 256;
                int vec = col_in_sb / 8;
                int coord = col_in_sb % 8;
                int sub = coord / sub_dim;
                int local = coord % sub_dim;
                int bit_off=0; for(int t=0;t<sub;t++) bit_off+=widths[t];
                int mask = (widths[sub]==32? (int)0xFFFFFFFF : ((1<<widths[sub])-1));
                int byte_base = (vec * k_bits)/8;
                int bit_shift = (vec * k_bits)%8;
                uint64_t window=0;
                for(int b=0;b<8;b++){
                    uint8_t byte=0;
                    if(byte_base+b < type_size) byte = qw[gn*row_bytes + sb*type_size + byte_base + b];
                    window |= (uint64_t)byte << (8*b);
                }
                uint64_t code = (window >> bit_shift) & (k_bits==64? ~0ULL : ((1ULL<<k_bits)-1));
                uint64_t sub_idx = (code >> bit_off) & (uint64_t)mask;
                int flat_idx = row_offset[gn] + table_base_arr[sub] + (int)sub_idx*sub_dim + local;
                uint8_t cb = cb_flat[flat_idx];
                out.__x = cb;
            }
            tileB_fp8[n_local*16 + k_local]=out;
        }
        __syncthreads();
        load_matrix_sync(fragA, tileA_fp8, 16);
        load_matrix_sync(fragB, tileB_fp8, 16);
        mma_sync(fragC, fragA, fragB, fragC);
        __syncthreads();
    }
    store_matrix_sync(tileC, fragC, 16, mem_row_major);
    __syncthreads();
    for(int i=threadIdx.x;i<256;i+=32){
        int r=i/16,c=i%16;
        int gr=row_base+r, gc=col_base+c;
        if(gr<M && gc<N) C[gr*N+gc]=tileC[r*16+c];
    }
}

// Utility to generate random E4M3 codebook and packed weights
hip_fp8_e4m3 make_fp8(float f){ return hip_fp8_e4m3(f); }

bool test_decode_one(int N,int K,int k_bits,int n_sub){
    int n_sb = K/256;
    int type_size = 4*k_bits;
    int row_bytes = n_sb*type_size;
    // codebook flat size
    auto widths = split_widths(k_bits,n_sub);
    int sub_dim = 8/n_sub;
    int flat_size=0;
    for(int w:widths) flat_size += (1<<w)*sub_dim;
    std::mt19937 rng(0xCB00 + k_bits + N*1000 + K);
    std::uniform_int_distribution<int> byte_dist(0,255);
    // Generate E4M3-valid bytes (avoid NaN 0x7F/0xFF) by sampling from small exact grid
    // Use float values from set that are exactly on E4M3 and converting via HIP
    std::vector<uint8_t> cb_flat(flat_size);
    for(int i=0;i<flat_size;i++){
        // Random pick from exact E4M3 grid values: use int range and convert via HIP to ensure valid
        float vals[9]={-2.f,-1.5f,-1.f,-0.5f,0.f,0.5f,1.f,1.5f,2.f};
        float v = vals[rng()%9];
        // Add some exponent variation: scale by power of two within E4M3 range
        int exp = (rng()%4)-1; // -1..2
        v *= (exp>=0? (1<<exp) : 1.0f/(1<<-exp));
        // Clamp to E4M3 max 448
        if(v>448) v=448; if(v<-448) v=-448;
        hip_fp8_e4m3 e(v);
        uint8_t b = e.__x;
        if(b==0x7F || b==0xFF || b==0x7E) b=0; // avoid NaN/Inf edge if any
        cb_flat[i]=b;
    }
    // Ensure some values are distinct to catch errors
    std::vector<int32_t> row_offset(N,0);
    // For fused layers, offset would be non-zero but for dense test 0.

    std::vector<uint8_t> qw(N*row_bytes);
    for(int i=0;i<N*row_bytes;i++) qw[i]= (uint8_t)byte_dist(rng);

    std::vector<float> weight_scale(N);
    for(int i=0;i<N;i++) weight_scale[i]= 0.5f + (rng()%100)/100.0f;

    // CPU reference unscaled
    auto cpu_out = cpu_decode_values(qw, cb_flat, row_offset, N,K,k_bits,n_sub,type_size);

    // GPU decode
    uint8_t *d_qw,*d_cb,*d_out;
    int32_t *d_off; float *d_scale; hip_bfloat16 *d_out_bf16;
    hipMalloc(&d_qw, N*row_bytes);
    hipMalloc(&d_cb, flat_size);
    hipMalloc(&d_off, N*sizeof(int32_t));
    hipMalloc(&d_out, N*K);
    hipMalloc(&d_scale, N*sizeof(float));
    hipMalloc(&d_out_bf16, N*K*sizeof(hip_bfloat16));
    hipMemcpy(d_qw, qw.data(), N*row_bytes, hipMemcpyHostToDevice);
    hipMemcpy(d_cb, cb_flat.data(), flat_size, hipMemcpyHostToDevice);
    hipMemcpy(d_off, row_offset.data(), N*sizeof(int32_t), hipMemcpyHostToDevice);
    hipMemcpy(d_scale, weight_scale.data(), N*sizeof(float), hipMemcpyHostToDevice);

    int threads=256;
    int blocks=(N*K+threads-1)/threads;
    hipLaunchKernelGGL(cb_decode_fp8_kernel, dim3(blocks), dim3(threads),0,0, d_qw,d_cb,d_off,d_out, N,K,k_bits,n_sub,type_size,row_bytes);
    hipDeviceSynchronize();
    std::vector<uint8_t> gpu_out(N*K);
    hipMemcpy(gpu_out.data(), d_out, N*K, hipMemcpyDeviceToHost);
    bool ok=true;
    int mism=0;
    for(int i=0;i<N*K;i++) if(gpu_out[i]!=cpu_out[i]){ if(mism<5) std::cout<<" mismatch at "<<i<<" gpu "<<(int)gpu_out[i]<<" cpu "<<(int)cpu_out[i]<<" row "<<i/K<<" col "<<i%K<<std::endl; mism++; ok=false; }
    std::cout<<"Decode unscaled k"<<k_bits<<" N"<<N<<" K"<<K<<" : "<<(ok?"PASS":"FAIL")<<" mism "<<mism<<"/"<<N*K<<std::endl;

    // Test scaled BF16 decode bit-exact vs CPU scaled
    // CPU scaled: convert each cpu_out byte to float via HIP, multiply by scale, convert to bf16 bits
    std::vector<hip_bfloat16> cpu_scaled(N*K);
    for(int i=0;i<N*K;i++){
        int row = i / K;
        hip_fp8_e4m3 fp8; fp8.__x = cpu_out[i];
        float f = (float)fp8;
        float scaled = f * weight_scale[row];
        cpu_scaled[i] = hip_bfloat16(scaled);
    }
    hipLaunchKernelGGL(cb_decode_scaled_bf16_kernel, dim3(blocks), dim3(threads),0,0, d_qw,d_cb,d_off,d_scale, d_out_bf16, N,K,k_bits,n_sub,type_size,row_bytes);
    hipDeviceSynchronize();
    std::vector<hip_bfloat16> gpu_scaled(N*K);
    hipMemcpy(gpu_scaled.data(), d_out_bf16, N*K*sizeof(hip_bfloat16), hipMemcpyDeviceToHost);
    int mism2=0;
    for(int i=0;i<N*K;i++){
        if(gpu_scaled[i].data != cpu_scaled[i].data){
            if(mism2<5) std::cout<<" scaled mismatch at "<<i<<" gpu "<<std::hex<<gpu_scaled[i].data<<" cpu "<<cpu_scaled[i].data<<std::dec<<std::endl;
            mism2++;
            ok=false;
        }
    }
    std::cout<<"Decode scaled BF16 k"<<k_bits<<" : "<<(mism2==0?"PASS":"FAIL")<<" mism "<<mism2<<std::endl;

    // Now test fused GEMM: Use decoded W (unscaled FP8) + activation FP8
    int M=16;
    // Activation as FP8 E4M3 (per-token dynamic quant mocked as random)
    std::vector<hip_fp8_e4m3> A_host(M*K);
    std::uniform_real_distribution<float> fdist(-2,2);
    for(int i=0;i<M*K;i++){ float v=fdist(rng); A_host[i]= hip_fp8_e4m3(v); }
    std::vector<hip_fp8_e4m3> W_fp8(N*K);
    for(int i=0;i<N*K;i++) W_fp8[i].__x = cpu_out[i];
    // CPU reference GEMM: Y = A * W^T scaled
    std::vector<float> A_f(M*K), W_f(N*K);
    for(int i=0;i<M*K;i++) A_f[i]= (float)A_host[i];
    for(int i=0;i<N*K;i++){ hip_fp8_e4m3 fp8; fp8.__x=cpu_out[i]; W_f[i]=(float)fp8; }
    std::vector<float> Y_ref(M*N,0);
    for(int m=0;m<M;m++) for(int n=0;n<N;n++){ float acc=0; for(int k=0;k<K;k++) acc += A_f[m*K+k]*W_f[n*K+k]; Y_ref[m*N+n]=acc*weight_scale[n]; }

    // GPU GEMM via WMMA FP8*FP8
    hip_fp8_e4m3 *dA; hip_fp8_e4m3 *dW; float *dY;
    hipMalloc(&dA, M*K*sizeof(hip_fp8_e4m3));
    hipMalloc(&dW, N*K*sizeof(hip_fp8_e4m3));
    hipMalloc(&dY, M*N*sizeof(float));
    hipMemcpy(dA, A_host.data(), M*K*sizeof(hip_fp8_e4m3), hipMemcpyHostToDevice);
    hipMemcpy(dW, W_fp8.data(), N*K*sizeof(hip_fp8_e4m3), hipMemcpyHostToDevice);
    hipMemset(dY,0,M*N*sizeof(float));
    dim3 grid((N+15)/16, (M+15)/16);
    dim3 block(32);
    // Unscaled GEMM, then scale rows
    hipLaunchKernelGGL(wmma_gemm_cb_fp8<hip_fp8_e4m3>, grid, block,0,0, dA, dW, d_scale, dY, M,N,K);
    hipDeviceSynchronize();
    float* dY2; hipMalloc(&dY2, M*N*sizeof(float));
    hipMemcpy(dY2, dY, M*N*sizeof(float), hipMemcpyDeviceToDevice);
    int total = M*N;
    int sblocks=(total+255)/256;
    hipLaunchKernelGGL(scale_rows_kernel, dim3(sblocks), dim3(256),0,0, dY2, d_scale, M,N);
    hipDeviceSynchronize();
    std::vector<float> Y_gpu(M*N);
    hipMemcpy(Y_gpu.data(), dY2, M*N*sizeof(float), hipMemcpyDeviceToHost);
    float max_abs=0, max_rel=0;
    int mism3=0;
    for(int i=0;i<M*N;i++){
        float diff = std::abs(Y_gpu[i]-Y_ref[i]);
        max_abs = std::max(max_abs, diff);
        float denom = std::max(std::abs(Y_ref[i]),1e-6f);
        max_rel = std::max(max_rel, diff/denom);
        if(diff>1e-3) mism3++;
    }
    std::cout<<"GEMM FP8 WMMA M"<<M<<" N"<<N<<" K"<<K<<" k"<<k_bits<<" max_abs "<<max_abs<<" max_rel "<<max_rel<<" mism "<<mism3<<" : "<<(max_abs<1e-2?"PASS":"FAIL")<<std::endl;

    // Also test BF16 fallback path: decode scaled BF16 + BF16 activation via BF16 WMMA
    // Use same A as BF16, W_scaled as BF16
    std::vector<hip_bfloat16> A_bf16(M*K), W_bf16(N*K);
    for(int i=0;i<M*K;i++) A_bf16[i]= hip_bfloat16(A_f[i]);
    for(int i=0;i<N*K;i++) W_bf16[i]= cpu_scaled[i];
    std::vector<float> Y_ref_bf16(M*N,0);
    for(int m=0;m<M;m++) for(int n=0;n<N;n++){ float acc=0; for(int k=0;k<K;k++) acc += (float)A_bf16[m*K+k]*(float)W_bf16[n*K+k]; Y_ref_bf16[m*N+n]=acc; }
    hip_bfloat16 *dA_bf16, *dW_bf16; float *dY_bf16;
    hipMalloc(&dA_bf16, M*K*sizeof(hip_bfloat16));
    hipMalloc(&dW_bf16, N*K*sizeof(hip_bfloat16));
    hipMalloc(&dY_bf16, M*N*sizeof(float));
    hipMemcpy(dA_bf16, A_bf16.data(), M*K*sizeof(hip_bfloat16), hipMemcpyHostToDevice);
    hipMemcpy(dW_bf16, W_bf16.data(), N*K*sizeof(hip_bfloat16), hipMemcpyHostToDevice);
    hipMemset(dY_bf16,0,M*N*sizeof(float));
    hipLaunchKernelGGL(wmma_gemm_bf16_cb, grid, block,0,0, dA_bf16, dW_bf16, dY_bf16, M,N,K);
    hipDeviceSynchronize();
    std::vector<float> Y_gpu_bf16(M*N);
    hipMemcpy(Y_gpu_bf16.data(), dY_bf16, M*N*sizeof(float), hipMemcpyDeviceToHost);
    float max_abs_bf16=0;
    int mism_bf16=0;
    for(int i=0;i<M*N;i++){ float diff=std::abs(Y_gpu_bf16[i]-Y_ref_bf16[i]); max_abs_bf16=std::max(max_abs_bf16,diff); if(diff>1e-2) mism_bf16++; }
    std::cout<<"GEMM BF16 WMMA fallback M"<<M<<" N"<<N<<" K"<<K<<" max_abs "<<max_abs_bf16<<" mism "<<mism_bf16<<" : "<<(max_abs_bf16<0.05?"PASS":"FAIL")<<std::endl;

    // Test FUSED decode+GEMM lane (same GEMM but decode inside)
    float *dY_fused;
    hipMalloc(&dY_fused, M*N*sizeof(float));
    hipMemset(dY_fused,0,M*N*sizeof(float));
    hipLaunchKernelGGL(wmma_gemm_cb_fused, grid, block,0,0, dA, d_qw,d_cb,d_off, dY_fused, M,N,K, k_bits,n_sub,type_size,row_bytes);
    hipDeviceSynchronize();
    float* dY_fused_scaled; hipMalloc(&dY_fused_scaled, M*N*sizeof(float));
    hipMemcpy(dY_fused_scaled, dY_fused, M*N*sizeof(float), hipMemcpyDeviceToDevice);
    hipLaunchKernelGGL(scale_rows_kernel, dim3(sblocks), dim3(256),0,0, dY_fused_scaled, d_scale, M,N);
    hipDeviceSynchronize();
    std::vector<float> Y_fused(M*N);
    hipMemcpy(Y_fused.data(), dY_fused_scaled, M*N*sizeof(float), hipMemcpyDeviceToHost);
    float max_abs_fused=0;
    int mism_fused=0;
    for(int i=0;i<M*N;i++){ float diff=std::abs(Y_fused[i]-Y_ref[i]); max_abs_fused=std::max(max_abs_fused,diff); if(diff>1e-3) mism_fused++; }
    std::cout<<"GEMM FUSED FP8 CB M"<<M<<" N"<<N<<" K"<<K<<" k"<<k_bits<<" max_abs "<<max_abs_fused<<" mism "<<mism_fused<<" : "<<(max_abs_fused<1e-2?"PASS":"FAIL")<<std::endl;
    if(mism_fused) ok=false;
    // Cross-check fused vs separate GPU GEMM (not just CPU): should be identical host-side fused vs separate (since both use same decoded W)
    // Y_gpu (separate) already computed as Y_gpu scaled; compare
    for(int i=0;i<M*N;i++) if(std::abs(Y_fused[i]-Y_gpu[i])>1e-6){ if(mism_fused<5) std::cout<<" fused vs separate mismatch "<<i<<" "<<Y_fused[i]<<" vs "<<Y_gpu[i]<<std::endl; mism_fused++; }

    hipFree(d_qw); hipFree(d_cb); hipFree(d_off); hipFree(d_out); hipFree(d_scale); hipFree(d_out_bf16);
    hipFree(dA); hipFree(dW); hipFree(dY); hipFree(dY2);
    hipFree(dA_bf16); hipFree(dW_bf16); hipFree(dY_bf16);
    hipFree(dY_fused); hipFree(dY_fused_scaled);
    return ok && mism==0 && mism2==0 && mism3==0 && mism_bf16==0 && mism_fused==0;
}

void bench_one(int M,int N,int K,int k_bits,int n_sub,int iters=300){
    int n_sb=K/256;
    int type_size=4*k_bits;
    int row_bytes=n_sb*type_size;
    int sub_dim=8/n_sub;
    int base=k_bits/n_sub, extra=k_bits%n_sub;
    int flat=0; for(int i=0;i<n_sub;i++){int w=base+(i<extra?1:0); flat += (1<<w)*sub_dim; }
    uint32_t rng_state=1;
    uint8_t* qw = new uint8_t[N*row_bytes]; for(int i=0;i<N*row_bytes;i++){ rng_state = rng_state*1103515245 + 12345; qw[i]= (rng_state>>16)%256; }
    uint8_t* cb = new uint8_t[flat]; for(int i=0;i<flat;i++){ rng_state = rng_state*1103515245 + 12345; float vals[5]={-1,0,0.5,1,2}; float v=vals[(rng_state>>16)%5]; hip_fp8_e4m3 e(v); cb[i]=e.__x; if(cb[i]==0x7F||cb[i]==0xFF) cb[i]=0; }
    // Host buffers as raw arrays to avoid vector of int/float/hip_fp8 which triggers macro collision
    int32_t* off = new int32_t[N](); // zero
    float* scale = new float[N]; for(int i=0;i<N;i++) scale[i]=1.0f;
    hip_fp8_e4m3* Ahost = new hip_fp8_e4m3[M*K]; for(int i=0;i<M*K;i++){ rng_state = rng_state*1103515245 + 12345; float v=((rng_state>>16)%9-4)*0.5f; Ahost[i]=hip_fp8_e4m3(v); }
    hip_fp8_e4m3* Whost = new hip_fp8_e4m3[N*K]; for(int i=0;i<N*K;i++){ rng_state = rng_state*1103515245 + 12345; float v=((rng_state>>16)%9-4)*0.5f; Whost[i]=hip_fp8_e4m3(v); }
    hip_fp8_e4m3 *dA,*dW; float *dC; uint8_t *d_qw,*d_cb; int32_t *d_off; float *d_scale,*dCfused;
    hipMalloc(&dA,M*K*sizeof(hip_fp8_e4m3));
    hipMalloc(&dW,N*K*sizeof(hip_fp8_e4m3));
    hipMalloc(&dC,M*N*sizeof(float));
    hipMalloc(&d_qw,N*row_bytes);
    hipMalloc(&d_cb,flat);
    hipMalloc(&d_off,N*sizeof(int32_t));
    hipMalloc(&d_scale,N*sizeof(float));
    hipMalloc(&dCfused,M*N*sizeof(float));
    hipMemcpy(dA,Ahost,M*K*sizeof(hip_fp8_e4m3),hipMemcpyHostToDevice);
    hipMemcpy(dW,Whost,N*K*sizeof(hip_fp8_e4m3),hipMemcpyHostToDevice);
    hipMemcpy(d_qw,qw,N*row_bytes,hipMemcpyHostToDevice);
    hipMemcpy(d_cb,cb,flat,hipMemcpyHostToDevice);
    hipMemcpy(d_off,off,N*sizeof(int32_t),hipMemcpyHostToDevice);
    hipMemcpy(d_scale,scale,N*sizeof(float),hipMemcpyHostToDevice);
    dim3 grid((N+15)/16,(M+15)/16);
    dim3 block(32);
    // warmup - use direct <<<>>> launch to avoid hipLaunchKernelGGL macro comma issues with std::vector locals
    for(int i=0;i<20;i++){ hipMemset(dC,0,M*N*sizeof(float)); wmma_plain_bench<<<grid,block>>>(dA,dW,d_scale,dC,M,N,K); hipDeviceSynchronize(); }
    hipEvent_t s,e; hipEventCreate(&s); hipEventCreate(&e);
    hipEventRecord(s);
    for(int i=0;i<iters;i++){ wmma_plain_bench<<<grid,block>>>(dA,dW,d_scale,dC,M,N,K); }
    hipEventRecord(e); hipEventSynchronize(e); float ms_plain; hipEventElapsedTime(&ms_plain,s,e); ms_plain/=iters;
    for(int i=0;i<20;i++){ hipMemset(dCfused,0,M*N*sizeof(float)); wmma_gemm_cb_fused<<<grid,block>>>(dA,d_qw,d_cb,d_off,dCfused,M,N,K,k_bits,n_sub,type_size,row_bytes); hipDeviceSynchronize(); }
    hipEventRecord(s);
    for(int i=0;i<iters;i++){ wmma_gemm_cb_fused<<<grid,block>>>(dA,d_qw,d_cb,d_off,dCfused,M,N,K,k_bits,n_sub,type_size,row_bytes); }
    hipEventRecord(e); hipEventSynchronize(e); float ms_fused; hipEventElapsedTime(&ms_fused,s,e); ms_fused/=iters;
    double tflops_plain=(2.0*M*N*K)/(ms_plain/1000.0)/1e12;
    double tflops_fused=(2.0*M*N*K)/(ms_fused/1000.0)/1e12;
    double bw_plain=(double)(M*K + N*K + M*N*4)/(ms_plain/1000.0)/1e9;
    double bw_fused=(double)(M*K + N*row_bytes + M*N*4)/(ms_fused/1000.0)/1e9;
    double ratio=ms_plain/ms_fused;
    std::cout<<"Bench M"<<M<<" N"<<N<<" K"<<K<<" k"<<k_bits<<" : plain "<<ms_plain<<"ms tflops "<<tflops_plain<<" bw "<<bw_plain<<"GB/s | fused "<<ms_fused<<"ms tflops "<<tflops_fused<<" bw_eff "<<bw_fused<<" ratio plain/fused "<<ratio<<" (weight bytes "<<N*K<<" vs "<<N*row_bytes<<" "<<(double)(N*K)/(N*row_bytes)<<"x) "<<(ratio>=0.97?"PASS":"SLOW")<<std::endl;
    hipFree(dA); hipFree(dW); hipFree(dC); hipFree(d_qw); hipFree(d_cb); hipFree(d_off); hipFree(d_scale); hipFree(dCfused);
    delete[] off; delete[] scale; delete[] Ahost; delete[] Whost; delete[] qw; delete[] cb;
    hipEventDestroy(s); hipEventDestroy(e);
}

int main(int argc,char** argv){
    hipDeviceProp_t prop; hipGetDeviceProperties(&prop,0);
    std::cout<<"Device "<<prop.name<<" "<<prop.gcnArchName<<std::endl;
    bool do_bench=false;
    for(int i=1;i<argc;i++) if(std::string(argv[i])=="--bench") do_bench=true;
    if(do_bench){
        std::cout<<"=== M3 Bench: plain FP8 WMMA vs CB fused (identical shapes) ==="<<std::endl;
        bench_one(16,4096,4096,32,4);
        bench_one(16,4096,4096,36,4);
        bench_one(16,4096,4096,40,4);
        bench_one(32,4096,4096,32,4);
        bench_one(64,4096,4096,32,4);
        bench_one(128,4096,4096,32,4);
        bench_one(16,1024,4096,32,4);
        bench_one(16,2048,1024,40,4);
        return 0;
    }
    bool all=true;
    // Test various k
    for(int k_bits: {32,36,40,44,48}){
        bool ok = test_decode_one(4, 512, k_bits, 4);
        all &= ok;
    }
    for(int k_bits: {28,30,32}){
        // 28,30 are ragged splits, test decode correctness even though not multiple of 4
        // But our kernel handles generic, so test.
        bool ok = test_decode_one(2, 256, k_bits, 4);
        all &= ok;
    }
    // Larger K with multiple superblocks
    all &= test_decode_one(8, 1024, 32, 4);

    std::cout<<"\nOverall "<<(all?"PASS":"FAIL")<<std::endl;
    return all?0:1;
}
