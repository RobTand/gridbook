// M7 persistent-B + fragment-direct for gfx1201 — dense FP8 CB serving kernels
// Implements M7.1 persistent-B tile reuse and M7.2 fragment-direct decode (LDS bypass)
// Build: hipcc --offload-arch=gfx1201 amd/gemm_m7_hip.cpp -o /tmp/gemm_m7 --std=c++17 -O2
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <hip/hip_fp8.h>
#include <iostream>
#include <vector>
#include <random>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <chrono>
using namespace rocwmma;

// ---- Plain optimized: direct global WMMA 16x16x16 / 16x16x32 ----
__global__ void plain_direct16(const hip_fp8_e4m3* A, const hip_fp8_e4m3* B, float* C, int M,int N,int K){
    int tm=blockIdx.y, tn=blockIdx.x;
    fragment<matrix_a,16,16,16,hip_fp8_e4m3,row_major> fa;
    fragment<matrix_b,16,16,16,hip_fp8_e4m3,col_major> fb;
    fragment<accumulator,16,16,16,float> fc; fill_fragment(fc,0.0f);
    for(int k0=0;k0<K;k0+=16){
        const hip_fp8_e4m3* ap = A + tm*16*K + k0;
        const hip_fp8_e4m3* bp = B + tn*16*K + k0;
        load_matrix_sync(fa, ap, K);
        load_matrix_sync(fb, bp, K);
        mma_sync(fc, fa, fb, fc);
    }
    float* cp = C + tm*16*N + tn*16;
    store_matrix_sync(cp, fc, N, mem_row_major);
}
__global__ void plain_direct32(const hip_fp8_e4m3* A, const hip_fp8_e4m3* B, float* C, int M,int N,int K){
    int tm=blockIdx.y, tn=blockIdx.x;
    fragment<matrix_a,16,16,32,hip_fp8_e4m3,row_major> fa;
    fragment<matrix_b,16,16,32,hip_fp8_e4m3,col_major> fb;
    fragment<accumulator,16,16,32,float> fc; fill_fragment(fc,0.0f);
    for(int k0=0;k0<K;k0+=32){
        const hip_fp8_e4m3* ap = A + tm*16*K + k0;
        const hip_fp8_e4m3* bp = B + tn*16*K + k0;
        load_matrix_sync(fa, ap, K);
        load_matrix_sync(fb, bp, K);
        mma_sync(fc, fa, fb, fc);
    }
    float* cp = C + tm*16*N + tn*16;
    store_matrix_sync(cp, fc, N, mem_row_major);
}

// ---- M6 repacked coalesced fused: 16x16x32, LDS staged qw (baseline for comparison) ----
__global__ void fused_m6_repacked_k32(const hip_fp8_e4m3* A, const uint8_t* qw_tiled, const uint8_t* cb, float* C, int M,int N,int K){
    int tm=blockIdx.y, tn=blockIdx.x;
    int Ntiles = N/16;
    __shared__ uint8_t s_cb[2048];
    __shared__ uint8_t s_qw[256];
    __shared__ hip_fp8_e4m3 s_B[512];
    fragment<matrix_a,16,16,32,hip_fp8_e4m3,row_major> fa;
    fragment<matrix_b,16,16,32,hip_fp8_e4m3,col_major> fb;
    fragment<accumulator,16,16,32,float> fc; fill_fragment(fc,0.0f);
    for(int i=threadIdx.x*8; i<2048; i+=32*8){
        if(i+8 <= 2048) *(uint64_t*)(s_cb+i) = *(const uint64_t*)(cb+i);
    }
    __syncthreads();
    for(int k0=0;k0<K;k0+=32){
        int kt = k0/32;
        int tile_base = (kt*Ntiles + tn)*256;
        for(int i=threadIdx.x*8; i<256; i+=32*8){
            *(uint64_t*)(s_qw+i) = *(const uint64_t*)(qw_tiled + tile_base + i);
        }
        __syncthreads();
        {
            int tid = threadIdx.x;
            for(int rep=0; rep<2; ++rep){
                int c = tid*2 + rep;
                if(c < 64){
                    int n_local = c / 4;
                    int vec = c % 4;
                    int qw_off = n_local*16 + vec*4;
                    uint32_t code = *(const uint32_t*)(s_qw + qw_off);
                    uint32_t c0=(code>>0)&0xFF, c1=(code>>8)&0xFF, c2=(code>>16)&0xFF, c3=(code>>24)&0xFF;
                    hip_fp8_e4m3 o0,o1,o2,o3,o4,o5,o6,o7;
                    o0.__x = s_cb[c0*2+0]; o1.__x = s_cb[c0*2+1];
                    o2.__x = s_cb[c1*2+512]; o3.__x = s_cb[c1*2+513];
                    o4.__x = s_cb[c2*2+1024]; o5.__x = s_cb[c2*2+1025];
                    o6.__x = s_cb[c3*2+1536]; o7.__x = s_cb[c3*2+1537];
                    s_B[n_local*32 + vec*8 + 0]=o0;
                    s_B[n_local*32 + vec*8 + 1]=o1;
                    s_B[n_local*32 + vec*8 + 2]=o2;
                    s_B[n_local*32 + vec*8 + 3]=o3;
                    s_B[n_local*32 + vec*8 + 4]=o4;
                    s_B[n_local*32 + vec*8 + 5]=o5;
                    s_B[n_local*32 + vec*8 + 6]=o6;
                    s_B[n_local*32 + vec*8 + 7]=o7;
                }
            }
        }
        __syncthreads();
        const hip_fp8_e4m3* ap = A + tm*16*K + k0;
        load_matrix_sync(fa, ap, K);
        load_matrix_sync(fb, s_B, 32);
        mma_sync(fc, fa, fb, fc);
        __syncthreads();
    }
    float* cp = C + tm*16*N + tn*16;
    store_matrix_sync(cp, fc, N, mem_row_major);
}

// ---- M7.1 PERSISTENT-B: decode N-tile ONCE per K chunk, reuse across all M tiles ----
// Grid: (N/16) blocks, each owns one N-tile (16 columns). Loops over K and inside K loops over Mt.
// LDS: s_cb 2048, s_qw 256, s_B 512. Accumulators: up to 8 (M128). Requires M multiple of 16.
__global__ void fused_persistent_k32(const hip_fp8_e4m3* A, const uint8_t* qw_tiled, const uint8_t* cb, float* C, int M,int N,int K){
    int tn = blockIdx.x;
    int Ntiles = N/16;
    if(tn >= Ntiles) return;
    int Mt = (M+15)/16;
    if(Mt>8) Mt=8; // limit to 8 for register budget (M128)
    __shared__ uint8_t s_cb[2048];
    __shared__ uint8_t s_qw[256];
    __shared__ hip_fp8_e4m3 s_B[512];
    // cb once per block
    for(int i=threadIdx.x*8; i<2048; i+=32*8){
        if(i+8 <= 2048) *(uint64_t*)(s_cb+i) = *(const uint64_t*)(cb+i);
    }
    __syncthreads();
    // accumulators for each M-tile
    fragment<accumulator,16,16,32,float> acc[8];
    for(int mt=0;mt<Mt;mt++) fill_fragment(acc[mt],0.0f);
    fragment<matrix_a,16,16,32,hip_fp8_e4m3,row_major> fa;
    fragment<matrix_b,16,16,32,hip_fp8_e4m3,col_major> fb;
    for(int k0=0;k0<K;k0+=32){
        int kt = k0/32;
        int tile_base = (kt*Ntiles + tn)*256;
        for(int i=threadIdx.x*8; i<256; i+=32*8){
            *(uint64_t*)(s_qw+i) = *(const uint64_t*)(qw_tiled + tile_base + i);
        }
        __syncthreads();
        // decode once per K chunk (amortized across Mt)
        {
            int tid = threadIdx.x;
            for(int rep=0; rep<2; ++rep){
                int c = tid*2 + rep;
                if(c < 64){
                    int n_local = c / 4;
                    int vec = c % 4;
                    int qw_off = n_local*16 + vec*4;
                    uint32_t code = *(const uint32_t*)(s_qw + qw_off);
                    uint32_t c0=(code>>0)&0xFF, c1=(code>>8)&0xFF, c2=(code>>16)&0xFF, c3=(code>>24)&0xFF;
                    s_B[n_local*32 + vec*8 + 0].__x = s_cb[c0*2+0];
                    s_B[n_local*32 + vec*8 + 1].__x = s_cb[c0*2+1];
                    s_B[n_local*32 + vec*8 + 2].__x = s_cb[c1*2+512];
                    s_B[n_local*32 + vec*8 + 3].__x = s_cb[c1*2+513];
                    s_B[n_local*32 + vec*8 + 4].__x = s_cb[c2*2+1024];
                    s_B[n_local*32 + vec*8 + 5].__x = s_cb[c2*2+1025];
                    s_B[n_local*32 + vec*8 + 6].__x = s_cb[c3*2+1536];
                    s_B[n_local*32 + vec*8 + 7].__x = s_cb[c3*2+1537];
                }
            }
        }
        __syncthreads();
        // reuse decoded s_B for each M-tile
        // Load B once per K chunk into fragment (shared across Mt to avoid repeated LDS loads? But load_matrix_sync is per mma, so load each time)
        // Instead, load fb once per K chunk and reuse? mma needs fb each time, but load can be reused: load fb once, then for each mt load fa and mma with same fb.
        load_matrix_sync(fb, s_B, 32);
        for(int mt=0;mt<Mt;mt++){
            const hip_fp8_e4m3* ap = A + mt*16*K + k0;
            load_matrix_sync(fa, ap, K);
            mma_sync(acc[mt], fa, fb, acc[mt]);
        }
        __syncthreads(); // protect s_B overwrite next iter; also ensures all mma done before next decode overwrites s_B (though fb already loaded)
    }
    // store each M-tile
    for(int mt=0;mt<Mt;mt++){
        float* cp = C + mt*16*N + tn*16;
        store_matrix_sync(cp, acc[mt], N, mem_row_major);
    }
}

// M7.1 persistent with 16x16 (K=16) variant for shapes where K not multiple of 32? (all 4096 so ok, but provide fallback)
__global__ void fused_persistent_k16(const hip_fp8_e4m3* A, const uint8_t* qw_tiled16, const uint8_t* cb, float* C, int M,int N,int K){
    int tn = blockIdx.x;
    int Ntiles=N/16;
    if(tn>=Ntiles) return;
    int Mt=(M+15)/16;
    if(Mt>8) Mt=8;
    __shared__ uint8_t s_cb[2048];
    __shared__ uint8_t s_qw[128]; // 16*8=128 for K=16: 32 codewords? Actually 16x16 tile: 16*16=256 elements, 32 codewords? Wait K=16 => 2 vec per row (16/8=2), 16 rows =>32 codewords =>128B
    __shared__ hip_fp8_e4m3 s_B[256]; // 16x16
    for(int i=threadIdx.x*8; i<2048; i+=32*8) *(uint64_t*)(s_cb+i)=*(uint64_t*)(cb+i);
    __syncthreads();
    fragment<accumulator,16,16,16,float> acc[8];
    for(int mt=0;mt<Mt;mt++) fill_fragment(acc[mt],0.0f);
    fragment<matrix_a,16,16,16,hip_fp8_e4m3,row_major> fa;
    fragment<matrix_b,16,16,16,hip_fp8_e4m3,col_major> fb;
    for(int k0=0;k0<K;k0+=16){
        int kt=k0/16;
        int tile_base=(kt*Ntiles+tn)*128;
        for(int i=threadIdx.x*8;i<128;i+=32*8) *(uint64_t*)(s_qw+i)=*(uint64_t*)(qw_tiled16+tile_base+i);
        __syncthreads();
        {
            int tid=threadIdx.x;
            if(tid<32){
                int n_local=tid/2;
                int vec=tid%2;
                int qw_off=n_local*8+vec*4;
                uint32_t code=*(uint32_t*)(s_qw+qw_off);
                uint32_t c0=(code>>0)&0xFF,c1=(code>>8)&0xFF,c2=(code>>16)&0xFF,c3=(code>>24)&0xFF;
                s_B[n_local*16+vec*8+0].__x=s_cb[c0*2+0];
                s_B[n_local*16+vec*8+1].__x=s_cb[c0*2+1];
                s_B[n_local*16+vec*8+2].__x=s_cb[c1*2+512];
                s_B[n_local*16+vec*8+3].__x=s_cb[c1*2+513];
                s_B[n_local*16+vec*8+4].__x=s_cb[c2*2+1024];
                s_B[n_local*16+vec*8+5].__x=s_cb[c2*2+1025];
                s_B[n_local*16+vec*8+6].__x=s_cb[c3*2+1536];
                s_B[n_local*16+vec*8+7].__x=s_cb[c3*2+1537];
            }
        }
        __syncthreads();
        load_matrix_sync(fb, s_B, 16);
        for(int mt=0;mt<Mt;mt++){
            const hip_fp8_e4m3* ap=A+mt*16*K+k0;
            load_matrix_sync(fa, ap, K);
            mma_sync(acc[mt], fa, fb, acc[mt]);
        }
        __syncthreads();
    }
    for(int mt=0;mt<Mt;mt++){
        float* cp=C+mt*16*N+tn*16;
        store_matrix_sync(cp, acc[mt], N, mem_row_major);
    }
}

// ---- M7.2 FRAGMENT-DIRECT: decode directly into WMMA fragment registers, no LDS B bounce ----
// For 16x16x16 tile: each lane decodes its assigned elements and fills fragment via operator[].
// Mapping derived empirically on gfx1201 (probe_frag): B col_major 16x16 -> lane's 8 elements correspond to n = lane%16, k = (lane/16)*8 + i
// This is PROBED, not assumed; see BENCH_M7 ISA/probe section.
// This kernel is persistent as well (reuses same fragment across Mt) but without s_B LDS.
__global__ void fused_persistent_frag_direct_k16(const hip_fp8_e4m3* A, const uint8_t* qw_tiled16, const uint8_t* cb, float* C, int M,int N,int K){
    int tn=blockIdx.x;
    int Ntiles=N/16;
    if(tn>=Ntiles) return;
    int Mt=(M+15)/16;
    if(Mt>8) Mt=8;
    __shared__ uint8_t s_cb[2048];
    __shared__ uint8_t s_qw[128];
    for(int i=threadIdx.x*8;i<2048;i+=32*8) *(uint64_t*)(s_cb+i)=*(uint64_t*)(cb+i);
    __syncthreads();
    fragment<accumulator,16,16,16,float> acc[8];
    for(int mt=0;mt<Mt;mt++) fill_fragment(acc[mt],0.0f);
    fragment<matrix_a,16,16,16,hip_fp8_e4m3,row_major> fa;
    fragment<matrix_b,16,16,16,hip_fp8_e4m3,col_major> fb;
    int lane = threadIdx.x % 32;
    int n_for_lane = lane % 16;
    int k_base = (lane / 16)*8;
    for(int k0=0;k0<K;k0+=16){
        int kt=k0/16;
        int tile_base=(kt*Ntiles+tn)*128;
        for(int i=threadIdx.x*8;i<128;i+=32*8) *(uint64_t*)(s_qw+i)=*(uint64_t*)(qw_tiled16+tile_base+i);
        __syncthreads();
        // decode directly into fb fragment registers (no LDS)
        // each lane holds 8 elements for its column n_for_lane and k_base .. k_base+7
        // qw layout for this tile: row n_for_lane has 2 codewords: vec0 and vec1 (each 4B)
        // n_for_lane row's qw offset: n_for_lane*8 + vec*4; vec = k/8 ?
        // For K=16 tile, there are 2 vec per row: vec = k_local/8 = 0 or 1
        // So for our lane's k_base, we need to map element i -> k_local = k_base + i
        // Determine vec and local j for each i: vec = k_local/8, j = k_local%8, but decode is per codeword (8 values) -> j corresponds to sub-table?
        // Actually decode pattern: one codeword gives 8 values (2 per sub). For K=16 tile, we have 32 codewords for 16x16=256 elements: each codeword produces 8.
        // Simpler: reuse same LDS s_qw gather but assign directly to fb[i] via cb lookup without writing to s_B.
        // For each i in fb (0..7), find its k_local and n_local, then decode that element's value via s_qw+s_cb.
        // n_local is fixed per lane (n_for_lane), k_local = k_base + i
        // Decode: vec_in_tile = k_local/8 (0 or1), coord = k_local%8, sub = coord/2, local = coord%2
        // Need codeword for this n_local, vec.
        for(int i=0;i<fb.num_elements;i++){
            int k_local = k_base + i;
            int vec = k_local / 8;
            int coord = k_local % 8;
            int qw_off = n_for_lane*8 + vec*4;
            uint32_t code = *(const uint32_t*)(s_qw + qw_off);
            // expand code's 4 bytes to 8 values: sub 0..3 each 2 values
            // code bytes: c0..c3
            uint32_t c0=(code>>0)&0xFF, c1=(code>>8)&0xFF, c2=(code>>16)&0xFF, c3=(code>>24)&0xFF;
            // sub mapping: sub = coord/2, but need to pick c based on sub
            // coord 0,1 -> c0 (sub0), 2,3->c1 (sub1), 4,5->c2, 6,7->c3
            int sub = coord / 2;
            int local = coord % 2;
            uint32_t cidx;
            int base;
            if(sub==0){cidx=c0; base=0;}
            else if(sub==1){cidx=c1; base=512;}
            else if(sub==2){cidx=c2; base=1024;}
            else {cidx=c3; base=1536;}
            fb[i].__x = s_cb[cidx*2 + local + base];
        }
        // Now fb holds correct B tile for this K chunk; reuse for each M tile
        // For large K, fb single load reused across Mt: no LDS bounce, no load_matrix_sync for B
        for(int mt=0;mt<Mt;mt++){
            const hip_fp8_e4m3* ap = A + mt*16*K + k0;
            load_matrix_sync(fa, ap, K);
            mma_sync(acc[mt], fa, fb, acc[mt]);
        }
        __syncthreads();
    }
    for(int mt=0;mt<Mt;mt++){
        float* cp=C+mt*16*N+tn*16;
        store_matrix_sync(cp, acc[mt], N, mem_row_major);
    }
}

// More waves variant for small-M latency hiding: 64 threads (2 waves) cooperatively load qw.
// For M16 (Mt=1), we can use 2 waves to increase MLP and occupancy.
// Simple version: 64 threads, half the tiles? But N-tile is 16, so 2 waves means we can split K decode across waves.
__global__ void fused_persistent_frag_direct_k16_2wave(const hip_fp8_e4m3* A, const uint8_t* qw_tiled16, const uint8_t* cb, float* C, int M,int N,int K){
    int tn=blockIdx.x;
    int Ntiles=N/16;
    if(tn>=Ntiles) return;
    int Mt=(M+15)/16;
    if(Mt>8) Mt=8;
    __shared__ uint8_t s_cb[2048];
    __shared__ uint8_t s_qw[128];
    for(int i=threadIdx.x*8;i<2048;i+=2048/8*8){} // placeholder
    // Use 64 threads: threadIdx.x 0..63, wave 0 lane 0..31, wave1 lane 0..31
    // For simplicity we keep 64-thread block but reuse same code as 32-thread but with 2x parallelism in qw load only.
    // Actual decode still per lane (need to know wave's lane). For brevity, delegate to 32-thread kernel when launched with 64.
    // This variant is stubbed to test occupancy; full impl would double-buffer qw staging.
    // To avoid complexity, we will just reuse single-wave logic but launched with 64 threads where extra wave idles.
    if(threadIdx.x>=32) return;
    // rest identical to frag_direct but with 64 block size effective occupancy 2 waves/SM
    // We'll just call same body as above but with blockDim 64; the extra wave does not participate in compute but increases occupancy? Not ideal.
    // Implement real 2-wave decode: each wave decodes its own half of B fragment? But fragment is wave-private, so each wave has its own fragment.
    // For simplicity we make 2 waves each compute same tile but we only need one wave's result (wave0). This is not useful.
    // Instead we duplicate work: not beneficial, so we leave as stub and document.
    // For measurement we will still launch with block 64 and only wave0 does work, which increases register pressure slightly but shows occupancy not helping without true cooperative.
}

// ---- repack helpers (host) ----
void repack_k32_for_m6(const uint8_t* qw, uint8_t* qw_tiled, int N, int K, int rbytes){
    int Ntiles=N/16; int Ktiles=K/32;
    for(int nt=0;nt<Ntiles;++nt) for(int kt=0;kt<Ktiles;++kt){
        int k0=kt*32;
        for(int n_local=0;n_local<16;++n_local){
            int gn=nt*16+n_local;
            for(int vec=0;vec<4;++vec){
                int gk=k0+vec*8; int sb=gk/256; int vec_in_sb=(gk%256)/8;
                int src=gn*rbytes+sb*128+vec_in_sb*4;
                int dst=(kt*Ntiles+nt)*256 + n_local*16 + vec*4;
                memcpy(qw_tiled+dst, qw+src, 4);
            }
        }
    }
}
void repack_k16_for_m7(const uint8_t* qw, uint8_t* qw_tiled, int N, int K, int rbytes){
    int Ntiles=N/16; int Ktiles=K/16;
    for(int nt=0;nt<Ntiles;++nt) for(int kt=0;kt<Ktiles;++kt){
        int k0=kt*16;
        for(int n_local=0;n_local<16;++n_local){
            int gn=nt*16+n_local;
            for(int vec=0;vec<2;++vec){
                int gk=k0+vec*8; int sb=gk/256; int vec_in_sb=(gk%256)/8;
                int src=gn*rbytes+sb*128+vec_in_sb*4;
                int dst=(kt*Ntiles+nt)*128 + n_local*8 + vec*4;
                memcpy(qw_tiled+dst, qw+src, 4);
            }
        }
    }
}

// ---- bench helpers ----
double bench_plain(int M,int N,int K,int use32,int iters=300){
    std::mt19937 rng(1);
    std::vector<hip_fp8_e4m3> A(M*K), B(N*K);
    for(auto &x:A){float v=(rng()%9-4)*0.5f; x=hip_fp8_e4m3(v);} for(auto &x:B){float v=(rng()%9-4)*0.5f; x=hip_fp8_e4m3(v);}
    hip_fp8_e4m3 *dA,*dB; float *dC; hipMalloc(&dA,M*K*sizeof(hip_fp8_e4m3)); hipMalloc(&dB,N*K*sizeof(hip_fp8_e4m3)); hipMalloc(&dC,M*N*sizeof(float));
    hipMemcpy(dA,A.data(),M*K*sizeof(hip_fp8_e4m3),hipMemcpyHostToDevice); hipMemcpy(dB,B.data(),N*K*sizeof(hip_fp8_e4m3),hipMemcpyHostToDevice);
    dim3 grid((N+15)/16,(M+15)/16);
    for(int i=0;i<20;i++){hipMemset(dC,0,M*N*sizeof(float)); if(use32) hipLaunchKernelGGL(plain_direct32,grid,dim3(32),0,0,dA,dB,dC,M,N,K); else hipLaunchKernelGGL(plain_direct16,grid,dim3(32),0,0,dA,dB,dC,M,N,K); hipDeviceSynchronize();}
    hipEvent_t s,e; hipEventCreate(&s); hipEventCreate(&e);
    hipEventRecord(s); for(int i=0;i<iters;i++){ if(use32) hipLaunchKernelGGL(plain_direct32,grid,dim3(32),0,0,dA,dB,dC,M,N,K); else hipLaunchKernelGGL(plain_direct16,grid,dim3(32),0,0,dA,dB,dC,M,N,K);} hipEventRecord(e); hipEventSynchronize(e); float ms; hipEventElapsedTime(&ms,s,e); ms/=iters;
    hipFree(dA); hipFree(dB); hipFree(dC); hipEventDestroy(s); hipEventDestroy(e); return ms;
}
double bench_fused_m6(int M,int N,int K,int iters=300){
    int kbits=32, nsub=4; int n_sb=K/256, tsize=4*kbits, rbytes=n_sb*tsize;
    int flat=0; for(int i=0;i<nsub;i++){int w=kbits/nsub+(i<kbits%nsub?1:0); flat+=(1<<w)*(8/nsub);}
    std::mt19937 rng(2);
    std::vector<uint8_t> qw(N*rbytes); for(auto &x:qw) x=rng()%256;
    std::vector<uint8_t> cb(flat); for(auto &x:cb){float v=(rng()%5)*0.5f; x=(uint8_t)(v*10); if(x==0x7F||x==0xFF)x=0;}
    std::vector<hip_fp8_e4m3> A(M*K); for(auto &x:A){float v=(rng()%9-4)*0.5f; x=hip_fp8_e4m3(v);}
    std::vector<uint8_t> qw_tiled; int Ntiles=N/16,Ktiles=K/32,tiled_bytes=Ntiles*Ktiles*256;
    qw_tiled.resize(tiled_bytes); repack_k32_for_m6(qw.data(),qw_tiled.data(),N,K,rbytes);
    hip_fp8_e4m3 *dA; float *dC; uint8_t *dqw,*dcb;
    hipMalloc(&dA,M*K*sizeof(hip_fp8_e4m3)); hipMalloc(&dC,M*N*sizeof(float)); hipMalloc(&dqw,tiled_bytes); hipMalloc(&dcb,flat);
    hipMemcpy(dA,A.data(),M*K*sizeof(hip_fp8_e4m3),hipMemcpyHostToDevice); hipMemcpy(dqw,qw_tiled.data(),tiled_bytes,hipMemcpyHostToDevice); hipMemcpy(dcb,cb.data(),flat,hipMemcpyHostToDevice);
    dim3 grid((N+15)/16,(M+15)/16);
    for(int i=0;i<20;i++){hipMemset(dC,0,M*N*sizeof(float)); hipLaunchKernelGGL(fused_m6_repacked_k32,grid,dim3(32),0,0,dA,dqw,dcb,dC,M,N,K); hipDeviceSynchronize();}
    hipEvent_t s,e; hipEventCreate(&s); hipEventCreate(&e);
    hipEventRecord(s); for(int i=0;i<iters;i++) hipLaunchKernelGGL(fused_m6_repacked_k32,grid,dim3(32),0,0,dA,dqw,dcb,dC,M,N,K); hipEventRecord(e); hipEventSynchronize(e); float ms; hipEventElapsedTime(&ms,s,e); ms/=iters;
    hipFree(dA); hipFree(dC); hipFree(dqw); hipFree(dcb); hipEventDestroy(s); hipEventDestroy(e); return ms;
}
double bench_persistent_k32(int M,int N,int K,int iters=300){
    int kbits=32,nsub=4,n_sb=K/256,tsize=4*kbits,rbytes=n_sb*tsize;
    int flat=0; for(int i=0;i<nsub;i++){int w=kbits/nsub+(i<kbits%nsub?1:0); flat+=(1<<w)*(8/nsub);}
    std::mt19937 rng(2);
    std::vector<uint8_t> qw(N*rbytes); for(auto &x:qw) x=rng()%256;
    std::vector<uint8_t> cb(flat); for(auto &x:cb){float v=(rng()%5)*0.5f; x=(uint8_t)(v*10); if(x==0x7F||x==0xFF)x=0;}
    std::vector<hip_fp8_e4m3> A(M*K); for(auto &x:A){float v=(rng()%9-4)*0.5f; x=hip_fp8_e4m3(v);}
    int Ntiles=N/16,Ktiles=K/32,tiled_bytes=Ntiles*Ktiles*256;
    std::vector<uint8_t> qw_tiled(tiled_bytes); repack_k32_for_m6(qw.data(),qw_tiled.data(),N,K,rbytes);
    hip_fp8_e4m3 *dA; float *dC; uint8_t *dqw,*dcb;
    hipMalloc(&dA,M*K*sizeof(hip_fp8_e4m3)); hipMalloc(&dC,M*N*sizeof(float)); hipMalloc(&dqw,tiled_bytes); hipMalloc(&dcb,flat);
    hipMemcpy(dA,A.data(),M*K*sizeof(hip_fp8_e4m3),hipMemcpyHostToDevice); hipMemcpy(dqw,qw_tiled.data(),tiled_bytes,hipMemcpyHostToDevice); hipMemcpy(dcb,cb.data(),flat,hipMemcpyHostToDevice);
    dim3 grid(Ntiles,1);
    for(int i=0;i<20;i++){hipMemset(dC,0,M*N*sizeof(float)); hipLaunchKernelGGL(fused_persistent_k32,grid,dim3(32),0,0,dA,dqw,dcb,dC,M,N,K); hipDeviceSynchronize();}
    hipEvent_t s,e; hipEventCreate(&s); hipEventCreate(&e);
    hipEventRecord(s); for(int i=0;i<iters;i++) hipLaunchKernelGGL(fused_persistent_k32,grid,dim3(32),0,0,dA,dqw,dcb,dC,M,N,K); hipEventRecord(e); hipEventSynchronize(e); float ms; hipEventElapsedTime(&ms,s,e); ms/=iters;
    hipFree(dA); hipFree(dC); hipFree(dqw); hipFree(dcb); hipEventDestroy(s); hipEventDestroy(e); return ms;
}
double bench_persistent_frag_k16(int M,int N,int K,int iters=300){
    int kbits=32,nsub=4,n_sb=K/256,tsize=4*kbits,rbytes=n_sb*tsize;
    int flat=2048;
    std::mt19937 rng(2);
    std::vector<uint8_t> qw(N*rbytes); for(auto &x:qw) x=rng()%256;
    std::vector<uint8_t> cb(flat); for(auto &x:cb){float v=(rng()%5)*0.5f; x=(uint8_t)(v*10); if(x==0x7F||x==0xFF)x=0;}
    std::vector<hip_fp8_e4m3> A(M*K); for(auto &x:A){float v=(rng()%9-4)*0.5f; x=hip_fp8_e4m3(v);}
    int Ntiles=N/16,Ktiles=K/16,tiled_bytes=Ntiles*Ktiles*128;
    std::vector<uint8_t> qw_tiled(tiled_bytes); repack_k16_for_m7(qw.data(),qw_tiled.data(),N,K,rbytes);
    hip_fp8_e4m3 *dA; float *dC; uint8_t *dqw,*dcb;
    hipMalloc(&dA,M*K*sizeof(hip_fp8_e4m3)); hipMalloc(&dC,M*N*sizeof(float)); hipMalloc(&dqw,tiled_bytes); hipMalloc(&dcb,flat);
    hipMemcpy(dA,A.data(),M*K*sizeof(hip_fp8_e4m3),hipMemcpyHostToDevice); hipMemcpy(dqw,qw_tiled.data(),tiled_bytes,hipMemcpyHostToDevice); hipMemcpy(dcb,cb.data(),flat,hipMemcpyHostToDevice);
    dim3 grid(Ntiles,1);
    for(int i=0;i<20;i++){hipMemset(dC,0,M*N*sizeof(float)); hipLaunchKernelGGL(fused_persistent_frag_direct_k16,grid,dim3(32),0,0,dA,dqw,dcb,dC,M,N,K); hipDeviceSynchronize();}
    hipEvent_t s,e; hipEventCreate(&s); hipEventCreate(&e);
    hipEventRecord(s); for(int i=0;i<iters;i++) hipLaunchKernelGGL(fused_persistent_frag_direct_k16,grid,dim3(32),0,0,dA,dqw,dcb,dC,M,N,K); hipEventRecord(e); hipEventSynchronize(e); float ms; hipEventElapsedTime(&ms,s,e); ms/=iters;
    hipFree(dA); hipFree(dC); hipFree(dqw); hipFree(dcb); hipEventDestroy(s); hipEventDestroy(e); return ms;
}

bool validate_persistent(){
    int M=16,N=32,K=256,kbits=32,nsub=4,n_sb=K/256,tsize=4*kbits,rbytes=n_sb*tsize;
    int subdim=8/nsub, flat=0; for(int i=0;i<nsub;i++){int w=kbits/nsub+(i<kbits%nsub?1:0); flat+=(1<<w)*subdim;}
    std::mt19937 rng(123);
    std::vector<uint8_t> qw(N*rbytes); for(auto &x:qw) x=rng()%256;
    std::vector<uint8_t> cb(flat); for(auto &x:cb){float v=(rng()%5)*0.5f; x=(uint8_t)(v*10); if(x==0x7F||x==0xFF)x=0;}
    std::vector<hip_fp8_e4m3> A(M*K); for(auto &a:A){float v=(rng()%9-4)*0.5f; a=hip_fp8_e4m3(v);}
    // CPU ref
    auto widths=[&](){std::vector<int> w(nsub); int b=kbits/nsub,e=kbits%nsub; for(int i=0;i<nsub;i++) w[i]=b+(i<e); return w;}();
    std::vector<int> tb(nsub,0); for(int i=1;i<nsub;i++) tb[i]=tb[i-1]+(1<<widths[i-1])*subdim;
    std::vector<uint8_t> W(N*K);
    for(int gn=0;gn<N;gn++) for(int sb=0;sb<n_sb;sb++) for(int v=0;v<32;v++){
        int bb=(v*kbits)/8, bs=(v*kbits)%8; uint64_t win=0; for(int b=0;b<8;b++){uint8_t by=0; if(bb+b<tsize) by=qw[gn*rbytes+sb*tsize+bb+b]; win|=(uint64_t)by<<(8*b);}
        uint64_t code=(win>>bs)&((1ULL<<kbits)-1);
        for(int j=0;j<8;j++){int sub=j/subdim, local=j%subdim; int off=0; for(int t=0;t<sub;t++) off+=widths[t]; int mask=(1<<widths[sub])-1; uint64_t sidx=(code>>off)&mask; int fidx=tb[sub]+sidx*subdim+local; W[gn*K+sb*256+v*8+j]=cb[fidx];}
    }
    std::vector<float> Af(M*K), Wf(N*K);
    for(int i=0;i<M*K;i++) Af[i]=(float)A[i];
    for(int i=0;i<N*K;i++){hip_fp8_e4m3 f; f.__x=W[i]; Wf[i]=(float)f;}
    std::vector<float> ref(M*N,0);
    for(int m=0;m<M;m++) for(int n=0;n<N;n++){float acc=0; for(int k=0;k<K;k++) acc+=Af[m*K+k]*Wf[n*K+k]; ref[m*N+n]=acc;}

    // test persistent k32
    {
        int Ntiles=N/16,Ktiles=K/32,tiled_bytes=Ntiles*Ktiles*256;
        std::vector<uint8_t> qw_tiled(tiled_bytes); repack_k32_for_m6(qw.data(),qw_tiled.data(),N,K,rbytes);
        hip_fp8_e4m3 *dA; uint8_t *dqw,*dcb; float *dC;
        hipMalloc(&dA,M*K*sizeof(hip_fp8_e4m3)); hipMalloc(&dqw,tiled_bytes); hipMalloc(&dcb,flat); hipMalloc(&dC,M*N*sizeof(float));
        hipMemcpy(dA,A.data(),M*K*sizeof(hip_fp8_e4m3),hipMemcpyHostToDevice); hipMemcpy(dqw,qw_tiled.data(),tiled_bytes,hipMemcpyHostToDevice); hipMemcpy(dcb,cb.data(),flat,hipMemcpyHostToDevice);
        hipMemset(dC,0,M*N*sizeof(float));
        dim3 grid(Ntiles,1);
        hipLaunchKernelGGL(fused_persistent_k32,grid,dim3(32),0,0,dA,dqw,dcb,dC,M,N,K);
        hipDeviceSynchronize();
        std::vector<float> gpu(M*N); hipMemcpy(gpu.data(),dC,M*N*sizeof(float),hipMemcpyDeviceToHost);
        float max_abs=0; int mism=0; for(int i=0;i<M*N;i++){float d=std::abs(gpu[i]-ref[i]); max_abs=std::max(max_abs,d); if(d>1e-3) mism++;}
        std::cout<<"M7 persistent_k32 M"<<M<<" N"<<N<<" K"<<K<<" max_abs "<<max_abs<<" mism "<<mism<<" "<<(max_abs<1e-3?"PASS":"FAIL")<<"\n";
        hipFree(dA); hipFree(dqw); hipFree(dcb); hipFree(dC);
        if(max_abs>=1e-3) return false;
    }
    // test persistent frag direct k16 with K=256
    {
        int Ntiles=N/16,Ktiles=K/16,tiled_bytes=Ntiles*Ktiles*128;
        std::vector<uint8_t> qw_tiled(tiled_bytes); repack_k16_for_m7(qw.data(),qw_tiled.data(),N,K,rbytes);
        hip_fp8_e4m3 *dA; uint8_t *dqw,*dcb; float *dC;
        hipMalloc(&dA,M*K*sizeof(hip_fp8_e4m3)); hipMalloc(&dqw,tiled_bytes); hipMalloc(&dcb,flat); hipMalloc(&dC,M*N*sizeof(float));
        hipMemcpy(dA,A.data(),M*K*sizeof(hip_fp8_e4m3),hipMemcpyHostToDevice); hipMemcpy(dqw,qw_tiled.data(),tiled_bytes,hipMemcpyHostToDevice); hipMemcpy(dcb,cb.data(),flat,hipMemcpyHostToDevice);
        hipMemset(dC,0,M*N*sizeof(float));
        dim3 grid(Ntiles,1);
        hipLaunchKernelGGL(fused_persistent_frag_direct_k16,grid,dim3(32),0,0,dA,dqw,dcb,dC,M,N,K);
        hipDeviceSynchronize();
        std::vector<float> gpu(M*N); hipMemcpy(gpu.data(),dC,M*N*sizeof(float),hipMemcpyDeviceToHost);
        float max_abs=0; int mism=0; for(int i=0;i<M*N;i++){float d=std::abs(gpu[i]-ref[i]); max_abs=std::max(max_abs,d); if(d>1e-3) mism++;}
        std::cout<<"M7 frag_direct_k16 M"<<M<<" N"<<N<<" K"<<K<<" max_abs "<<max_abs<<" mism "<<mism<<" "<<(max_abs<1e-3?"PASS":"FAIL")<<"\n";
        hipFree(dA); hipFree(dqw); hipFree(dcb); hipFree(dC);
        if(max_abs>=1e-3) return false;
    }
    // larger persistent test M64
    {
        int M2=64,N2=32,K2=256;
        std::vector<hip_fp8_e4m3> A2(M2*K2); for(auto &a:A2){float v=(rng()%9-4)*0.5f; a=hip_fp8_e4m3(v);}
        std::vector<uint8_t> qw2(N2*rbytes); for(auto &x:qw2) x=rng()%256;
        // recompute W2 for N2
        std::vector<uint8_t> W2(N2*K2);
        for(int gn=0;gn<N2;gn++) for(int sb=0;sb<n_sb;sb++) for(int v=0;v<32;v++){
            int bb=(v*kbits)/8, bs=(v*kbits)%8; uint64_t win=0; for(int b=0;b<8;b++){uint8_t by=0; if(bb+b<tsize) by=qw2[gn*rbytes+sb*tsize+bb+b]; win|=(uint64_t)by<<(8*b);}
            uint64_t code=(win>>bs)&((1ULL<<kbits)-1);
            for(int j=0;j<8;j++){int sub=j/subdim, local=j%subdim; int off=0; for(int t=0;t<sub;t++) off+=widths[t]; int mask=(1<<widths[sub])-1; uint64_t sidx=(code>>off)&mask; int fidx=tb[sub]+sidx*subdim+local; W2[gn*K2+sb*256+v*8+j]=cb[fidx];}
        }
        std::vector<float> Af2(M2*K2),Wf2(N2*K2);
        for(int i=0;i<M2*K2;i++) Af2[i]=(float)A2[i];
        for(int i=0;i<N2*K2;i++){hip_fp8_e4m3 f; f.__x=W2[i]; Wf2[i]=(float)f;}
        std::vector<float> ref2(M2*N2,0);
        for(int m=0;m<M2;m++) for(int n=0;n<N2;n++){float acc=0; for(int k=0;k<K2;k++) acc+=Af2[m*K2+k]*Wf2[n*K2+k]; ref2[m*N2+n]=acc;}
        int Ntiles=N2/16,Ktiles=K2/32,tiled_bytes=Ntiles*Ktiles*256;
        std::vector<uint8_t> qw_tiled(tiled_bytes); repack_k32_for_m6(qw2.data(),qw_tiled.data(),N2,K2,rbytes);
        hip_fp8_e4m3 *dA; uint8_t *dqw,*dcb; float *dC;
        hipMalloc(&dA,M2*K2*sizeof(hip_fp8_e4m3)); hipMalloc(&dqw,tiled_bytes); hipMalloc(&dcb,flat); hipMalloc(&dC,M2*N2*sizeof(float));
        hipMemcpy(dA,A2.data(),M2*K2*sizeof(hip_fp8_e4m3),hipMemcpyHostToDevice); hipMemcpy(dqw,qw_tiled.data(),tiled_bytes,hipMemcpyHostToDevice); hipMemcpy(dcb,cb.data(),flat,hipMemcpyHostToDevice);
        hipMemset(dC,0,M2*N2*sizeof(float));
        dim3 grid(Ntiles,1);
        hipLaunchKernelGGL(fused_persistent_k32,grid,dim3(32),0,0,dA,dqw,dcb,dC,M2,N2,K2);
        hipDeviceSynchronize();
        std::vector<float> gpu(M2*N2); hipMemcpy(gpu.data(),dC,M2*N2*sizeof(float),hipMemcpyDeviceToHost);
        float max_abs=0; int mism=0; for(int i=0;i<M2*N2;i++){float d=std::abs(gpu[i]-ref2[i]); max_abs=std::max(max_abs,d); if(d>1e-3) mism++;}
        std::cout<<"M7 persistent_k32 M"<<M2<<" N"<<N2<<" K"<<K2<<" max_abs "<<max_abs<<" mism "<<mism<<" "<<(max_abs<1e-3?"PASS":"FAIL")<<"\n";
        hipFree(dA); hipFree(dqw); hipFree(dcb); hipFree(dC);
        if(max_abs>=1e-3) return false;
    }
    return true;
}

static double median_vec(std::vector<double> v){ std::sort(v.begin(),v.end()); return v[v.size()/2]; }

int main(){
    hipDeviceProp_t p; hipGetDeviceProperties(&p,0);
    std::cout<<"Device "<<p.name<<" "<<p.gcnArchName<<" "<<p.multiProcessorCount<<" CUs warp "<<p.warpSize<<"\n";
    if(!validate_persistent()){ std::cout<<"VALIDATION FAILED\n"; return 1; }
    std::cout<<"All validations PASS\n";
    struct Shape{int M,N,K;}; std::vector<Shape> shapes={{16,4096,4096},{32,4096,4096},{64,4096,4096},{128,4096,4096},{16,1024,4096},{16,2048,1024}};
    std::cout<<"\n=== Plain vs M6 (non-persistent) median of 10 ===\n";
    for(auto s:shapes){
        std::vector<double> vplain, vm6;
        for(int r=0;r<10;r++){ vplain.push_back(bench_plain(s.M,s.N,s.K,0,300)); vm6.push_back(bench_fused_m6(s.M,s.N,s.K,300));}
        std::cout<<"M"<<s.M<<" N"<<s.N<<" K"<<s.K<<" plain med "<<median_vec(vplain)<<" m6 med "<<median_vec(vm6)<<" ratio "<<median_vec(vplain)/median_vec(vm6)<<"\n";
    }
    std::cout<<"\n=== Persistent k32 (LDS) median of 10 ===\n";
    for(auto s:shapes){
        std::vector<double> vplain, vper;
        for(int r=0;r<10;r++){ vplain.push_back(bench_plain(s.M,s.N,s.K,0,300)); vper.push_back(bench_persistent_k32(s.M,s.N,s.K,300));}
        double mp=median_vec(vplain), mf=median_vec(vper);
        std::cout<<"M"<<s.M<<" N"<<s.N<<" K"<<s.K<<" plain "<<mp<<" pers "<<mf<<" ratio "<<mp/mf<<" "<<(mp/mf>=0.97?"PASS":"SLOW")<<"\n";
    }
    std::cout<<"\n=== Frag-direct k16 persistent median of 10 ===\n";
    for(auto s:shapes){
        if(s.K%16!=0) continue;
        std::vector<double> vplain, vfrag;
        for(int r=0;r<10;r++){ vplain.push_back(bench_plain(s.M,s.N,s.K,0,300)); vfrag.push_back(bench_persistent_frag_k16(s.M,s.N,s.K,300));}
        double mp=median_vec(vplain), mf=median_vec(vfrag);
        std::cout<<"M"<<s.M<<" N"<<s.N<<" K"<<s.K<<" plain "<<mp<<" frag "<<mf<<" ratio "<<mp/mf<<" "<<(mp/mf>=0.97?"PASS":"SLOW")<<"\n";
    }
    // isolated decode probe vs saving
    {
        std::cout<<"\n=== Isolated decode vs saving at 420 GB/s ===\n";
        double bw=420; // approximate measured plain BW
        for(auto s:shapes){
            double saving_bytes = (double)s.N*s.K - s.N*(s.K/256*128);
            double saving_ms = saving_bytes / (bw*1e9) *1000;
            std::cout<<"M"<<s.M<<" N"<<s.N<<" K"<<s.K<<" saving "<<saving_bytes/1e6<<"MB "<<saving_ms<<"ms\n";
        }
    }
    return 0;
}
