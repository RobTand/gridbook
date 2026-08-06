// M6 coalesced fused GEMM for gfx1201 — repacked coalesced qw + LDS decode + double-buffer attempt
// Build: hipcc --offload-arch=gfx1201 amd/gemm_m6_hip.cpp -o /tmp/gemm_m6 --std=c++17 -O2
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

// ---- M5 pipelined fused for reference (re-included for bench parity) ----
__global__ void fused_m5_k32(const hip_fp8_e4m3* A, const uint8_t* qw, const uint8_t* cb, float* C, int M,int N,int K,int rbytes){
    int tm=blockIdx.y, tn=blockIdx.x;
    int cb0 = tn*16;
    __shared__ uint8_t s_cb[2048];
    __shared__ hip_fp8_e4m3 s_B[2][512];
    fragment<matrix_a,16,16,32,hip_fp8_e4m3,row_major> fa;
    fragment<matrix_b,16,16,32,hip_fp8_e4m3,col_major> fb;
    fragment<accumulator,16,16,32,float> fc; fill_fragment(fc,0.0f);
    for(int i=threadIdx.x*8; i<2048; i+=32*8){
        if(i+8 <= 2048) *(uint64_t*)(s_cb+i) = *(const uint64_t*)(cb+i);
    }
    __syncthreads();
    for(int k0=0;k0<K;k0+=32){
        int buf = (k0/32) % 2;
        {
            int tid = threadIdx.x;
            for(int rep=0; rep<2; ++rep){
                int c = tid*2 + rep;
                if(c < 64){
                    int n_local = c / 4;
                    int vec = c % 4;
                    int gn = cb0 + n_local;
                    int gk_base = k0 + vec*8;
                    hip_fp8_e4m3 out[8];
                    if(gn < N && gk_base < K){
                        int sb = gk_base / 256;
                        int vec_in_sb = (gk_base % 256)/8;
                        int base_off = gn * rbytes + sb*128 + vec_in_sb*4;
                        uint32_t code = *(const uint32_t*)(qw + base_off);
                        uint32_t c0=(code>>0)&0xFF, c1=(code>>8)&0xFF, c2=(code>>16)&0xFF, c3=(code>>24)&0xFF;
                        out[0].__x = s_cb[c0*2+0]; out[1].__x = s_cb[c0*2+1];
                        out[2].__x = s_cb[c1*2+512]; out[3].__x = s_cb[c1*2+513];
                        out[4].__x = s_cb[c2*2+1024]; out[5].__x = s_cb[c2*2+1025];
                        out[6].__x = s_cb[c3*2+1536]; out[7].__x = s_cb[c3*2+1537];
                    } else {
                        for(int j=0;j<8;++j) out[j].__x=0;
                    }
                    #pragma unroll
                    for(int j=0;j<8;++j){
                        int k_local = vec*8 + j;
                        s_B[buf][n_local*32 + k_local] = out[j];
                    }
                }
            }
        }
        __syncthreads();
        const hip_fp8_e4m3* ap = A + tm*16*K + k0;
        load_matrix_sync(fa, ap, K);
        load_matrix_sync(fb, (hip_fp8_e4m3*)s_B[buf], 32);
        mma_sync(fc, fa, fb, fc);
    }
    float* cp = C + tm*16*N + tn*16;
    store_matrix_sync(cp, fc, N, mem_row_major);
}

// ---- M6 repacked coalesced fused: 16x16x32, LDS staged qw (coalesced), LDS cb, single sync per tile ----
__global__ void fused_m6_repacked_k32(const hip_fp8_e4m3* A, const uint8_t* qw_tiled, const uint8_t* cb, float* C, int M,int N,int K){
    int tm=blockIdx.y, tn=blockIdx.x;
    int cb0 = tn*16;
    int Ntiles = N/16;
    __shared__ uint8_t s_cb[2048];
    __shared__ uint8_t s_qw[256]; // one tile's packed bytes, coalesced
    __shared__ hip_fp8_e4m3 s_B[512];
    fragment<matrix_a,16,16,32,hip_fp8_e4m3,row_major> fa;
    fragment<matrix_b,16,16,32,hip_fp8_e4m3,col_major> fb;
    fragment<accumulator,16,16,32,float> fc; fill_fragment(fc,0.0f);
    // cb LDS
    for(int i=threadIdx.x*8; i<2048; i+=32*8){
        if(i+8 <= 2048) *(uint64_t*)(s_cb+i) = *(const uint64_t*)(cb+i);
    }
    __syncthreads();
    for(int k0=0;k0<K;k0+=32){
        int kt = k0/32;
        int tile_base = (kt*Ntiles + tn)*256;
        // coalesced load 256B tile packed into s_qw
        for(int i=threadIdx.x*8; i<256; i+=32*8){
            *(uint64_t*)(s_qw+i) = *(const uint64_t*)(qw_tiled + tile_base + i);
        }
        __syncthreads();
        // decode from LDS s_qw (no global)
        {
            int tid = threadIdx.x;
            for(int rep=0; rep<2; ++rep){
                int c = tid*2 + rep;
                if(c < 64){
                    int n_local = c / 4;
                    int vec = c % 4;
                    // s_qw layout: row*16 + vec*4  (16B per row)
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
        // ensure B not overwritten before next tile's decode needs s_B
        // next iteration will overwrite s_B after its qw load; mma already consumed s_B (sync), safe
        // need a sync to protect s_qw overwrite? s_qw will be overwritten at top of next loop after this mma, but that load is guarded by __syncthreads before decode; since we are at bottom, next loop's s_qw load will wait for all threads to finish mma (which is synchronous). No extra sync needed beyond the one at top of loop, but add one to serialize s_qw reuse if needed
        __syncthreads();
    }
    float* cp = C + tm*16*N + tn*16;
    store_matrix_sync(cp, fc, N, mem_row_major);
}

// Double-buffered variant: s_qw[2][256] + s_B[2][512] to overlap next tile's qw load with current WMMA (software pipeline illusion)
__global__ void fused_m6_repacked_db_k32(const hip_fp8_e4m3* A, const uint8_t* qw_tiled, const uint8_t* cb, float* C, int M,int N,int K){
    int tm=blockIdx.y, tn=blockIdx.x;
    int Ntiles = N/16;
    __shared__ uint8_t s_cb[2048];
    __shared__ uint8_t s_qw[2][256];
    __shared__ hip_fp8_e4m3 s_B[2][512];
    fragment<matrix_a,16,16,32,hip_fp8_e4m3,row_major> fa;
    fragment<matrix_b,16,16,32,hip_fp8_e4m3,col_major> fb;
    fragment<accumulator,16,16,32,float> fc; fill_fragment(fc,0.0f);
    for(int i=threadIdx.x*8; i<2048; i+=32*8){
        if(i+8 <= 2048) *(uint64_t*)(s_cb+i) = *(const uint64_t*)(cb+i);
    }
    __syncthreads();
    // preload first tile's qw into buf 0
    if(K>=32){
        int tile_base0 = (0*Ntiles + tn)*256;
        for(int i=threadIdx.x*8; i<256; i+=32*8) *(uint64_t*)(s_qw[0]+i) = *(const uint64_t*)(qw_tiled+tile_base0+i);
    }
    __syncthreads();
    for(int k0=0;k0<K;k0+=32){
        int buf = (k0/32)%2;
        int nbuf = buf^1;
        // prefetch next tile's qw into nbuf while we decode current
        if(k0+32 < K){
            int kt_next = (k0+32)/32;
            int tile_base_n = (kt_next*Ntiles + tn)*256;
            // Note: this load is concurrent with current tile's decode+mma? Not truly async without separate sync, but we place it after current tile's qw sync and before next iteration's decode. To allow overlap, we would need to not sync here.
            // For now we prefetch after sync, still serial but using double buffer to reduce sync count.
        }
        // decode current buf's s_qw -> s_B[buf]
        {
            int tid=threadIdx.x;
            for(int rep=0;rep<2;++rep){
                int c=tid*2+rep;
                if(c<64){
                    int n_local=c/4, vec=c%4;
                    int qw_off=n_local*16+vec*4;
                    uint32_t code=*(const uint32_t*)(s_qw[buf]+qw_off);
                    uint32_t c0=(code>>0)&0xFF, c1=(code>>8)&0xFF, c2=(code>>16)&0xFF, c3=(code>>24)&0xFF;
                    s_B[buf][n_local*32+vec*8+0].__x=s_cb[c0*2+0];
                    s_B[buf][n_local*32+vec*8+1].__x=s_cb[c0*2+1];
                    s_B[buf][n_local*32+vec*8+2].__x=s_cb[c1*2+512];
                    s_B[buf][n_local*32+vec*8+3].__x=s_cb[c1*2+513];
                    s_B[buf][n_local*32+vec*8+4].__x=s_cb[c2*2+1024];
                    s_B[buf][n_local*32+vec*8+5].__x=s_cb[c2*2+1025];
                    s_B[buf][n_local*32+vec*8+6].__x=s_cb[c3*2+1536];
                    s_B[buf][n_local*32+vec*8+7].__x=s_cb[c3*2+1537];
                }
            }
        }
        __syncthreads();
        const hip_fp8_e4m3* ap = A + tm*16*K + k0;
        load_matrix_sync(fa, ap, K);
        load_matrix_sync(fb, s_B[buf], 32);
        mma_sync(fc, fa, fb, fc);
        __syncthreads();
        // load next tile's qw into nbuf for next iteration (after mma, before next decode)
        if(k0+32 < K){
            int kt_next = (k0+32)/32;
            int tile_base_n = (kt_next*Ntiles + tn)*256;
            for(int i=threadIdx.x*8; i<256; i+=32*8) *(uint64_t*)(s_qw[nbuf]+i) = *(const uint64_t*)(qw_tiled+tile_base_n+i);
            __syncthreads();
        }
    }
    float* cp = C + tm*16*N + tn*16;
    store_matrix_sync(cp, fc, N, mem_row_major);
}

// Larger K-chunk 64 variant (16x16x64, double tile width) — halves iterations to 64 for K=4096, but needs 128 codewords per tile (512B qw per tile)
__global__ void fused_m6_repacked_k64(const hip_fp8_e4m3* A, const uint8_t* qw_tiled64, const uint8_t* cb, float* C, int M,int N,int K){
    // K must be multiple of 64, N multiple of 16
    int tm=blockIdx.y, tn=blockIdx.x;
    int Ntiles=N/16;
    __shared__ uint8_t s_cb[2048];
    __shared__ uint8_t s_qw[512]; // 16 rows * 8 codewords*4B
    __shared__ hip_fp8_e4m3 s_B[1024]; // 16x64=1024
    fragment<matrix_a,16,16,64,hip_fp8_e4m3,row_major> fa;
    fragment<matrix_b,16,16,64,hip_fp8_e4m3,col_major> fb;
    fragment<accumulator,16,16,64,float> fc; fill_fragment(fc,0.0f);
    for(int i=threadIdx.x*8; i<2048; i+=32*8){
        if(i+8<=2048) *(uint64_t*)(s_cb+i)=*(const uint64_t*)(cb+i);
    }
    __syncthreads();
    for(int k0=0;k0<K;k0+=64){
        int kt=k0/64;
        int tile_base=(kt*Ntiles+tn)*512;
        for(int i=threadIdx.x*8; i<512; i+=32*8) *(uint64_t*)(s_qw+i)=*(const uint64_t*)(qw_tiled64+tile_base+i);
        __syncthreads();
        // 128 codewords per tile, 32 threads => 4 per thread
        for(int rep=0; rep<4; ++rep){
            int c=threadIdx.x*4+rep;
            if(c<128){
                int n_local=c/8, vec=c%8; // 8 vecs per row (64/8)
                int qw_off=n_local*32+vec*4; // 32B per row (8*4)
                uint32_t code=*(const uint32_t*)(s_qw+qw_off);
                uint32_t c0=(code>>0)&0xFF, c1=(code>>8)&0xFF, c2=(code>>16)&0xFF, c3=(code>>24)&0xFF;
                s_B[n_local*64+vec*8+0].__x=s_cb[c0*2+0];
                s_B[n_local*64+vec*8+1].__x=s_cb[c0*2+1];
                s_B[n_local*64+vec*8+2].__x=s_cb[c1*2+512];
                s_B[n_local*64+vec*8+3].__x=s_cb[c1*2+513];
                s_B[n_local*64+vec*8+4].__x=s_cb[c2*2+1024];
                s_B[n_local*64+vec*8+5].__x=s_cb[c2*2+1025];
                s_B[n_local*64+vec*8+6].__x=s_cb[c3*2+1536];
                s_B[n_local*64+vec*8+7].__x=s_cb[c3*2+1537];
            }
        }
        __syncthreads();
        const hip_fp8_e4m3* ap = A + tm*16*K + k0;
        load_matrix_sync(fa, ap, K);
        load_matrix_sync(fb, s_B, 64);
        mma_sync(fc, fa, fb, fc);
        __syncthreads();
    }
    float* cp = C + tm*16*N + tn*16;
    store_matrix_sync(cp, fc, N, mem_row_major);
}

// ---- repack helper (host) ----
void repack_k32_for_m6(const uint8_t* qw, uint8_t* qw_tiled, int N, int K, int rbytes){
    int Ntiles=N/16;
    int Ktiles=K/32;
    for(int nt=0; nt<Ntiles; ++nt){
        for(int kt=0; kt<Ktiles; ++kt){
            int k0=kt*32;
            for(int n_local=0;n_local<16;++n_local){
                int gn=nt*16+n_local;
                for(int vec=0; vec<4; ++vec){
                    int gk=k0+vec*8;
                    int sb=gk/256;
                    int vec_in_sb=(gk%256)/8;
                    int src=gn*rbytes+sb*128+vec_in_sb*4;
                    int dst=(kt*Ntiles+nt)*256 + n_local*16 + vec*4;
                    memcpy(qw_tiled+dst, qw+src, 4);
                }
            }
        }
    }
}
void repack_k64_for_m6(const uint8_t* qw, uint8_t* qw_tiled, int N, int K, int rbytes){
    int Ntiles=N/16;
    int Ktiles=K/64;
    for(int nt=0; nt<Ntiles; ++nt){
        for(int kt=0; kt<Ktiles; ++kt){
            int k0=kt*64;
            for(int n_local=0;n_local<16;++n_local){
                int gn=nt*16+n_local;
                for(int vec=0; vec<8; ++vec){
                    int gk=k0+vec*8;
                    int sb=gk/256;
                    int vec_in_sb=(gk%256)/8;
                    int src=gn*rbytes+sb*128+vec_in_sb*4;
                    int dst=(kt*Ntiles+nt)*512 + n_local*32 + vec*4;
                    memcpy(qw_tiled+dst, qw+src, 4);
                }
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
double bench_fused_m5(int M,int N,int K,int iters=300){
    int kbits=32, nsub=4;
    int n_sb=K/256, tsize=4*kbits, rbytes=n_sb*tsize;
    int base=kbits/nsub, extra=kbits%nsub, subdim=8/nsub, flat=0; for(int i=0;i<nsub;i++){int w=base+(i<extra?1:0); flat+=(1<<w)*subdim;}
    std::mt19937 rng(2);
    std::vector<uint8_t> qw(N*rbytes); for(auto &x:qw) x=rng()%256;
    std::vector<uint8_t> cb(flat); for(auto &x:cb){float v=(rng()%5)*0.5f; x=(uint8_t)(v*10); if(x==0x7F||x==0xFF)x=0;}
    std::vector<hip_fp8_e4m3> A(M*K); for(auto &x:A){float v=(rng()%9-4)*0.5f; x=hip_fp8_e4m3(v);}
    hip_fp8_e4m3 *dA; float *dC; uint8_t *dqw,*dcb;
    hipMalloc(&dA,M*K*sizeof(hip_fp8_e4m3)); hipMalloc(&dC,M*N*sizeof(float)); hipMalloc(&dqw,N*rbytes); hipMalloc(&dcb,flat);
    hipMemcpy(dA,A.data(),M*K*sizeof(hip_fp8_e4m3),hipMemcpyHostToDevice); hipMemcpy(dqw,qw.data(),N*rbytes,hipMemcpyHostToDevice); hipMemcpy(dcb,cb.data(),flat,hipMemcpyHostToDevice);
    dim3 grid((N+15)/16,(M+15)/16);
    for(int i=0;i<20;i++){hipMemset(dC,0,M*N*sizeof(float)); hipLaunchKernelGGL(fused_m5_k32,grid,dim3(32),0,0,dA,dqw,dcb,dC,M,N,K,rbytes); hipDeviceSynchronize();}
    hipEvent_t s,e; hipEventCreate(&s); hipEventCreate(&e);
    hipEventRecord(s); for(int i=0;i<iters;i++) hipLaunchKernelGGL(fused_m5_k32,grid,dim3(32),0,0,dA,dqw,dcb,dC,M,N,K,rbytes); hipEventRecord(e); hipEventSynchronize(e); float ms; hipEventElapsedTime(&ms,s,e); ms/=iters;
    hipFree(dA); hipFree(dC); hipFree(dqw); hipFree(dcb); hipEventDestroy(s); hipEventDestroy(e); return ms;
}
double bench_fused_m6(int M,int N,int K,int variant,int iters=300){
    int kbits=32, nsub=4;
    int n_sb=K/256, tsize=4*kbits, rbytes=n_sb*tsize;
    int flat=0; for(int i=0;i<nsub;i++){int w=kbits/nsub+(i<kbits%nsub?1:0); flat+=(1<<w)*(8/nsub);}
    std::mt19937 rng(2);
    std::vector<uint8_t> qw(N*rbytes); for(auto &x:qw) x=rng()%256;
    std::vector<uint8_t> cb(flat); for(auto &x:cb){float v=(rng()%5)*0.5f; x=(uint8_t)(v*10); if(x==0x7F||x==0xFF)x=0;}
    std::vector<hip_fp8_e4m3> A(M*K); for(auto &x:A){float v=(rng()%9-4)*0.5f; x=hip_fp8_e4m3(v);}
    // repack
    std::vector<uint8_t> qw_tiled;
    int tiled_bytes=0;
    if(variant==0 || variant==1){
        int Ntiles=N/16, Ktiles=K/32;
        tiled_bytes=Ntiles*Ktiles*256;
        qw_tiled.resize(tiled_bytes);
        repack_k32_for_m6(qw.data(), qw_tiled.data(), N,K,rbytes);
    } else {
        int Ntiles=N/16, Ktiles=K/64;
        tiled_bytes=Ntiles*Ktiles*512;
        qw_tiled.resize(tiled_bytes);
        repack_k64_for_m6(qw.data(), qw_tiled.data(), N,K,rbytes);
    }
    hip_fp8_e4m3 *dA; float *dC; uint8_t *dqw,*dcb;
    hipMalloc(&dA,M*K*sizeof(hip_fp8_e4m3)); hipMalloc(&dC,M*N*sizeof(float)); hipMalloc(&dqw,tiled_bytes); hipMalloc(&dcb,flat);
    hipMemcpy(dA,A.data(),M*K*sizeof(hip_fp8_e4m3),hipMemcpyHostToDevice); hipMemcpy(dqw,qw_tiled.data(),tiled_bytes,hipMemcpyHostToDevice); hipMemcpy(dcb,cb.data(),flat,hipMemcpyHostToDevice);
    dim3 grid((N+15)/16,(M+15)/16);
    for(int i=0;i<20;i++){hipMemset(dC,0,M*N*sizeof(float));
        if(variant==0) hipLaunchKernelGGL(fused_m6_repacked_k32,grid,dim3(32),0,0,dA,dqw,dcb,dC,M,N,K);
        else if(variant==1) hipLaunchKernelGGL(fused_m6_repacked_db_k32,grid,dim3(32),0,0,dA,dqw,dcb,dC,M,N,K);
        else hipLaunchKernelGGL(fused_m6_repacked_k64,grid,dim3(32),0,0,dA,dqw,dcb,dC,M,N,K);
        hipDeviceSynchronize();}
    hipEvent_t s,e; hipEventCreate(&s); hipEventCreate(&e);
    hipEventRecord(s); for(int i=0;i<iters;i++){
        if(variant==0) hipLaunchKernelGGL(fused_m6_repacked_k32,grid,dim3(32),0,0,dA,dqw,dcb,dC,M,N,K);
        else if(variant==1) hipLaunchKernelGGL(fused_m6_repacked_db_k32,grid,dim3(32),0,0,dA,dqw,dcb,dC,M,N,K);
        else hipLaunchKernelGGL(fused_m6_repacked_k64,grid,dim3(32),0,0,dA,dqw,dcb,dC,M,N,K);
    } hipEventRecord(e); hipEventSynchronize(e); float ms; hipEventElapsedTime(&ms,s,e); ms/=iters;
    hipFree(dA); hipFree(dC); hipFree(dqw); hipFree(dcb); hipEventDestroy(s); hipEventDestroy(e); return ms;
}
// ---- isolated decode kernels for timer evidence ----
__global__ void decode_only_scattered(const uint8_t* qw, const uint8_t* cb, uint8_t* out, int N,int K,int rbytes){
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    int total=N*(K/8); // codewords = N*K/8 for k32
    if(tid>=total) return;
    int gn=tid/(K/8);
    int v=tid%(K/8);
    int sb=(v*8)/256, vec=(v*8)%256/8;
    int base_off = gn*rbytes + sb*128 + vec*4;
    uint32_t code = *(const uint32_t*)(qw + base_off);
    // decode to out (8 bytes)
    uint32_t c0=(code>>0)&0xFF, c1=(code>>8)&0xFF, c2=(code>>16)&0xFF, c3=(code>>24)&0xFF;
    // cb is 2048 bytes, need global load
    // We use cb as LUT, but here we use global cb (not LDS) to mimic scattered path
    out[tid*8+0]=cb[c0*2+0]; out[tid*8+1]=cb[c0*2+1];
    out[tid*8+2]=cb[c1*2+512]; out[tid*8+3]=cb[c1*2+513];
    out[tid*8+4]=cb[c2*2+1024]; out[tid*8+5]=cb[c2*2+1025];
    out[tid*8+6]=cb[c3*2+1536]; out[tid*8+7]=cb[c3*2+1537];
}
__global__ void decode_only_repacked(const uint8_t* qw_tiled, const uint8_t* cb, uint8_t* out, int N,int K){
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    int total=N*(K/8);
    if(tid>=total) return;
    // coalesced: qw_tiled is tile-contiguous 256B per 16x32 tile
    // Need to map tid -> tile + offset
    int Ntiles=N/16, Ktiles=K/32;
    int codewords_per_tile=64;
    int tiles=Ntiles*Ktiles;
    int tile = tid / codewords_per_tile;
    int c_in_tile = tid % codewords_per_tile;
    int nt = tile % Ntiles;
    int kt = tile / Ntiles;
    int n_local = c_in_tile / 4, vec = c_in_tile % 4;
    int tile_base = (kt*Ntiles + nt)*256;
    int qw_off = tile_base + n_local*16 + vec*4;
    uint32_t code = *(const uint32_t*)(qw_tiled + qw_off);
    uint32_t c0=(code>>0)&0xFF, c1=(code>>8)&0xFF, c2=(code>>16)&0xFF, c3=(code>>24)&0xFF;
    // Use cb in global for this probe (LDS variant would be separate)
    // To keep similar to fused's LDS cb, we use global but coalesced cb loads are small
    out[tid*8+0]=cb[c0*2+0]; out[tid*8+1]=cb[c0*2+1];
    out[tid*8+2]=cb[c1*2+512]; out[tid*8+3]=cb[c1*2+513];
    out[tid*8+4]=cb[c2*2+1024]; out[tid*8+5]=cb[c2*2+1025];
    out[tid*8+6]=cb[c3*2+1536]; out[tid*8+7]=cb[c3*2+1537];
}
double bench_decode_isolated(int N,int K, bool repacked, int iters=300){
    int kbits=32, nsub=4,n_sb=K/256, tsize=4*kbits, rbytes=n_sb*tsize;
    int flat=2048; // k32
    std::mt19937 rng(2);
    std::vector<uint8_t> qw(N*rbytes); for(auto &x:qw) x=rng()%256;
    std::vector<uint8_t> cb(flat); for(auto &x:cb) x=rng()%256;
    std::vector<uint8_t> qw_tiled;
    int tiled_bytes=0;
    uint8_t *dqw; uint8_t *dcb; uint8_t *dout;
    hipMalloc(&dcb, flat); hipMemcpy(dcb, cb.data(), flat, hipMemcpyHostToDevice);
    hipMalloc(&dout, (size_t)N*K);
    if(repacked){
        int Ntiles=N/16, Ktiles=K/32;
        tiled_bytes=Ntiles*Ktiles*256;
        qw_tiled.resize(tiled_bytes);
        repack_k32_for_m6(qw.data(), qw_tiled.data(), N,K,rbytes);
        hipMalloc(&dqw, tiled_bytes); hipMemcpy(dqw, qw_tiled.data(), tiled_bytes, hipMemcpyHostToDevice);
    } else {
        hipMalloc(&dqw, N*rbytes); hipMemcpy(dqw, qw.data(), N*rbytes, hipMemcpyHostToDevice);
    }
    int total=N*(K/8);
    dim3 grid((total+255)/256,1); dim3 block(256);
    for(int i=0;i<20;i++){ if(repacked) hipLaunchKernelGGL(decode_only_repacked,grid,block,0,0,dqw,dcb,dout,N,K); else hipLaunchKernelGGL(decode_only_scattered,grid,block,0,0,dqw,dcb,dout,N,K,rbytes); hipDeviceSynchronize(); }
    hipEvent_t s,e; hipEventCreate(&s); hipEventCreate(&e);
    hipEventRecord(s); for(int i=0;i<iters;i++){ if(repacked) hipLaunchKernelGGL(decode_only_repacked,grid,block,0,0,dqw,dcb,dout,N,K); else hipLaunchKernelGGL(decode_only_scattered,grid,block,0,0,dqw,dcb,dout,N,K,rbytes); } hipEventRecord(e); hipEventSynchronize(e); float ms; hipEventElapsedTime(&ms,s,e); ms/=iters;
    hipFree(dqw); hipFree(dcb); hipFree(dout); hipEventDestroy(s); hipEventDestroy(e);
    return ms;
}

bool validate_m6(){
    int M=16,N=32,K=256,kbits=32,nsub=4;
    int n_sb=K/256, tsize=4*kbits, rbytes=n_sb*tsize;
    int subdim=8/nsub;
    int flat=0; for(int i=0;i<nsub;i++){int w=kbits/nsub+(i<kbits%nsub?1:0); flat+=(1<<w)*subdim;}
    std::mt19937 rng(123);
    std::vector<uint8_t> qw(N*rbytes); for(auto &x:qw) x=rng()%256;
    std::vector<uint8_t> cb(flat); for(auto &x:cb){float v=(rng()%5)*0.5f; x=(uint8_t)(v*10); if(x==0x7F||x==0xFF)x=0;}
    std::vector<hip_fp8_e4m3> A(M*K); for(auto &a:A){float v=(rng()%9-4)*0.5f; a=hip_fp8_e4m3(v);}
    // CPU reference decode W
    std::vector<int> widths(nsub); int b=kbits/nsub,e=kbits%nsub; for(int i=0;i<nsub;i++) widths[i]=b+(i<e);
    std::vector<int> tb(nsub,0); for(int i=1;i<nsub;i++) tb[i]=tb[i-1]+(1<<widths[i-1])*subdim;
    std::vector<uint8_t> W(N*K);
    for(int gn=0;gn<N;gn++) for(int sb=0;sb<n_sb;sb++) for(int v=0;v<32;v++){
        int bb=(v*kbits)/8, bs=(v*kbits)%8;
        uint64_t win=0; for(int bb2=0;bb2<8;bb2++){uint8_t by=0; if(bb+bb2<tsize) by=qw[gn*rbytes+sb*tsize+bb+bb2]; win|=(uint64_t)by<<(8*bb2);}
        uint64_t code=(win>>bs)&((1ULL<<kbits)-1);
        for(int j=0;j<8;j++){int sub=j/subdim, local=j%subdim; int off=0; for(int t=0;t<sub;t++) off+=widths[t]; int mask=(1<<widths[sub])-1; uint64_t sidx=(code>>off)&mask; int fidx=tb[sub]+sidx*subdim+local; W[gn*K+sb*256+v*8+j]=cb[fidx];}
    }
    std::vector<float> Af(M*K), Wf(N*K);
    for(int i=0;i<M*K;i++) Af[i]=(float)A[i];
    for(int i=0;i<N*K;i++){hip_fp8_e4m3 f; f.__x=W[i]; Wf[i]=(float)f;}
    std::vector<float> ref(M*N,0);
    for(int m=0;m<M;m++) for(int n=0;n<N;n++){float acc=0; for(int k=0;k<K;k++) acc+=Af[m*K+k]*Wf[n*K+k]; ref[m*N+n]=acc;}
    // repack
    int Ntiles=N/16, Ktiles=K/32;
    int tiled_bytes=Ntiles*Ktiles*256;
    std::vector<uint8_t> qw_tiled(tiled_bytes);
    repack_k32_for_m6(qw.data(), qw_tiled.data(), N,K,rbytes);
    hip_fp8_e4m3 *dA; uint8_t *dqw,*dcb; float *dC;
    hipMalloc(&dA,M*K*sizeof(hip_fp8_e4m3)); hipMalloc(&dqw,tiled_bytes); hipMalloc(&dcb,flat); hipMalloc(&dC,M*N*sizeof(float));
    hipMemcpy(dA,A.data(),M*K*sizeof(hip_fp8_e4m3),hipMemcpyHostToDevice);
    hipMemcpy(dqw,qw_tiled.data(),tiled_bytes,hipMemcpyHostToDevice);
    hipMemcpy(dcb,cb.data(),flat,hipMemcpyHostToDevice);
    hipMemset(dC,0,M*N*sizeof(float));
    dim3 grid((N+15)/16,(M+15)/16);
    hipLaunchKernelGGL(fused_m6_repacked_k32,grid,dim3(32),0,0,dA,dqw,dcb,dC,M,N,K);
    hipDeviceSynchronize();
    std::vector<float> gpu(M*N);
    hipMemcpy(gpu.data(),dC,M*N*sizeof(float),hipMemcpyDeviceToHost);
    float max_abs=0; int mism=0;
    for(int i=0;i<M*N;i++){float diff=std::abs(gpu[i]-ref[i]); max_abs=std::max(max_abs,diff); if(diff>1e-3) mism++;}
    std::cout<<"M6 validate repacked_k32 M"<<M<<" N"<<N<<" K"<<K<<" max_abs "<<max_abs<<" mism "<<mism<<" "<<(max_abs<1e-3?"PASS":"FAIL")<<"\n";
    bool ok=max_abs<1e-3;
    // also validate db and k64 if ok
    if(ok){
        hipMemset(dC,0,M*N*sizeof(float));
        hipLaunchKernelGGL(fused_m6_repacked_db_k32,grid,dim3(32),0,0,dA,dqw,dcb,dC,M,N,K);
        hipDeviceSynchronize();
        hipMemcpy(gpu.data(),dC,M*N*sizeof(float),hipMemcpyDeviceToHost);
        max_abs=0; mism=0; for(int i=0;i<M*N;i++){float diff=std::abs(gpu[i]-ref[i]); max_abs=std::max(max_abs,diff); if(diff>1e-3) mism++;}
        std::cout<<"M6 validate repacked_db_k32 max_abs "<<max_abs<<" mism "<<mism<<" "<<(max_abs<1e-3?"PASS":"FAIL")<<"\n";
        ok&=max_abs<1e-3;
        // k64 validate with K=256 (Ktiles=4 for 64) — need separate tiled64 buffer
        {
            int tiled64 = Ntiles*(K/64)*512;
            std::vector<uint8_t> qw64(tiled64);
            repack_k64_for_m6(qw.data(), qw64.data(), N,K,rbytes);
            uint8_t *dqw64; hipMalloc(&dqw64,tiled64); hipMemcpy(dqw64,qw64.data(),tiled64,hipMemcpyHostToDevice);
            hipMemset(dC,0,M*N*sizeof(float));
            hipLaunchKernelGGL(fused_m6_repacked_k64,grid,dim3(32),0,0,dA,dqw64,dcb,dC,M,N,K);
            hipDeviceSynchronize();
            hipMemcpy(gpu.data(),dC,M*N*sizeof(float),hipMemcpyDeviceToHost);
            max_abs=0; mism=0; for(int i=0;i<M*N;i++){float diff=std::abs(gpu[i]-ref[i]); max_abs=std::max(max_abs,diff); if(diff>1e-3) mism++;}
            std::cout<<"M6 validate repacked_k64 max_abs "<<max_abs<<" mism "<<mism<<" "<<(max_abs<1e-3?"PASS":"FAIL")<<"\n";
            ok&=max_abs<1e-3;
            hipFree(dqw64);
        }
    }
    hipFree(dA); hipFree(dqw); hipFree(dcb); hipFree(dC);
    return ok;
}

static double median_vec(std::vector<double> &v){ std::sort(v.begin(), v.end()); return v[v.size()/2]; }
int main(){
    hipDeviceProp_t p; hipGetDeviceProperties(&p,0);
    std::cout<<"Device "<<p.name<<" "<<p.gcnArchName<<" "<<p.multiProcessorCount<<" CUs\n";
    if(!validate_m6()) std::cout<<"VALIDATION FAILED - bench still runs for diagnosis\n";
    struct Shape{int M,N,K;}; std::vector<Shape> shapes={{16,4096,4096},{32,4096,4096},{64,4096,4096},{128,4096,4096},{16,1024,4096},{16,2048,1024}};
    // stable plain vs plain32 with median of 20 reps (each rep = 300 iters avg)
    std::cout<<"=== Plain optimized (direct16 vs direct32) median of 20 reps (300 iters each) ===\n";
    for(auto s:shapes){
        std::vector<double> v16,v32;
        for(int r=0;r<20;r++){ v16.push_back(bench_plain(s.M,s.N,s.K,0,300)); v32.push_back(bench_plain(s.M,s.N,s.K,1,300)); }
        double ms16=median_vec(v16), ms32=median_vec(v32);
        std::cout<<"M"<<s.M<<" N"<<s.N<<" K"<<s.K<<" plain16 median "<<ms16<<"ms (min "<<*std::min_element(v16.begin(),v16.end())<<" max "<<*std::max_element(v16.begin(),v16.end())<<") | plain32 median "<<ms32<<"ms\n";
    }
    std::cout<<"\n=== M5 baseline fused_m5_k32 vs plain (median per shape, interleaved) ===\n";
    for(auto s:shapes){
        std::vector<double> ratios; std::vector<double> plains, fuseds;
        for(int r=0;r<20;r++){
            double ms_plain=bench_plain(s.M,s.N,s.K,0,300);
            double ms_m5=bench_fused_m5(s.M,s.N,s.K,300);
            plains.push_back(ms_plain); fuseds.push_back(ms_m5); ratios.push_back(ms_plain/ms_m5);
        }
        double ms_plain=median_vec(plains), ms_m5=median_vec(fuseds), ratio=median_vec(ratios);
        std::cout<<"M"<<s.M<<" N"<<s.N<<" K"<<s.K<<" plain median "<<ms_plain<<" ms_fused_m5 median "<<ms_m5<<" ratio median "<<ratio<<" "<<(ratio>=0.97?"PASS":"SLOW")<<"\n";
    }
    std::cout<<"\n=== M6 repacked_k32 (coalesced) vs plain (median) ===\n";
    for(auto s:shapes){
        std::vector<double> ratios, plains, fuseds;
        for(int r=0;r<20;r++){
            double ms_plain=bench_plain(s.M,s.N,s.K,0,300);
            double ms_m6=bench_fused_m6(s.M,s.N,s.K,0,300);
            plains.push_back(ms_plain); fuseds.push_back(ms_m6); ratios.push_back(ms_plain/ms_m6);
        }
        double ms_plain=median_vec(plains), ms_m6=median_vec(fuseds), ratio=median_vec(ratios);
        double tf_plain=(2.0*s.M*s.N*s.K)/(ms_plain/1000)/1e12;
        double tf_m6=(2.0*s.M*s.N*s.K)/(ms_m6/1000)/1e12;
        double bw_plain=(double)(s.M*s.K + s.N*s.K + s.M*s.N*4)/(ms_plain/1000)/1e9;
        double bw_eff=(double)(s.M*s.K + s.N*(s.K/256*128) + s.M*s.N*4)/(ms_m6/1000)/1e9;
        std::cout<<"M"<<s.M<<" N"<<s.N<<" K"<<s.K<<" plain median "<<ms_plain<<"ms TF "<<tf_plain<<" BW "<<bw_plain<<" | m6_repacked median "<<ms_m6<<"ms TF "<<tf_m6<<" effBW "<<bw_eff<<" ratio "<<ratio<<" "<<(ratio>=0.97?"PASS":"SLOW")<<"\n";
    }
    std::cout<<"\n=== M6 repacked_db_k32 (double-buffer) median ===\n";
    for(auto s:shapes){
        std::vector<double> ratios, plains, fuseds;
        for(int r=0;r<20;r++){
            double ms_plain=bench_plain(s.M,s.N,s.K,0,300);
            double ms_m6=bench_fused_m6(s.M,s.N,s.K,1,300);
            plains.push_back(ms_plain); fuseds.push_back(ms_m6); ratios.push_back(ms_plain/ms_m6);
        }
        double ms_plain=median_vec(plains), ms_m6=median_vec(fuseds), ratio=median_vec(ratios);
        std::cout<<"M"<<s.M<<" N"<<s.N<<" K"<<s.K<<" plain median "<<ms_plain<<" m6_db median "<<ms_m6<<" ratio "<<ratio<<" "<<(ratio>=0.97?"PASS":"SLOW")<<"\n";
    }
    std::cout<<"\n=== M6 repacked_k64 (16x16x64) median ===\n";
    for(auto s:shapes){
        if(s.K%64!=0) continue;
        std::vector<double> ratios, plains, fuseds;
        for(int r=0;r<20;r++){
            double ms_plain=bench_plain(s.M,s.N,s.K,0,300);
            double ms_m6=bench_fused_m6(s.M,s.N,s.K,2,300);
            plains.push_back(ms_plain); fuseds.push_back(ms_m6); ratios.push_back(ms_plain/ms_m6);
        }
        double ms_plain=median_vec(plains), ms_m6=median_vec(fuseds), ratio=median_vec(ratios);
        std::cout<<"M"<<s.M<<" N"<<s.N<<" K"<<s.K<<" plain median "<<ms_plain<<" m6_k64 median "<<ms_m6<<" ratio "<<ratio<<" "<<(ratio>=0.97?"PASS":"SLOW")<<"\n";
    }
    // isolated decode probe + repack cost
    {
        std::cout<<"\n=== Isolated decode probe (N=4096 K=4096 k32, 8 MB packed ->16 MB decoded) ===\n";
        double ms_scat = bench_decode_isolated(4096,4096,false,300);
        double ms_rep = bench_decode_isolated(4096,4096,true,300);
        double bw_scat = (double)(4096*4096)/(ms_scat/1000)/1e9; // decoded bytes
        double bw_rep_packed = (double)(4096*2048)/(ms_rep/1000)/1e9;
        double bw_rep_decoded = (double)(4096*4096)/(ms_rep/1000)/1e9;
        std::cout<<"scattered decode "<<ms_scat<<"ms decoded BW "<<bw_scat<<" GB/s\n";
        std::cout<<"repacked coalesced decode "<<ms_rep<<"ms packed BW "<<bw_rep_packed<<" decoded BW "<<bw_rep_decoded<<" GB/s\n";
        double plain_bw=517; // from earlier plain median 0.040ms ~340 GB/s actual, roof 620
        double saving_bytes=4096*4096 - 4096*2048; // 8 MB
        double saving_ms = saving_bytes / (plain_bw*1e9) *1000;
        std::cout<<"plain saving at "<<plain_bw<<" GB/s = "<<saving_ms<<" ms; repacked decode "<<ms_rep<<" ms "<<(ms_rep > saving_ms ? "> saving (cannot hide)" : "< saving")<<"\n";
    }
    // repack host cost microbench: offline preprocessing time (reuse chrono)
    {
        int N=4096,K=4096; int n_sb=K/256, rbytes=n_sb*128;
        std::vector<uint8_t> qw(N*rbytes); for(auto &x:qw) x=rand()%256;
        std::vector<uint8_t> tiled(N/16*(K/32)*256);
        auto t0=std::chrono::high_resolution_clock::now();
        for(int i=0;i<20;i++) repack_k32_for_m6(qw.data(), tiled.data(), N,K,rbytes);
        auto t1=std::chrono::high_resolution_clock::now();
        double ms=std::chrono::duration<double,std::milli>(t1-t0).count()/20;
        double bw=(double)N*rbytes/(ms/1000)/1e9;
        std::cout<<"\nRepack host cost (once per weight, not per GEMM): "<<ms<<" ms for N="<<N<<" K="<<K<<" ("<<N*rbytes/(1<<20)<<" MB) BW "<<bw<<" GB/s\n";
        std::cout<<"Repack amortizes: weight is stationary, repack is offline preprocessing (legal per M6 spec), not counted in GEMM time.\n";
    }
    return 0;
}
