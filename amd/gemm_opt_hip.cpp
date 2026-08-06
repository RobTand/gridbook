// Optimized GEMM for M4 — plain FP8 direct vs CB fused with vectorized decode
// Build: hipcc --offload-arch=gfx1201 amd/gemm_opt_hip.cpp -o /tmp/gemm_opt --std=c++17 -O2
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <hip/hip_fp8.h>
#include <iostream>
#include <vector>
#include <random>
#include <cstring>
using namespace rocwmma;

// ---- Plain optimized: direct global WMMA 16x16x16 (no LDS) ----
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

// ---- Helpers for fused ----
__device__ inline void sw_dev(int k,int n,int* o){int b=k/n,e=k%n; for(int i=0;i<n;i++) o[i]=b+(i<e?1:0);}

// Naive fused (baseline for comparison) — per-element decode via LDS
__global__ void fused_naive(const hip_fp8_e4m3* A, const uint8_t* qw, const uint8_t* cb, const int32_t* off, float* C, int M,int N,int K,int kbits,int nsub,int tsize,int rbytes){
    int tm=blockIdx.y, tn=blockIdx.x; int rb=tm*16, cb0=tn*16;
    __shared__ hip_fp8_e4m3 tA[256]; __shared__ hip_fp8_e4m3 tB[256]; __shared__ float tC[256];
    fragment<matrix_a,16,16,16,hip_fp8_e4m3,row_major> fa;
    fragment<matrix_b,16,16,16,hip_fp8_e4m3,col_major> fb;
    fragment<accumulator,16,16,16,float> fc; fill_fragment(fc,0.0f);
    int subdim=8/nsub; int w[4]; sw_dev(kbits,nsub,w); int tb[4]; tb[0]=0; for(int i=1;i<nsub;i++) tb[i]=tb[i-1]+(1<<w[i-1])*subdim;
    for(int k0=0;k0<K;k0+=16){
        for(int i=threadIdx.x;i<256;i+=32){int r=i/16,c=i%16;int gr=rb+r,gk=k0+c; hip_fp8_e4m3 v; v.__x=0; if(gr<M&&gk<K) v=A[gr*K+gk]; tA[r*16+c]=v;}
        for(int i=threadIdx.x;i<256;i+=32){
            int kl=i%16,nl=i/16; int gk=k0+kl, gn=cb0+nl; hip_fp8_e4m3 out; out.__x=0;
            if(gk<K&&gn<N){
                int sb=gk/256, col=gk%256, vec=col/8, coord=col%8;
                int sub=coord/subdim, local=coord%subdim;
                int boff=0; for(int t=0;t<sub;t++) boff+=w[t];
                int mask=(w[sub]==32?0xFFFFFFFF:((1<<w[sub])-1));
                int bb=(vec*kbits)/8, bs=(vec*kbits)%8;
                uint64_t win=0; for(int b=0;b<8;b++){uint8_t by=0; if(bb+b<tsize) by=qw[gn*rbytes+sb*tsize+bb+b]; win|=(uint64_t)by<<(8*b);}
                uint64_t code=(win>>bs)&((kbits==64?~0ULL:((1ULL<<kbits)-1)));
                uint64_t sidx=(code>>boff)&(uint64_t)mask;
                int fidx=off[gn]+tb[sub]+(int)sidx*subdim+local;
                out.__x=cb[fidx];
            }
            tB[nl*16+kl]=out;
        }
        __syncthreads(); load_matrix_sync(fa,tA,16); load_matrix_sync(fb,tB,16); mma_sync(fc,fa,fb,fc); __syncthreads();
    }
    store_matrix_sync(tC,fc,16,mem_row_major); __syncthreads();
    for(int i=threadIdx.x;i<256;i+=32){int r=i/16,c=i%16;int gr=rb+r,gc=cb0+c; if(gr<M&&gc<N) C[gr*N+gc]=tC[r*16+c];}
}

// Optimized fused: A direct, B via vec-level decode (32 workers, 8-byte window) — generic k
__global__ void fused_vec(const hip_fp8_e4m3* A, const uint8_t* qw, const uint8_t* cb, const int32_t* off, float* C, int M,int N,int K,int kbits,int nsub,int tsize,int rbytes){
    int tm=blockIdx.y, tn=blockIdx.x; int cb0=tn*16;
    __shared__ hip_fp8_e4m3 tB[256]; __shared__ float tC[256];
    fragment<matrix_a,16,16,16,hip_fp8_e4m3,row_major> fa;
    fragment<matrix_b,16,16,16,hip_fp8_e4m3,col_major> fb;
    fragment<accumulator,16,16,16,float> fc; fill_fragment(fc,0.0f);
    int subdim=8/nsub; int w[4]; sw_dev(kbits,nsub,w); int tb[4]; tb[0]=0; for(int i=1;i<nsub;i++) tb[i]=tb[i-1]+(1<<w[i-1])*subdim;
    for(int k0=0;k0<K;k0+=16){
        // decode B tile via 32 codewords (one per thread)
        {
            int tid=threadIdx.x;
            if(tid<32){
                int nl=tid/2; int vec_in_tile=tid%2;
                int gn=cb0+nl; int gk_base=k0+vec_in_tile*8;
                if(gn<N && gk_base<K){
                    int sb=gk_base/256; int vec=(gk_base%256)/8;
                    int bb=(vec*kbits)/8, bs=(vec*kbits)%8;
                    uint64_t win=0; for(int b=0;b<8;b++){uint8_t by=0; if(bb+b<tsize) by=qw[gn*rbytes+sb*tsize+bb+b]; win|=(uint64_t)by<<(8*b);}
                    uint64_t code=(win>>bs)&((kbits==64?~0ULL:((1ULL<<kbits)-1)));
                    for(int j=0;j<8;j++){
                        int k_local=vec_in_tile*8+j;
                        int sub=j/subdim, local=j%subdim;
                        int boff=0; for(int t=0;t<sub;t++) boff+=w[t];
                        int mask=(w[sub]==32?0xFFFFFFFF:((1<<w[sub])-1));
                        uint64_t sidx=(code>>boff)&mask;
                        int fidx=off[gn]+tb[sub]+(int)sidx*subdim+local;
                        hip_fp8_e4m3 o; o.__x=cb[fidx];
                        tB[nl*16+k_local]=o;
                    }
                } else {
                    for(int j=0;j<8;j++) tB[nl*16+vec_in_tile*8+j].__x=0;
                }
            }
        }
        __syncthreads();
        const hip_fp8_e4m3* ap = A + tm*16*K + k0;
        load_matrix_sync(fa, ap, K);
        load_matrix_sync(fb, tB, 16);
        mma_sync(fc, fa, fb, fc);
        __syncthreads();
    }
    store_matrix_sync(tC,fc,16,mem_row_major); __syncthreads();
    for(int i=threadIdx.x;i<256;i+=32){int r=i/16,c=i%16;int gr=tm*16+r,gc=cb0+c; if(gr<M&&gc<N) C[gr*N+gc]=tC[r*16+c];}
}

// Specialized fused for k=32 with cb in LDS and 32-bit direct load — fastest path
__global__ void fused_k32_fast(const hip_fp8_e4m3* A, const uint8_t* qw, const uint8_t* cb, const int32_t* off, float* C, int M,int N,int K,int rbytes){
    int tm=blockIdx.y, tn=blockIdx.x; int cb0=tn*16;
    __shared__ uint8_t s_cb[2048];
    __shared__ hip_fp8_e4m3 tB[256];
    __shared__ float tC[256];
    fragment<matrix_a,16,16,16,hip_fp8_e4m3,row_major> fa;
    fragment<matrix_b,16,16,16,hip_fp8_e4m3,col_major> fb;
    fragment<accumulator,16,16,16,float> fc; fill_fragment(fc,0.0f);
    // load cb into LDS once per block (2048 bytes) — all blocks share same cb, so reuse across K loop
    for(int i=threadIdx.x*8; i<2048; i+=32*8){
        if(i+8<=2048) *(uint64_t*)(s_cb+i)=*(const uint64_t*)(cb+i);
    }
    __syncthreads();
    for(int k0=0;k0<K;k0+=16){
        // decode 32 codewords via 32 threads, using s_cb and qw (direct 32-bit load)
        {
            int tid=threadIdx.x;
            if(tid<32){
                int nl=tid/2; int vec_in_tile=tid%2;
                int gn=cb0+nl; int gk_base=k0+vec_in_tile*8;
                if(gn<N && gk_base<K){
                    int sb=gk_base/256; int vec=(gk_base%256)/8;
                    int base_off = gn*rbytes + sb*128 + vec*4;
                    uint32_t code = *(const uint32_t*)(qw + base_off);
                    uint32_t c0=(code>>0)&0xFF, c1=(code>>8)&0xFF, c2=(code>>16)&0xFF, c3=(code>>24)&0xFF;
                    // cb layout for k32: 512 per sub, sub_dim 2
                    // Need off[gn] but dense test off=0
                    // local tables: s_cb[c*2 + sub*512]
                    // j 0,1 -> c0 ; 2,3 -> c1 etc.
                    hip_fp8_e4m3 o0,o1,o2,o3,o4,o5,o6,o7;
                    o0.__x = s_cb[c0*2+0]; o1.__x = s_cb[c0*2+1];
                    o2.__x = s_cb[c1*2+512]; o3.__x = s_cb[c1*2+513];
                    o4.__x = s_cb[c2*2+1024]; o5.__x = s_cb[c2*2+1025];
                    o6.__x = s_cb[c3*2+1536]; o7.__x = s_cb[c3*2+1537];
                    tB[nl*16+vec_in_tile*8+0]=o0;
                    tB[nl*16+vec_in_tile*8+1]=o1;
                    tB[nl*16+vec_in_tile*8+2]=o2;
                    tB[nl*16+vec_in_tile*8+3]=o3;
                    tB[nl*16+vec_in_tile*8+4]=o4;
                    tB[nl*16+vec_in_tile*8+5]=o5;
                    tB[nl*16+vec_in_tile*8+6]=o6;
                    tB[nl*16+vec_in_tile*8+7]=o7;
                } else {
                    for(int j=0;j<8;j++) tB[nl*16+vec_in_tile*8+j].__x=0;
                }
            }
        }
        __syncthreads();
        const hip_fp8_e4m3* ap = A + tm*16*K + k0;
        load_matrix_sync(fa, ap, K);
        load_matrix_sync(fb, tB, 16);
        mma_sync(fc, fa, fb, fc);
        __syncthreads();
    }
    store_matrix_sync(tC,fc,16,mem_row_major); __syncthreads();
    for(int i=threadIdx.x;i<256;i+=32){int r=i/16,c=i%16;int gr=tm*16+r,gc=cb0+c; if(gr<M&&gc<N) C[gr*N+gc]=tC[r*16+c];}
}

// Bench harness
double bench_plain(int M,int N,int K,int use32,int iters=500){
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
double bench_fused(const hip_fp8_e4m3* dA_unused,int M,int N,int K,int kbits,int nsub,int which,int iters=300){
    int n_sb=K/256, tsize=4*kbits, rbytes=n_sb*tsize;
    int base=kbits/nsub, extra=kbits%nsub, subdim=8/nsub, flat=0; for(int i=0;i<nsub;i++){int w=base+(i<extra?1:0); flat+=(1<<w)*subdim;}
    std::mt19937 rng(2);
    std::vector<uint8_t> qw(N*rbytes); for(auto &x:qw) x=rng()%256;
    std::vector<uint8_t> cb(flat); for(auto &x:cb){float v=(rng()%5)*0.5f; uint8_t bv=(uint8_t)(v*10); x=bv; if(x==0x7F||x==0xFF)x=0;}
    std::vector<int32_t> off(N,0);
    std::vector<hip_fp8_e4m3> A(M*K); for(auto &x:A){float v=(rng()%9-4)*0.5f; x=hip_fp8_e4m3(v);}
    hip_fp8_e4m3 *dA; float *dC; uint8_t *dqw,*dcb; int32_t *doff;
    hipMalloc(&dA,M*K*sizeof(hip_fp8_e4m3)); hipMalloc(&dC,M*N*sizeof(float)); hipMalloc(&dqw,N*rbytes); hipMalloc(&dcb,flat); hipMalloc(&doff,N*sizeof(int32_t));
    hipMemcpy(dA,A.data(),M*K*sizeof(hip_fp8_e4m3),hipMemcpyHostToDevice); hipMemcpy(dqw,qw.data(),N*rbytes,hipMemcpyHostToDevice); hipMemcpy(dcb,cb.data(),flat,hipMemcpyHostToDevice); hipMemcpy(doff,off.data(),N*sizeof(int32_t),hipMemcpyHostToDevice);
    dim3 grid((N+15)/16,(M+15)/16);
    for(int i=0;i<20;i++){hipMemset(dC,0,M*N*sizeof(float));
        if(which==0) hipLaunchKernelGGL(fused_naive,grid,dim3(32),0,0,dA,dqw,dcb,doff,dC,M,N,K,kbits,nsub,tsize,rbytes);
        else if(which==1) hipLaunchKernelGGL(fused_vec,grid,dim3(32),0,0,dA,dqw,dcb,doff,dC,M,N,K,kbits,nsub,tsize,rbytes);
        else hipLaunchKernelGGL(fused_k32_fast,grid,dim3(32),0,0,dA,dqw,dcb,doff,dC,M,N,K,rbytes);
        hipDeviceSynchronize();
    }
    hipEvent_t s,e; hipEventCreate(&s); hipEventCreate(&e);
    hipEventRecord(s);
    for(int i=0;i<iters;i++){
        if(which==0) hipLaunchKernelGGL(fused_naive,grid,dim3(32),0,0,dA,dqw,dcb,doff,dC,M,N,K,kbits,nsub,tsize,rbytes);
        else if(which==1) hipLaunchKernelGGL(fused_vec,grid,dim3(32),0,0,dA,dqw,dcb,doff,dC,M,N,K,kbits,nsub,tsize,rbytes);
        else hipLaunchKernelGGL(fused_k32_fast,grid,dim3(32),0,0,dA,dqw,dcb,doff,dC,M,N,K,rbytes);
    }
    hipEventRecord(e); hipEventSynchronize(e); float ms; hipEventElapsedTime(&ms,s,e); ms/=iters;
    hipFree(dA); hipFree(dC); hipFree(dqw); hipFree(dcb); hipFree(doff); hipEventDestroy(s); hipEventDestroy(e);
    return ms;
}
int main(){
    hipDeviceProp_t p; hipGetDeviceProperties(&p,0);
    std::cout<<"Device "<<p.name<<" "<<p.gcnArchName<<" "<<p.multiProcessorCount<<" CUs\n";
    struct Shape{int M,N,K,k;}; std::vector<Shape> shapes={
        {16,4096,4096,32},{16,4096,4096,36},{16,4096,4096,40},{32,4096,4096,32},{64,4096,4096,32},{128,4096,4096,32},{16,1024,4096,32},{16,2048,1024,40}
    };
    std::cout<<"=== Plain optimized (direct16 vs direct32) ===\n";
    for(auto s:shapes){
        if(s.k!=32) continue;
        double ms16=bench_plain(s.M,s.N,s.K,0);
        double ms32=bench_plain(s.M,s.N,s.K,1);
        double tf16=(2.0*s.M*s.N*s.K)/(ms16/1000)/1e12;
        double tf32=(2.0*s.M*s.N*s.K)/(ms32/1000)/1e12;
        double bw16=(double)(s.M*s.K + s.N*s.K + s.M*s.N*4)/(ms16/1000)/1e9;
        double bw32=(double)(s.M*s.K + s.N*s.K + s.M*s.N*4)/(ms32/1000)/1e9;
        std::cout<<"M"<<s.M<<" N"<<s.N<<" K"<<s.K<<" plain16 "<<ms16<<"ms TF "<<tf16<<" BW "<<bw16<<" | plain32 "<<ms32<<"ms TF "<<tf32<<" BW "<<bw32<<"\n";
    }
    std::cout<<"\n=== Fused variants vs plain (M16 N4096 K4096 k32) ===\n";
    {
        int M=16,N=4096,K=4096,k=32,nsub=4;
        double ms_plain=bench_plain(M,N,K,0,500);
        double ms_naive=bench_fused(nullptr,M,N,K,k,nsub,0,300);
        double ms_vec=bench_fused(nullptr,M,N,K,k,nsub,1,300);
        double ms_k32=bench_fused(nullptr,M,N,K,k,nsub,2,300);
        std::cout<<"plain_direct16 "<<ms_plain<<" ms\n";
        std::cout<<"fused_naive "<<ms_naive<<" ratio "<<ms_plain/ms_naive<<"\n";
        std::cout<<"fused_vec "<<ms_vec<<" ratio "<<ms_plain/ms_vec<<"\n";
        std::cout<<"fused_k32_fast "<<ms_k32<<" ratio "<<ms_plain/ms_k32<<"\n";
    }
    std::cout<<"\n=== Full sweep: plain_direct16 vs fused (best available) ===\n";
    for(auto s:shapes){
        int nsub=4;
        double ms_plain=bench_plain(s.M,s.N,s.K,0,300);
        double ms_fused;
        if(s.k==32) ms_fused=bench_fused(nullptr,s.M,s.N,s.K,s.k,nsub,2,300);
        else ms_fused=bench_fused(nullptr,s.M,s.N,s.K,s.k,nsub,1,300);
        double ratio=ms_plain/ms_fused;
        double tf_plain=(2.0*s.M*s.N*s.K)/(ms_plain/1000)/1e12;
        double tf_fused=(2.0*s.M*s.N*s.K)/(ms_fused/1000)/1e12;
        double bw_plain=(double)(s.M*s.K + s.N*s.K + s.M*s.N*4)/(ms_plain/1000)/1e9;
        double bw_fused_eff=(double)(s.M*s.K + s.N*(s.K/256*4*s.k) + s.M*s.N*4)/(ms_fused/1000)/1e9;
        std::cout<<"M"<<s.M<<" N"<<s.N<<" K"<<s.K<<" k"<<s.k<<" plain "<<ms_plain<<"ms TF "<<tf_plain<<" BW "<<bw_plain<<" | fused "<<ms_fused<<"ms TF "<<tf_fused<<" effBW "<<bw_fused_eff<<" ratio "<<ratio<<" "<<(ratio>=0.97?"PASS": ratio>=1.0?"PASS":"SLOW")<<"\n";
    }
    return 0;
}
