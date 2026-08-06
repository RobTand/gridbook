// M5 pipelined fused GEMM for gfx1201 — K=32, double-buffered LDS, vectorized qw
// Build: hipcc --offload-arch=gfx1201 amd/gemm_m5_hip.cpp -o /tmp/gemm_m5 --std=c++17 -O2
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <hip/hip_fp8.h>
#include <iostream>
#include <vector>
#include <random>
#include <cstring>
using namespace rocwmma;

// ---- Plain optimized: direct global WMMA 16x16x16 / 16x16x32 (same as M4) ----
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

// ---- M5 pipelined fused K32: 16x16x32 tile, double-buffered LDS B (2x512), vectorized qw, LDS cb ----
__global__ void fused_m5_k32(const hip_fp8_e4m3* A, const uint8_t* qw, const uint8_t* cb, float* C, int M,int N,int K,int rbytes){
    int tm=blockIdx.y, tn=blockIdx.x;
    int cb0 = tn*16;
    // LDS: codebook 2048B + two B buffers 512*1B each = 3072B
    __shared__ uint8_t s_cb[2048];
    __shared__ hip_fp8_e4m3 s_B[2][512]; // double buffered
    __shared__ float tC[256];
    fragment<matrix_a,16,16,32,hip_fp8_e4m3,row_major> fa;
    fragment<matrix_b,16,16,32,hip_fp8_e4m3,col_major> fb;
    fragment<accumulator,16,16,32,float> fc; fill_fragment(fc,0.0f);

    // Load cb into LDS once per block via 64-bit vector loads (coalesced)
    for(int i=threadIdx.x*8; i<2048; i+=32*8){
        if(i+8 <= 2048) *(uint64_t*)(s_cb+i) = *(const uint64_t*)(cb+i);
    }
    __syncthreads();

    for(int k0=0;k0<K;k0+=32){
        int buf = (k0/32) % 2;
        // ---- decode B tile for this k0 into s_B[buf] ----
        {
            int tid = threadIdx.x;
            // 64 codewords per tile, 32 threads => 2 per thread
            // Each codeword produces 8 FP8 values, but we need to place them col_major: tB[n*16 + k_local]? For 16x32 col_major, layout is n*32 + k? Actually shared layout for WMMA col_major 16x32 expects stride 16? Let's use same as before: tile col_major with ldm=16 for 16x16, but for 16x32 ldm=16 still, storing as [n][k] with n*32 + k? Need to verify rocWMMA col_major 16x32 load expects pointer to 16x32 col_major with leading dimension K? The load_matrix_sync for B col_major with size 16x32 uses K as leading dimension? In plain_direct32, it loads from global with ldm=K (e.g., 4096). For LDS version, ldm is 16? Wait for 16x16 LDS, ldm=16. For 16x32, ldm should be 16 as well? Actually tile is 16 (N) x 32 (K), col_major means columns are contiguous K=32? No, col_major for B means K is row dimension, N is col. So loading 16x32 col_major tile with 16 columns, 32 rows: each column contiguous 32. So ldm = 32? But earlier for 16x16 we used ldm=16, which matches. For 16x32, ldm should be 32 if using LDS? However rocWMMA example for 16x32 often still uses ldm=16? Let's check spec: fragment<matrix_b,16,16,32,hip_fp8_e4m3,col_major> means M=16,N=16,K=32. So B is KxN =32x16 col_major, so leading dimension is K=32. So LDS tile should be 32*16 =512 with stride 32. So element (k,n) at n*32 + k.
            // We'll implement that.
            for(int rep=0; rep<2; ++rep){
                int c = tid*2 + rep; // 0..63
                if(c < 64){
                    int n_local = c / 4; // 0..15
                    int vec = c % 4;     // 0..3
                    int gn = cb0 + n_local;
                    int gk_base = k0 + vec*8;
                    hip_fp8_e4m3 out[8];
                    if(gn < N && gk_base < K){
                        int sb = gk_base / 256;
                        int vec_in_sb = (gk_base % 256)/8;
                        int base_off = gn * rbytes + sb*128 + vec_in_sb*4;
                        uint32_t code = *(const uint32_t*)(qw + base_off);
                        uint32_t c0=(code>>0)&0xFF, c1=(code>>8)&0xFF, c2=(code>>16)&0xFF, c3=(code>>24)&0xFF;
                        // expand 4 codewords -> 8*? Actually each code is 1 codeword = 8 values, but we have 4 codewords per vec_in_tile? Wait vec is per tile 32/8=4 codewords, each codeword 4 bytes, each code cb lookup 2*? For k32 split is 4 sub-tables each 8 bits, sub_dim=2
                        // Our earlier k32_fast did: c0 corresponds to vec*4 +0..3 etc per 8 values: j 0,1->c0, 2,3->c1 etc? Actually k32_fast expanded per codeword: one codeword 32 bits = 4 sub-indices each 8 bits, each sub has 2 values. So 8 values per codeword were interleaved across sub tables.
                        // For 16x32 tile, vec=0..3 per row, each vec yields 8 values, but our loop handles one vec per iteration (8 values). So we need to expand current codeword into 8 values.
                        // code for vec: 4 bytes c0..c3 correspond to 4 sub-tables? No, k32_fast used c0..c3 as 4 separate codewords? Wait k32_fast decoded 32 codewords per tile (16x16) where each codeword is 4 bytes? Then it expanded each codeword's 4 bytes into 8 values via s_cb lookups.
                        // So for k32, each codeword is 4 bytes = 32 bits = 4*8 bits, each byte is sub-index for one sub-table (since n_sub=4, each sub width 8, sub_dim=2). Then per codeword, it produced 8 values: 2 per sub.
                        // So our current code loads one 32-bit word per vec (which is one codeword). Good.
                        // Now expand that codeword's 4 bytes into 8 values and store into s_B:
                        out[0].__x = s_cb[c0*2+0]; out[1].__x = s_cb[c0*2+1];
                        out[2].__x = s_cb[c1*2+512]; out[3].__x = s_cb[c1*2+513];
                        out[4].__x = s_cb[c2*2+1024]; out[5].__x = s_cb[c2*2+1025];
                        out[6].__x = s_cb[c3*2+1536]; out[7].__x = s_cb[c3*2+1537];
                    } else {
                        for(int j=0;j<8;++j) out[j].__x=0;
                    }
                    // store into s_B[buf] col_major: s_B[n_local*32 + vec*8 + j]
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
        // No second __syncthreads needed because next iteration's decode writes to other buffer
        // But need to ensure previous buffer not overwritten until mma done. Since mma_sync is synchronous, it's safe to not sync.
        // However we still need a barrier to ensure decode of next tile doesn't start before mma's load_matrix_sync has consumed s_B[buf]? But s_B[buf] is read via load_matrix_sync which is also synchronous and has waitcnt. So after mma_sync, the buffer is free.
        // We still need __syncthreads at start of next iteration to ensure all threads have finished mma before reusing? But we use double buffering, so no reuse until 2 tiles later. So we can avoid extra sync.
        // To be safe, add a lightweight s_barrier only if needed, but we already have __syncthreads at next iteration's decode target? Actually next iteration's decode writes to other buffer, no conflict.
        // So we can proceed without second sync.
        // For correctness, we keep a single __syncthreads at top of loop (already done), so second is not needed.
    }
    float* cp = C + tm*16*N + tn*16;
    store_matrix_sync(cp, fc, N, mem_row_major);
}

// Bench harness (same as gemm_opt but with new kernel)
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

bool validate_m5(){
    int M=16,N=32,K=256,kbits=32,nsub=4;
    int n_sb=K/256, tsize=4*kbits, rbytes=n_sb*tsize;
    int subdim=8/nsub;
    int flat=0; for(int i=0;i<nsub;i++){int w=kbits/nsub+(i<kbits%nsub?1:0); flat+=(1<<w)*subdim;}
    std::mt19937 rng(123);
    std::vector<uint8_t> qw(N*rbytes); for(auto &x:qw) x=rng()%256;
    std::vector<uint8_t> cb(flat); for(auto &x:cb){float v=(rng()%5)*0.5f; x=(uint8_t)(v*10); if(x==0x7F||x==0xFF)x=0;}
    std::vector<hip_fp8_e4m3> A(M*K); for(auto &a:A){float v=(rng()%9-4)*0.5f; a=hip_fp8_e4m3(v);}
    // CPU decode W
    auto widths=[&](){std::vector<int> w(nsub); int b=kbits/nsub,e=kbits%nsub; for(int i=0;i<nsub;i++) w[i]=b+(i<e); return w;}();
    std::vector<int> tb(nsub,0); for(int i=1;i<nsub;i++) tb[i]=tb[i-1]+(1<<widths[i-1])*subdim;
    std::vector<uint8_t> W(N*K);
    for(int gn=0;gn<N;gn++) for(int sb=0;sb<n_sb;sb++) for(int v=0;v<32;v++){
        int bb=(v*kbits)/8, bs=(v*kbits)%8;
        uint64_t win=0; for(int b=0;b<8;b++){uint8_t by=0; if(bb+b<tsize) by=qw[gn*rbytes+sb*tsize+bb+b]; win|=(uint64_t)by<<(8*b);}
        uint64_t code=(win>>bs)&((1ULL<<kbits)-1);
        for(int j=0;j<8;j++){int sub=j/subdim, local=j%subdim; int off=0; for(int t=0;t<sub;t++) off+=widths[t]; int mask=(1<<widths[sub])-1; uint64_t sidx=(code>>off)&mask; int fidx=tb[sub]+sidx*subdim+local; W[gn*K+sb*256+v*8+j]=cb[fidx];}
    }
    std::vector<float> Af(M*K), Wf(N*K);
    for(int i=0;i<M*K;i++) Af[i]=(float)A[i];
    for(int i=0;i<N*K;i++){hip_fp8_e4m3 f; f.__x=W[i]; Wf[i]=(float)f;}
    std::vector<float> ref(M*N,0);
    for(int m=0;m<M;m++) for(int n=0;n<N;n++){float acc=0; for(int k=0;k<K;k++) acc+=Af[m*K+k]*Wf[n*K+k]; ref[m*N+n]=acc;}
    hip_fp8_e4m3 *dA; uint8_t *dqw,*dcb; float *dC;
    hipMalloc(&dA,M*K*sizeof(hip_fp8_e4m3)); hipMalloc(&dqw,N*rbytes); hipMalloc(&dcb,flat); hipMalloc(&dC,M*N*sizeof(float));
    hipMemcpy(dA,A.data(),M*K*sizeof(hip_fp8_e4m3),hipMemcpyHostToDevice);
    hipMemcpy(dqw,qw.data(),N*rbytes,hipMemcpyHostToDevice);
    hipMemcpy(dcb,cb.data(),flat,hipMemcpyHostToDevice);
    hipMemset(dC,0,M*N*sizeof(float));
    dim3 grid((N+15)/16,(M+15)/16);
    hipLaunchKernelGGL(fused_m5_k32,grid,dim3(32),0,0,dA,dqw,dcb,dC,M,N,K,rbytes);
    hipDeviceSynchronize();
    std::vector<float> gpu(M*N);
    hipMemcpy(gpu.data(),dC,M*N*sizeof(float),hipMemcpyDeviceToHost);
    float max_abs=0; int mism=0;
    for(int i=0;i<M*N;i++){float diff=std::abs(gpu[i]-ref[i]); max_abs=std::max(max_abs,diff); if(diff>1e-3) mism++;}
    std::cout<<"M5 validate M"<<M<<" N"<<N<<" K"<<K<<" max_abs "<<max_abs<<" mism "<<mism<<" "<<(max_abs<1e-3?"PASS":"FAIL")<<"\n";
    hipFree(dA); hipFree(dqw); hipFree(dcb); hipFree(dC);
    return max_abs<1e-3;
}

int main(){
    hipDeviceProp_t p; hipGetDeviceProperties(&p,0);
    std::cout<<"Device "<<p.name<<" "<<p.gcnArchName<<" "<<p.multiProcessorCount<<" CUs\n";
    if(!validate_m5()) std::cout<<"VALIDATION FAILED - bench still runs for diagnosis\n";
    struct Shape{int M,N,K;}; std::vector<Shape> shapes={{16,4096,4096},{32,4096,4096},{64,4096,4096},{128,4096,4096},{16,1024,4096},{16,2048,1024}};
    std::cout<<"=== Plain optimized (direct16 vs direct32) ===\n";
    for(auto s:shapes){
        double ms16=bench_plain(s.M,s.N,s.K,0);
        double ms32=bench_plain(s.M,s.N,s.K,1);
        std::cout<<"M"<<s.M<<" N"<<s.N<<" K"<<s.K<<" plain16 "<<ms16<<"ms | plain32 "<<ms32<<"ms\n";
    }
    std::cout<<"\n=== M5 fused_k32_pipe vs plain ===\n";
    for(auto s:shapes){
        double ms_plain=bench_plain(s.M,s.N,s.K,0,300);
        double ms_fused=bench_fused_m5(s.M,s.N,s.K,300);
        double ratio=ms_plain/ms_fused;
        double tf_plain=(2.0*s.M*s.N*s.K)/(ms_plain/1000)/1e12;
        double tf_fused=(2.0*s.M*s.N*s.K)/(ms_fused/1000)/1e12;
        double bw_plain=(double)(s.M*s.K + s.N*s.K + s.M*s.N*4)/(ms_plain/1000)/1e9;
        double bw_fused=(double)(s.M*s.K + s.N*(s.K/256*128) + s.M*s.N*4)/(ms_fused/1000)/1e9;
        std::cout<<"M"<<s.M<<" N"<<s.N<<" K"<<s.K<<" plain "<<ms_plain<<"ms TF "<<tf_plain<<" BW "<<bw_plain<<" | fused_m5 "<<ms_fused<<"ms TF "<<tf_fused<<" effBW "<<bw_fused<<" ratio "<<ratio<<" "<<(ratio>=0.97?"PASS":ratio>=1.0?"PASS":"SLOW")<<"\n";
    }
    return 0;
}
