// bench_verify_m7 — M7 verification gate: one alloc set, alternating plain/persistent/frag-direct, min/median, repack cost
// Build: hipcc --offload-arch=gfx1201 amd/bench_verify_m7.cpp -o /tmp/bench_verify_m7 --std=c++17 -O2
// Also uses kernels from gemm_m7_hip.cpp via duplication (to keep single file gate)
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <hip/hip_fp8.h>
#include <iostream>
#include <vector>
#include <random>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <cmath>
using namespace rocwmma;

// ---- kernels duplicated from gemm_m7_hip.cpp ----
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
        const hip_fp8_e4m3* ap = A + tm*16*K + k0;
        load_matrix_sync(fa, ap, K);
        load_matrix_sync(fb, s_B, 32);
        mma_sync(fc, fa, fb, fc);
        __syncthreads();
    }
    float* cp = C + tm*16*N + tn*16;
    store_matrix_sync(cp, fc, N, mem_row_major);
}
__global__ void fused_persistent_k32(const hip_fp8_e4m3* A, const uint8_t* qw_tiled, const uint8_t* cb, float* C, int M,int N,int K){
    int tn = blockIdx.x;
    int Ntiles = N/16;
    if(tn >= Ntiles) return;
    int Mt = (M+15)/16;
    if(Mt>8) Mt=8;
    __shared__ uint8_t s_cb[2048];
    __shared__ uint8_t s_qw[256];
    __shared__ hip_fp8_e4m3 s_B[512];
    for(int i=threadIdx.x*8; i<2048; i+=32*8){
        if(i+8 <= 2048) *(uint64_t*)(s_cb+i) = *(const uint64_t*)(cb+i);
    }
    __syncthreads();
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
        load_matrix_sync(fb, s_B, 32);
        for(int mt=0;mt<Mt;mt++){
            const hip_fp8_e4m3* ap = A + mt*16*K + k0;
            load_matrix_sync(fa, ap, K);
            mma_sync(acc[mt], fa, fb, acc[mt]);
        }
        __syncthreads();
    }
    for(int mt=0;mt<Mt;mt++){
        float* cp = C + mt*16*N + tn*16;
        store_matrix_sync(cp, acc[mt], N, mem_row_major);
    }
}
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
        for(int i=0;i<fb.num_elements;i++){
            int k_local = k_base + i;
            int vec = k_local / 8;
            int coord = k_local % 8;
            int qw_off = n_for_lane*8 + vec*4;
            uint32_t code = *(const uint32_t*)(s_qw + qw_off);
            uint32_t c0=(code>>0)&0xFF, c1=(code>>8)&0xFF, c2=(code>>16)&0xFF, c3=(code>>24)&0xFF;
            int sub = coord / 2;
            int local = coord % 2;
            uint32_t cidx; int base;
            if(sub==0){cidx=c0; base=0;} else if(sub==1){cidx=c1; base=512;} else if(sub==2){cidx=c2; base=1024;} else {cidx=c3; base=1536;}
            fb[i].__x = s_cb[cidx*2 + local + base];
        }
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
static double median_vec(std::vector<double> v){ std::sort(v.begin(), v.end()); return v[v.size()/2]; }

int main(){
    hipDeviceProp_t prop; hipGetDeviceProperties(&prop,0);
    std::cout<<"Device "<<prop.name<<" "<<prop.gcnArchName<<" "<<prop.multiProcessorCount<<" CUs warp "<<prop.warpSize<<"\n";
    std::cout<<"ROCm "<<prop.major<<"."<<prop.minor<<"\n";
    // Repack cost
    {
        int N=4096,K=4096; int n_sb=K/256, rbytes=n_sb*128;
        int Ntiles=N/16,Ktiles=K/32; int tiled_bytes=Ntiles*Ktiles*256;
        std::vector<uint8_t> qw(N*rbytes); std::vector<uint8_t> tiled(tiled_bytes);
        std::mt19937 rng(0x1234); for(auto &x:qw) x=rng()%256;
        repack_k32_for_m6(qw.data(), tiled.data(), N,K,rbytes);
        volatile uint64_t sink=0;
        const int reps=50;
        std::vector<double> times;
        for(int r=0;r<reps;++r){
            auto t0=std::chrono::high_resolution_clock::now();
            repack_k32_for_m6(qw.data(), tiled.data(), N,K,rbytes);
            auto t1=std::chrono::high_resolution_clock::now();
            double ms=std::chrono::duration<double,std::milli>(t1-t0).count();
            times.push_back(ms);
            uint64_t c=0; for(int i=0;i<tiled_bytes;i+=4096) c+=tiled[i]; sink+=c;
        }
        double med=median_vec(times);
        double minv=*std::min_element(times.begin(),times.end());
        double maxv=*std::max_element(times.begin(),times.end());
        double bytes=(double)N*rbytes;
        double bw_med=bytes/(med/1000)/1e9;
        std::cout<<"\n=== Repack host cost (k32, N=4096 K=4096, 8 MB) ===\n";
        std::cout<<"repack median "<<med<<" ms min "<<minv<<" max "<<maxv<<" sink "<<sink<<"\n";
        std::cout<<"host BW median "<<bw_med<<" GB/s\n";
    }
    {
        int N=4096,K=4096; int n_sb=K/256, rbytes=n_sb*128;
        int Ntiles=N/16,Ktiles=K/16; int tiled_bytes=Ntiles*Ktiles*128;
        std::vector<uint8_t> qw(N*rbytes); std::vector<uint8_t> tiled(tiled_bytes);
        std::mt19937 rng(0x1234); for(auto &x:qw) x=rng()%256;
        repack_k16_for_m7(qw.data(), tiled.data(), N,K,rbytes);
        volatile uint64_t sink=0;
        std::vector<double> times;
        for(int r=0;r<50;++r){
            auto t0=std::chrono::high_resolution_clock::now();
            repack_k16_for_m7(qw.data(), tiled.data(), N,K,rbytes);
            auto t1=std::chrono::high_resolution_clock::now();
            double ms=std::chrono::duration<double,std::milli>(t1-t0).count();
            times.push_back(ms);
            uint64_t c=0; for(int i=0;i<tiled_bytes;i+=4096) c+=tiled[i]; sink+=c;
        }
        double med=median_vec(times);
        double bytes=(double)N*rbytes;
        double bw=bytes/(med/1000)/1e9;
        std::cout<<"\n=== Repack k16 cost median "<<med<<" ms BW "<<bw<<" GB/s sink "<<sink<<" ===\n";
    }
    struct Shape{int M,N,K; const char* label;};
    std::vector<Shape> shapes={{16,4096,4096,"M16 N4096 K4096"},{32,4096,4096,"M32 N4096 K4096"},{64,4096,4096,"M64 N4096 K4096"},{128,4096,4096,"M128 N4096 K4096"},{16,1024,4096,"M16 N1024 K4096"},{16,2048,1024,"M16 N2048 K1024"}};
    const int iters=300; const int reps=20; const int warmup=20;
    std::cout<<"\n=== Verification gate: one alloc set, alternating plain/fused, "<<reps<<" reps ===\n";
    for(auto sh: shapes){
        int M=sh.M,N=sh.N,K=sh.K;
        int n_sb=K/256,rbytes=n_sb*128, flat=2048;
        int Ntiles=N/16,Ktiles32=K/32,tiled32=Ntiles*Ktiles32*256;
        int Ktiles16=K/16,tiled16=Ntiles*Ktiles16*128;
        std::mt19937 rng_plain(1), rng_fused(2);
        std::vector<hip_fp8_e4m3> hA(M*K); for(auto &x:hA){float v=(rng_plain()%9-4)*0.5f; x=hip_fp8_e4m3(v);}
        std::vector<hip_fp8_e4m3> hB(N*K); for(auto &x:hB){float v=(rng_plain()%9-4)*0.5f; x=hip_fp8_e4m3(v);}
        std::vector<uint8_t> hqw(N*rbytes); for(auto &x:hqw) x=rng_fused()%256;
        std::vector<uint8_t> hcb(flat); for(auto &x:hcb){float v=(rng_fused()%5)*0.5f; uint8_t bv=(uint8_t)(v*10); if(bv==0x7F||bv==0xFF) bv=0; x=bv;}
        std::vector<uint8_t> hqw_tiled32(tiled32); repack_k32_for_m6(hqw.data(),hqw_tiled32.data(),N,K,rbytes);
        std::vector<uint8_t> hqw_tiled16(tiled16); repack_k16_for_m7(hqw.data(),hqw_tiled16.data(),N,K,rbytes);
        hip_fp8_e4m3 *dA,*dB; uint8_t *dqw32,*dqw16,*dcb; float *dC_plain,*dC_m6,*dC_pers,*dC_frag;
        hipMalloc(&dA,M*K*sizeof(hip_fp8_e4m3)); hipMalloc(&dB,N*K*sizeof(hip_fp8_e4m3));
        hipMalloc(&dqw32,tiled32); hipMalloc(&dqw16,tiled16); hipMalloc(&dcb,flat);
        hipMalloc(&dC_plain,M*N*sizeof(float)); hipMalloc(&dC_m6,M*N*sizeof(float)); hipMalloc(&dC_pers,M*N*sizeof(float)); hipMalloc(&dC_frag,M*N*sizeof(float));
        hipMemcpy(dA,hA.data(),M*K*sizeof(hip_fp8_e4m3),hipMemcpyHostToDevice);
        hipMemcpy(dB,hB.data(),N*K*sizeof(hip_fp8_e4m3),hipMemcpyHostToDevice);
        hipMemcpy(dqw32,hqw_tiled32.data(),tiled32,hipMemcpyHostToDevice);
        hipMemcpy(dqw16,hqw_tiled16.data(),tiled16,hipMemcpyHostToDevice);
        hipMemcpy(dcb,hcb.data(),flat,hipMemcpyHostToDevice);
        dim3 grid_m6((N+15)/16,(M+15)/16);
        dim3 grid_pers(Ntiles,1);
        for(int w=0;w<warmup;++w){
            hipMemset(dC_plain,0,M*N*sizeof(float)); hipLaunchKernelGGL(plain_direct16,grid_m6,dim3(32),0,0,dA,dB,dC_plain,M,N,K);
            hipLaunchKernelGGL(fused_m6_repacked_k32,grid_m6,dim3(32),0,0,dA,dqw32,dcb,dC_m6,M,N,K);
            hipLaunchKernelGGL(fused_persistent_k32,grid_pers,dim3(32),0,0,dA,dqw32,dcb,dC_pers,M,N,K);
            hipLaunchKernelGGL(fused_persistent_frag_direct_k16,grid_pers,dim3(32),0,0,dA,dqw16,dcb,dC_frag,M,N,K);
            hipDeviceSynchronize();
        }
        hipEvent_t s,e; hipEventCreate(&s); hipEventCreate(&e);
        std::vector<double> plains,m6s,perss,frags;
        for(int r=0;r<reps;++r){
            hipEventRecord(s); for(int i=0;i<iters;++i) hipLaunchKernelGGL(plain_direct16,grid_m6,dim3(32),0,0,dA,dB,dC_plain,M,N,K); hipEventRecord(e); hipEventSynchronize(e); float ms; hipEventElapsedTime(&ms,s,e); plains.push_back(ms/iters);
            hipEventRecord(s); for(int i=0;i<iters;++i) hipLaunchKernelGGL(fused_m6_repacked_k32,grid_m6,dim3(32),0,0,dA,dqw32,dcb,dC_m6,M,N,K); hipEventRecord(e); hipEventSynchronize(e); hipEventElapsedTime(&ms,s,e); m6s.push_back(ms/iters);
            hipEventRecord(s); for(int i=0;i<iters;++i) hipLaunchKernelGGL(fused_persistent_k32,grid_pers,dim3(32),0,0,dA,dqw32,dcb,dC_pers,M,N,K); hipEventRecord(e); hipEventSynchronize(e); hipEventElapsedTime(&ms,s,e); perss.push_back(ms/iters);
            hipEventRecord(s); for(int i=0;i<iters;++i) hipLaunchKernelGGL(fused_persistent_frag_direct_k16,grid_pers,dim3(32),0,0,dA,dqw16,dcb,dC_frag,M,N,K); hipEventRecord(e); hipEventSynchronize(e); hipEventElapsedTime(&ms,s,e); frags.push_back(ms/iters);
        }
        hipEventDestroy(s); hipEventDestroy(e);
        double plain_min=*std::min_element(plains.begin(),plains.end()), plain_med=median_vec(plains);
        double m6_min=*std::min_element(m6s.begin(),m6s.end()), m6_med=median_vec(m6s);
        double pers_min=*std::min_element(perss.begin(),perss.end()), pers_med=median_vec(perss);
        double frag_min=*std::min_element(frags.begin(),frags.end()), frag_med=median_vec(frags);
        std::cout<<"\n"<<sh.label<<" plain16 vs m6 vs pers_k32 vs frag_k16 (one alloc, alternating, 20 reps)\n";
        std::cout<<"  plain min "<<plain_min<<" med "<<plain_med<<" | m6 min "<<m6_min<<" med "<<m6_med<<" ratio "<<plain_min/m6_min<<"\n";
        std::cout<<"  pers min "<<pers_min<<" med "<<pers_med<<" ratio plain_min/pers_min "<<plain_min/pers_min<<" plain_med/pers_med "<<plain_med/pers_med<<"\n";
        std::cout<<"  frag min "<<frag_min<<" med "<<frag_med<<" ratio plain_min/frag_min "<<plain_min/frag_min<<"\n";
        double bw_plain_min=(double)(M*K+N*K+M*N*4)/(plain_min/1000)/1e9;
        std::cout<<"  BW plain min "<<bw_plain_min<<" GB/s\n";
        for(int i=0;i<reps;++i) std::cout<<"    rep "<<i<<" plain "<<plains[i]<<" m6 "<<m6s[i]<<" pers "<<perss[i]<<" frag "<<frags[i]<<"\n";
        hipFree(dA); hipFree(dB); hipFree(dqw32); hipFree(dqw16); hipFree(dcb); hipFree(dC_plain); hipFree(dC_m6); hipFree(dC_pers); hipFree(dC_frag);
    }
    std::cout<<"\n=== Profiler attestation ===\n";
    std::cout<<"rocprofv3 PMCs unavailable on gfx1201, timer+ISA remain.\n";
    return 0;
}
