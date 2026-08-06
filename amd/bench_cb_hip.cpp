// Benchmark CB fused vs plain FP8 WMMA on gfx1201
// Build: hipcc --offload-arch=gfx1201 bench_cb_hip.cpp -o /tmp/bench_cb --std=c++17 -O2
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <hip/hip_fp8.h>
#include <iostream>
#include <vector>
#include <random>

using namespace rocwmma;

// Reuse kernels from cb_decode_hip.cpp? duplicate minimal for bench
__device__ inline void split_widths_device_bench(int k_bits,int n_sub,int* out){
    int base = k_bits / n_sub;
    int extra = k_bits % n_sub;
    for(int i=0;i<n_sub;i++) out[i]= base + (i<extra?1:0);
}

__global__ void wmma_gemm_plain2(
    const hip_fp8_e4m3* A,
    const hip_fp8_e4m3* B,
    float* C,
    int M,int N,int K)
{
    int tileM=blockIdx.y;
    int tileN=blockIdx.x;
    int row_base=tileM*16, col_base=tileN*16;
    __shared__ hip_fp8_e4m3 tileA[256];
    __shared__ hip_fp8_e4m3 tileB[256];
    __shared__ float tileC[256];
    fragment<matrix_a,16,16,16, hip_fp8_e4m3, row_major> fragA;
    fragment<matrix_b,16,16,16, hip_fp8_e4m3, col_major> fragB;
    fragment<accumulator,16,16,16, float> fragC;
    fill_fragment(fragC,0.0f);
    for(int k0=0;k0<K;k0+=16){
        for(int i=threadIdx.x;i<256;i+=32){ int r=i/16,c=i%16; int gr=row_base+r,gk=k0+c; hip_fp8_e4m3 v{}; v.__x=0; if(gr<M&&gk<K) v=A[gr*K+gk]; tileA[r*16+c]=v; }
        for(int i=threadIdx.x;i<256;i+=32){ int kl=i%16,nl=i/16; int gk=k0+kl,gn=col_base+nl; hip_fp8_e4m3 v; v.__x=0; if(gk<K&&gn<N) v=B[gn*K+gk]; tileB[nl*16+kl]=v; }
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

__global__ void wmma_gemm_fused_bench(
    const hip_fp8_e4m3* A,
    const uint8_t* qw,
    const uint8_t* cb_flat,
    const int32_t* row_offset,
    float* C,
    int M,int N,int K, int k_bits,int n_sub,int type_size,int row_bytes)
{
    int tileM=blockIdx.y, tileN=blockIdx.x;
    int row_base=tileM*16, col_base=tileN*16;
    __shared__ hip_fp8_e4m3 tileA[256];
    __shared__ hip_fp8_e4m3 tileB[256];
    __shared__ float tileC[256];
    fragment<matrix_a,16,16,16, hip_fp8_e4m3, row_major> fragA;
    fragment<matrix_b,16,16,16, hip_fp8_e4m3, col_major> fragB;
    fragment<accumulator,16,16,16, float> fragC;
    fill_fragment(fragC,0.0f);
    int sub_dim=8/n_sub;
    int widths[4]; split_widths_device_bench(k_bits,n_sub,widths);
    int tbase[4]; tbase[0]=0; for(int i=1;i<n_sub;i++) tbase[i]=tbase[i-1]+(1<<widths[i-1])*sub_dim;
    for(int k0=0;k0<K;k0+=16){
        for(int i=threadIdx.x;i<256;i+=32){ int r=i/16,c=i%16; int gr=row_base+r,gk=k0+c; hip_fp8_e4m3 v; v.__x=0; if(gr<M&&gk<K) v=A[gr*K+gk]; tileA[r*16+c]=v; }
        for(int i=threadIdx.x;i<256;i+=32){
            int kl=i%16,nl=i/16; int gk=k0+kl,gn=col_base+nl; hip_fp8_e4m3 out; out.__x=0;
            if(gk<K&&gn<N){
                int sb=gk/256, col_in_sb=gk%256, vec=col_in_sb/8, coord=col_in_sb%8;
                int sub=coord/sub_dim, local=coord%sub_dim;
                int bit_off=0; for(int t=0;t<sub;t++) bit_off+=widths[t];
                int mask=(widths[sub]==32?0xFFFFFFFF:((1<<widths[sub])-1));
                int byte_base=(vec*k_bits)/8, bit_shift=(vec*k_bits)%8;
                uint64_t window=0; for(int b=0;b<8;b++){ uint8_t byte=0; if(byte_base+b<type_size) byte=qw[gn*row_bytes+sb*type_size+byte_base+b]; window|=(uint64_t)byte<<(8*b); }
                uint64_t code=(window>>bit_shift)&(k_bits==64?~0ULL:((1ULL<<k_bits)-1));
                uint64_t sub_idx=(code>>bit_off)&(uint64_t)mask;
                int flat_idx=row_offset[gn]+tbase[sub]+(int)sub_idx*sub_dim+local;
                out.__x=cb_flat[flat_idx];
            }
            tileB[nl*16+kl]=out;
        }
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

__global__ void scale_rows(float* C,const float* s,int M,int N){ int idx=blockIdx.x*blockDim.x+threadIdx.x; int tot=M*N; if(idx>=tot) return; int n=idx%N; C[idx]*=s[n]; }

double bench_kernel(int M,int N,int K,int k_bits,int n_sub, int iters=500){
    int n_sb=K/256;
    int type_size=4*k_bits;
    int row_bytes=n_sb*type_size;
    auto widths=[&](){ std::vector<int> w(n_sub); int b=k_bits/n_sub,e=k_bits%n_sub; for(int i=0;i<n_sub;i++) w[i]=b+(i<e?1:0); return w; }();
    int sub_dim=8/n_sub;
    int flat=0; for(int w:widths) flat+=(1<<w)*sub_dim;
    std::mt19937 rng(1);
    std::vector<uint8_t> qw(N*row_bytes); for(auto &x:qw) x=rng()%256;
    std::vector<uint8_t> cb(flat); for(auto &x:cb){ float vals[5]={-1,0,0.5,1,2}; float v=vals[rng()%5]; hip_fp8_e4m3 e(v); x=e.__x; if(x==0x7F||x==0xFF) x=0; }
    std::vector<int32_t> off(N,0);
    std::vector<float> scale(N,1.0f);
    std::vector<hip_fp8_e4m3> A(M*K); for(auto &x:A){ float v=(rng()%9-4)*0.5f; x=hip_fp8_e4m3(v); }
    // decoded W for plain
    // quick CPU decode to produce W bytes for plain kernel (use same logic as fused, but offline)
    // For bench we just generate random W bytes (perf same, not correctness)
    std::vector<hip_fp8_e4m3> W(N*K); for(auto &x:W){ float v=(rng()%9-4)*0.5f; x=hip_fp8_e4m3(v); }

    hip_fp8_e4m3 *dA,*dW; float *dC; uint8_t *d_qw,*d_cb; int32_t *d_off; float *d_scale,*dCfused;
    hipMalloc(&dA,M*K*sizeof(hip_fp8_e4m3));
    hipMalloc(&dW,N*K*sizeof(hip_fp8_e4m3));
    hipMalloc(&dC,M*N*sizeof(float));
    hipMalloc(&d_qw,N*row_bytes);
    hipMalloc(&d_cb,flat);
    hipMalloc(&d_off,N*sizeof(int32_t));
    hipMalloc(&d_scale,N*sizeof(float));
    hipMalloc(&dCfused,M*N*sizeof(float));
    hipMemcpy(dA,A.data(),M*K*sizeof(hip_fp8_e4m3),hipMemcpyHostToDevice);
    hipMemcpy(dW,W.data(),N*K*sizeof(hip_fp8_e4m3),hipMemcpyHostToDevice);
    hipMemcpy(d_qw,qw.data(),N*row_bytes,hipMemcpyHostToDevice);
    hipMemcpy(d_cb,cb.data(),flat,hipMemcpyHostToDevice);
    hipMemcpy(d_off,off.data(),N*sizeof(int32_t),hipMemcpyHostToDevice);
    hipMemcpy(d_scale,scale.data(),N*sizeof(float),hipMemcpyHostToDevice);
    dim3 gridDim((N+15)/16,(M+15)/16);
    // warmup
    for(int i=0;i<20;i++){ hipMemset(dC,0,M*N*sizeof(float)); hipLaunchKernelGGL(wmma_gemm_plain2,gridDim,dim3(32),0,0,dA,dW,dC,M,N,K); hipDeviceSynchronize(); }
    hipEvent_t start, stop; hipEventCreate(&start); hipEventCreate(&stop);
    hipEventRecord(start);
    for(int i=0;i<iters;i++){ hipLaunchKernelGGL(wmma_gemm_plain2,gridDim,dim3(32),0,0,dA,dW,dC,M,N,K); }
    hipEventRecord(stop); hipEventSynchronize(stop); float ms_plain; hipEventElapsedTime(&ms_plain,start,stop); ms_plain/=iters;
    // fused
    for(int i=0;i<20;i++){ hipMemset(dCfused,0,M*N*sizeof(float)); hipLaunchKernelGGL(wmma_gemm_fused_bench,gridDim,dim3(32),0,0,dA,d_qw,d_cb,d_off,dCfused,M,N,K,k_bits,n_sub,type_size,row_bytes); hipDeviceSynchronize(); }
    hipEventRecord(start);
    for(int i=0;i<iters;i++){ hipLaunchKernelGGL(wmma_gemm_fused_bench,gridDim,dim3(32),0,0,dA,d_qw,d_cb,d_off,dCfused,M,N,K,k_bits,n_sub,type_size,row_bytes); }
    hipEventRecord(stop); hipEventSynchronize(stop); float ms_fused; hipEventElapsedTime(&ms_fused,start,stop); ms_fused/=iters;
    double tflops_plain = (2.0*M*N*K)/(ms_plain/1000.0)/1e12;
    double tflops_fused = (2.0*M*N*K)/(ms_fused/1000.0)/1e12;
    double bw_plain_GBs = (double)(M*K + N*K + M*N*4)/(ms_plain/1000.0)/1e9;
    double bw_fused_GBs = (double)(M*K + N*row_bytes + M*N*4)/(ms_fused/1000.0)/1e9;
    double eff_bytes_plain = (double)(N*K); // weight bytes streamed
    double eff_bytes_fused = (double)(N*row_bytes);
    double ratio = ms_plain / ms_fused; // >1 means fused faster
    std::cout<<"Shape M"<<M<<" N"<<N<<" K"<<K<<" k"<<k_bits<<" : plain "<<ms_plain<<" ms tflops "<<tflops_plain<<" bw "<<bw_plain_GBs<<"GB/s | fused "<<ms_fused<<" ms tflops "<<tflops_fused<<" eff_bw_fused "<<bw_fused_GBs<<" ratio plain/fused "<<ratio<<" (eff weight bytes "<<eff_bytes_plain<<" vs "<<eff_bytes_fused<<" ratio "<<eff_bytes_plain/eff_bytes_fused<<")"<<std::endl;
    hipFree(dA); hipFree(dW); hipFree(dC); hipFree(d_qw); hipFree(d_cb); hipFree(d_off); hipFree(d_scale); hipFree(dCfused);
    hipEventDestroy(start); hipEventDestroy(stop);
    return ratio;
}

int main(){
    hipDeviceProp_t p; hipGetDeviceProperties(&p,0);
    std::cout<<"Device "<<p.name<<" "<<p.gcnArchName<<" "<<p.multiProcessorCount<<" CUs"<<std::endl;
    // Memory-bound small M vs compute-bound large M
    bench_kernel(16, 4096, 4096, 32,4);
    bench_kernel(16, 4096, 4096, 36,4);
    bench_kernel(16, 4096, 4096, 40,4);
    bench_kernel(32, 4096, 4096, 32,4);
    bench_kernel(64, 4096, 4096, 32,4);
    bench_kernel(128, 4096, 4096, 32,4);
    bench_kernel(16, 1024, 4096, 32,4);
    bench_kernel(16, 2048, 1024, 40,4);
    return 0;
}
