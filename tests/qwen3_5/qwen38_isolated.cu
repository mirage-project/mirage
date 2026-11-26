// Production templates at Qwen3.8-27B geometry, with explicit pointer partitions.
#include "tasks/common/qwen35_prefill.cuh"
#include <cuda_runtime.h>
using BF16=type::bfloat16_t;
constexpr int SCRATCH=kernel::QWEN35_CHUNK_SCRATCH+64*48*128/2;
__global__ void decode(BF16 const*x,BF16 const*z,BF16 const*ab,float const*p,BF16 const*w,
    float*s,int64_t const*pos,int64_t const*ids,BF16*raw,BF16*out) {
  int h=blockIdx.x;
  kernel::qwen35_delta_grouped_task<BF16,BF16,48,16>(x,z+h*128,ab+h,p+h,w,s+h*16384,pos,ids+h,raw+h*128,out+h*128);
}
extern "C" int launch_decode(void const*x,void const*z,void const*ab,void const*p,void const*w,
    void*s,void const*pos,void const*ids,void*raw,void*out) {
  decode<<<48,256>>>((BF16 const*)x,(BF16 const*)z,(BF16 const*)ab,(float const*)p,(BF16 const*)w,
    (float*)s,(int64_t const*)pos,(int64_t const*)ids,(BF16*)raw,(BF16*)out);
  return cudaGetLastError();
}
__global__ void prefill(BF16 const*x,BF16 const*z,BF16 const*ab,float const*p,BF16 const*w,
    float const*s,int64_t const*length,int64_t const*ids,BF16*out,float*next,float*scratch) {
  int h=blockIdx.x;
  kernel::qwen35_prefill_delta_grouped_task<BF16,BF16,48,16,128>(x,z+h*128,ab,p+h,w,s+h*16384,
    length,ids+h,out+h*128,next+h*16384,scratch+h*SCRATCH);
}
extern "C" int launch_prefill(void const*x,void const*z,void const*ab,void const*p,void const*w,
    void const*s,void const*length,void const*ids,void*out,void*next,void*scratch) {
  prefill<<<48,256>>>((BF16 const*)x,(BF16 const*)z,(BF16 const*)ab,(float const*)p,(BF16 const*)w,
    (float const*)s,(int64_t const*)length,(int64_t const*)ids,(BF16*)out,(float*)next,(float*)scratch);
  return cudaGetLastError();
}
template<bool SDPA> __global__ void attention(BF16 const*q,BF16 const*g,int64_t const*pos,
    BF16 const*k,BF16 const*v,int64_t const*heads,int64_t const*config,BF16*out,float*scratch) {
  int i=blockIdx.x;
  if constexpr(SDPA) kernel::qwen35_prefill_sdpa_task<BF16,1024,4,6>(q+i*256,g+i*256,pos+i,k,v,heads+i,config,out+i*256,scratch+i*4112);
  else kernel::qwen35_prefill_eager_task<BF16,1024,4,6>(q+i*256,g+i*256,pos+i,k,v,heads+i,config,out+i*256,scratch+i*4112);
}
extern "C" int launch_attention(void const*q,void const*g,void const*pos,void const*k,void const*v,
    void const*heads,void const*config,void*out,void*scratch,int rows,int sdpa) {
  if(sdpa) attention<true><<<rows*24,256>>>((BF16 const*)q,(BF16 const*)g,(int64_t const*)pos,(BF16 const*)k,(BF16 const*)v,(int64_t const*)heads,(int64_t const*)config,(BF16*)out,(float*)scratch);
  else attention<false><<<rows*24,256>>>((BF16 const*)q,(BF16 const*)g,(int64_t const*)pos,(BF16 const*)k,(BF16 const*)v,(int64_t const*)heads,(int64_t const*)config,(BF16*)out,(float*)scratch);
  return cudaGetLastError();
}
__global__ void norm(BF16 const*x,float const*w,BF16*out,int rows) {
  int row=blockIdx.x;
  if(rows==1)kernel::qwen35_norm_task<BF16,5120>(x,w,out);
  else {int64_t n=rows;kernel::qwen35_prefill_norm_task<BF16,5120>(x+row*5120,w,&n,out+row*5120);}
}
extern "C" int launch_norm(void const*x,void const*w,void*out,int rows) {
  norm<<<rows,256>>>((BF16 const*)x,(float const*)w,(BF16*)out,rows);
  return cudaGetLastError();
}

__global__ void qk(BF16 const*qg,BF16 const*k,float const*qw,float const*kw,BF16 const*rope,
    BF16*query,BF16*key,BF16*gate) {
  kernel::qwen35_qk_rope_task<BF16,24,4>(qg,k,qw,kw,rope,query,key,gate);
}
extern "C" int launch_qk(void const*qg,void const*k,void const*qw,void const*kw,void const*rope,
    void*query,void*key,void*gate) {
  qk<<<1,256>>>((BF16 const*)qg,(BF16 const*)k,(float const*)qw,(float const*)kw,(BF16 const*)rope,
    (BF16*)query,(BF16*)key,(BF16*)gate);
  return cudaGetLastError();
}

template<int N> __global__ void projection(BF16 const*x,BF16 const*w,BF16*y,int rows) {
  int64_t full_rows=rows;
  if(rows==1) kernel::qwen35_prefill_linear_task<BF16,5120,N,1,3,1728>(x,w+blockIdx.x*64*5120,y+blockIdx.x*64);
  else kernel::qwen35_prefill_linear_dispatch<BF16,5120,N>(x,w+blockIdx.x*64*5120,&full_rows,y+blockIdx.x*64);
}
extern "C" int launch_projection(void const*x,void const*w,void*y,int n,int rows) {
  if(n==128) projection<128><<<2,256>>>((BF16 const*)x,(BF16 const*)w,(BF16*)y,rows);
  else if(n==1024) projection<1024><<<16,256>>>((BF16 const*)x,(BF16 const*)w,(BF16*)y,rows);
  else return int(cudaErrorInvalidValue);
  return cudaGetLastError();
}

__global__ void down(BF16 const*x,BF16 const*w,BF16*y,int rows) {
  int64_t full_rows=rows;
  kernel::qwen35_prefill_linear_dispatch<BF16,17408,5120>(x,w+blockIdx.x*64*17408,&full_rows,y+blockIdx.x*64);
}
extern "C" int launch_down(void const*x,void const*w,void*y,int rows) {
  down<<<80,256>>>((BF16 const*)x,(BF16 const*)w,(BF16*)y,rows);
  return cudaGetLastError();
}
