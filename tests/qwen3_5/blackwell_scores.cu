#include "tasks/common/qwen35_attention.cuh"
using BF16 = type::bfloat16_t;
template<int MMA_K, bool FORCE>
__global__ void scores(BF16 const *q, int64_t const *position, BF16 const *keys,
                       int64_t const *heads, float *output) {
  int h=blockIdx.x, b=blockIdx.y;
  kernel::qwen35_scores_task<BF16,1024,FORCE,MMA_K,4,6>(
      q+(b*24+h)*256,position,keys+b*1024*4*256,heads+h,output+(b*24+h)*1024,int(gridDim.y));
}
__global__ void gemv_scores(BF16 const *q,int64_t const *position,BF16 const *keys,
                            float *output,int configuration) {
  int h=blockIdx.x,b=blockIdx.y;
  bool tiled=configuration & (1<<15);
  int parts=configuration&63;
  int chunk=1<<((configuration>>6)&7);
  bool strided=configuration&64;
  bool serial=tiled ? ((configuration>>9)&3)==2 : configuration&128;
  bool ascending=tiled ? ((configuration>>9)&3)==1 : configuration&256;
  q+=(b*24+h)*256;keys+=b*1024*4*256+(h/6)*256;output+=(b*24+h)*1024;
  for(int t=threadIdx.x;t<=*position;t+=blockDim.x){
    float partial[32]={};
    for(int d=0;d<256;++d){
      int lane=tiled ? (d/chunk)%parts : strided?d%parts:d/(256/parts);
      partial[lane]=__fmaf_rn(float(q[d]),float(keys[t*4*256+d]),partial[lane]);
    }
    if(serial){for(int i=1;i<parts;++i)partial[0]=__fadd_rn(partial[0],partial[i]);}
    else if(ascending){for(int offset=1;offset<parts;offset<<=1)
      for(int i=0;i<parts;i+=2*offset)partial[i]=__fadd_rn(partial[i],partial[i+offset]);}
    else for(int offset=parts/2;offset;offset>>=1)
      for(int i=0;i<offset;++i)partial[i]=__fadd_rn(partial[i],partial[i+offset]);
    output[t]=float(BF16(__fmul_rn(float(BF16(partial[0])),.0625f)));
  }
}
extern "C" int launch_scores(void const *q,void const *p,void const *k,void const *h,
                             void *out,int batch,int mode) {
  if(mode>=100)gemv_scores<<<dim3(24,batch),256>>>((BF16 const*)q,(int64_t const*)p,(BF16 const*)k,(float*)out,mode-100);
  else if(mode==8) scores<8,true><<<dim3(24,batch),256>>>((BF16 const*)q,(int64_t const*)p,(BF16 const*)k,(int64_t const*)h,(float*)out);
  else if(mode==16) scores<16,true><<<dim3(24,batch),256>>>((BF16 const*)q,(int64_t const*)p,(BF16 const*)k,(int64_t const*)h,(float*)out);
  else scores<16,false><<<dim3(24,batch),256>>>((BF16 const*)q,(int64_t const*)p,(BF16 const*)k,(int64_t const*)h,(float*)out);
  return cudaGetLastError();
}
