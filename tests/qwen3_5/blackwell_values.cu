#include "tasks/common/qwen35_attention.cuh"
using BF16=type::bfloat16_t;
__global__ void gemv_values(BF16 const *p,BF16 const *gate,int64_t const *position,
                            BF16 const *cache,BF16 *out,int configuration) {
  int h=blockIdx.x,b=blockIdx.y,d=threadIdx.x,length=int(*position)+1;
  int parts=configuration&127, chunk_code=(configuration>>7)&15;
  int vector = chunk_code >= 6 ? 1 << (chunk_code-5) : 1;
  int chunk=chunk_code && chunk_code < 6 ? (1<<(chunk_code-1)) :
      ((length+vector*parts-1)/(vector*parts))*vector;
  int order=(configuration>>11)&3;
  int inner=1<<((configuration>>13)&3),inner_order=(configuration>>15)&3;
  bool contiguous_inner=configuration&(1<<17);
  p+=(b*24+h)*1024;gate+=(b*24+h)*256;
  cache+=b*1024*4*256+(h/6)*256;out+=(b*24+h)*256;
  float partial[64]={};
  for(int t=0;t<length;++t){
    int lane=(t/chunk)%parts;
    int at=(t/(chunk*parts))*chunk+t%chunk;
  int sub=contiguous_inner ? (at/((chunk+inner-1)/inner))%inner : at%inner;
    int index=lane*inner+sub;
    partial[index]=__fmaf_rn(float(p[t]),float(cache[t*4*256+d]),partial[index]);
  }
  for(int lane=0;lane<parts;++lane){
    float *s=partial+lane*inner;
    if(inner_order==2){for(int i=1;i<inner;++i)s[0]=__fadd_rn(s[0],s[i]);}
    else if(inner_order==1){for(int offset=1;offset<inner;offset<<=1)
      for(int i=0;i<inner;i+=2*offset)s[i]=__fadd_rn(s[i],s[i+offset]);}
    else for(int offset=inner/2;offset;offset>>=1)
      for(int i=0;i<offset;++i)s[i]=__fadd_rn(s[i],s[i+offset]);
    partial[lane]=s[0];
  }
  if(order==2){for(int i=1;i<parts;++i)partial[0]=__fadd_rn(partial[0],partial[i]);}
  else if(order==1){for(int offset=1;offset<parts;offset<<=1)
    for(int i=0;i<parts;i+=2*offset)partial[i]=__fadd_rn(partial[i],partial[i+offset]);}
  else for(int offset=parts/2;offset;offset>>=1)
    for(int i=0;i<offset;++i)partial[i]=__fadd_rn(partial[i],partial[i+offset]);
  BF16 sigmoid=BF16(1.f/(1.f+expf(-float(gate[d]))));
  out[d]=BF16(__fmul_rn(float(BF16(partial[0])),float(sigmoid)));
}
__global__ void wave64_values(BF16 const *p,BF16 const *gate,int64_t const *position,
                              BF16 const *cache,BF16 *out) {
  int h=blockIdx.x,b=blockIdx.y,d=threadIdx.x,length=int(*position)+1;
  int chunk=(length+7)/8;
  p+=(b*24+h)*1024;gate+=(b*24+h)*256;
  cache+=b*1024*4*256+(h/6)*256;out+=(b*24+h)*256;
  float lane_sum[8]={};
  for(int lane=0;lane<8;++lane){
    int end=min((lane+1)*chunk,length);
    for(int begin=lane*chunk;begin<end;begin+=64){
      float tile_sum=0.f;
      for(int t=begin;t<min(begin+64,end);++t)
        tile_sum=__fmaf_rn(float(p[t]),float(cache[t*4*256+d]),tile_sum);
      lane_sum[lane]=__fadd_rn(lane_sum[lane],tile_sum);
    }
  }
  for(int lane=1;lane<8;++lane)lane_sum[0]=__fadd_rn(lane_sum[0],lane_sum[lane]);
  BF16 sigmoid=BF16(1.f/(1.f+expf(-float(gate[d]))));
  out[d]=BF16(__fmul_rn(float(BF16(lane_sum[0])),float(sigmoid)));
}
template<int MMA_K,int RESIDUE,bool FORCE>
__global__ void values(BF16 const *p,BF16 const *gate,int64_t const *position,
                       BF16 const *cache,int64_t const *heads,BF16 *out) {
  int h=blockIdx.x,b=blockIdx.y;
  kernel::qwen35_value_task<BF16,1024,FORCE,MMA_K,RESIDUE,4,6>(
      p+(b*24+h)*1024,gate+(b*24+h)*256,position,cache+b*1024*4*256,
      heads+h,out+(b*24+h)*256,int(*position)+1,int(gridDim.y));
}
extern "C" int launch_values(void const *p,void const *gate,void const *position,
                              void const *cache,void const *heads,void *out,int batch,int mode) {
#define RUN_VALUE(K,R,F) values<K,R,F><<<dim3(24,batch),256>>>((BF16 const*)p,(BF16 const*)gate,(int64_t const*)position,(BF16 const*)cache,(int64_t const*)heads,(BF16*)out)
  if(mode==9999)wave64_values<<<dim3(24,batch),256>>>((BF16 const*)p,(BF16 const*)gate,(int64_t const*)position,(BF16 const*)cache,(BF16*)out);
  else if(mode>=100)gemv_values<<<dim3(24,batch),256>>>((BF16 const*)p,(BF16 const*)gate,(int64_t const*)position,(BF16 const*)cache,(BF16*)out,mode-100);
  else if(mode==64){RUN_VALUE(16,64,true);}else if(mode==32){RUN_VALUE(16,32,true);}
  else if(mode==16){RUN_VALUE(16,0,true);}else if(mode==8){RUN_VALUE(8,0,true);}
  else{RUN_VALUE(16,0,false);}
  return cudaGetLastError();
}
