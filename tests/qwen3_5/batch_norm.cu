#include "tasks/common/qwen35_gdn.cuh"
#include "tasks/common/qwen35_attention.cuh"
using BF16=type::bfloat16_t;
template<int WIDTH,int ROWS>
__global__ void norm(BF16 const *x,float const *w,BF16 *out) {
  kernel::qwen35_norm_task<BF16,WIDTH,ROWS>(x+blockIdx.x*WIDTH,w,out+blockIdx.x*WIDTH);
}
template<int ROWS>
__global__ void qk(BF16 const *qg,BF16 const *k,float const *qw,float const *kw,
                   BF16 const *rope,BF16 *qout,BF16 *kout,BF16 *gate) {
  int b=blockIdx.x;
  kernel::qwen35_qk_rope_task<BF16,24,4,ROWS>(qg+b*24*512,k+b*4*256,qw,kw,rope,qout+b*24*256,kout+b*4*256,gate+b*24*256);
}
template<int ROWS>
int launch(void const *x,void const *w,void *out,int batch,int width) {
  if(width==5120)norm<5120,ROWS><<<batch,256>>>((BF16 const*)x,(float const*)w,(BF16*)out);
  else norm<1024,ROWS><<<batch,256>>>((BF16 const*)x,(float const*)w,(BF16*)out);
  return cudaGetLastError();
}
extern "C" int launch_norm(void const *x,void const *w,void *out,int batch,int width) {
  if(batch>=16)return launch<16>(x,w,out,batch,width);
  if(batch>=8)return launch<8>(x,w,out,batch,width);
  if(batch>=4)return launch<4>(x,w,out,batch,width);
  if(batch==3)return launch<3>(x,w,out,batch,width);
  if(batch==2)return launch<2>(x,w,out,batch,width);
  return launch<1>(x,w,out,batch,width);
}
extern "C" int launch_qk(void const *qg,void const *k,void const *qw,void const *kw,void const *rope,
                         void *qout,void *kout,void *gate,int batch) {
#define RUN_QK(R) qk<R><<<batch,256>>>((BF16 const*)qg,(BF16 const*)k,(float const*)qw,(float const*)kw,(BF16 const*)rope,(BF16*)qout,(BF16*)kout,(BF16*)gate)
  if(batch>=4){RUN_QK(4);}else if(batch==3){RUN_QK(3);}else if(batch==2){RUN_QK(2);}else{RUN_QK(1);}
  return cudaGetLastError();
}
