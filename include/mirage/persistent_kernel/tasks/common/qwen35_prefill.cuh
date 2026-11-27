/* Copyright 2026 The Mirage Team. Licensed under the Apache License, Version 2.0. */
#pragma once
#include "tasks/common/qwen35_gdn.cuh"
#include "tasks/common/qwen35_attention.cuh"
#include "tasks/common/qwen35_prefill_attention.cuh"
#include "tasks/common/qwen35_model.cuh"

namespace kernel {
// Reproduce PyTorch's vectorized mean reduction for a batched matrix.
// Its lane count depends on the full call's row count, not the 64-row tile.
template <typename T, int D>
__device__ __forceinline__ float qwen35_prefill_inv_rms(T const *x,int rows) {
  int height=1;
  while(height*2<=rows && height<16) height*=2;
  int width=min(D/4,512/height);
  __shared__ float sums[256];
  __shared__ float result;
  float parts[4]={};
  if(threadIdx.x<width) {
    for(int at=threadIdx.x*4;at<D;at+=width*4)
      for(int j=0;j<4;++j) parts[j]=__fadd_rn(parts[j],__fmul_rn(float(x[at+j]),float(x[at+j])));
  }
  float sum=__fadd_rn(__fadd_rn(__fadd_rn(parts[0],parts[1]),parts[2]),parts[3]);
  sums[threadIdx.x]=sum;
  for(int offset=width/2;offset>=32;offset/=2) {
    __syncthreads();
    if(threadIdx.x<offset) {sum=__fadd_rn(sum,sums[threadIdx.x+offset]);sums[threadIdx.x]=sum;}
  }
  __syncthreads();
  if(threadIdx.x<32) {
    for(int offset=1;offset<32;offset*=2) sum=__fadd_rn(sum,__shfl_down_sync(0xffffffff,sum,offset));
    if(threadIdx.x==0) result=rsqrtf(__fadd_rn(__fmul_rn(sum,1.f/D),1.e-6f));
  }
  __syncthreads();
  float inverse=result;
  __syncthreads();
  return inverse;
}

template <typename T, int HIDDEN=5120>
__device__ __forceinline__ void qwen35_prefill_norm_task(
    void const *input,void const *weight,void const *row_count,void *output) {
  auto x=static_cast<T const *>(input); auto w=static_cast<float const *>(weight);
  auto y=static_cast<T *>(output);
  float inverse=qwen35_prefill_inv_rms<T,HIDDEN>(x,int(*static_cast<int64_t const *>(row_count)));
  for(int i=threadIdx.x;i<HIDDEN;i+=blockDim.x)
    y[i]=T(__fmul_rn(__fmul_rn(float(x[i]),inverse),1.f+w[i]));
  __syncthreads();
}

template <typename T, int Q_HEADS=24, int KV_HEADS=4>
__device__ __forceinline__ void qwen35_prefill_qk_rope_task(
    void const *qg_ptr, void const *k_ptr, void const *qw_ptr,
    void const *kw_ptr, void const *rope_ptr, void const *row_count, void *q_out, void *k_out,
    void *gate_out) {
  auto qg = static_cast<T const *>(qg_ptr);
  auto k = static_cast<T const *>(k_ptr);
  auto qw = static_cast<float const *>(qw_ptr);
  auto kw = static_cast<float const *>(kw_ptr);
  auto rope = static_cast<T const *>(rope_ptr); // [cos(64), sin(64)]
  auto qo = static_cast<T *>(q_out);
  auto ko = static_cast<T *>(k_out);
  auto gate = static_cast<T *>(gate_out);
  __shared__ float inv[Q_HEADS+KV_HEADS];
  int rows=int(*static_cast<int64_t const *>(row_count));
  for(int h=0;h<Q_HEADS+KV_HEADS;++h) {
    auto x=h<Q_HEADS ? qg+h*512 : k+(h-Q_HEADS)*256;
    float inverse=qwen35_prefill_inv_rms<T,256>(x,rows*(h<Q_HEADS ? Q_HEADS : KV_HEADS));
    if(threadIdx.x==0) inv[h]=inverse;
    __syncthreads();
  }
  for (int i = threadIdx.x; i < (Q_HEADS+KV_HEADS) * 256; i += blockDim.x) {
    int h = i / 256, d = i % 256;
    auto x = h < Q_HEADS ? qg + h * 512 : k + (h - Q_HEADS) * 256;
    auto w = h < Q_HEADS ? qw : kw;
    auto destination=h<Q_HEADS ? qo+h*256 : ko+(h-Q_HEADS)*256;
    destination[d] = T(__fmul_rn(__fmul_rn(float(x[d]), inv[h]), 1.f + w[d]));
  }
  __syncthreads();
  // Reuse output storage for normalized Q/K. Load both rotary partners
  // before any thread overwrites them, avoiding another 5 KiB static buffer.
  for(int h=0;h<Q_HEADS+KV_HEADS;++h) {
    int d=threadIdx.x;
    auto destination=h<Q_HEADS ? qo+h*256 : ko+(h-Q_HEADS)*256;
    T value=destination[d];
    int other=d<32 ? d+32 : d-32;
    float rotated=d<64 ? float(destination[other])*(d<32 ? -1.f : 1.f) : 0.f;
    __syncthreads();
    if(d<64) {
      T left=T(__fmul_rn(float(value),float(rope[d])));
      T right=T(__fmul_rn(rotated,float(rope[64+d])));
      value=T(__fadd_rn(float(left),float(right)));
    }
    destination[d]=value;
    if(h<Q_HEADS) gate[h*256+d]=qg[h*512+256+d];
    __syncthreads();
  }
}

template <typename T, int CAPACITY, int KV_HEADS=4, int Q_PER_KV=6>
__device__ __forceinline__ void qwen35_prefill_eager_task(
    void const *query, void const *gate, void const *position,
    void const *key_cache, void const *value_cache, void const *head_id,
    void const *split_config, void *gated_output, void *scratch) {
  int64_t pos = *static_cast<int64_t const *>(position);
  if (pos < 0 || pos >= CAPACITY) return;
  // Keep the score vector in head-private global scratch. Another 4 KiB of
  // static shared storage exceeds the persistent worker budget on H100.
  auto scores=static_cast<float *>(scratch);
  auto p=reinterpret_cast<T *>(scores+CAPACITY);
  __shared__ float max_score, denominator;
  // The pinned cuBLAS batched path uses m16n8k8 for unaligned multi-row
  // attention, and a 16-wide reduction for aligned matrices. This decision
  // belongs to the full call, not the causal prefix visible to this query.
  bool narrow_mma=*static_cast<int64_t const *>(split_config)==8;
  if(narrow_mma)
    qwen35_scores_task<T, CAPACITY, true, 8, KV_HEADS, Q_PER_KV>(query, position, key_cache, head_id, scores);
  else
    qwen35_scores_task<T, CAPACITY, true, 16, KV_HEADS, Q_PER_KV>(query, position, key_cache, head_id, scores);
  if (threadIdx.x == 0) {
    float peak = scores[0];
    for (int t = 1; t <= pos; ++t) peak = fmaxf(peak, scores[t]);
    max_score = peak;
  }
  __syncthreads();
  for (int t = threadIdx.x; t <= pos; t += blockDim.x)
    scores[t] = expf(scores[t] - max_score);
  __syncthreads();
  if (threadIdx.x < 32) {
    // Match PyTorch PersistentSoftmax: strided per-lane sums, then descending
    // XOR shuffles. A serial denominator differs at long-context BF16 ties.
    float sum = 0.f;
    for (int t = threadIdx.x; t <= pos; t += 32) sum = __fadd_rn(sum, scores[t]);
    for (int offset = 16; offset; offset >>= 1)
      sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffff, sum, offset));
    if (threadIdx.x == 0) denominator = sum;
  }
  __syncthreads();
  for (int t = threadIdx.x; t < CAPACITY; t += blockDim.x) {
    T probability = t <= pos ? T(scores[t] / denominator) : T(0.f);
    p[t] = probability;
    if (t <= pos) scores[t] = float(probability);
  }
  __syncthreads();
  if(narrow_mma)
    qwen35_value_task<T, CAPACITY, true, 8, 32, KV_HEADS, Q_PER_KV>(p, gate, position,
        value_cache, head_id, gated_output, int(static_cast<int64_t const *>(split_config)[1]));
  else
    qwen35_value_task<T, CAPACITY, true, 16, 0, KV_HEADS, Q_PER_KV>(p, gate, position,
                                          value_cache, head_id, gated_output);
  __syncthreads();
}
// Fixed 64-token compute tiles, with an explicit valid length for the final
// tile. Buffers/state belong to a single request, separate from decode state
// until the host commits a successfully completed prefill operation.
constexpr int QWEN35_CHUNK = 64;
constexpr int QWEN35_CHUNK_SCRATCH = 9 * 8192 + 3 * 4096 + 64;

template <typename T, int KV_HEADS=4>
__device__ __forceinline__ void qwen35_prefill_kv_task(
    void const *key, void const *value, void const *metadata,
    void *key_cache, void *value_cache) {
  auto meta=static_cast<int64_t const *>(metadata);
  int past=int(meta[0]),length=int(meta[1]);
  auto k=static_cast<T const *>(key),v=static_cast<T const *>(value);
  auto kc=static_cast<T *>(key_cache),vc=static_cast<T *>(value_cache);
  for (int i=threadIdx.x; i<length*(KV_HEADS*256); i+=blockDim.x) {
    kc[past*(KV_HEADS*256)+i]=k[i]; vc[past*(KV_HEADS*256)+i]=v[i];
  }
  __syncthreads();
}

template <typename T, int INTERMEDIATE=17408>
__device__ __forceinline__ void qwen35_prefill_silu_task(void const *input,void *output) {
  auto x=static_cast<T const *>(input); auto y=static_cast<T *>(output);
  for (int i=threadIdx.x;i<INTERMEDIATE;i+=blockDim.x) {
    float gate=float(x[i]);
    T activated=T(gate/(1.f+expf(-gate)));
    y[i]=T(__fmul_rn(float(activated),float(x[INTERMEDIATE+i])));
  }
  __syncthreads();
}

template <typename T, int HIDDEN=5120>
__device__ __forceinline__ void qwen35_prefill_add_task(void const *a,void const *b,void *output) {
  auto x=static_cast<T const *>(a),y=static_cast<T const *>(b);
  auto z=static_cast<T *>(output);
  for (int i=threadIdx.x;i<HIDDEN;i+=blockDim.x) z[i]=T(float(x[i])+float(y[i]));
  __syncthreads();
}

template <typename T, int HIDDEN=5120>
__device__ __forceinline__ void qwen35_prefill_last_task(
    void const *input,void const *length_ptr,void *output) {
  int last=int(*static_cast<int64_t const *>(length_ptr))-1;
  auto x=static_cast<T const *>(input);auto y=static_cast<T *>(output);
  for(int i=threadIdx.x;i<HIDDEN;i+=blockDim.x) y[i]=x[last*HIDDEN+i];
  __syncthreads();
}

template <typename T, int K, int N, int M=64, int SPLITS=1, int SPAN=K/SPLITS>
__device__ __forceinline__ void qwen35_prefill_linear_task(
    void const *input, void const *weight, void *output) {
  static_assert(K%16==0 && SPAN%16==0 && SPLITS*SPAN>=K);
  auto x = static_cast<T const *>(input);
  auto w = static_cast<T const *>(weight); // one 64-column slice
  auto y = static_cast<T *>(output);      // same slice, full N row stride
  int lane=threadIdx.x%32, warp=threadIdx.x/32;
  int group=lane/4, pair=(lane%4)*2;
  for (int row=0; row<M; row+=16) {
    float total[4]={};
    #pragma unroll
    for (int split=0; split<SPLITS; ++split) {
      float c[4]={};
      for (int base=split*SPAN; base<min((split+1)*SPAN,K); base+=16) {
        uint32_t a[4]={}, b[2]={};
        #pragma unroll
        for (int half=0; half<2; ++half) {
          #pragma unroll
          for (int j=0; j<2; ++j) {
            int d=base+pair+half*8+j;
            if (row+group<M) a[half*2] |= uint32_t(__bfloat16_as_ushort(
                __float2bfloat16(float(x[(row+group)*K+d])))) << (16*j);
            if (row+group+8<M) a[half*2+1] |= uint32_t(__bfloat16_as_ushort(
                __float2bfloat16(float(x[(row+group+8)*K+d])))) << (16*j);
            b[half] |= uint32_t(__bfloat16_as_ushort(
                __float2bfloat16(float(w[(warp*8+group)*K+d])))) << (16*j);
          }
        }
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
      }
      #pragma unroll
      for (int j=0; j<4; ++j) total[j]=__fadd_rn(total[j], c[j]);
    }
    #pragma unroll
    for (int j=0; j<2; ++j) {
      if (row+group<M) y[(row+group)*N+warp*8+pair+j]=T(total[j]);
      if (row+group+8<M) y[(row+group+8)*N+warp*8+pair+j]=T(total[j+2]);
    }
  }
  __syncthreads();
}

// Pinned Torch 2.7/cuBLAS 12.8 on the 132-SM H100 selects different split-K
// kernels for the narrow 27B projections. These shape boundaries were checked
// against separate native A/B and K/V calls for every length from 1 to 1024.
// The choice affects FP32 reduction order only; every path computes the same
// matrix product. Use the full call's length, including a partial final tile.
__device__ __forceinline__ int qwen38_projection_splits(int rows,int columns) {
  if(columns==128)
    return ((rows>=641 && rows<=984) || (rows>=992 && rows%8==0)) ? 2 : 3;
  if(rows<=24 || rows==32) return 3;
  if(rows<=79 || (rows>=96 && rows<=112) || (rows>=121 && rows<=128) ||
      (rows>=161 && rows<=168)) return 2;
  return 1;
}

__device__ __forceinline__ int qwen38_down_splits(int rows) {
  if((rows>=57 && rows<=63) || (rows>=121 && rows<=128) ||
      (rows>=313 && rows<=319)) return 2;
  if((rows>=40 && rows<=56) || (rows>=64 && rows<=120) ||
      (rows>=129 && rows<=184) || (rows>=193 && rows<=200) ||
      rows==208 || rows==320) return 3;
  return 1;
}

template <typename T,int K,int N>
__device__ __forceinline__ void qwen35_prefill_linear_dispatch(
    void const *input,void const *weight,void const *rows,void *output) {
  if constexpr(K==17408 && N==5120) {
    int splits=qwen38_down_splits(int(*static_cast<int64_t const *>(rows)));
    if(splits==3) qwen35_prefill_linear_task<T,K,N,64,3,5824>(input,weight,output);
    else if(splits==2) qwen35_prefill_linear_task<T,K,N,64,2>(input,weight,output);
    else qwen35_prefill_linear_task<T,K,N,64,1>(input,weight,output);
  } else if constexpr(K==5120 && (N==128 || N==1024)) {
    int splits=qwen38_projection_splits(int(*static_cast<int64_t const *>(rows)),N);
    if(splits==3) qwen35_prefill_linear_task<T,K,N,64,3,1728>(input,weight,output);
    else if(splits==2) qwen35_prefill_linear_task<T,K,N,64,2>(input,weight,output);
    else qwen35_prefill_linear_task<T,K,N,64,1>(input,weight,output);
  } else qwen35_prefill_linear_task<T,K,N,64,1>(input,weight,output);
}

template <typename T, int CHANNELS=6144>
__device__ __forceinline__ void qwen35_prefill_conv_task(
    void const *input, void const *weight, void const *history,
    void const *length_ptr, void *output, void *next_history) {
  auto x=static_cast<T const *>(input);
  auto w=static_cast<T const *>(weight);
  auto old=static_cast<T const *>(history);
  auto y=static_cast<T *>(output);
  auto state=static_cast<T *>(next_history);
  int length=int(*static_cast<int64_t const *>(length_ptr));
  // Runtime partitions the channel dimension into 48 groups of 128.
  for (int index=threadIdx.x; index<64*128; index+=blockDim.x) {
    int t=index/128, c=index%128;
    float sum=0.f;
    if (t<length) {
      for (int j=0; j<4; ++j) {
        int at=t+j-3;
        float value=at<0 ? float(old[c*3+at+3]) : float(x[at*CHANNELS+c]);
        sum=__fadd_rn(sum, __fmul_rn(value, float(w[c*4+j])));
      }
      sum=float(T(sum));
      sum=sum/(1.f+expf(-sum));
    }
    y[t*CHANNELS+c]=T(sum);
  }
  for (int index=threadIdx.x; index<128*3; index+=blockDim.x) {
    int c=index/3, at=length-3+index%3;
    state[index]=at<0 ? old[c*3+at+3] : x[at*CHANNELS+c];
  }
  __syncthreads();
}

// Native FP32 matmuls and separate elementwise operations follow the
// Transformers chunk-delta formulation; no recurrent-token substitution.
// One task owns a head, and all time rows of that head have full tensor strides.
template <typename T, typename NormT, int HEADS=48, int KEY_HEADS=16, int AB_WIDTH=128>
__device__ __forceinline__ void qwen35_prefill_delta_task(
    void const *conv, void const *z_ptr, void const *ab_ptr,
    void const *parameters, void const *norm_weight, void const *state_ptr,
    void const *length_ptr, void *raw_ptr, void *gated_ptr,
    void *state_out_ptr, void *scratch_ptr, void const *head_id=nullptr) {
  auto x=static_cast<T const *>(conv);
  int vh=head_id ? int(*static_cast<int64_t const *>(head_id)) : 0;
  int kh=vh/(HEADS/KEY_HEADS);
  constexpr int CHANNELS=(2*KEY_HEADS+HEADS)*128;
  auto qx=x+kh*128, kx=x+(KEY_HEADS+kh)*128, vx=x+(2*KEY_HEADS+vh)*128;
  auto z=static_cast<T const *>(z_ptr);
  auto ab=static_cast<T const *>(ab_ptr);
  auto p=static_cast<float const *>(parameters);
  auto old=static_cast<float const *>(state_ptr);
  auto next=static_cast<float *>(state_out_ptr);
  auto raw=static_cast<T *>(raw_ptr);
  auto gated=static_cast<T *>(gated_ptr);
  int length=int(*static_cast<int64_t const *>(length_ptr));
  auto q=static_cast<float *>(scratch_ptr);
  auto k=q+8192, kb=k+8192, vb=kb+8192, value=vb+8192;
  auto kd=value+8192, residual=kd+8192, qg=residual+8192, kg=qg+8192;
  auto decay=kg+8192, correction=decay+4096, attention=correction+4096;
  auto g=attention+4096;
  __shared__ float qi, ki, beta;
  for (int t=0; t<64; ++t) {
    int d=threadIdx.x;
    float qv=d<128 && t<length ? float(qx[t*CHANNELS+d]) : 0.f;
    float kv=d<128 && t<length ? float(kx[t*CHANNELS+d]) : 0.f;
    float qs=qwen35_sum_128(float(T(__fmul_rn(qv,qv))));
    float ks=qwen35_sum_128(float(T(__fmul_rn(kv,kv))));
    if (d==0) {
      qi=float(T(rsqrtf(float(T(float(T(qs))+1.e-6f)))));
      ki=float(T(rsqrtf(float(T(float(T(ks))+1.e-6f)))));
      float dt=float(ab[t*AB_WIDTH])+p[HEADS];
      float softplus=dt>20.f ? dt : log1pf(expf(dt));
      g[t]=t<length ? -expf(p[0])*softplus : 0.f;
      beta=t<length ? float(T(1.f/(1.f+expf(-float(ab[t*AB_WIDTH+HEADS]))))) : 0.f;
    }
    __syncthreads();
    if (d<128) {
      q[t*128+d]=__fmul_rn(float(T(__fmul_rn(qv,qi))), rsqrtf(128.f));
      k[t*128+d]=float(T(__fmul_rn(kv,ki)));
      kb[t*128+d]=__fmul_rn(k[t*128+d], beta);
      vb[t*128+d]=t<length ? __fmul_rn(float(vx[t*CHANNELS+d]),beta) : 0.f;
    }
    __syncthreads();
  }
  // PyTorch ScanUtils.cuh chooses scan width from the total number of rows
  // (16 heads * full-call chunks). At three or more chunks it scans two
  // 32-element segments, carrying the first segment into the next before
  // its Sklansky scan. A full 64-element scan has different FP32 rounding.
  int full_length=int(static_cast<int64_t const *>(length_ptr)[1]);
  int scan_threads=HEADS*((full_length+63)/64)>32 ? 16 : 32;
  for(int block=0;block<64;block+=2*scan_threads) {
    if(threadIdx.x==0 && block) g[block]=__fadd_rn(g[block],g[block-1]);
    __syncthreads();
    for (int size=1; size<=scan_threads; size*=2) {
      if (threadIdx.x<scan_threads) {
        int start=block+(threadIdx.x/size)*(2*size)+size;
        int dst=start+threadIdx.x%size;
        g[dst]=__fadd_rn(g[start-1],g[dst]);
      }
      __syncthreads();
    }
  }
  __syncthreads();
  for (int index=threadIdx.x; index<4096; index+=blockDim.x) {
    int r=index/64, c=index%64;
    float dd=c<=r ? expf(__fadd_rn(g[r],-g[c])) : 0.f;
    decay[index]=dd;
    float dot=0.f;
    for (int d=0; d<128; ++d) dot=__fmaf_rn(kb[r*128+d],k[c*128+d],dot);
    correction[index]=c<r ? -__fmul_rn(dot,dd) : 0.f;
  }
  __syncthreads();
  for (int r=1; r<64; ++r) {
    float value=0.f;
    if (threadIdx.x<r) {
      int c=threadIdx.x;
      float partial[4]={};
      for (int t=0; t<r; ++t)
        partial[t%4]=__fadd_rn(partial[t%4],__fmul_rn(correction[r*64+t],correction[t*64+c]));
      value=__fadd_rn(__fadd_rn(__fadd_rn(partial[0],partial[1]),partial[2]),partial[3]);
      value=__fadd_rn(correction[r*64+c],value);
    }
    __syncthreads();
    if (threadIdx.x<r) correction[r*64+threadIdx.x]=value;
    __syncthreads();
  }
  if (threadIdx.x<64) correction[threadIdx.x*65]=1.f;
  for (int index=threadIdx.x; index<8192; index+=blockDim.x) {
    int r=index/128;
    kg[index]=__fmul_rn(kb[index],expf(g[r]));
    qg[index]=__fmul_rn(q[index],expf(g[r]));
  }
  __syncthreads();
  for (int index=threadIdx.x; index<8192; index+=blockDim.x) {
    int r=index/128,d=index%128;
    float a=0.f,b=0.f;
    for (int t=0; t<64; ++t) {
      a=__fmaf_rn(correction[r*64+t],vb[t*128+d],a);
      b=__fmaf_rn(correction[r*64+t],kg[t*128+d],b);
    }
    value[index]=a; kd[index]=b;
  }
  for (int index=threadIdx.x; index<4096; index+=blockDim.x) {
    int r=index/64,c=index%64;
    float sum=0.f;
    for (int d=0; d<128; ++d) sum=__fmaf_rn(q[r*128+d],k[c*128+d],sum);
    attention[index]=__fmul_rn(sum,decay[index]);
  }
  __syncthreads();
  for (int index=threadIdx.x; index<8192; index+=blockDim.x) {
    int r=index/128,d=index%128;
    float sum=0.f;
    for (int j=0; j<128; ++j) sum=__fmaf_rn(kd[r*128+j],old[j*128+d],sum);
    residual[index]=__fadd_rn(value[index],-sum);
    kg[index]=__fmul_rn(k[index],expf(__fadd_rn(g[63],-g[r])));
  }
  __syncthreads();
  for (int index=threadIdx.x; index<8192; index+=blockDim.x) {
    int r=index/128,d=index%128;
    float a=0.f,b=0.f;
    for (int j=0; j<128; ++j) a=__fmaf_rn(qg[r*128+j],old[j*128+d],a);
    for (int t=0; t<64; ++t) b=__fmaf_rn(attention[r*64+t],residual[t*128+d],b);
    raw[r*(HEADS*128)+d]=T(__fadd_rn(a,b));
  }
  for (int index=threadIdx.x; index<16384; index+=blockDim.x) {
    int r=index/128,d=index%128;
    float sum=0.f;
    for (int t=0; t<64; ++t) sum=__fmaf_rn(kg[t*128+r],residual[t*128+d],sum);
    next[index]=__fadd_rn(__fmul_rn(old[index],expf(g[63])),sum);
  }
  __syncthreads();
  for (int t=0; t<64; ++t)
    qwen35_norm_gate_task<T,NormT,128>(raw+t*(HEADS*128),z+t*(HEADS*128),norm_weight,gated+t*(HEADS*128));
}
template <typename T, typename NormT, int HEADS, int KEY_HEADS, int AB_WIDTH>
__device__ __forceinline__ void qwen35_prefill_delta_grouped_task(
    void const *conv, void const *z, void const *ab, void const *parameters,
    void const *weight, void const *state, void const *length, void const *head_id,
    void *gated, void *next, void *scratch) {
  void *raw=static_cast<float *>(scratch)+QWEN35_CHUNK_SCRATCH;
  qwen35_prefill_delta_task<T,NormT,HEADS,KEY_HEADS,AB_WIDTH>(
      conv,z,static_cast<T const *>(ab)+*static_cast<int64_t const *>(head_id),
      parameters,weight,state,length,raw,gated,next,scratch,head_id);
}
} // namespace kernel
