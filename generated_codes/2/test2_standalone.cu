#define NUM_GPUS 1
#define USE_NVSHMEM false
#define MIRAGE_BLACKWELL
#include "runtime.h"
// debug用的头文件
#include <threadblock/utils.h>
#include <random>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cutlass/util/print_error.hpp>
#include <cute/util/debug.hpp>
#include <cutlass/gemm/collective/builders/sm100_common.inl>
// debug用的头文件
using namespace cute;

constexpr int Gemm_M = 512;
constexpr int Gemm_N = 1024; 
constexpr int Gemm_K = 256;
constexpr int Tiled_M = 256;
constexpr int Tiled_N = 256;

// 添加参考GEMM实现
template <class AccType,
          class TensorA, class TensorB,
          class TensorC, class TensorD,
          class Alpha, class Beta>
void
reference_gemm(TensorA const& tensor_A, TensorB const& tensor_B,
               TensorC const& tensor_C, TensorD      & tensor_D,
               Alpha alpha, Beta beta)
{
  using namespace cute;
  for (int m = 0; m < size<0>(tensor_D); ++m) {
    for (int n = 0; n < size<1>(tensor_D); ++n) {
      AccType c = AccType(0.f);
      for (int k = 0; k < size<1>(tensor_A); ++k) {
        c += tensor_A(m,k) * tensor_B(n,k);
      }
      tensor_D(m,n) = alpha * c + beta * tensor_C(m,n);
    }
  }
}

template <class Tensor>
void
initialize_tensor(Tensor& tensor, cute::tuple<int, int> value_range = {-5, 5}, bool m_input = false)
{
  using DataType = typename Tensor::element_type;
  auto [min, max] = value_range;
  for (int i = 0; i < cute::size(tensor); i++) {
    // tensor(i) = DataType(int((max-min)*(rand() / double(RAND_MAX)) + min));
    tensor(i) = DataType(1);
    // tensor(i) = DataType(rand() % 3);
  }
  tensor(0) = DataType(3);
  // if (m_input){
  //   for (int i = 0; i < 10; i++){
  //     tensor(0, i) = DataType(3);
  //   }
  // }
  // else{
  //   for (int i = 0; i < 10; i++){
  //     tensor(i, 0) = DataType(3);
  //   }
  // }
  // print_tensor(tensor);
}
template <class TMA_10000003, class TMA_10000004>
__global__ void  __launch_bounds__(256) custom_kernel_0(CUTE_GRID_CONSTANT TMA_10000003 const tma_10000003, CUTE_GRID_CONSTANT TMA_10000004 const tma_10000004,  half_t* dtensor10000005_ptr, half_t const* dtensor10000003_ptr, half_t const* dtensor10000004_ptr) {
int thread_idx = threadIdx.x;
static constexpr int NUM_THREADS = 128;
static constexpr int CONSUMER_NUM_THREADS = 128;
auto cluster_shape = make_shape(Int<4>{}, Int<4>{}, Int<1>{});
uint32_t elect_one_warp = (threadIdx.x / 32 == 0); 


// STensors
extern __shared__ char buf[];

uint64_t *mma_barrier_ptr = (uint64_t*)(buf + 16);
uint32_t *tmem_base_ptr = (uint32_t*)(buf + 0);
half_t *stensor20000015_ptr = (half_t*)(buf + 128);
half_t *stensor20000013_ptr = (half_t*)(buf + 32896);
half_t *stensor20000012_ptr = (half_t*)(buf + 128);
// Tensory Memory Allocation
using TmemAllocator = cute::TMEM::Allocator2Sm;
TmemAllocator tmem_allocator{};
if (elect_one_warp) { 
tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns, tmem_base_ptr); 
}

__syncthreads();


auto tiled_mma = cutlass::gemm::collective::detail::sm100_make_2sm_trivial_tiled_mma<half_t, half_t, half_t, Shape<Int<256>, Int<256>>, decltype(cluster_shape), UMMA::Major::K, UMMA::Major::MN>();
Layout cluster_layout_vmnk = tiled_divide(make_layout(cluster_shape),
                                          make_tile(typename decltype(tiled_mma)::AtomThrID{}));

auto mma_tiler = make_shape(tile_size<0>(tiled_mma), tile_size<1>(tiled_mma), tile_size<2>(tiled_mma)*_2{});
int cta_rank = cute::block_rank_in_cluster();
auto cta_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank);
auto elect_one_cta = get<0>(cta_in_cluster_coord_vmnk) == Int<0>{};
using Matmul20000015LayoutA = decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<32>, Int<128>>, Stride<Int<1>, Int<32>>>{}));
using Matmul20000015LayoutB = decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<256>, Int<32>>, Stride<Int<1>, Int<256>>>{}));
using Matmul20000015LayoutC = Layout<Shape<Int<256>, Int<128>>, Stride<Int<1>, Int<256>>>;
using Matmul20000015Kernel = tb::Blackwell_Matmul<half_t, true, false, Matmul20000015LayoutA, Matmul20000015LayoutB, Matmul20000015LayoutC, NUM_THREADS, 0, false, false, true, true, 4, decltype(cluster_shape), decltype(tiled_mma), decltype(mma_tiler)>;
auto matmul_20000015_accum = Matmul20000015Kernel::get_mma_tC(blockIdx.x, blockIdx.y, *tmem_base_ptr);

// MMA mbarrier Initialization
uint16_t pair_mask = (1 << blockIdx.y * gridDim.x + (blockIdx.x / 2) * 2) | (1 << blockIdx.y * gridDim.x + (blockIdx.x / 2) * 2 + 1);
// uint16_t pair_mask = static_cast<uint16_t>((1u << size(cluster_shape)) - 1u);
// uint16_t pair_mask = create_tma_multicast_mask<0,1>(cluster_layout_vmnk, cta_in_cluster_coord_vmnk) |
//                               create_tma_multicast_mask<0,2>(cluster_layout_vmnk, cta_in_cluster_coord_vmnk);
if (elect_one_warp && cute::elect_one_sync()) {
  int num_mcast_participants = size<1>(cluster_layout_vmnk) + size<2>(cluster_layout_vmnk) - 1;
  // printf("num_mcast_participants is %d\n", num_mcast_participants);
  // printf("pair_mask is %x\n", pair_mask);
  cute::initialize_barrier(*mma_barrier_ptr, 8);
  // cute::initialize_barrier(*mma_barrier_ptr, 8 * size(cluster_shape) / 2);
}
__syncthreads();
// G->S copy atoms
// Copy for G->S: dtensor 10000003 -> stensor 20000012
using DTensor10000003TileLayout = Layout<Shape<Int<32>, Int<128>>, Stride<Int<1>, Int<256>>>;
tb::BlackwellAsyncPipeline<4, decltype(cluster_shape)> blackwell_async_pipeline_20000012((void *) (buf + 98432), (tb::warpgroup_id() == 1 && tb::warp_id() % mirage::config::NUM_WARPS_PER_GROUP == 0), tb::warpgroup_id() < 1, 16384, 1, elect_one_cta);
using STensor20000012InputAtom = tb::InputTMAAsyncCopy_Blackwell<half_t, decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<32>>, Stride<Int<32>, Int<1>>>{})), Layout<Shape<Int<512>, Int<256>>, Stride<Int<256>, Int<1>>>, decltype(tma_10000003), decltype(blackwell_async_pipeline_20000012), true, 8, decltype(tiled_mma), decltype(mma_tiler), decltype(cluster_shape)>;
// Copy for G->S: dtensor 10000004 -> stensor 20000013
using DTensor10000004TileLayout = Layout<Shape<Int<256>, Int<32>>, Stride<Int<1>, Int<1024>>>;
tb::BlackwellAsyncPipeline<4, decltype(cluster_shape)> blackwell_async_pipeline_20000013((void *) (buf + 98496), (tb::warpgroup_id() == 1 && tb::warp_id() % mirage::config::NUM_WARPS_PER_GROUP == 0), tb::warpgroup_id() < 1, 16384, 1, elect_one_cta);
using STensor20000013InputAtom = tb::InputTMAAsyncCopy_Blackwell<half_t, decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<256>, Int<32>>, Stride<Int<1>, Int<256>>>{})), Layout<Shape<Int<1024>, Int<256>>, Stride<Int<1>, Int<1024>>>, decltype(tma_10000004), decltype(blackwell_async_pipeline_20000013), false, 8, decltype(tiled_mma), decltype(mma_tiler), decltype(cluster_shape)>;

__syncthreads();
  *((uint128_t*)buf) = 0ul;
  
  
  
  // S->G copy atoms
  // Copy for S->G: stensor 20000015 -> dtensor 10000005
  half_t *dtensor10000005_tile_ptr = dtensor10000005_ptr  + blockIdx.x*128*1024 + blockIdx.y*256*1;
  using DTensor10000005TileLayout = Layout<Shape<Int<256>, Int<128>>, Stride<Int<1>, Int<1024>>>;
  using STensor20000015OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000005TileLayout, Layout<Shape<Int<256>, Int<128>>, Stride<Int<1>, Int<256>>>, NUM_THREADS>;
  
  
  int warpgroup_id = tb::warpgroup_id();
  if (warpgroup_id == 1) {
    if (tb::warp_id_in_wg() == 0) {
      for (uint32_t for_idx = 0; for_idx < 8; for_idx++) {
        // printf("producer for loop %d, blockIdx.x is %d, blockIdx.y is %d, threadIdx.x is %d\n", for_idx, blockIdx.x, blockIdx.y, threadIdx.x);
        STensor20000012InputAtom::run(tma_10000003, stensor20000012_ptr, tiled_mma, mma_tiler, for_idx, blackwell_async_pipeline_20000012);
        STensor20000013InputAtom::run(tma_10000004, stensor20000013_ptr, tiled_mma, mma_tiler, for_idx, blackwell_async_pipeline_20000013);
      }
    }
  }
  else {
    // Consumer main loop
    for (uint32_t for_idx = 0; for_idx < 8; for_idx++) {
      {
        // if (elect_one_cta) {
        //   printf("consumer for loop %d, blockIdx.x is %d, blockIdx.y is %d, threadIdx.x is %d\n", for_idx, blockIdx.x, blockIdx.y, threadIdx.x);
        // }
        // OP type: tb_matmul_op
        if (elect_one_cta && elect_one_warp) {
          int read_idx_20000012 = blackwell_async_pipeline_20000012.consumer_wait();
          int read_idx_20000013 = blackwell_async_pipeline_20000013.consumer_wait();
          Matmul20000015Kernel::run(matmul_20000015_accum, stensor20000012_ptr, stensor20000013_ptr, for_idx, tiled_mma, read_idx_20000012, blackwell_async_pipeline_20000012, blackwell_async_pipeline_20000013);

          cutlass::arch::umma_arrive_multicast_2x1SM(mma_barrier_ptr, pair_mask);
          
        }
      }
      if (elect_one_cta && elect_one_warp) {
        blackwell_async_pipeline_20000012.consumer_release();
        blackwell_async_pipeline_20000013.consumer_release();
    }
    }

    cute::wait_barrier(*mma_barrier_ptr, 0);

    // Matmul20000015Kernel::write_tC_to_gC(dtensor10000005_tile_ptr-(blockIdx.x % 2)*128*1024, matmul_20000015_accum, thread_idx);

    Matmul20000015Kernel::write_tC_to_sC(stensor20000015_ptr, matmul_20000015_accum, thread_idx);
    // The epilogue (kernels outside the loop)
    tb::wg_sync<CONSUMER_NUM_THREADS>(8);
    {
      // OP type: tb_output_op
      STensor20000015OutputAtom::run(dtensor10000005_tile_ptr, stensor20000015_ptr, thread_idx);
    }
  }
  cluster_sync();
  // __syncthreads();
  if (elect_one_warp) { 
  tmem_allocator.release_allocation_lock(); 
  tmem_allocator.free(*tmem_base_ptr, TmemAllocator::Sm100TmemCapacityColumns); 
}
}



static void _init() {
}

void _execute_mugraph(std::vector<void const *> input_tensors, std::vector<void*> output_tensors, void* buf, cudaStream_t stream, void * profiler_buffer){
  {
    // OP type: kn_input_op
  }
  {
    // OP type: kn_input_op
  }
  {
    // OP type: kn_customized_op
    half_t *dtensor10000005 = (half_t*)output_tensors.at(0);  // 更改为half_t*
    half_t *dtensor10000003 = (half_t*)input_tensors.at(0);
    half_t *dtensor10000004 = (half_t*)input_tensors.at(1);

    dim3 block_dim(256, 1, 1);
    size_t smem_size = 196864 + 32 + 128;  // Add extra space for tmem_base_ptr with alignment padding
    auto cluster_shape = make_shape(Int<4>{}, Int<4>{}, Int<1>{});
    
    dim3 cluster_dim(size<0>(cluster_shape), size<1>(cluster_shape), size<2>(cluster_shape));
    dim3 grid_dim(((Gemm_M + Tiled_M - 1) / Tiled_M + cluster_dim.x - 1) / cluster_dim.x * cluster_dim.x, ((Gemm_N + Tiled_N - 1) / Tiled_N + cluster_dim.y - 1) / cluster_dim.y * cluster_dim.y, 1);

    // define tmas
    TiledMMA tiled_mma = cutlass::gemm::collective::detail::sm100_make_2sm_trivial_tiled_mma<half_t, half_t, half_t, Shape<Int<256>, Int<256>>, decltype(cluster_shape), UMMA::Major::K, UMMA::Major::MN>();
    Layout cluster_layout_vmnk = tiled_divide(make_layout(cluster_shape), make_tile(typename decltype(tiled_mma)::AtomThrID{}));
    auto mma_tiler = make_shape(tile_size<0>(tiled_mma), tile_size<1>(tiled_mma), tile_size<2>(tiled_mma)*_2{});
    
    std::vector<bool> minputs = {true, false};
    
    static constexpr cute::UMMA::Major UMMAMajor_10000003 = UMMA::Major::K;
    // using DstMNKLayout_10000003 = decltype(partition_shape_A(tiled_mma, make_shape(size<0>(mma_tiler), size<2>(mma_tiler))));
    using DstMNKLayout_origin_10000003 = decltype(composition(Swizzle<2, 4, 3>{}, Layout<Shape<Int<128>, Int<32>>, Stride<Int<32>, Int<1>>>{}));
    using DstMNKLayout_10000003 = decltype(partition_shape_A(tiled_mma, make_shape(shape<0>(DstMNKLayout_origin_10000003{})*_2{}, shape<1>(DstMNKLayout_origin_10000003{}))));
    using SrcMNKLayout_10000003 = Layout<Shape<Int<512>, Int<256>>, Stride<Int<256>, Int<1>>>;
    using SmemLayoutAtom_10000003 = decltype(cutlass::gemm::collective::detail::sm100_smem_selector<UMMAMajor_10000003, half_t, decltype(get<0>(mma_tiler)), decltype(get<2>(mma_tiler))>());
    // *2{} since 2 cta multicas
    // using SmemLayoutAtom_10000003 = decltype(cutlass::gemm::collective::detail::sm100_smem_selector<UMMAMajor_10000003, half_t, decltype(shape<0>(DstMNKLayout_10000003{})), decltype(shape<1>(DstMNKLayout_10000003{}))>());
    
    
    // zy: add a stage to last dim
    using DstPipeLayout_10000003 = decltype(UMMA::tile_to_mma_shape(SmemLayoutAtom_10000003{}, append(DstMNKLayout_10000003{}, Int<4>{}), Step<_1,_2,_3>{}));
    // using DstPipeLayout_10000003 = decltype(UMMA::tile_to_mma_shape(SmemLayoutAtom_10000003{}, make_shape(shape<0>(DstMNKLayout_10000003{}), shape<1>(DstMNKLayout_10000003{}), _4{}), Step<_1,_2,_3>{}));
    
    auto g_tensor_10000003 = make_tensor(make_gmem_ptr<half_t>(dtensor10000003), SrcMNKLayout_10000003{});
    auto tma_10000003 = make_tma_atom_A_sm100(SM100_TMA_2SM_LOAD_MULTICAST{}, g_tensor_10000003, DstPipeLayout_10000003{}(_,_,_,Int<0>{}), mma_tiler, tiled_mma, cluster_layout_vmnk);
    

    static constexpr cute::UMMA::Major UMMAMajor_10000004 = UMMA::Major::MN;
    // static constexpr cute::UMMA::Major UMMAMajor_10000004 = UMMA::Major::K;
    using DstMNKLayout_10000004 = decltype(partition_shape_B(tiled_mma, make_shape(size<1>(mma_tiler), size<2>(mma_tiler))));
    // change to K major for B
    using SrcMNKLayout_10000004 = Layout<Shape<Int<1024>, Int<256>>, Stride<Int<1>, Int<1024>>>;
    // using SrcMNKLayout_10000004 = Layout<Shape<Int<1024>, Int<256>>, Stride<Int<256>, Int<1>>>;
    // using SrcMNKLayout_10000004 = Layout<Shape<Int<1024>, Int<256>>, Stride<Int<1>, Int<1024>>>;
    using SmemLayoutAtom_10000004 = decltype(cutlass::gemm::collective::detail::sm100_smem_selector<UMMAMajor_10000004, half_t, decltype(get<1>(mma_tiler)), decltype(get<2>(mma_tiler))>());
    // using SmemLayoutAtom_10000004 = decltype(cutlass::gemm::collective::detail::sm100_smem_selector<UMMAMajor_10000004, half_t, decltype(get<1>(mma_tiler)), decltype(get<2>(mma_tiler))>());
    // using DstPipeLayout_10000004 = decltype(UMMA::tile_to_mma_shape(SmemLayoutAtom_10000004{}, (DstMNKLayout_10000004{})));
    using DstPipeLayout_10000004 = decltype(UMMA::tile_to_mma_shape(SmemLayoutAtom_10000004{}, append(DstMNKLayout_10000004{}, Int<4>{}), Step<_2,_1,_3>{}));
    auto g_tensor_10000004 = make_tensor(make_gmem_ptr<half_t>(dtensor10000004), SrcMNKLayout_10000004{});
    auto tma_10000004 = make_tma_atom_B_sm100(SM100_TMA_2SM_LOAD_MULTICAST{}, g_tensor_10000004, DstPipeLayout_10000004{}(_,_,_,Int<0>{}), mma_tiler, tiled_mma, cluster_layout_vmnk);
    
    // // zy: change to add the DstPipeLayout
    auto kernel_ptr = &custom_kernel_0<decltype(tma_10000003), decltype(tma_10000004)>;
    cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    
    cutlass::ClusterLaunchParams params = {grid_dim, block_dim, cluster_dim, smem_size};
    cutlass::launch_kernel_on_cluster(params, (void const*) kernel_ptr, tma_10000003, tma_10000004,  dtensor10000005, dtensor10000003, dtensor10000004);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel execution failed: %s\n", cudaGetErrorString(err));
        exit(1);
    }
  }
  {
    // OP type: kn_output_op
  }
}

int main() {
    printf("Starting CUDA kernel testing...\n");
    
    // Initialize CUDA
    cudaSetDevice(0);
    
    // Set random seed
    srand(time(nullptr));
    
    // A: 512 x 256 (K-major)
    // B: 1024 x 256 (N-major) 
    // D: 512 x 1024 
    
    printf("Running GEMM problem size (MxNxK): %dx%dx%d\n", Gemm_M, Gemm_N, Gemm_K);
    
    ////////////////////////////////////////////////////////////
    //
    // Create tensor layouts and data type definitions
    //
    ////////////////////////////////////////////////////////////
    
    // Define data types
    using TypeA = cutlass::half_t; // MMA A data type
    auto type_str_a = "half_t";
    using TypeB = cutlass::half_t; // MMA B data type
    auto type_str_b = "half_t";
    using TypeC = half_t;           // MMA C data type
    auto type_str_c = "half_t";
    using TypeD = half_t;           // MMA D data type
    auto type_str_d = "half_t";
    using TypeAccumulator = half_t; // Accumulator type
    
    // A tensor MxK K-major (Layout T = Row-Major)
    // Layout layout_A = make_layout(make_shape (Gemm_M,   Gemm_K),
    //                               make_stride(Gemm_K, Int<1>{}));   // (Gemm_M,Gemm_K):(Gemm_K,_1)
    // // B tensor NxK K-major (Layout N = Column-Major)
    // Layout layout_B = make_layout(make_shape (Gemm_N,   Gemm_K),
    //                               make_stride(Gemm_K, Int<1>{}));   // (Gemm_N,Gemm_K):(Gemm_K,_1)
    // // C tensor MxN N-major (Layout T = Row-Major) 
    // Layout layout_C = make_layout(make_shape (Gemm_M,   Gemm_N),
    //                               make_stride(Gemm_N, Int<1>{}));   // (Gemm_M,Gemm_N):(Gemm_N,_1)
    // // D tensor MxN N-major (Layout T = Row-Major)
    // Layout layout_D = make_layout(make_shape (Gemm_M,   Gemm_N),
                                  // make_stride(Gemm_N, Int<1>{}));   // (Gemm_M,Gemm_N):(Gemm_N,_1)
      // Layout layout_D = make_layout(make_shape (Gemm_M,   Gemm_N),
      //                           make_stride(Int<1>{}, Gemm_M));   // (Gemm_M,Gemm_N):(Gemm_N,_1)
    
    Layout layout_A = make_layout(make_shape (Gemm_M,   Gemm_K),
                                  make_stride(Gemm_K, Int<1>{}));   // (Gemm_M,Gemm_K):(Gemm_K,_1)
    // B tensor NxK K-major (Layout N = Column-Major)
    Layout layout_B = make_layout(make_shape (Gemm_N,   Gemm_K),
                                  make_stride(Int<1>{}, Gemm_N));   // (Gemm_N,Gemm_K):(Gemm_K,_1)
    // C tensor MxN N-major (Layout T = Row-Major) 
    Layout layout_C = make_layout(make_shape (Gemm_M,   Gemm_N),
                                  make_stride(Gemm_N, Int<1>{}));   // (Gemm_M,Gemm_N):(Gemm_N,_1)
    // D tensor MxN N-major (Layout T = Row-Major)
    Layout layout_D = make_layout(make_shape (Gemm_M,   Gemm_N),
                                  make_stride(Gemm_N, Int<1>{}));   // (Gemm_M,Gemm_N):(Gemm_N,_1)


    ////////////////////////////////////////////////////////////
    //
    // Host memory allocation and tensor creation
    //
    ////////////////////////////////////////////////////////////
    
    // Use thrust for host allocation
    thrust::host_vector<TypeA>   host_A(Gemm_M * Gemm_K);
    Tensor host_tensor_A = make_tensor(host_A.data(), layout_A);
    printf("host_tensor_A:\t"); print(host_tensor_A); printf("\n");
    
    thrust::host_vector<TypeB>   host_B(Gemm_N * Gemm_K);
    Tensor host_tensor_B = make_tensor(host_B.data(), layout_B);
    printf("host_tensor_B:\t"); print(host_tensor_B); printf("\n");
    
    thrust::host_vector<TypeC>   host_C(Gemm_M * Gemm_N);
    Tensor host_tensor_C = make_tensor(host_C.data(), layout_C);
    printf("host_tensor_C:\t"); print(host_tensor_C); printf("\n");
    
    // For storing device output results
    thrust::host_vector<TypeD>   host_D_result(Gemm_M * Gemm_N);
    Tensor host_tensor_D_result = make_tensor(host_D_result.data(), layout_D);
    printf("host_tensor_D_result:\t"); print(host_tensor_D_result); printf("\n");
    
    
    ////////////////////////////////////////////////////////////
    //
    // Initialize tensor data
    //
    ////////////////////////////////////////////////////////////
    

    initialize_tensor(host_tensor_A, make_tuple(-2, 2), true);
    initialize_tensor(host_tensor_B, make_tuple(-2, 2), false);
    

    
    // Debug output: check input data
    printf("Sample input data A (first 10 elements): ");
    for(int i = 0; i < 10 && i < size(host_tensor_A); i++) {
        printf("%.4f ", (float)host_tensor_A(i));
    }
    printf("\n");
    printf("Sample input data B (first 10 elements): ");
    for(int i = 0; i < 10 && i < size(host_tensor_B); i++) {
        printf("%.4f ", (float)host_tensor_B(i));
    }
    printf("\n");
    
    ////////////////////////////////////////////////////////////
    //
    // Prepare device memory and execute custom kernel
    //
    ////////////////////////////////////////////////////////////
    
    // Allocate device memory
    thrust::device_vector<TypeA> device_A = host_A;
    thrust::device_vector<TypeB> device_B = host_B;
    thrust::device_vector<TypeC> device_C = host_C;
    thrust::device_vector<TypeD> device_D(Gemm_M * Gemm_N);
    
    Tensor mA = make_tensor(make_gmem_ptr<half_t>(device_A.data().get()), layout_A);
    Tensor mB = make_tensor(make_gmem_ptr<half_t>(device_B.data().get()), layout_B);
    Tensor mC = make_tensor(make_gmem_ptr<half_t>(device_C.data().get()), layout_C);
    printf("mA:\t"); print(mA); printf("\n");
    printf("mB:\t"); print(mB); printf("\n");
    printf("mC:\t"); print(mC); printf("\n");

  
    half_t *d_input1 = device_A.data().get();
    half_t *d_input2 = device_B.data().get();
    half_t *d_output = device_C.data().get(); 
    printf("d_output:\t"); print(d_output); printf("\n");
    
    
    // Allocate buffers
    const size_t buf_size = 196864 + 1024;
    const size_t profiler_buf_size = 0;
    void *d_buffer, *d_profiler_buffer;
    cudaError_t err = cudaMalloc(&d_buffer, buf_size);
    if (err != cudaSuccess) {
        printf("cudaMalloc failed for d_buffer: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    err = cudaMalloc(&d_profiler_buffer, profiler_buf_size);
    if (err != cudaSuccess) {
        printf("cudaMalloc failed for d_profiler_buffer: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    
    // Prepare input/output vectors
    std::vector<void const *> input_tensors = {d_input1, d_input2};
    std::vector<void*> output_tensors = {d_output};  // Now half_t* type
    
    ////////////////////////////////////////////////////////////
    //
    // Performance Testing: Warmup + 2000 runs
    //
    ////////////////////////////////////////////////////////////
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    printf("Starting warmup phase...\n");
    // Warmup: run 50 times
    for(int i = 0; i < 0; i++) {
        
        _execute_mugraph(input_tensors, output_tensors, d_buffer, 0, d_profiler_buffer);
        cudaDeviceSynchronize();
    }
    printf("Warmup phase completed.\n");
    
    printf("Starting performance test (2000 runs)...\n");
    // Performance test: run 2000 times and record times
    float total_time = 0.0f;
    std::vector<float> times;
    
    for(int i = 0; i < 1; i++) {
        // Record start time
        cudaEventRecord(start);
        
        // Execute kernel
        _execute_mugraph(input_tensors, output_tensors, d_buffer, 0, d_profiler_buffer);
        
        // Record end time
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        // Calculate execution time
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        times.push_back(milliseconds);
        total_time += milliseconds;
        
        printf("Run %d: %.4f ms\n", i+1, milliseconds);
    }
    
    // Calculate statistics
    float avg_time = total_time / 2000.0f;
    float min_time = *std::min_element(times.begin(), times.end());
    float max_time = *std::max_element(times.begin(), times.end());
    
    // Calculate standard deviation
    float variance = 0.0f;
    for(float time : times) {
        variance += (time - avg_time) * (time - avg_time);
    }
    variance /= 2000.0f;
    float std_dev = sqrt(variance);
    
    printf("\n=== Mirage Custom Kernel Performance Statistics ===\n");
    printf("GEMM problem size: %dx%dx%d\n", Gemm_M, Gemm_N, Gemm_K);
    printf("Number of runs: 2000\n");
    printf("Average time: %.4f ms\n", avg_time);
    printf("Minimum time: %.4f ms\n", min_time);
    printf("Maximum time: %.4f ms\n", max_time);
    printf("Standard deviation: %.4f ms\n", std_dev);
    printf("Total FLOPS: %.2f GFLOPS\n", (2.0 * Gemm_M * Gemm_N * Gemm_K) / 1e9);
    printf("Performance (based on average time): %.2f GFLOPS\n", (2.0 * Gemm_M * Gemm_N * Gemm_K) / (avg_time * 1e6));
    printf("=======================================\n\n");
    
    // Copy results back to host for verification
    thrust::host_vector<half_t> temp_output(Gemm_M * Gemm_N);  // Direct use of half_t
    err = cudaMemcpy(temp_output.data(), d_output, Gemm_M * Gemm_N * sizeof(half_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("cudaMemcpy failed for output: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    
    // Direct copy of float results, no conversion needed
    for (int i = 0; i < Gemm_M * Gemm_N; ++i) {
        host_D_result[i] = temp_output[i];
        // if (temp_output[i] != ) {
        //     printf("copy from device to host: %.4f, %.4f\n", temp_output[i], host_D_result[i]);
        // }
    }
    
    printf("Custom kernel execution completed.\n");
    
    ////////////////////////////////////////////////////////////
    //
    // Execute reference GEMM kernel
    //
    ////////////////////////////////////////////////////////////
    
    printf("Executing reference GEMM implementation...\n");
    thrust::host_vector<TypeD> host_reference_D(Gemm_M * Gemm_N);
    auto host_reference_tensor_D = make_tensor(host_reference_D.data(), layout_D);
    
    using Alpha = float;
    using Beta = float;
    Alpha alpha = 1.0f;  // Corresponds to D = alpha * A * B^T + beta * C
    Beta beta = 0.0f;    // Our kernel is actually D = A * B^T
    
    reference_gemm<TypeAccumulator>(host_tensor_A, host_tensor_B, host_tensor_C, host_reference_tensor_D, alpha, beta);
    
    ////////////////////////////////////////////////////////////
    //
    // Compare results
    //
    ////////////////////////////////////////////////////////////
    
    printf("Comparing results...\n");
    auto relative_error = print_matrix_multiply_mollified_relative_error(type_str_a, host_tensor_A,
                                                                         type_str_b, host_tensor_B,
                                                                         type_str_d, host_tensor_D_result, host_reference_tensor_D);
    
    // Print some sample results for debugging
    printf("\nSample results (first 10 elements):\n");
    for(int i = 0; i < Gemm_M*Gemm_N; i++) {
      if (host_D_result[i] != host_reference_D[i]) {
        // printf("Incorrect output: ");
        printf("x = %d, y = %d, %.4f, %.4f \n", i/Gemm_N, i%Gemm_N, (float)host_D_result[i], (float)host_reference_D[i]);
        break;
      }
    }

    printf("\n");
    printf("Actual output: ");
    for(int i = 0; i < 16; i++) {
        printf("%.4f ", (float)host_D_result[i]);
    }
    printf("\n Reference output: ");
    for(int i = 0; i < 16; i++) {
        printf("%.4f ", (float)host_reference_D[i]);
    }
    printf("\n");
    
    // printf("full output: \n");
    // for(int i = 0; i < Gemm_M*Gemm_N; i++) {
    //   printf("%.4f ", (float)host_D_result[i]);
    //   if (i % Gemm_N == Gemm_N - 1) {
    //     printf("\n");
    //   }
    // }
    // printf("\n");

    // Success criteria - for half precision, relative error should be small
    bool success = relative_error <= 0;  // 1% tolerance, considering half precision limitations
    printf("Relative error: %.6e\n", relative_error);
    printf("Verification result: %s\n", success ? "PASSED" : "FAILED");
    
    ////////////////////////////////////////////////////////////
    //
    // Clean up resources
    //
    ////////////////////////////////////////////////////////////
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_buffer);
    cudaFree(d_profiler_buffer);
    
    printf("\nProgram execution %s.\n", success ? "SUCCESSFUL" : "FAILED");
    return success ? 0 : 1;
} 