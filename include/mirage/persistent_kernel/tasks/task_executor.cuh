/* Copyright 2025 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

template <typename TaskImpl>
class Task {
public:
  struct Config {
    int input_pages[TaskImpl::NUM_INPUT];
    int weight_pages[TaskImpl::NUM_WEIGHT];
    int output_pages;

    TensorLifetime input_lifetime;
    TensorLifetime weight_lifetime;
    TensorLifetime output_lifetime;

    bool prefetch;
  };

  __device__ static constexpr Config get_config() {
    return TaskImpl::config();
  }

  __device__ static int prefetch(typename TaskImpl::Params &params,
                                 TensorDesc &input_desc,
                                 TensorDesc &weight_desc,
                                 TensorDesc &output_desc) {
    return TaskImpl::prefetch(
        params, input_desc, weight_desc, output_desc, temp_desc);
  }

  __device__ static void mainloop(typename TaskImpl::Params &params,
                                  TensorDesc &input_desc,
                                  TensorDesc &weight_desc,
                                  TensorDesc &output_desc,
                                  int &previous_copy_done_flag) {
    TaskImpl::mainloop(params,
                       input_desc,
                       weight_desc,
                       output_desc,
                       temp_desc,
                       previous_copy_done_flag);
  }

  __device__ static void epilogue(typename TaskImpl::Params &params,
                                  TensorDesc &output_desc) {
    TaskImpl::epilogue(params, output_desc);
  }
};

class TaskExecutor {
private:
  SharedMemoryManager *smem_mgr;
  __shared__ int previous_copy_done_flag;

public:
  // Generic task execution with automatic overlapping
  template <typename CurrentTaskImpl, typename NextTaskImpl>
  __device__ void execute(typename CurrentTaskImpl::Params &params,
                          typename NextTaskImpl::Params *next_params = nullptr,
                          TaskDesc const &current_task,
                          TaskDesc const &next_task) // For prefetch
  {
    using Config = typename CurrentTaskImpl::Config;
    Config cfg = CurrentTaskImpl::get_config();

    // allocate pages for current task
    for (int i = 0; i < CurrentTaskImpl::NUM_INPUT, i++) {
      current_task.input[i].page_id =
          smem_mgr->allocate_pages(cfg.input_pages[i], cfg.input_lifetime);
      current_task.input[i].num_pages = cfg.input_pages;
      current_task.input[i].lifetime = cfg.input_lifetime;
      current_task.input[i].base_ptr =
          smem_mgr->get_page_ptr(current_task.input[i].page_id);
    }

    for (int i = 0; i < CurrentTaskImpl::NUM_WEIGHT; i++) {
      current_task.weight[i].page_id =
          smem_mgr->allocate_pages(cfg.weight_pages, cfg.weight_lifetime);
      current_task.weight[i].num_pages = cfg.weight_pages;
      current_task.weight[i].lifetime = cfg.weight_lifetime;
      current_task.weight[i].ptr =
          smem_mgr->get_page_ptr(current_task.weight[i].page_id);
    }

    current_task.output.page_id =
        smem_mgr->allocate_pages(cfg.output_pages, cfg.output_lifetime);
    current_task.output.num_pages = cfg.output_pages;
    current_task.output.lifetime = cfg.output_lifetime;
    current_task.output.ptr =
        smem_mgr->get_page_ptr(current_task.output[i].page_id);

    __syncthreads();
    // compute current task
    CurrentTaskImpl::mainloop(params,
                              current_task.input,
                              current_task.weight[0],
                              current_task.output[0],
                              task.temp,
                              previous_copy_done_flag);

    prefetch_next_task<NextTaskImpl>(next_task, next_params);
    // free all tensors used in current task
    smem_mgr->release_all_pages(current_task.input.page_id,
                                task.input.num_pages);

    // don't free output pages here, as they may be used later
    //  smem_mgr->release_all_pages(current_task.output[i].page_id,
    //                                      current_task.output[i].num_pages);
    smem_mgr->release_all_pages(current_task.weight[i].page_id,
                                current_task.weight[i].num_pages);
  }

private:
  template <typename TaskImpl>
  __device__ void prefetch_next_task(typename TaskImpl::Params &next_params,
                                     TaskDesc const &next_task) {
    // wait for previous copy to finish
    asm volatile("bar.sync %0, %1;\n" ::"r"(previous_copy_done_flag), "n"(128));
    using Config = typename TaskImpl::Config;
    Config cfg = TaskImpl::get_config();

    // Check if we have space
    if (!smem_mgr->can_allocate(cfg.weight_pages)) {
      return; // Not enough space, skip prefetch
    }

    for (int i = 0; i < CurrentTaskImpl::NUM_WEIGHT; i++) {
      next_task.weight[i].page_id =
          smem_mgr->allocate_pages(cfg.weight_pages, cfg.weight_lifetime);
      next_task.weight[i].num_pages = cfg.weight_pages;
      next_task.weight[i].lifetime = cfg.weight_lifetime;
      next_task.weight[i].ptr =
          smem_mgr->get_page_ptr(next_task.weight[i].page_id);
    }

    TaskImpl::prefetch();
  }
};