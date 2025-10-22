#include "persistent_kernel.cuh"
#include <nlohmann/json.hpp>
#include <fstream>
#include <filesystem>
using json = nlohmann::json;
using namespace mirage::runtime;
size_t get_event_id(int my_gpu_id, size_t event_pos, bool nvshmem_event) {
  size_t event_id = ((static_cast<size_t>(my_gpu_id) << 32) | event_pos);
  if (nvshmem_event) {
    event_id = event_id | EVENT_NVSHMEM_TAG;
  }
  return event_id;
}

void construct_task_graph(int num_gpus,
                          int my_gpu_id,
                          std::vector<FullTaskDesc> &all_tasks,
                          std::vector<EventDesc> &all_events,
                          std::vector<TaskId> &first_tasks,
                          std::map<std::string, void*> const &all_tensors) {
  std::filesystem::path file_path(__FILE__);
  std::ifstream json_file(file_path.parent_path().string()+"/task_graph.json");
  nlohmann::json json_task_graph;
  json_file >> json_task_graph;
  for (json const &task : json_task_graph["all_tasks"]) {
    FullTaskDesc task_desc(static_cast<TaskType>(task.at("task_type")),
                task.at("variant_id"));
    task_desc.request_id = task.at("request_id").get<int>();
    if (task.at("trigger_event").is_number_integer()) {
      task_desc.trigger_event = task.at("trigger_event").get<unsigned long long int>();
    }
    else {
      assert(false);
    }
    if (task.at("dependent_event").is_number_integer()) {
      task_desc.dependent_event = task.at("dependent_event").get<unsigned long long int>();
    }
    else {
      assert(false);
    }
    task_desc.num_inputs = 0;
    for (json const &tensor : task["inputs"]) {
      TensorDesc input;
      std::string name = tensor.at("base_ptr").get<std::string>();
      assert(all_tensors.find(name) != all_tensors.end());
      off_t offset = tensor.at("offset").get<off_t>();
      input.base_ptr = static_cast<char*>(all_tensors.at(name))+offset;
      assert(tensor.at("dims").size() == tensor.at("strides").size());
      input.num_dims = tensor.at("dims").size();
      input.data_type = tensor.at("data_type").get<int>();
      for (int i = 0; i < input.num_dims; i++) {
        input.dim[i] = tensor["dims"][i].get<int>();
        input.stride[i] = tensor["strides"][i].get<int>();
      }
      task_desc.inputs[task_desc.num_inputs++] = input;
    }
    task_desc.num_outputs = 0;
    for (json const &tensor : task["outputs"]) {
      TensorDesc output;
      std::string name = tensor.at("base_ptr").get<std::string>();
      assert(all_tensors.find(name) != all_tensors.end());
      off_t offset = tensor.at("offset").get<off_t>();
      output.base_ptr = static_cast<char*>(all_tensors.at(name))+offset;
      assert(tensor.at("dims").size() == tensor.at("strides").size());
      output.num_dims = tensor.at("dims").size();
      output.data_type = tensor.at("data_type").get<int>();
      for (int i = 0; i < output.num_dims; i++) {
        output.dim[i] = tensor["dims"][i];
        output.stride[i] = tensor["strides"][i];
      }
      task_desc.outputs[task_desc.num_outputs++] = output;
    }
    #ifdef MPK_ENABLE_TMA
    if (task.at("task_type") > TASK_HOPPER_TASK_BEGIN && task.at("task_type") < TASK_HOPPER_TASK_END) {
      create_tma_desc_by_task(task_desc);
    }
    #endif
    all_tasks.push_back(task_desc);
  }
  for (json const &e : json_task_graph["all_events"]) {
    EventType event_type = static_cast<EventType>(e.at("event_type").get<int>());
    int num_triggers = e.at("num_triggers").get<int>();
    int first_task_id = e.at("first_task_id").get<int>();
    int last_task_id = e.at("last_task_id").get<int>();
    all_events.push_back(EventDesc(event_type, num_triggers, first_task_id, last_task_id));
  }
  for (json const &t : json_task_graph["first_tasks"]) {
    first_tasks.push_back(t.get<int>());
  }
}

static void _init_persistent_kernel(std::vector<FullTaskDesc> &all_tasks,
                                    std::vector<EventDesc> &all_events,
                                  std::vector<TaskId> &first_tasks,
                                  int num_gpus,
                                  int my_gpu_id) {
  assert(num_gpus = 1);
  std::map<std::string, void*> all_tensors;
  char *input_token = (char*)(0x70d49e4e4200);
  all_tensors["input_token"] = input_token;
  char *cos_position_embedding = (char*)(0x70d40c800000);
  all_tensors["cos_position_embedding"] = cos_position_embedding;
  char *sin_position_embedding = (char*)(0x70d40d000000);
  all_tensors["sin_position_embedding"] = sin_position_embedding;
  void *embed_out;
  cudaMalloc(&embed_out, 65536);
  all_tensors["embed_out"] = embed_out;
  void *rmsnorm_out;
  cudaMalloc(&rmsnorm_out, 65536);
  all_tensors["rmsnorm_out"] = rmsnorm_out;
  void *attn_in;
  cudaMalloc(&attn_in, 98304);
  all_tensors["attn_in"] = attn_in;
  void *attn_out;
  cudaMalloc(&attn_out, 65536);
  all_tensors["attn_out"] = attn_out;
  void *attn_proj_out;
  cudaMalloc(&attn_proj_out, 65536);
  all_tensors["attn_proj_out"] = attn_proj_out;
  void *all_reduce_buf;
  cudaMalloc(&all_reduce_buf, 65536);
  all_tensors["all_reduce_buf"] = all_reduce_buf;
  void *attn_allreduce_out;
  cudaMalloc(&attn_allreduce_out, 65536);
  all_tensors["attn_allreduce_out"] = attn_allreduce_out;
  void *mlp_mid;
  cudaMalloc(&mlp_mid, 393216);
  all_tensors["mlp_mid"] = mlp_mid;
  void *silu_mul_out;
  cudaMalloc(&silu_mul_out, 196608);
  all_tensors["silu_mul_out"] = silu_mul_out;
  void *mlp_out;
  cudaMalloc(&mlp_out, 65536);
  all_tensors["mlp_out"] = mlp_out;
  void *mlp_final;
  cudaMalloc(&mlp_final, 65536);
  all_tensors["mlp_final"] = mlp_final;
  void *argmax_in;
  cudaMalloc(&argmax_in, 2457600);
  all_tensors["argmax_in"] = argmax_in;
  void *argmax_part_value;
  cudaMalloc(&argmax_part_value, 2048);
  all_tensors["argmax_part_value"] = argmax_part_value;
  void *argmax_part_index;
  cudaMalloc(&argmax_part_index, 8192);
  all_tensors["argmax_part_index"] = argmax_part_index;
  char *output_token = (char*)(0x70d49e4e4400);
  all_tensors["output_token"] = output_token;
  char *embed_tokens = (char*)(0x70d454000000);
  all_tensors["embed_tokens"] = embed_tokens;
  char *layer_0_input_layernorm = (char*)(0x70d49e404e00);
  all_tensors["layer_0_input_layernorm"] = layer_0_input_layernorm;
  char *layer_0_qkv_proj;
  cudaMalloc(&layer_0_qkv_proj, 50331648);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_0_qkv_proj + 0), 6291456, reinterpret_cast<const void *>(0x70d0e0800000), 4194304, 4194304, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_0_qkv_proj + 4194304), 6291456, reinterpret_cast<const void *>(0x70d0de000000), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_0_qkv_proj + 5242880), 6291456, reinterpret_cast<const void *>(0x70d0e2800000), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice);
  all_tensors["layer_0_qkv_proj"] = layer_0_qkv_proj;
  char *layer_0_q_norm = (char*)(0x70d49e408e00);
  all_tensors["layer_0_q_norm"] = layer_0_q_norm;
  char *layer_0_k_norm = (char*)(0x70d49e400200);
  all_tensors["layer_0_k_norm"] = layer_0_k_norm;
  char *layer_0_k_cache = (char*)(0x70d5c0000000);
  all_tensors["layer_0_k_cache"] = layer_0_k_cache;
  char *layer_0_v_cache = (char*)(0x70d4a0000000);
  all_tensors["layer_0_v_cache"] = layer_0_v_cache;
  char *layer_0_o_proj = (char*)(0x70d0de800000);
  all_tensors["layer_0_o_proj"] = layer_0_o_proj;
  char *layer_0_post_attn_layernorm = (char*)(0x70d49e406e00);
  all_tensors["layer_0_post_attn_layernorm"] = layer_0_post_attn_layernorm;
  char *layer_0_gatedup_proj;
  cudaMalloc(&layer_0_gatedup_proj, 201326592);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_0_gatedup_proj + 0), 4194304, reinterpret_cast<const void *>(0x70d0d2000000), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_0_gatedup_proj + 2097152), 4194304, reinterpret_cast<const void *>(0x70d0d8000000), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice);
  all_tensors["layer_0_gatedup_proj"] = layer_0_gatedup_proj;
  char *layer_0_down_proj = (char*)(0x70d0cc000000);
  all_tensors["layer_0_down_proj"] = layer_0_down_proj;
  char *layer_1_input_layernorm = (char*)(0x70d49e409000);
  all_tensors["layer_1_input_layernorm"] = layer_1_input_layernorm;
  char *layer_1_qkv_proj;
  cudaMalloc(&layer_1_qkv_proj, 50331648);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_1_qkv_proj + 0), 6291456, reinterpret_cast<const void *>(0x70d0f7800000), 4194304, 4194304, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_1_qkv_proj + 4194304), 6291456, reinterpret_cast<const void *>(0x70d0f5000000), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_1_qkv_proj + 5242880), 6291456, reinterpret_cast<const void *>(0x70d0f9800000), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice);
  all_tensors["layer_1_qkv_proj"] = layer_1_qkv_proj;
  char *layer_1_q_norm = (char*)(0x70d49e40d200);
  all_tensors["layer_1_q_norm"] = layer_1_q_norm;
  char *layer_1_k_norm = (char*)(0x70d49e40d000);
  all_tensors["layer_1_k_norm"] = layer_1_k_norm;
  char *layer_1_k_cache = (char*)(0x70d5c8000000);
  all_tensors["layer_1_k_cache"] = layer_1_k_cache;
  char *layer_1_v_cache = (char*)(0x70d4a8000000);
  all_tensors["layer_1_v_cache"] = layer_1_v_cache;
  char *layer_1_o_proj = (char*)(0x70d0f5800000);
  all_tensors["layer_1_o_proj"] = layer_1_o_proj;
  char *layer_1_post_attn_layernorm = (char*)(0x70d49e40b000);
  all_tensors["layer_1_post_attn_layernorm"] = layer_1_post_attn_layernorm;
  char *layer_1_gatedup_proj;
  cudaMalloc(&layer_1_gatedup_proj, 201326592);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_1_gatedup_proj + 0), 4194304, reinterpret_cast<const void *>(0x70d0e9000000), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_1_gatedup_proj + 2097152), 4194304, reinterpret_cast<const void *>(0x70d0ef000000), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice);
  all_tensors["layer_1_gatedup_proj"] = layer_1_gatedup_proj;
  char *layer_1_down_proj = (char*)(0x70d0e3000000);
  all_tensors["layer_1_down_proj"] = layer_1_down_proj;
  char *layer_2_input_layernorm = (char*)(0x70d49e40d400);
  all_tensors["layer_2_input_layernorm"] = layer_2_input_layernorm;
  char *layer_2_qkv_proj;
  cudaMalloc(&layer_2_qkv_proj, 50331648);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_2_qkv_proj + 0), 6291456, reinterpret_cast<const void *>(0x70d10e800000), 4194304, 4194304, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_2_qkv_proj + 4194304), 6291456, reinterpret_cast<const void *>(0x70d10c000000), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_2_qkv_proj + 5242880), 6291456, reinterpret_cast<const void *>(0x70d110800000), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice);
  all_tensors["layer_2_qkv_proj"] = layer_2_qkv_proj;
  char *layer_2_q_norm = (char*)(0x70d49e411600);
  all_tensors["layer_2_q_norm"] = layer_2_q_norm;
  char *layer_2_k_norm = (char*)(0x70d49e411400);
  all_tensors["layer_2_k_norm"] = layer_2_k_norm;
  char *layer_2_k_cache = (char*)(0x70d5d0000000);
  all_tensors["layer_2_k_cache"] = layer_2_k_cache;
  char *layer_2_v_cache = (char*)(0x70d4b0000000);
  all_tensors["layer_2_v_cache"] = layer_2_v_cache;
  char *layer_2_o_proj = (char*)(0x70d10c800000);
  all_tensors["layer_2_o_proj"] = layer_2_o_proj;
  char *layer_2_post_attn_layernorm = (char*)(0x70d49e40f400);
  all_tensors["layer_2_post_attn_layernorm"] = layer_2_post_attn_layernorm;
  char *layer_2_gatedup_proj;
  cudaMalloc(&layer_2_gatedup_proj, 201326592);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_2_gatedup_proj + 0), 4194304, reinterpret_cast<const void *>(0x70d100000000), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_2_gatedup_proj + 2097152), 4194304, reinterpret_cast<const void *>(0x70d106000000), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice);
  all_tensors["layer_2_gatedup_proj"] = layer_2_gatedup_proj;
  char *layer_2_down_proj = (char*)(0x70d0fa000000);
  all_tensors["layer_2_down_proj"] = layer_2_down_proj;
  char *layer_3_input_layernorm = (char*)(0x70d49e411800);
  all_tensors["layer_3_input_layernorm"] = layer_3_input_layernorm;
  char *layer_3_qkv_proj;
  cudaMalloc(&layer_3_qkv_proj, 50331648);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_3_qkv_proj + 0), 6291456, reinterpret_cast<const void *>(0x70d125800000), 4194304, 4194304, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_3_qkv_proj + 4194304), 6291456, reinterpret_cast<const void *>(0x70d123000000), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_3_qkv_proj + 5242880), 6291456, reinterpret_cast<const void *>(0x70d127800000), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice);
  all_tensors["layer_3_qkv_proj"] = layer_3_qkv_proj;
  char *layer_3_q_norm = (char*)(0x70d49e415a00);
  all_tensors["layer_3_q_norm"] = layer_3_q_norm;
  char *layer_3_k_norm = (char*)(0x70d49e415800);
  all_tensors["layer_3_k_norm"] = layer_3_k_norm;
  char *layer_3_k_cache = (char*)(0x70d5d8000000);
  all_tensors["layer_3_k_cache"] = layer_3_k_cache;
  char *layer_3_v_cache = (char*)(0x70d4b8000000);
  all_tensors["layer_3_v_cache"] = layer_3_v_cache;
  char *layer_3_o_proj = (char*)(0x70d123800000);
  all_tensors["layer_3_o_proj"] = layer_3_o_proj;
  char *layer_3_post_attn_layernorm = (char*)(0x70d49e413800);
  all_tensors["layer_3_post_attn_layernorm"] = layer_3_post_attn_layernorm;
  char *layer_3_gatedup_proj;
  cudaMalloc(&layer_3_gatedup_proj, 201326592);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_3_gatedup_proj + 0), 4194304, reinterpret_cast<const void *>(0x70d117000000), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_3_gatedup_proj + 2097152), 4194304, reinterpret_cast<const void *>(0x70d11d000000), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice);
  all_tensors["layer_3_gatedup_proj"] = layer_3_gatedup_proj;
  char *layer_3_down_proj = (char*)(0x70d111000000);
  all_tensors["layer_3_down_proj"] = layer_3_down_proj;
  char *layer_4_input_layernorm = (char*)(0x70d49e415c00);
  all_tensors["layer_4_input_layernorm"] = layer_4_input_layernorm;
  char *layer_4_qkv_proj;
  cudaMalloc(&layer_4_qkv_proj, 50331648);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_4_qkv_proj + 0), 6291456, reinterpret_cast<const void *>(0x70d13c800000), 4194304, 4194304, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_4_qkv_proj + 4194304), 6291456, reinterpret_cast<const void *>(0x70d13a000000), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_4_qkv_proj + 5242880), 6291456, reinterpret_cast<const void *>(0x70d13e800000), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice);
  all_tensors["layer_4_qkv_proj"] = layer_4_qkv_proj;
  char *layer_4_q_norm = (char*)(0x70d49e419e00);
  all_tensors["layer_4_q_norm"] = layer_4_q_norm;
  char *layer_4_k_norm = (char*)(0x70d49e419c00);
  all_tensors["layer_4_k_norm"] = layer_4_k_norm;
  char *layer_4_k_cache = (char*)(0x70d5e0000000);
  all_tensors["layer_4_k_cache"] = layer_4_k_cache;
  char *layer_4_v_cache = (char*)(0x70d4c0000000);
  all_tensors["layer_4_v_cache"] = layer_4_v_cache;
  char *layer_4_o_proj = (char*)(0x70d13a800000);
  all_tensors["layer_4_o_proj"] = layer_4_o_proj;
  char *layer_4_post_attn_layernorm = (char*)(0x70d49e417c00);
  all_tensors["layer_4_post_attn_layernorm"] = layer_4_post_attn_layernorm;
  char *layer_4_gatedup_proj;
  cudaMalloc(&layer_4_gatedup_proj, 201326592);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_4_gatedup_proj + 0), 4194304, reinterpret_cast<const void *>(0x70d12e000000), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_4_gatedup_proj + 2097152), 4194304, reinterpret_cast<const void *>(0x70d134000000), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice);
  all_tensors["layer_4_gatedup_proj"] = layer_4_gatedup_proj;
  char *layer_4_down_proj = (char*)(0x70d128000000);
  all_tensors["layer_4_down_proj"] = layer_4_down_proj;
  char *layer_5_input_layernorm = (char*)(0x70d49e41a000);
  all_tensors["layer_5_input_layernorm"] = layer_5_input_layernorm;
  char *layer_5_qkv_proj;
  cudaMalloc(&layer_5_qkv_proj, 50331648);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_5_qkv_proj + 0), 6291456, reinterpret_cast<const void *>(0x70d153800000), 4194304, 4194304, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_5_qkv_proj + 4194304), 6291456, reinterpret_cast<const void *>(0x70d151000000), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_5_qkv_proj + 5242880), 6291456, reinterpret_cast<const void *>(0x70d155800000), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice);
  all_tensors["layer_5_qkv_proj"] = layer_5_qkv_proj;
  char *layer_5_q_norm = (char*)(0x70d49e41e200);
  all_tensors["layer_5_q_norm"] = layer_5_q_norm;
  char *layer_5_k_norm = (char*)(0x70d49e41e000);
  all_tensors["layer_5_k_norm"] = layer_5_k_norm;
  char *layer_5_k_cache = (char*)(0x70d5e8000000);
  all_tensors["layer_5_k_cache"] = layer_5_k_cache;
  char *layer_5_v_cache = (char*)(0x70d4c8000000);
  all_tensors["layer_5_v_cache"] = layer_5_v_cache;
  char *layer_5_o_proj = (char*)(0x70d151800000);
  all_tensors["layer_5_o_proj"] = layer_5_o_proj;
  char *layer_5_post_attn_layernorm = (char*)(0x70d49e41c000);
  all_tensors["layer_5_post_attn_layernorm"] = layer_5_post_attn_layernorm;
  char *layer_5_gatedup_proj;
  cudaMalloc(&layer_5_gatedup_proj, 201326592);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_5_gatedup_proj + 0), 4194304, reinterpret_cast<const void *>(0x70d145000000), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_5_gatedup_proj + 2097152), 4194304, reinterpret_cast<const void *>(0x70d14b000000), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice);
  all_tensors["layer_5_gatedup_proj"] = layer_5_gatedup_proj;
  char *layer_5_down_proj = (char*)(0x70d13f000000);
  all_tensors["layer_5_down_proj"] = layer_5_down_proj;
  char *layer_6_input_layernorm = (char*)(0x70d49e41e400);
  all_tensors["layer_6_input_layernorm"] = layer_6_input_layernorm;
  char *layer_6_qkv_proj;
  cudaMalloc(&layer_6_qkv_proj, 50331648);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_6_qkv_proj + 0), 6291456, reinterpret_cast<const void *>(0x70d16a800000), 4194304, 4194304, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_6_qkv_proj + 4194304), 6291456, reinterpret_cast<const void *>(0x70d168000000), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_6_qkv_proj + 5242880), 6291456, reinterpret_cast<const void *>(0x70d16c800000), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice);
  all_tensors["layer_6_qkv_proj"] = layer_6_qkv_proj;
  char *layer_6_q_norm = (char*)(0x70d49e422600);
  all_tensors["layer_6_q_norm"] = layer_6_q_norm;
  char *layer_6_k_norm = (char*)(0x70d49e422400);
  all_tensors["layer_6_k_norm"] = layer_6_k_norm;
  char *layer_6_k_cache = (char*)(0x70d5f0000000);
  all_tensors["layer_6_k_cache"] = layer_6_k_cache;
  char *layer_6_v_cache = (char*)(0x70d4d0000000);
  all_tensors["layer_6_v_cache"] = layer_6_v_cache;
  char *layer_6_o_proj = (char*)(0x70d168800000);
  all_tensors["layer_6_o_proj"] = layer_6_o_proj;
  char *layer_6_post_attn_layernorm = (char*)(0x70d49e420400);
  all_tensors["layer_6_post_attn_layernorm"] = layer_6_post_attn_layernorm;
  char *layer_6_gatedup_proj;
  cudaMalloc(&layer_6_gatedup_proj, 201326592);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_6_gatedup_proj + 0), 4194304, reinterpret_cast<const void *>(0x70d15c000000), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_6_gatedup_proj + 2097152), 4194304, reinterpret_cast<const void *>(0x70d162000000), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice);
  all_tensors["layer_6_gatedup_proj"] = layer_6_gatedup_proj;
  char *layer_6_down_proj = (char*)(0x70d156000000);
  all_tensors["layer_6_down_proj"] = layer_6_down_proj;
  char *layer_7_input_layernorm = (char*)(0x70d49e440800);
  all_tensors["layer_7_input_layernorm"] = layer_7_input_layernorm;
  char *layer_7_qkv_proj;
  cudaMalloc(&layer_7_qkv_proj, 50331648);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_7_qkv_proj + 0), 6291456, reinterpret_cast<const void *>(0x70d16d800000), 4194304, 4194304, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_7_qkv_proj + 4194304), 6291456, reinterpret_cast<const void *>(0x70d16d000000), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_7_qkv_proj + 5242880), 6291456, reinterpret_cast<const void *>(0x70d16f800000), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice);
  all_tensors["layer_7_qkv_proj"] = layer_7_qkv_proj;
  char *layer_7_q_norm = (char*)(0x70d49e444a00);
  all_tensors["layer_7_q_norm"] = layer_7_q_norm;
  char *layer_7_k_norm = (char*)(0x70d49e444800);
  all_tensors["layer_7_k_norm"] = layer_7_k_norm;
  char *layer_7_k_cache = (char*)(0x70d5f8000000);
  all_tensors["layer_7_k_cache"] = layer_7_k_cache;
  char *layer_7_v_cache = (char*)(0x70d4d8000000);
  all_tensors["layer_7_v_cache"] = layer_7_v_cache;
  char *layer_7_o_proj = (char*)(0x70d22e000000);
  all_tensors["layer_7_o_proj"] = layer_7_o_proj;
  char *layer_7_post_attn_layernorm = (char*)(0x70d49e442800);
  all_tensors["layer_7_post_attn_layernorm"] = layer_7_post_attn_layernorm;
  char *layer_7_gatedup_proj;
  cudaMalloc(&layer_7_gatedup_proj, 201326592);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_7_gatedup_proj + 0), 4194304, reinterpret_cast<const void *>(0x70d222000000), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_7_gatedup_proj + 2097152), 4194304, reinterpret_cast<const void *>(0x70d228000000), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice);
  all_tensors["layer_7_gatedup_proj"] = layer_7_gatedup_proj;
  char *layer_7_down_proj = (char*)(0x70d21c000000);
  all_tensors["layer_7_down_proj"] = layer_7_down_proj;
  char *layer_8_input_layernorm = (char*)(0x70d49e444c00);
  all_tensors["layer_8_input_layernorm"] = layer_8_input_layernorm;
  char *layer_8_qkv_proj;
  cudaMalloc(&layer_8_qkv_proj, 50331648);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_8_qkv_proj + 0), 6291456, reinterpret_cast<const void *>(0x70d244800000), 4194304, 4194304, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_8_qkv_proj + 4194304), 6291456, reinterpret_cast<const void *>(0x70d242000000), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_8_qkv_proj + 5242880), 6291456, reinterpret_cast<const void *>(0x70d246800000), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice);
  all_tensors["layer_8_qkv_proj"] = layer_8_qkv_proj;
  char *layer_8_q_norm = (char*)(0x70d49e448e00);
  all_tensors["layer_8_q_norm"] = layer_8_q_norm;
  char *layer_8_k_norm = (char*)(0x70d49e448c00);
  all_tensors["layer_8_k_norm"] = layer_8_k_norm;
  char *layer_8_k_cache = (char*)(0x70d600000000);
  all_tensors["layer_8_k_cache"] = layer_8_k_cache;
  char *layer_8_v_cache = (char*)(0x70d4e0000000);
  all_tensors["layer_8_v_cache"] = layer_8_v_cache;
  char *layer_8_o_proj = (char*)(0x70d242800000);
  all_tensors["layer_8_o_proj"] = layer_8_o_proj;
  char *layer_8_post_attn_layernorm = (char*)(0x70d49e446c00);
  all_tensors["layer_8_post_attn_layernorm"] = layer_8_post_attn_layernorm;
  char *layer_8_gatedup_proj;
  cudaMalloc(&layer_8_gatedup_proj, 201326592);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_8_gatedup_proj + 0), 4194304, reinterpret_cast<const void *>(0x70d236000000), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_8_gatedup_proj + 2097152), 4194304, reinterpret_cast<const void *>(0x70d23c000000), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice);
  all_tensors["layer_8_gatedup_proj"] = layer_8_gatedup_proj;
  char *layer_8_down_proj = (char*)(0x70d230000000);
  all_tensors["layer_8_down_proj"] = layer_8_down_proj;
  char *layer_9_input_layernorm = (char*)(0x70d49e449000);
  all_tensors["layer_9_input_layernorm"] = layer_9_input_layernorm;
  char *layer_9_qkv_proj;
  cudaMalloc(&layer_9_qkv_proj, 50331648);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_9_qkv_proj + 0), 6291456, reinterpret_cast<const void *>(0x70d25b800000), 4194304, 4194304, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_9_qkv_proj + 4194304), 6291456, reinterpret_cast<const void *>(0x70d259000000), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_9_qkv_proj + 5242880), 6291456, reinterpret_cast<const void *>(0x70d25d800000), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice);
  all_tensors["layer_9_qkv_proj"] = layer_9_qkv_proj;
  char *layer_9_q_norm = (char*)(0x70d49e44d200);
  all_tensors["layer_9_q_norm"] = layer_9_q_norm;
  char *layer_9_k_norm = (char*)(0x70d49e44d000);
  all_tensors["layer_9_k_norm"] = layer_9_k_norm;
  char *layer_9_k_cache = (char*)(0x70d608000000);
  all_tensors["layer_9_k_cache"] = layer_9_k_cache;
  char *layer_9_v_cache = (char*)(0x70d4e8000000);
  all_tensors["layer_9_v_cache"] = layer_9_v_cache;
  char *layer_9_o_proj = (char*)(0x70d259800000);
  all_tensors["layer_9_o_proj"] = layer_9_o_proj;
  char *layer_9_post_attn_layernorm = (char*)(0x70d49e44b000);
  all_tensors["layer_9_post_attn_layernorm"] = layer_9_post_attn_layernorm;
  char *layer_9_gatedup_proj;
  cudaMalloc(&layer_9_gatedup_proj, 201326592);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_9_gatedup_proj + 0), 4194304, reinterpret_cast<const void *>(0x70d24d000000), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_9_gatedup_proj + 2097152), 4194304, reinterpret_cast<const void *>(0x70d253000000), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice);
  all_tensors["layer_9_gatedup_proj"] = layer_9_gatedup_proj;
  char *layer_9_down_proj = (char*)(0x70d247000000);
  all_tensors["layer_9_down_proj"] = layer_9_down_proj;
  char *layer_10_input_layernorm = (char*)(0x70d49e422800);
  all_tensors["layer_10_input_layernorm"] = layer_10_input_layernorm;
  char *layer_10_qkv_proj;
  cudaMalloc(&layer_10_qkv_proj, 50331648);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_10_qkv_proj + 0), 6291456, reinterpret_cast<const void *>(0x70d184800000), 4194304, 4194304, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_10_qkv_proj + 4194304), 6291456, reinterpret_cast<const void *>(0x70d182000000), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_10_qkv_proj + 5242880), 6291456, reinterpret_cast<const void *>(0x70d186800000), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice);
  all_tensors["layer_10_qkv_proj"] = layer_10_qkv_proj;
  char *layer_10_q_norm = (char*)(0x70d49e426a00);
  all_tensors["layer_10_q_norm"] = layer_10_q_norm;
  char *layer_10_k_norm = (char*)(0x70d49e426800);
  all_tensors["layer_10_k_norm"] = layer_10_k_norm;
  char *layer_10_k_cache = (char*)(0x70d610000000);
  all_tensors["layer_10_k_cache"] = layer_10_k_cache;
  char *layer_10_v_cache = (char*)(0x70d4f0000000);
  all_tensors["layer_10_v_cache"] = layer_10_v_cache;
  char *layer_10_o_proj = (char*)(0x70d182800000);
  all_tensors["layer_10_o_proj"] = layer_10_o_proj;
  char *layer_10_post_attn_layernorm = (char*)(0x70d49e424800);
  all_tensors["layer_10_post_attn_layernorm"] = layer_10_post_attn_layernorm;
  char *layer_10_gatedup_proj;
  cudaMalloc(&layer_10_gatedup_proj, 201326592);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_10_gatedup_proj + 0), 4194304, reinterpret_cast<const void *>(0x70d176000000), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_10_gatedup_proj + 2097152), 4194304, reinterpret_cast<const void *>(0x70d17c000000), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice);
  all_tensors["layer_10_gatedup_proj"] = layer_10_gatedup_proj;
  char *layer_10_down_proj = (char*)(0x70d170000000);
  all_tensors["layer_10_down_proj"] = layer_10_down_proj;
  char *layer_11_input_layernorm = (char*)(0x70d49e426c00);
  all_tensors["layer_11_input_layernorm"] = layer_11_input_layernorm;
  char *layer_11_qkv_proj;
  cudaMalloc(&layer_11_qkv_proj, 50331648);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_11_qkv_proj + 0), 6291456, reinterpret_cast<const void *>(0x70d19b800000), 4194304, 4194304, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_11_qkv_proj + 4194304), 6291456, reinterpret_cast<const void *>(0x70d199000000), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_11_qkv_proj + 5242880), 6291456, reinterpret_cast<const void *>(0x70d19d800000), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice);
  all_tensors["layer_11_qkv_proj"] = layer_11_qkv_proj;
  char *layer_11_q_norm = (char*)(0x70d49e42ae00);
  all_tensors["layer_11_q_norm"] = layer_11_q_norm;
  char *layer_11_k_norm = (char*)(0x70d49e42ac00);
  all_tensors["layer_11_k_norm"] = layer_11_k_norm;
  char *layer_11_k_cache = (char*)(0x70d618000000);
  all_tensors["layer_11_k_cache"] = layer_11_k_cache;
  char *layer_11_v_cache = (char*)(0x70d4f8000000);
  all_tensors["layer_11_v_cache"] = layer_11_v_cache;
  char *layer_11_o_proj = (char*)(0x70d199800000);
  all_tensors["layer_11_o_proj"] = layer_11_o_proj;
  char *layer_11_post_attn_layernorm = (char*)(0x70d49e428c00);
  all_tensors["layer_11_post_attn_layernorm"] = layer_11_post_attn_layernorm;
  char *layer_11_gatedup_proj;
  cudaMalloc(&layer_11_gatedup_proj, 201326592);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_11_gatedup_proj + 0), 4194304, reinterpret_cast<const void *>(0x70d18d000000), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_11_gatedup_proj + 2097152), 4194304, reinterpret_cast<const void *>(0x70d193000000), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice);
  all_tensors["layer_11_gatedup_proj"] = layer_11_gatedup_proj;
  char *layer_11_down_proj = (char*)(0x70d187000000);
  all_tensors["layer_11_down_proj"] = layer_11_down_proj;
  char *layer_12_input_layernorm = (char*)(0x70d49e42b000);
  all_tensors["layer_12_input_layernorm"] = layer_12_input_layernorm;
  char *layer_12_qkv_proj;
  cudaMalloc(&layer_12_qkv_proj, 50331648);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_12_qkv_proj + 0), 6291456, reinterpret_cast<const void *>(0x70d1b2800000), 4194304, 4194304, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_12_qkv_proj + 4194304), 6291456, reinterpret_cast<const void *>(0x70d1b0000000), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_12_qkv_proj + 5242880), 6291456, reinterpret_cast<const void *>(0x70d1b4800000), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice);
  all_tensors["layer_12_qkv_proj"] = layer_12_qkv_proj;
  char *layer_12_q_norm = (char*)(0x70d49e42f200);
  all_tensors["layer_12_q_norm"] = layer_12_q_norm;
  char *layer_12_k_norm = (char*)(0x70d49e42f000);
  all_tensors["layer_12_k_norm"] = layer_12_k_norm;
  char *layer_12_k_cache = (char*)(0x70d620000000);
  all_tensors["layer_12_k_cache"] = layer_12_k_cache;
  char *layer_12_v_cache = (char*)(0x70d500000000);
  all_tensors["layer_12_v_cache"] = layer_12_v_cache;
  char *layer_12_o_proj = (char*)(0x70d1b0800000);
  all_tensors["layer_12_o_proj"] = layer_12_o_proj;
  char *layer_12_post_attn_layernorm = (char*)(0x70d49e42d000);
  all_tensors["layer_12_post_attn_layernorm"] = layer_12_post_attn_layernorm;
  char *layer_12_gatedup_proj;
  cudaMalloc(&layer_12_gatedup_proj, 201326592);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_12_gatedup_proj + 0), 4194304, reinterpret_cast<const void *>(0x70d1a4000000), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_12_gatedup_proj + 2097152), 4194304, reinterpret_cast<const void *>(0x70d1aa000000), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice);
  all_tensors["layer_12_gatedup_proj"] = layer_12_gatedup_proj;
  char *layer_12_down_proj = (char*)(0x70d19e000000);
  all_tensors["layer_12_down_proj"] = layer_12_down_proj;
  char *layer_13_input_layernorm = (char*)(0x70d49e42f400);
  all_tensors["layer_13_input_layernorm"] = layer_13_input_layernorm;
  char *layer_13_qkv_proj;
  cudaMalloc(&layer_13_qkv_proj, 50331648);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_13_qkv_proj + 0), 6291456, reinterpret_cast<const void *>(0x70d1c9800000), 4194304, 4194304, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_13_qkv_proj + 4194304), 6291456, reinterpret_cast<const void *>(0x70d1c7000000), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_13_qkv_proj + 5242880), 6291456, reinterpret_cast<const void *>(0x70d1cb800000), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice);
  all_tensors["layer_13_qkv_proj"] = layer_13_qkv_proj;
  char *layer_13_q_norm = (char*)(0x70d49e433600);
  all_tensors["layer_13_q_norm"] = layer_13_q_norm;
  char *layer_13_k_norm = (char*)(0x70d49e433400);
  all_tensors["layer_13_k_norm"] = layer_13_k_norm;
  char *layer_13_k_cache = (char*)(0x70d628000000);
  all_tensors["layer_13_k_cache"] = layer_13_k_cache;
  char *layer_13_v_cache = (char*)(0x70d508000000);
  all_tensors["layer_13_v_cache"] = layer_13_v_cache;
  char *layer_13_o_proj = (char*)(0x70d1c7800000);
  all_tensors["layer_13_o_proj"] = layer_13_o_proj;
  char *layer_13_post_attn_layernorm = (char*)(0x70d49e431400);
  all_tensors["layer_13_post_attn_layernorm"] = layer_13_post_attn_layernorm;
  char *layer_13_gatedup_proj;
  cudaMalloc(&layer_13_gatedup_proj, 201326592);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_13_gatedup_proj + 0), 4194304, reinterpret_cast<const void *>(0x70d1bb000000), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_13_gatedup_proj + 2097152), 4194304, reinterpret_cast<const void *>(0x70d1c1000000), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice);
  all_tensors["layer_13_gatedup_proj"] = layer_13_gatedup_proj;
  char *layer_13_down_proj = (char*)(0x70d1b5000000);
  all_tensors["layer_13_down_proj"] = layer_13_down_proj;
  char *layer_14_input_layernorm = (char*)(0x70d49e433800);
  all_tensors["layer_14_input_layernorm"] = layer_14_input_layernorm;
  char *layer_14_qkv_proj;
  cudaMalloc(&layer_14_qkv_proj, 50331648);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_14_qkv_proj + 0), 6291456, reinterpret_cast<const void *>(0x70d1e0800000), 4194304, 4194304, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_14_qkv_proj + 4194304), 6291456, reinterpret_cast<const void *>(0x70d1de000000), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_14_qkv_proj + 5242880), 6291456, reinterpret_cast<const void *>(0x70d1e2800000), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice);
  all_tensors["layer_14_qkv_proj"] = layer_14_qkv_proj;
  char *layer_14_q_norm = (char*)(0x70d49e437a00);
  all_tensors["layer_14_q_norm"] = layer_14_q_norm;
  char *layer_14_k_norm = (char*)(0x70d49e437800);
  all_tensors["layer_14_k_norm"] = layer_14_k_norm;
  char *layer_14_k_cache = (char*)(0x70d630000000);
  all_tensors["layer_14_k_cache"] = layer_14_k_cache;
  char *layer_14_v_cache = (char*)(0x70d510000000);
  all_tensors["layer_14_v_cache"] = layer_14_v_cache;
  char *layer_14_o_proj = (char*)(0x70d1de800000);
  all_tensors["layer_14_o_proj"] = layer_14_o_proj;
  char *layer_14_post_attn_layernorm = (char*)(0x70d49e435800);
  all_tensors["layer_14_post_attn_layernorm"] = layer_14_post_attn_layernorm;
  char *layer_14_gatedup_proj;
  cudaMalloc(&layer_14_gatedup_proj, 201326592);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_14_gatedup_proj + 0), 4194304, reinterpret_cast<const void *>(0x70d1d2000000), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_14_gatedup_proj + 2097152), 4194304, reinterpret_cast<const void *>(0x70d1d8000000), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice);
  all_tensors["layer_14_gatedup_proj"] = layer_14_gatedup_proj;
  char *layer_14_down_proj = (char*)(0x70d1cc000000);
  all_tensors["layer_14_down_proj"] = layer_14_down_proj;
  char *layer_15_input_layernorm = (char*)(0x70d49e437c00);
  all_tensors["layer_15_input_layernorm"] = layer_15_input_layernorm;
  char *layer_15_qkv_proj;
  cudaMalloc(&layer_15_qkv_proj, 50331648);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_15_qkv_proj + 0), 6291456, reinterpret_cast<const void *>(0x70d1f7800000), 4194304, 4194304, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_15_qkv_proj + 4194304), 6291456, reinterpret_cast<const void *>(0x70d1f5000000), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_15_qkv_proj + 5242880), 6291456, reinterpret_cast<const void *>(0x70d1f9800000), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice);
  all_tensors["layer_15_qkv_proj"] = layer_15_qkv_proj;
  char *layer_15_q_norm = (char*)(0x70d49e43be00);
  all_tensors["layer_15_q_norm"] = layer_15_q_norm;
  char *layer_15_k_norm = (char*)(0x70d49e43bc00);
  all_tensors["layer_15_k_norm"] = layer_15_k_norm;
  char *layer_15_k_cache = (char*)(0x70d638000000);
  all_tensors["layer_15_k_cache"] = layer_15_k_cache;
  char *layer_15_v_cache = (char*)(0x70d518000000);
  all_tensors["layer_15_v_cache"] = layer_15_v_cache;
  char *layer_15_o_proj = (char*)(0x70d1f5800000);
  all_tensors["layer_15_o_proj"] = layer_15_o_proj;
  char *layer_15_post_attn_layernorm = (char*)(0x70d49e439c00);
  all_tensors["layer_15_post_attn_layernorm"] = layer_15_post_attn_layernorm;
  char *layer_15_gatedup_proj;
  cudaMalloc(&layer_15_gatedup_proj, 201326592);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_15_gatedup_proj + 0), 4194304, reinterpret_cast<const void *>(0x70d1e9000000), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_15_gatedup_proj + 2097152), 4194304, reinterpret_cast<const void *>(0x70d1ef000000), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice);
  all_tensors["layer_15_gatedup_proj"] = layer_15_gatedup_proj;
  char *layer_15_down_proj = (char*)(0x70d1e3000000);
  all_tensors["layer_15_down_proj"] = layer_15_down_proj;
  char *layer_16_input_layernorm = (char*)(0x70d49e43c000);
  all_tensors["layer_16_input_layernorm"] = layer_16_input_layernorm;
  char *layer_16_qkv_proj;
  cudaMalloc(&layer_16_qkv_proj, 50331648);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_16_qkv_proj + 0), 6291456, reinterpret_cast<const void *>(0x70d20e800000), 4194304, 4194304, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_16_qkv_proj + 4194304), 6291456, reinterpret_cast<const void *>(0x70d20c000000), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_16_qkv_proj + 5242880), 6291456, reinterpret_cast<const void *>(0x70d210800000), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice);
  all_tensors["layer_16_qkv_proj"] = layer_16_qkv_proj;
  char *layer_16_q_norm = (char*)(0x70d49e440200);
  all_tensors["layer_16_q_norm"] = layer_16_q_norm;
  char *layer_16_k_norm = (char*)(0x70d49e440000);
  all_tensors["layer_16_k_norm"] = layer_16_k_norm;
  char *layer_16_k_cache = (char*)(0x70d640000000);
  all_tensors["layer_16_k_cache"] = layer_16_k_cache;
  char *layer_16_v_cache = (char*)(0x70d520000000);
  all_tensors["layer_16_v_cache"] = layer_16_v_cache;
  char *layer_16_o_proj = (char*)(0x70d20c800000);
  all_tensors["layer_16_o_proj"] = layer_16_o_proj;
  char *layer_16_post_attn_layernorm = (char*)(0x70d49e43e000);
  all_tensors["layer_16_post_attn_layernorm"] = layer_16_post_attn_layernorm;
  char *layer_16_gatedup_proj;
  cudaMalloc(&layer_16_gatedup_proj, 201326592);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_16_gatedup_proj + 0), 4194304, reinterpret_cast<const void *>(0x70d200000000), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_16_gatedup_proj + 2097152), 4194304, reinterpret_cast<const void *>(0x70d206000000), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice);
  all_tensors["layer_16_gatedup_proj"] = layer_16_gatedup_proj;
  char *layer_16_down_proj = (char*)(0x70d1fa000000);
  all_tensors["layer_16_down_proj"] = layer_16_down_proj;
  char *layer_17_input_layernorm = (char*)(0x70d49e44d400);
  all_tensors["layer_17_input_layernorm"] = layer_17_input_layernorm;
  char *layer_17_qkv_proj;
  cudaMalloc(&layer_17_qkv_proj, 50331648);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_17_qkv_proj + 0), 6291456, reinterpret_cast<const void *>(0x70d219800000), 4194304, 4194304, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_17_qkv_proj + 4194304), 6291456, reinterpret_cast<const void *>(0x70d217000000), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_17_qkv_proj + 5242880), 6291456, reinterpret_cast<const void *>(0x70d21b800000), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice);
  all_tensors["layer_17_qkv_proj"] = layer_17_qkv_proj;
  char *layer_17_q_norm = (char*)(0x70d49e440600);
  all_tensors["layer_17_q_norm"] = layer_17_q_norm;
  char *layer_17_k_norm = (char*)(0x70d49e440400);
  all_tensors["layer_17_k_norm"] = layer_17_k_norm;
  char *layer_17_k_cache = (char*)(0x70d648000000);
  all_tensors["layer_17_k_cache"] = layer_17_k_cache;
  char *layer_17_v_cache = (char*)(0x70d528000000);
  all_tensors["layer_17_v_cache"] = layer_17_v_cache;
  char *layer_17_o_proj = (char*)(0x70d217800000);
  all_tensors["layer_17_o_proj"] = layer_17_o_proj;
  char *layer_17_post_attn_layernorm = (char*)(0x70d49e44f400);
  all_tensors["layer_17_post_attn_layernorm"] = layer_17_post_attn_layernorm;
  char *layer_17_gatedup_proj;
  cudaMalloc(&layer_17_gatedup_proj, 201326592);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_17_gatedup_proj + 0), 4194304, reinterpret_cast<const void *>(0x70d211000000), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_17_gatedup_proj + 2097152), 4194304, reinterpret_cast<const void *>(0x70d264000000), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice);
  all_tensors["layer_17_gatedup_proj"] = layer_17_gatedup_proj;
  char *layer_17_down_proj = (char*)(0x70d25e000000);
  all_tensors["layer_17_down_proj"] = layer_17_down_proj;
  char *layer_18_input_layernorm = (char*)(0x70d49e451400);
  all_tensors["layer_18_input_layernorm"] = layer_18_input_layernorm;
  char *layer_18_qkv_proj;
  cudaMalloc(&layer_18_qkv_proj, 50331648);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_18_qkv_proj + 0), 6291456, reinterpret_cast<const void *>(0x70d27e800000), 4194304, 4194304, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_18_qkv_proj + 4194304), 6291456, reinterpret_cast<const void *>(0x70d27c000000), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_18_qkv_proj + 5242880), 6291456, reinterpret_cast<const void *>(0x70d280800000), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice);
  all_tensors["layer_18_qkv_proj"] = layer_18_qkv_proj;
  char *layer_18_q_norm = (char*)(0x70d49e455600);
  all_tensors["layer_18_q_norm"] = layer_18_q_norm;
  char *layer_18_k_norm = (char*)(0x70d49e455400);
  all_tensors["layer_18_k_norm"] = layer_18_k_norm;
  char *layer_18_k_cache = (char*)(0x70d650000000);
  all_tensors["layer_18_k_cache"] = layer_18_k_cache;
  char *layer_18_v_cache = (char*)(0x70d530000000);
  all_tensors["layer_18_v_cache"] = layer_18_v_cache;
  char *layer_18_o_proj = (char*)(0x70d27c800000);
  all_tensors["layer_18_o_proj"] = layer_18_o_proj;
  char *layer_18_post_attn_layernorm = (char*)(0x70d49e453400);
  all_tensors["layer_18_post_attn_layernorm"] = layer_18_post_attn_layernorm;
  char *layer_18_gatedup_proj;
  cudaMalloc(&layer_18_gatedup_proj, 201326592);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_18_gatedup_proj + 0), 4194304, reinterpret_cast<const void *>(0x70d270000000), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_18_gatedup_proj + 2097152), 4194304, reinterpret_cast<const void *>(0x70d276000000), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice);
  all_tensors["layer_18_gatedup_proj"] = layer_18_gatedup_proj;
  char *layer_18_down_proj = (char*)(0x70d26a000000);
  all_tensors["layer_18_down_proj"] = layer_18_down_proj;
  char *layer_19_input_layernorm = (char*)(0x70d49e455800);
  all_tensors["layer_19_input_layernorm"] = layer_19_input_layernorm;
  char *layer_19_qkv_proj;
  cudaMalloc(&layer_19_qkv_proj, 50331648);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_19_qkv_proj + 0), 6291456, reinterpret_cast<const void *>(0x70d295800000), 4194304, 4194304, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_19_qkv_proj + 4194304), 6291456, reinterpret_cast<const void *>(0x70d293000000), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_19_qkv_proj + 5242880), 6291456, reinterpret_cast<const void *>(0x70d297800000), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice);
  all_tensors["layer_19_qkv_proj"] = layer_19_qkv_proj;
  char *layer_19_q_norm = (char*)(0x70d49e459a00);
  all_tensors["layer_19_q_norm"] = layer_19_q_norm;
  char *layer_19_k_norm = (char*)(0x70d49e459800);
  all_tensors["layer_19_k_norm"] = layer_19_k_norm;
  char *layer_19_k_cache = (char*)(0x70d658000000);
  all_tensors["layer_19_k_cache"] = layer_19_k_cache;
  char *layer_19_v_cache = (char*)(0x70d538000000);
  all_tensors["layer_19_v_cache"] = layer_19_v_cache;
  char *layer_19_o_proj = (char*)(0x70d293800000);
  all_tensors["layer_19_o_proj"] = layer_19_o_proj;
  char *layer_19_post_attn_layernorm = (char*)(0x70d49e457800);
  all_tensors["layer_19_post_attn_layernorm"] = layer_19_post_attn_layernorm;
  char *layer_19_gatedup_proj;
  cudaMalloc(&layer_19_gatedup_proj, 201326592);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_19_gatedup_proj + 0), 4194304, reinterpret_cast<const void *>(0x70d287000000), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_19_gatedup_proj + 2097152), 4194304, reinterpret_cast<const void *>(0x70d28d000000), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice);
  all_tensors["layer_19_gatedup_proj"] = layer_19_gatedup_proj;
  char *layer_19_down_proj = (char*)(0x70d281000000);
  all_tensors["layer_19_down_proj"] = layer_19_down_proj;
  char *layer_20_input_layernorm = (char*)(0x70d49e459c00);
  all_tensors["layer_20_input_layernorm"] = layer_20_input_layernorm;
  char *layer_20_qkv_proj;
  cudaMalloc(&layer_20_qkv_proj, 50331648);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_20_qkv_proj + 0), 6291456, reinterpret_cast<const void *>(0x70d2ac800000), 4194304, 4194304, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_20_qkv_proj + 4194304), 6291456, reinterpret_cast<const void *>(0x70d2aa000000), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_20_qkv_proj + 5242880), 6291456, reinterpret_cast<const void *>(0x70d2ae800000), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice);
  all_tensors["layer_20_qkv_proj"] = layer_20_qkv_proj;
  char *layer_20_q_norm = (char*)(0x70d49e45de00);
  all_tensors["layer_20_q_norm"] = layer_20_q_norm;
  char *layer_20_k_norm = (char*)(0x70d49e45dc00);
  all_tensors["layer_20_k_norm"] = layer_20_k_norm;
  char *layer_20_k_cache = (char*)(0x70d660000000);
  all_tensors["layer_20_k_cache"] = layer_20_k_cache;
  char *layer_20_v_cache = (char*)(0x70d540000000);
  all_tensors["layer_20_v_cache"] = layer_20_v_cache;
  char *layer_20_o_proj = (char*)(0x70d2aa800000);
  all_tensors["layer_20_o_proj"] = layer_20_o_proj;
  char *layer_20_post_attn_layernorm = (char*)(0x70d49e45bc00);
  all_tensors["layer_20_post_attn_layernorm"] = layer_20_post_attn_layernorm;
  char *layer_20_gatedup_proj;
  cudaMalloc(&layer_20_gatedup_proj, 201326592);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_20_gatedup_proj + 0), 4194304, reinterpret_cast<const void *>(0x70d29e000000), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_20_gatedup_proj + 2097152), 4194304, reinterpret_cast<const void *>(0x70d2a4000000), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice);
  all_tensors["layer_20_gatedup_proj"] = layer_20_gatedup_proj;
  char *layer_20_down_proj = (char*)(0x70d298000000);
  all_tensors["layer_20_down_proj"] = layer_20_down_proj;
  char *layer_21_input_layernorm = (char*)(0x70d49e45e000);
  all_tensors["layer_21_input_layernorm"] = layer_21_input_layernorm;
  char *layer_21_qkv_proj;
  cudaMalloc(&layer_21_qkv_proj, 50331648);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_21_qkv_proj + 0), 6291456, reinterpret_cast<const void *>(0x70d2c3800000), 4194304, 4194304, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_21_qkv_proj + 4194304), 6291456, reinterpret_cast<const void *>(0x70d2c1000000), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_21_qkv_proj + 5242880), 6291456, reinterpret_cast<const void *>(0x70d2c5800000), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice);
  all_tensors["layer_21_qkv_proj"] = layer_21_qkv_proj;
  char *layer_21_q_norm = (char*)(0x70d49e462200);
  all_tensors["layer_21_q_norm"] = layer_21_q_norm;
  char *layer_21_k_norm = (char*)(0x70d49e462000);
  all_tensors["layer_21_k_norm"] = layer_21_k_norm;
  char *layer_21_k_cache = (char*)(0x70d668000000);
  all_tensors["layer_21_k_cache"] = layer_21_k_cache;
  char *layer_21_v_cache = (char*)(0x70d548000000);
  all_tensors["layer_21_v_cache"] = layer_21_v_cache;
  char *layer_21_o_proj = (char*)(0x70d2c1800000);
  all_tensors["layer_21_o_proj"] = layer_21_o_proj;
  char *layer_21_post_attn_layernorm = (char*)(0x70d49e460000);
  all_tensors["layer_21_post_attn_layernorm"] = layer_21_post_attn_layernorm;
  char *layer_21_gatedup_proj;
  cudaMalloc(&layer_21_gatedup_proj, 201326592);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_21_gatedup_proj + 0), 4194304, reinterpret_cast<const void *>(0x70d2b5000000), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_21_gatedup_proj + 2097152), 4194304, reinterpret_cast<const void *>(0x70d2bb000000), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice);
  all_tensors["layer_21_gatedup_proj"] = layer_21_gatedup_proj;
  char *layer_21_down_proj = (char*)(0x70d2af000000);
  all_tensors["layer_21_down_proj"] = layer_21_down_proj;
  char *layer_22_input_layernorm = (char*)(0x70d49e462400);
  all_tensors["layer_22_input_layernorm"] = layer_22_input_layernorm;
  char *layer_22_qkv_proj;
  cudaMalloc(&layer_22_qkv_proj, 50331648);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_22_qkv_proj + 0), 6291456, reinterpret_cast<const void *>(0x70d2da800000), 4194304, 4194304, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_22_qkv_proj + 4194304), 6291456, reinterpret_cast<const void *>(0x70d2d8000000), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_22_qkv_proj + 5242880), 6291456, reinterpret_cast<const void *>(0x70d2dc800000), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice);
  all_tensors["layer_22_qkv_proj"] = layer_22_qkv_proj;
  char *layer_22_q_norm = (char*)(0x70d49e466600);
  all_tensors["layer_22_q_norm"] = layer_22_q_norm;
  char *layer_22_k_norm = (char*)(0x70d49e466400);
  all_tensors["layer_22_k_norm"] = layer_22_k_norm;
  char *layer_22_k_cache = (char*)(0x70d670000000);
  all_tensors["layer_22_k_cache"] = layer_22_k_cache;
  char *layer_22_v_cache = (char*)(0x70d550000000);
  all_tensors["layer_22_v_cache"] = layer_22_v_cache;
  char *layer_22_o_proj = (char*)(0x70d2d8800000);
  all_tensors["layer_22_o_proj"] = layer_22_o_proj;
  char *layer_22_post_attn_layernorm = (char*)(0x70d49e464400);
  all_tensors["layer_22_post_attn_layernorm"] = layer_22_post_attn_layernorm;
  char *layer_22_gatedup_proj;
  cudaMalloc(&layer_22_gatedup_proj, 201326592);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_22_gatedup_proj + 0), 4194304, reinterpret_cast<const void *>(0x70d2cc000000), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_22_gatedup_proj + 2097152), 4194304, reinterpret_cast<const void *>(0x70d2d2000000), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice);
  all_tensors["layer_22_gatedup_proj"] = layer_22_gatedup_proj;
  char *layer_22_down_proj = (char*)(0x70d2c6000000);
  all_tensors["layer_22_down_proj"] = layer_22_down_proj;
  char *layer_23_input_layernorm = (char*)(0x70d49e466800);
  all_tensors["layer_23_input_layernorm"] = layer_23_input_layernorm;
  char *layer_23_qkv_proj;
  cudaMalloc(&layer_23_qkv_proj, 50331648);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_23_qkv_proj + 0), 6291456, reinterpret_cast<const void *>(0x70d2f1800000), 4194304, 4194304, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_23_qkv_proj + 4194304), 6291456, reinterpret_cast<const void *>(0x70d2ef000000), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_23_qkv_proj + 5242880), 6291456, reinterpret_cast<const void *>(0x70d2f3800000), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice);
  all_tensors["layer_23_qkv_proj"] = layer_23_qkv_proj;
  char *layer_23_q_norm = (char*)(0x70d49e46aa00);
  all_tensors["layer_23_q_norm"] = layer_23_q_norm;
  char *layer_23_k_norm = (char*)(0x70d49e46a800);
  all_tensors["layer_23_k_norm"] = layer_23_k_norm;
  char *layer_23_k_cache = (char*)(0x70d678000000);
  all_tensors["layer_23_k_cache"] = layer_23_k_cache;
  char *layer_23_v_cache = (char*)(0x70d558000000);
  all_tensors["layer_23_v_cache"] = layer_23_v_cache;
  char *layer_23_o_proj = (char*)(0x70d2ef800000);
  all_tensors["layer_23_o_proj"] = layer_23_o_proj;
  char *layer_23_post_attn_layernorm = (char*)(0x70d49e468800);
  all_tensors["layer_23_post_attn_layernorm"] = layer_23_post_attn_layernorm;
  char *layer_23_gatedup_proj;
  cudaMalloc(&layer_23_gatedup_proj, 201326592);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_23_gatedup_proj + 0), 4194304, reinterpret_cast<const void *>(0x70d2e3000000), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_23_gatedup_proj + 2097152), 4194304, reinterpret_cast<const void *>(0x70d2e9000000), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice);
  all_tensors["layer_23_gatedup_proj"] = layer_23_gatedup_proj;
  char *layer_23_down_proj = (char*)(0x70d2dd000000);
  all_tensors["layer_23_down_proj"] = layer_23_down_proj;
  char *layer_24_input_layernorm = (char*)(0x70d49e46ac00);
  all_tensors["layer_24_input_layernorm"] = layer_24_input_layernorm;
  char *layer_24_qkv_proj;
  cudaMalloc(&layer_24_qkv_proj, 50331648);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_24_qkv_proj + 0), 6291456, reinterpret_cast<const void *>(0x70d308800000), 4194304, 4194304, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_24_qkv_proj + 4194304), 6291456, reinterpret_cast<const void *>(0x70d306000000), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_24_qkv_proj + 5242880), 6291456, reinterpret_cast<const void *>(0x70d30a800000), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice);
  all_tensors["layer_24_qkv_proj"] = layer_24_qkv_proj;
  char *layer_24_q_norm = (char*)(0x70d49e46ee00);
  all_tensors["layer_24_q_norm"] = layer_24_q_norm;
  char *layer_24_k_norm = (char*)(0x70d49e46ec00);
  all_tensors["layer_24_k_norm"] = layer_24_k_norm;
  char *layer_24_k_cache = (char*)(0x70d680000000);
  all_tensors["layer_24_k_cache"] = layer_24_k_cache;
  char *layer_24_v_cache = (char*)(0x70d560000000);
  all_tensors["layer_24_v_cache"] = layer_24_v_cache;
  char *layer_24_o_proj = (char*)(0x70d306800000);
  all_tensors["layer_24_o_proj"] = layer_24_o_proj;
  char *layer_24_post_attn_layernorm = (char*)(0x70d49e46cc00);
  all_tensors["layer_24_post_attn_layernorm"] = layer_24_post_attn_layernorm;
  char *layer_24_gatedup_proj;
  cudaMalloc(&layer_24_gatedup_proj, 201326592);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_24_gatedup_proj + 0), 4194304, reinterpret_cast<const void *>(0x70d2fa000000), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_24_gatedup_proj + 2097152), 4194304, reinterpret_cast<const void *>(0x70d300000000), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice);
  all_tensors["layer_24_gatedup_proj"] = layer_24_gatedup_proj;
  char *layer_24_down_proj = (char*)(0x70d2f4000000);
  all_tensors["layer_24_down_proj"] = layer_24_down_proj;
  char *layer_25_input_layernorm = (char*)(0x70d49e46f000);
  all_tensors["layer_25_input_layernorm"] = layer_25_input_layernorm;
  char *layer_25_qkv_proj;
  cudaMalloc(&layer_25_qkv_proj, 50331648);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_25_qkv_proj + 0), 6291456, reinterpret_cast<const void *>(0x70d31f800000), 4194304, 4194304, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_25_qkv_proj + 4194304), 6291456, reinterpret_cast<const void *>(0x70d31d000000), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_25_qkv_proj + 5242880), 6291456, reinterpret_cast<const void *>(0x70d321800000), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice);
  all_tensors["layer_25_qkv_proj"] = layer_25_qkv_proj;
  char *layer_25_q_norm = (char*)(0x70d49e473200);
  all_tensors["layer_25_q_norm"] = layer_25_q_norm;
  char *layer_25_k_norm = (char*)(0x70d49e473000);
  all_tensors["layer_25_k_norm"] = layer_25_k_norm;
  char *layer_25_k_cache = (char*)(0x70d688000000);
  all_tensors["layer_25_k_cache"] = layer_25_k_cache;
  char *layer_25_v_cache = (char*)(0x70d568000000);
  all_tensors["layer_25_v_cache"] = layer_25_v_cache;
  char *layer_25_o_proj = (char*)(0x70d31d800000);
  all_tensors["layer_25_o_proj"] = layer_25_o_proj;
  char *layer_25_post_attn_layernorm = (char*)(0x70d49e471000);
  all_tensors["layer_25_post_attn_layernorm"] = layer_25_post_attn_layernorm;
  char *layer_25_gatedup_proj;
  cudaMalloc(&layer_25_gatedup_proj, 201326592);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_25_gatedup_proj + 0), 4194304, reinterpret_cast<const void *>(0x70d311000000), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_25_gatedup_proj + 2097152), 4194304, reinterpret_cast<const void *>(0x70d317000000), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice);
  all_tensors["layer_25_gatedup_proj"] = layer_25_gatedup_proj;
  char *layer_25_down_proj = (char*)(0x70d30b000000);
  all_tensors["layer_25_down_proj"] = layer_25_down_proj;
  char *layer_26_input_layernorm = (char*)(0x70d49e473400);
  all_tensors["layer_26_input_layernorm"] = layer_26_input_layernorm;
  char *layer_26_qkv_proj;
  cudaMalloc(&layer_26_qkv_proj, 50331648);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_26_qkv_proj + 0), 6291456, reinterpret_cast<const void *>(0x70d336800000), 4194304, 4194304, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_26_qkv_proj + 4194304), 6291456, reinterpret_cast<const void *>(0x70d334000000), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_26_qkv_proj + 5242880), 6291456, reinterpret_cast<const void *>(0x70d338800000), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice);
  all_tensors["layer_26_qkv_proj"] = layer_26_qkv_proj;
  char *layer_26_q_norm = (char*)(0x70d49e477600);
  all_tensors["layer_26_q_norm"] = layer_26_q_norm;
  char *layer_26_k_norm = (char*)(0x70d49e477400);
  all_tensors["layer_26_k_norm"] = layer_26_k_norm;
  char *layer_26_k_cache = (char*)(0x70d690000000);
  all_tensors["layer_26_k_cache"] = layer_26_k_cache;
  char *layer_26_v_cache = (char*)(0x70d570000000);
  all_tensors["layer_26_v_cache"] = layer_26_v_cache;
  char *layer_26_o_proj = (char*)(0x70d334800000);
  all_tensors["layer_26_o_proj"] = layer_26_o_proj;
  char *layer_26_post_attn_layernorm = (char*)(0x70d49e475400);
  all_tensors["layer_26_post_attn_layernorm"] = layer_26_post_attn_layernorm;
  char *layer_26_gatedup_proj;
  cudaMalloc(&layer_26_gatedup_proj, 201326592);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_26_gatedup_proj + 0), 4194304, reinterpret_cast<const void *>(0x70d328000000), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_26_gatedup_proj + 2097152), 4194304, reinterpret_cast<const void *>(0x70d32e000000), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice);
  all_tensors["layer_26_gatedup_proj"] = layer_26_gatedup_proj;
  char *layer_26_down_proj = (char*)(0x70d322000000);
  all_tensors["layer_26_down_proj"] = layer_26_down_proj;
  char *layer_27_input_layernorm = (char*)(0x70d49e477c00);
  all_tensors["layer_27_input_layernorm"] = layer_27_input_layernorm;
  char *layer_27_qkv_proj;
  cudaMalloc(&layer_27_qkv_proj, 50331648);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_27_qkv_proj + 0), 6291456, reinterpret_cast<const void *>(0x70d347800000), 4194304, 4194304, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_27_qkv_proj + 4194304), 6291456, reinterpret_cast<const void *>(0x70d345000000), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_27_qkv_proj + 5242880), 6291456, reinterpret_cast<const void *>(0x70d349800000), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice);
  all_tensors["layer_27_qkv_proj"] = layer_27_qkv_proj;
  char *layer_27_q_norm = (char*)(0x70d49e477a00);
  all_tensors["layer_27_q_norm"] = layer_27_q_norm;
  char *layer_27_k_norm = (char*)(0x70d49e477800);
  all_tensors["layer_27_k_norm"] = layer_27_k_norm;
  char *layer_27_k_cache = (char*)(0x70d698000000);
  all_tensors["layer_27_k_cache"] = layer_27_k_cache;
  char *layer_27_v_cache = (char*)(0x70d578000000);
  all_tensors["layer_27_v_cache"] = layer_27_v_cache;
  char *layer_27_o_proj = (char*)(0x70d345800000);
  all_tensors["layer_27_o_proj"] = layer_27_o_proj;
  char *layer_27_post_attn_layernorm = (char*)(0x70d49e479c00);
  all_tensors["layer_27_post_attn_layernorm"] = layer_27_post_attn_layernorm;
  char *layer_27_gatedup_proj;
  cudaMalloc(&layer_27_gatedup_proj, 201326592);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_27_gatedup_proj + 0), 4194304, reinterpret_cast<const void *>(0x70d339000000), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_27_gatedup_proj + 2097152), 4194304, reinterpret_cast<const void *>(0x70d33f000000), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice);
  all_tensors["layer_27_gatedup_proj"] = layer_27_gatedup_proj;
  char *layer_27_down_proj = (char*)(0x70d34a000000);
  all_tensors["layer_27_down_proj"] = layer_27_down_proj;
  char *layer_28_input_layernorm = (char*)(0x70d49e47bc00);
  all_tensors["layer_28_input_layernorm"] = layer_28_input_layernorm;
  char *layer_28_qkv_proj;
  cudaMalloc(&layer_28_qkv_proj, 50331648);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_28_qkv_proj + 0), 6291456, reinterpret_cast<const void *>(0x70d364800000), 4194304, 4194304, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_28_qkv_proj + 4194304), 6291456, reinterpret_cast<const void *>(0x70d362000000), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_28_qkv_proj + 5242880), 6291456, reinterpret_cast<const void *>(0x70d366800000), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice);
  all_tensors["layer_28_qkv_proj"] = layer_28_qkv_proj;
  char *layer_28_q_norm = (char*)(0x70d49e47fe00);
  all_tensors["layer_28_q_norm"] = layer_28_q_norm;
  char *layer_28_k_norm = (char*)(0x70d49e47fc00);
  all_tensors["layer_28_k_norm"] = layer_28_k_norm;
  char *layer_28_k_cache = (char*)(0x70d6a0000000);
  all_tensors["layer_28_k_cache"] = layer_28_k_cache;
  char *layer_28_v_cache = (char*)(0x70d580000000);
  all_tensors["layer_28_v_cache"] = layer_28_v_cache;
  char *layer_28_o_proj = (char*)(0x70d362800000);
  all_tensors["layer_28_o_proj"] = layer_28_o_proj;
  char *layer_28_post_attn_layernorm = (char*)(0x70d49e47dc00);
  all_tensors["layer_28_post_attn_layernorm"] = layer_28_post_attn_layernorm;
  char *layer_28_gatedup_proj;
  cudaMalloc(&layer_28_gatedup_proj, 201326592);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_28_gatedup_proj + 0), 4194304, reinterpret_cast<const void *>(0x70d356000000), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_28_gatedup_proj + 2097152), 4194304, reinterpret_cast<const void *>(0x70d35c000000), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice);
  all_tensors["layer_28_gatedup_proj"] = layer_28_gatedup_proj;
  char *layer_28_down_proj = (char*)(0x70d350000000);
  all_tensors["layer_28_down_proj"] = layer_28_down_proj;
  char *layer_29_input_layernorm = (char*)(0x70d49e480000);
  all_tensors["layer_29_input_layernorm"] = layer_29_input_layernorm;
  char *layer_29_qkv_proj;
  cudaMalloc(&layer_29_qkv_proj, 50331648);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_29_qkv_proj + 0), 6291456, reinterpret_cast<const void *>(0x70d37b800000), 4194304, 4194304, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_29_qkv_proj + 4194304), 6291456, reinterpret_cast<const void *>(0x70d379000000), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_29_qkv_proj + 5242880), 6291456, reinterpret_cast<const void *>(0x70d37d800000), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice);
  all_tensors["layer_29_qkv_proj"] = layer_29_qkv_proj;
  char *layer_29_q_norm = (char*)(0x70d49e484200);
  all_tensors["layer_29_q_norm"] = layer_29_q_norm;
  char *layer_29_k_norm = (char*)(0x70d49e484000);
  all_tensors["layer_29_k_norm"] = layer_29_k_norm;
  char *layer_29_k_cache = (char*)(0x70d6a8000000);
  all_tensors["layer_29_k_cache"] = layer_29_k_cache;
  char *layer_29_v_cache = (char*)(0x70d588000000);
  all_tensors["layer_29_v_cache"] = layer_29_v_cache;
  char *layer_29_o_proj = (char*)(0x70d379800000);
  all_tensors["layer_29_o_proj"] = layer_29_o_proj;
  char *layer_29_post_attn_layernorm = (char*)(0x70d49e482000);
  all_tensors["layer_29_post_attn_layernorm"] = layer_29_post_attn_layernorm;
  char *layer_29_gatedup_proj;
  cudaMalloc(&layer_29_gatedup_proj, 201326592);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_29_gatedup_proj + 0), 4194304, reinterpret_cast<const void *>(0x70d36d000000), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_29_gatedup_proj + 2097152), 4194304, reinterpret_cast<const void *>(0x70d373000000), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice);
  all_tensors["layer_29_gatedup_proj"] = layer_29_gatedup_proj;
  char *layer_29_down_proj = (char*)(0x70d367000000);
  all_tensors["layer_29_down_proj"] = layer_29_down_proj;
  char *layer_30_input_layernorm = (char*)(0x70d49e484400);
  all_tensors["layer_30_input_layernorm"] = layer_30_input_layernorm;
  char *layer_30_qkv_proj;
  cudaMalloc(&layer_30_qkv_proj, 50331648);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_30_qkv_proj + 0), 6291456, reinterpret_cast<const void *>(0x70d392800000), 4194304, 4194304, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_30_qkv_proj + 4194304), 6291456, reinterpret_cast<const void *>(0x70d390000000), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_30_qkv_proj + 5242880), 6291456, reinterpret_cast<const void *>(0x70d394800000), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice);
  all_tensors["layer_30_qkv_proj"] = layer_30_qkv_proj;
  char *layer_30_q_norm = (char*)(0x70d49e488600);
  all_tensors["layer_30_q_norm"] = layer_30_q_norm;
  char *layer_30_k_norm = (char*)(0x70d49e488400);
  all_tensors["layer_30_k_norm"] = layer_30_k_norm;
  char *layer_30_k_cache = (char*)(0x70d6b0000000);
  all_tensors["layer_30_k_cache"] = layer_30_k_cache;
  char *layer_30_v_cache = (char*)(0x70d590000000);
  all_tensors["layer_30_v_cache"] = layer_30_v_cache;
  char *layer_30_o_proj = (char*)(0x70d390800000);
  all_tensors["layer_30_o_proj"] = layer_30_o_proj;
  char *layer_30_post_attn_layernorm = (char*)(0x70d49e486400);
  all_tensors["layer_30_post_attn_layernorm"] = layer_30_post_attn_layernorm;
  char *layer_30_gatedup_proj;
  cudaMalloc(&layer_30_gatedup_proj, 201326592);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_30_gatedup_proj + 0), 4194304, reinterpret_cast<const void *>(0x70d384000000), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_30_gatedup_proj + 2097152), 4194304, reinterpret_cast<const void *>(0x70d38a000000), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice);
  all_tensors["layer_30_gatedup_proj"] = layer_30_gatedup_proj;
  char *layer_30_down_proj = (char*)(0x70d37e000000);
  all_tensors["layer_30_down_proj"] = layer_30_down_proj;
  char *layer_31_input_layernorm = (char*)(0x70d49e488800);
  all_tensors["layer_31_input_layernorm"] = layer_31_input_layernorm;
  char *layer_31_qkv_proj;
  cudaMalloc(&layer_31_qkv_proj, 50331648);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_31_qkv_proj + 0), 6291456, reinterpret_cast<const void *>(0x70d3a9800000), 4194304, 4194304, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_31_qkv_proj + 4194304), 6291456, reinterpret_cast<const void *>(0x70d3a7000000), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_31_qkv_proj + 5242880), 6291456, reinterpret_cast<const void *>(0x70d3ab800000), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice);
  all_tensors["layer_31_qkv_proj"] = layer_31_qkv_proj;
  char *layer_31_q_norm = (char*)(0x70d49e48ca00);
  all_tensors["layer_31_q_norm"] = layer_31_q_norm;
  char *layer_31_k_norm = (char*)(0x70d49e48c800);
  all_tensors["layer_31_k_norm"] = layer_31_k_norm;
  char *layer_31_k_cache = (char*)(0x70d6b8000000);
  all_tensors["layer_31_k_cache"] = layer_31_k_cache;
  char *layer_31_v_cache = (char*)(0x70d598000000);
  all_tensors["layer_31_v_cache"] = layer_31_v_cache;
  char *layer_31_o_proj = (char*)(0x70d3a7800000);
  all_tensors["layer_31_o_proj"] = layer_31_o_proj;
  char *layer_31_post_attn_layernorm = (char*)(0x70d49e48a800);
  all_tensors["layer_31_post_attn_layernorm"] = layer_31_post_attn_layernorm;
  char *layer_31_gatedup_proj;
  cudaMalloc(&layer_31_gatedup_proj, 201326592);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_31_gatedup_proj + 0), 4194304, reinterpret_cast<const void *>(0x70d39b000000), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_31_gatedup_proj + 2097152), 4194304, reinterpret_cast<const void *>(0x70d3a1000000), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice);
  all_tensors["layer_31_gatedup_proj"] = layer_31_gatedup_proj;
  char *layer_31_down_proj = (char*)(0x70d395000000);
  all_tensors["layer_31_down_proj"] = layer_31_down_proj;
  char *layer_32_input_layernorm = (char*)(0x70d49e48cc00);
  all_tensors["layer_32_input_layernorm"] = layer_32_input_layernorm;
  char *layer_32_qkv_proj;
  cudaMalloc(&layer_32_qkv_proj, 50331648);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_32_qkv_proj + 0), 6291456, reinterpret_cast<const void *>(0x70d3c0800000), 4194304, 4194304, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_32_qkv_proj + 4194304), 6291456, reinterpret_cast<const void *>(0x70d3be000000), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_32_qkv_proj + 5242880), 6291456, reinterpret_cast<const void *>(0x70d3c2800000), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice);
  all_tensors["layer_32_qkv_proj"] = layer_32_qkv_proj;
  char *layer_32_q_norm = (char*)(0x70d49e490e00);
  all_tensors["layer_32_q_norm"] = layer_32_q_norm;
  char *layer_32_k_norm = (char*)(0x70d49e490c00);
  all_tensors["layer_32_k_norm"] = layer_32_k_norm;
  char *layer_32_k_cache = (char*)(0x70d6c0000000);
  all_tensors["layer_32_k_cache"] = layer_32_k_cache;
  char *layer_32_v_cache = (char*)(0x70d5a0000000);
  all_tensors["layer_32_v_cache"] = layer_32_v_cache;
  char *layer_32_o_proj = (char*)(0x70d3be800000);
  all_tensors["layer_32_o_proj"] = layer_32_o_proj;
  char *layer_32_post_attn_layernorm = (char*)(0x70d49e48ec00);
  all_tensors["layer_32_post_attn_layernorm"] = layer_32_post_attn_layernorm;
  char *layer_32_gatedup_proj;
  cudaMalloc(&layer_32_gatedup_proj, 201326592);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_32_gatedup_proj + 0), 4194304, reinterpret_cast<const void *>(0x70d3b2000000), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_32_gatedup_proj + 2097152), 4194304, reinterpret_cast<const void *>(0x70d3b8000000), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice);
  all_tensors["layer_32_gatedup_proj"] = layer_32_gatedup_proj;
  char *layer_32_down_proj = (char*)(0x70d3ac000000);
  all_tensors["layer_32_down_proj"] = layer_32_down_proj;
  char *layer_33_input_layernorm = (char*)(0x70d49e491000);
  all_tensors["layer_33_input_layernorm"] = layer_33_input_layernorm;
  char *layer_33_qkv_proj;
  cudaMalloc(&layer_33_qkv_proj, 50331648);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_33_qkv_proj + 0), 6291456, reinterpret_cast<const void *>(0x70d3d7800000), 4194304, 4194304, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_33_qkv_proj + 4194304), 6291456, reinterpret_cast<const void *>(0x70d3d5000000), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_33_qkv_proj + 5242880), 6291456, reinterpret_cast<const void *>(0x70d3d9800000), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice);
  all_tensors["layer_33_qkv_proj"] = layer_33_qkv_proj;
  char *layer_33_q_norm = (char*)(0x70d49e495200);
  all_tensors["layer_33_q_norm"] = layer_33_q_norm;
  char *layer_33_k_norm = (char*)(0x70d49e495000);
  all_tensors["layer_33_k_norm"] = layer_33_k_norm;
  char *layer_33_k_cache = (char*)(0x70d6c8000000);
  all_tensors["layer_33_k_cache"] = layer_33_k_cache;
  char *layer_33_v_cache = (char*)(0x70d5a8000000);
  all_tensors["layer_33_v_cache"] = layer_33_v_cache;
  char *layer_33_o_proj = (char*)(0x70d3d5800000);
  all_tensors["layer_33_o_proj"] = layer_33_o_proj;
  char *layer_33_post_attn_layernorm = (char*)(0x70d49e493000);
  all_tensors["layer_33_post_attn_layernorm"] = layer_33_post_attn_layernorm;
  char *layer_33_gatedup_proj;
  cudaMalloc(&layer_33_gatedup_proj, 201326592);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_33_gatedup_proj + 0), 4194304, reinterpret_cast<const void *>(0x70d3c9000000), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_33_gatedup_proj + 2097152), 4194304, reinterpret_cast<const void *>(0x70d3cf000000), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice);
  all_tensors["layer_33_gatedup_proj"] = layer_33_gatedup_proj;
  char *layer_33_down_proj = (char*)(0x70d3c3000000);
  all_tensors["layer_33_down_proj"] = layer_33_down_proj;
  char *layer_34_input_layernorm = (char*)(0x70d49e495400);
  all_tensors["layer_34_input_layernorm"] = layer_34_input_layernorm;
  char *layer_34_qkv_proj;
  cudaMalloc(&layer_34_qkv_proj, 50331648);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_34_qkv_proj + 0), 6291456, reinterpret_cast<const void *>(0x70d3ee800000), 4194304, 4194304, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_34_qkv_proj + 4194304), 6291456, reinterpret_cast<const void *>(0x70d3ec000000), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_34_qkv_proj + 5242880), 6291456, reinterpret_cast<const void *>(0x70d3f0800000), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice);
  all_tensors["layer_34_qkv_proj"] = layer_34_qkv_proj;
  char *layer_34_q_norm = (char*)(0x70d49e499600);
  all_tensors["layer_34_q_norm"] = layer_34_q_norm;
  char *layer_34_k_norm = (char*)(0x70d49e499400);
  all_tensors["layer_34_k_norm"] = layer_34_k_norm;
  char *layer_34_k_cache = (char*)(0x70d6d0000000);
  all_tensors["layer_34_k_cache"] = layer_34_k_cache;
  char *layer_34_v_cache = (char*)(0x70d5b0000000);
  all_tensors["layer_34_v_cache"] = layer_34_v_cache;
  char *layer_34_o_proj = (char*)(0x70d3ec800000);
  all_tensors["layer_34_o_proj"] = layer_34_o_proj;
  char *layer_34_post_attn_layernorm = (char*)(0x70d49e497400);
  all_tensors["layer_34_post_attn_layernorm"] = layer_34_post_attn_layernorm;
  char *layer_34_gatedup_proj;
  cudaMalloc(&layer_34_gatedup_proj, 201326592);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_34_gatedup_proj + 0), 4194304, reinterpret_cast<const void *>(0x70d3e0000000), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_34_gatedup_proj + 2097152), 4194304, reinterpret_cast<const void *>(0x70d3e6000000), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice);
  all_tensors["layer_34_gatedup_proj"] = layer_34_gatedup_proj;
  char *layer_34_down_proj = (char*)(0x70d3da000000);
  all_tensors["layer_34_down_proj"] = layer_34_down_proj;
  char *layer_35_input_layernorm = (char*)(0x70d49e499800);
  all_tensors["layer_35_input_layernorm"] = layer_35_input_layernorm;
  char *layer_35_qkv_proj;
  cudaMalloc(&layer_35_qkv_proj, 50331648);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_35_qkv_proj + 0), 6291456, reinterpret_cast<const void *>(0x70d405800000), 4194304, 4194304, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_35_qkv_proj + 4194304), 6291456, reinterpret_cast<const void *>(0x70d403000000), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_35_qkv_proj + 5242880), 6291456, reinterpret_cast<const void *>(0x70d407800000), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice);
  all_tensors["layer_35_qkv_proj"] = layer_35_qkv_proj;
  char *layer_35_q_norm = (char*)(0x70d49e49da00);
  all_tensors["layer_35_q_norm"] = layer_35_q_norm;
  char *layer_35_k_norm = (char*)(0x70d49e49d800);
  all_tensors["layer_35_k_norm"] = layer_35_k_norm;
  char *layer_35_k_cache = (char*)(0x70d6d8000000);
  all_tensors["layer_35_k_cache"] = layer_35_k_cache;
  char *layer_35_v_cache = (char*)(0x70d5b8000000);
  all_tensors["layer_35_v_cache"] = layer_35_v_cache;
  char *layer_35_o_proj = (char*)(0x70d403800000);
  all_tensors["layer_35_o_proj"] = layer_35_o_proj;
  char *layer_35_post_attn_layernorm = (char*)(0x70d49e49b800);
  all_tensors["layer_35_post_attn_layernorm"] = layer_35_post_attn_layernorm;
  char *layer_35_gatedup_proj;
  cudaMalloc(&layer_35_gatedup_proj, 201326592);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_35_gatedup_proj + 0), 4194304, reinterpret_cast<const void *>(0x70d3f7000000), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice);
  cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_35_gatedup_proj + 2097152), 4194304, reinterpret_cast<const void *>(0x70d3fd000000), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice);
  all_tensors["layer_35_gatedup_proj"] = layer_35_gatedup_proj;
  char *layer_35_down_proj = (char*)(0x70d3f1000000);
  all_tensors["layer_35_down_proj"] = layer_35_down_proj;
  char *model_norm_weight = (char*)(0x70d49e49dc00);
  all_tensors["model_norm_weight"] = model_norm_weight;
  char *lm_head = (char*)(0x70cf96000000);
  all_tensors["lm_head"] = lm_head;
  all_tensors["nullptr"] = nullptr;
  construct_task_graph(num_gpus, my_gpu_id, all_tasks, all_events, first_tasks, all_tensors);
  cudaDeviceSynchronize();
}

__device__ __forceinline__
void _execute_task(TaskDesc const* task_desc,
                   RuntimeConfig const &runtime_config) {
  if (task_desc->task_type == TASK_ARGMAX_PARTIAL && task_desc->variant_id == 0) {
      kernel::argmax_partial_kernel<bfloat16, 8, 1200, 128>(
      task_desc->input_ptrs[0],
      task_desc->output_ptrs[0],
      task_desc->output_ptrs[1],
      runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS]);

  }
  else if (task_desc->task_type == TASK_ARGMAX_REDUCE && task_desc->variant_id == 0) {
      kernel::argmax_reduce_kernel<bfloat16, 8, 1200, 128>(
      task_desc->input_ptrs[0],
      task_desc->input_ptrs[1],
      task_desc->output_ptrs[0],
      runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS]);

  }
  else if (task_desc->task_type == TASK_PAGED_ATTENTION_HOPPER && task_desc->variant_id == 0) {
      using TMA_Q = kernel::tma::tma_3d<bfloat16, 3, 3, 3, 8, 6, 128, 8, 4, 64, 6144, 128, 1, 1, 2, 2048, true>;
  using TMA_KV = kernel::tma::tma_3d<bfloat16, 3, 3, 3, 8, 6, 128, 8, 1, 64, 6144, 128, 1, 1, 2, 512, true>;
  using TMA_PAGED_KV_CACHE = kernel::tma::tma_4d<bfloat16, 3, 3, 3, 1, 4096, 8, 128, 1, 64, 4, 64, 4194304, 524288, 128, 1, 1, 2, 4096, true>;
  using TMA_OUTPUT = kernel::tma::tma_3d<bfloat16, 3, 3, 3, 8, 32, 128, 8, 4, 64, 8192, 128, 1, 1, 2, 2048, true>;
  TMA_Q  tma_q (static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[0][0]));
  TMA_KV tma_k (static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[0][1]));
  TMA_KV tma_v (static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[0][2]));
  TMA_PAGED_KV_CACHE tma_paged_k_cache(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[1][0]));
  TMA_PAGED_KV_CACHE tma_paged_v_cache(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[2][0]));
  TMA_OUTPUT tma_output(static_cast<CUtensorMap*>(task_desc->output_tma_desc_ptrs[0][0]));
  kernel::multitoken_paged_attention_hopper_impl<bfloat16, 4, 1, 1024, 6144, 4096, 128, 512, 4096, TMA_Q, TMA_KV, TMA_PAGED_KV_CACHE, TMA_OUTPUT, 8>(
      tma_q,
      tma_k,
      tma_v,
      tma_paged_k_cache,
      tma_paged_v_cache,
      tma_output,
      task_desc->input_ptrs[1],
      task_desc->input_ptrs[2],
      runtime_config.qo_indptr_buffer,
      runtime_config.paged_kv_indptr_buffer,
      runtime_config.paged_kv_indices_buffer,
      runtime_config.paged_kv_last_page_len_buffer,
      task_desc->request_id,
      true,
      true,
      task_desc->input_ptrs[3],
      task_desc->input_ptrs[4],
      task_desc->input_ptrs[5],
      task_desc->input_ptrs[6],
      1e-6f,
      1e-6f,
      task_desc->output_ptrs[0],
      task_desc->head_group);

  }
  else if (task_desc->task_type == TASK_RMS_NORM_HOPPER && task_desc->variant_id == 0) {
      kernel::rms_norm_hopper_impl<bfloat16, 1, 4096>(
      task_desc->input_ptrs[0],
      task_desc->input_ptrs[1],
      task_desc->output_ptrs[0],
      1e-6f);

  }
  else if (task_desc->task_type == TASK_LINEAR_SWAPAB_HOPPER && task_desc->variant_id == 0) {
      using TMA_B = kernel::tma::tma_2d<bfloat16, 3, 3, 3, 8, 4096, 8, 64, 4096, 1, 1, 4, 1024, true>;
  using TMA_A = kernel::tma::tma_2d<bfloat16, 3, 3, 3, 64, 4096, 64, 64, 4096, 1, 1, 4, 4096, true>;
  using TMA_OUT = kernel::tma::tma_2d<bfloat16, 3, 3, 3, 8, 64, 8, 64, 6144, 1, 1, 1, 1024, true>;
    TMA_A tma_a(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[1][0]));
    TMA_B tma_b(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[0][0]));
    TMA_OUT tma_out(static_cast<CUtensorMap*>(task_desc->output_tma_desc_ptrs[0][0]));
    kernel::linear_swapAB_kernel_hopper<bfloat16, 8, 64, 4096, 5, TMA_A, TMA_B, TMA_OUT, void, 6144>(
        tma_a,
        tma_b,
        tma_out, 
        nullptr
    );

  }
  else if (task_desc->task_type == TASK_LINEAR_SWAPAB_HOPPER && task_desc->variant_id == 1) {
      using TMA_B = kernel::tma::tma_2d<bfloat16, 3, 3, 3, 8, 4096, 8, 64, 4096, 1, 1, 4, 1024, true>;
  using TMA_A = kernel::tma::tma_2d<bfloat16, 3, 3, 3, 256, 4096, 64, 64, 4096, 1, 1, 4, 4096, true>;
  using TMA_OUT = kernel::tma::tma_2d<bfloat16, 3, 3, 3, 8, 256, 8, 64, 24576, 1, 1, 1, 1024, true>;
    TMA_A tma_a(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[1][0]));
    TMA_B tma_b(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[0][0]));
    TMA_OUT tma_out(static_cast<CUtensorMap*>(task_desc->output_tma_desc_ptrs[0][0]));
    kernel::linear_swapAB_kernel_hopper<bfloat16, 8, 256, 4096, 5, TMA_A, TMA_B, TMA_OUT, void, 24576>(
        tma_a,
        tma_b,
        tma_out, 
        nullptr
    );

  }
  else if (task_desc->task_type == TASK_LINEAR_SWAPAB_HOPPER && task_desc->variant_id == 2) {
      using TMA_B = kernel::tma::tma_2d<bfloat16, 3, 3, 3, 8, 4096, 8, 64, 4096, 1, 1, 4, 1024, true>;
  using TMA_A = kernel::tma::tma_2d<bfloat16, 3, 3, 3, 1600, 4096, 64, 64, 4096, 1, 1, 4, 4096, true>;
  using TMA_OUT = kernel::tma::tma_2d<bfloat16, 3, 3, 3, 8, 1600, 8, 64, 153600, 1, 1, 1, 1024, true>;
    TMA_A tma_a(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[1][0]));
    TMA_B tma_b(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[0][0]));
    TMA_OUT tma_out(static_cast<CUtensorMap*>(task_desc->output_tma_desc_ptrs[0][0]));
    kernel::linear_swapAB_kernel_hopper<bfloat16, 8, 1600, 4096, 5, TMA_A, TMA_B, TMA_OUT, void, 153600>(
        tma_a,
        tma_b,
        tma_out, 
        nullptr
    );

  }
  else if (task_desc->task_type == TASK_LINEAR_SWAPAB_WITH_RESIDUAL_HOPPER && task_desc->variant_id == 0) {
      using TMA_B = kernel::tma::tma_2d<bfloat16, 3, 3, 3, 8, 4096, 8, 64, 4096, 1, 1, 4, 1024, true>;
  using TMA_A = kernel::tma::tma_2d<bfloat16, 3, 3, 3, 64, 4096, 64, 64, 4096, 1, 1, 4, 4096, true>;
  using TMA_RESIDUAL = kernel::tma::tma_2d<bfloat16, 0, 0, 0, 8, 64, 8, 64, 4096, 1, 1, 1, 1024, true>;
  using TMA_OUT = kernel::tma::tma_2d<bfloat16, 3, 3, 3, 8, 64, 8, 64, 4096, 1, 1, 1, 1024, true>;
    TMA_A tma_a(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[1][0]));
    TMA_B tma_b(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[0][0]));
    TMA_RESIDUAL tma_residual(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[2][0]));
    TMA_OUT tma_out(static_cast<CUtensorMap*>(task_desc->output_tma_desc_ptrs[0][0]));
    kernel::linear_swapAB_kernel_hopper<bfloat16, 8, 64, 4096, 5, TMA_A, TMA_B, TMA_OUT, TMA_RESIDUAL, 4096>(
        tma_a,
        tma_b,
        tma_out, 
        &tma_residual
    );

  }
  else if (task_desc->task_type == TASK_LINEAR_SWAPAB_WITH_RESIDUAL_HOPPER && task_desc->variant_id == 1) {
      using TMA_B = kernel::tma::tma_2d<bfloat16, 3, 3, 3, 8, 12288, 8, 64, 12288, 1, 1, 4, 1024, true>;
  using TMA_A = kernel::tma::tma_2d<bfloat16, 3, 3, 3, 64, 12288, 64, 64, 12288, 1, 1, 4, 4096, true>;
  using TMA_RESIDUAL = kernel::tma::tma_2d<bfloat16, 0, 0, 0, 8, 64, 8, 64, 4096, 1, 1, 1, 1024, true>;
  using TMA_OUT = kernel::tma::tma_2d<bfloat16, 3, 3, 3, 8, 64, 8, 64, 4096, 1, 1, 1, 1024, true>;
    TMA_A tma_a(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[1][0]));
    TMA_B tma_b(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[0][0]));
    TMA_RESIDUAL tma_residual(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[2][0]));
    TMA_OUT tma_out(static_cast<CUtensorMap*>(task_desc->output_tma_desc_ptrs[0][0]));
    kernel::linear_swapAB_kernel_hopper<bfloat16, 8, 64, 12288, 5, TMA_A, TMA_B, TMA_OUT, TMA_RESIDUAL, 4096>(
        tma_a,
        tma_b,
        tma_out, 
        &tma_residual
    );

  }
  else if (task_desc->task_type == TASK_SILU_MUL_HOPPER && task_desc->variant_id == 0) {
      kernel::silu_mul_task_impl_hopper<bfloat16, 8, 256, 24576, 12288>(
      task_desc->input_ptrs[0],
      task_desc->output_ptrs[0],
      runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS]);

  }
  else if (task_desc->task_type == TASK_EMBEDDING_HOPPER && task_desc->variant_id == 0) {
      kernel::embedding_kernel_hopper<bfloat16, 8, 4096, 4096>(
      task_desc->input_ptrs[0],
      task_desc->input_ptrs[1],
      task_desc->output_ptrs[0]);

  }
}

#include <Python.h>
#include <cuda_runtime.h>

static PyObject *init_func(PyObject *self, PyObject *args) {
  PyObject *meta_list, *py_profiler_buffer;
  std::vector<void*> meta_tensors;
  int my_mpi_rank, num_workers, num_local_schedulers, num_remote_schedulers, max_seq_length, total_num_requests;
  long long eos_token_id;
  void *profiler_buffer;

  if (!PyArg_ParseTuple(args, "OOiiiiiiL", &meta_list, &py_profiler_buffer, &my_mpi_rank, &num_workers, &num_local_schedulers, &num_remote_schedulers, &max_seq_length, &total_num_requests, &eos_token_id)) {
    PyErr_SetString(PyExc_TypeError, "Invalid parameters");
    return NULL;
  }

  if(!PyList_Check(meta_list)) {
    PyErr_SetString(PyExc_TypeError, "arg1 must be a list.");
    return NULL;
  }

  Py_ssize_t meta_size = PyList_Size(meta_list);

  for(Py_ssize_t i = 0; i < meta_size; i++) {
    PyObject *item = PyList_GetItem(meta_list, i);
    void* tensor = PyLong_AsVoidPtr(item);
    if(!tensor) {
      PyErr_Format(PyExc_TypeError, "Failed to convert item %d (meta) to void pointer", i);
      return NULL;
    }
    meta_tensors.push_back(PyLong_AsVoidPtr(item));
  }
  profiler_buffer = PyLong_AsVoidPtr(py_profiler_buffer);

  init_persistent_kernel(meta_tensors, profiler_buffer, my_mpi_rank, num_workers, num_local_schedulers, num_remote_schedulers, max_seq_length, total_num_requests, eos_token_id);

  Py_RETURN_NONE;
}

static PyObject *launch_func(PyObject *self, PyObject *args) {
  launch_persistent_kernel();

  Py_RETURN_NONE;
}

static PyObject *finalize_func(PyObject *self, PyObject *args) {
  finalize_persistent_kernel();

  Py_RETURN_NONE;
}

static PyMethodDef ModuleMethods[] = {
  {"init_func", init_func, METH_VARARGS, "initialize persistent kernel"},
  {"launch_func", launch_func, METH_VARARGS, "launch persistent kernel"},
  {"finalize_func", finalize_func, METH_VARARGS, "finalize persistent kernel"},
  {NULL, NULL, 0, NULL} // sentinel
};

static struct PyModuleDef ModuleDef = {
  PyModuleDef_HEAD_INIT,
  "__mirage_launcher",
  NULL, //documentation
  -1, //size
  ModuleMethods,
  NULL, // m_slots
  NULL, // m_traverse
  NULL, // m_clear
  NULL  // m_free
};

PyMODINIT_FUNC PyInit___mirage_launcher(void) {
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {
    return NULL;
  }
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}
