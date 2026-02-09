#include "persistent_kernel.cuh"
#include <nlohmann/json.hpp>
#include <fstream>
#include <filesystem>
using json = nlohmann::json;
using namespace mirage::runtime;

// Global variable for runtime JSON path (referenced by Python for kernel reuse)
std::string g_task_graph_json_path;

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
    std::string json_path = g_task_graph_json_path;
    if (json_path.empty()) {
        // Fall back to __FILE__ based path for backward compatibility
        std::filesystem::path file_path(__FILE__);
        json_path = file_path.parent_path().string()+"/task_graph.json";
    }
    std::ifstream json_file(json_path);
    if (!json_file.is_open()) {
        fprintf(stderr, "ERROR: Failed to open task graph JSON file: %s\n", json_path.c_str());
        abort();
    }
  nlohmann::json json_task_graph;
  json_file >> json_task_graph;
  for (json const &task : json_task_graph["all_tasks"]) {
    FullTaskDesc task_desc(static_cast<TaskType>(task.at("task_type")),
                task.at("variant_id"));
    task_desc.task_metadata.request_id = task.at("request_id").get<int>();
    task_desc.task_metadata.expert_offset = task.at("expert_offset").get<int>();
    task_desc.task_metadata.kv_idx = task.at("kv_idx").get<int>();
    task_desc.task_metadata.merge_task_offset = task.at("merge_task_offset").get<int>();
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
    if (task.at("task_type") > TASK_SM100_TMA_START_TASK && task.at("task_type") < TASK_SM100_TMA_END_TASK) {
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
                                  int my_gpu_id,
                                  std::map<std::string, void*> const &model_tensors) {
  assert(num_gpus = 1);
  std::map<std::string, void*> all_tensors;
  char *input_token = static_cast<char*>(model_tensors.at("input_token"));
  all_tensors["input_token"] = input_token;
  char *cos_position_embedding = static_cast<char*>(model_tensors.at("cos_position_embedding"));
  all_tensors["cos_position_embedding"] = cos_position_embedding;
  char *sin_position_embedding = static_cast<char*>(model_tensors.at("sin_position_embedding"));
  all_tensors["sin_position_embedding"] = sin_position_embedding;
  void *embed_out;
  CUDA_CHECK(cudaMalloc(&embed_out, 65536));
  all_tensors["embed_out"] = embed_out;
  void *rmsnorm_out;
  CUDA_CHECK(cudaMalloc(&rmsnorm_out, 65536));
  all_tensors["rmsnorm_out"] = rmsnorm_out;
  void *attn_in;
  CUDA_CHECK(cudaMalloc(&attn_in, 98304));
  all_tensors["attn_in"] = attn_in;
  void *attn_out;
  CUDA_CHECK(cudaMalloc(&attn_out, 65536));
  all_tensors["attn_out"] = attn_out;
  void *attn_proj_out;
  CUDA_CHECK(cudaMalloc(&attn_proj_out, 65536));
  all_tensors["attn_proj_out"] = attn_proj_out;
  void *all_reduce_buf;
  CUDA_CHECK(cudaMalloc(&all_reduce_buf, 65536));
  all_tensors["all_reduce_buf"] = all_reduce_buf;
  void *attn_allreduce_out;
  CUDA_CHECK(cudaMalloc(&attn_allreduce_out, 65536));
  all_tensors["attn_allreduce_out"] = attn_allreduce_out;
  void *mlp_mid;
  CUDA_CHECK(cudaMalloc(&mlp_mid, 393216));
  all_tensors["mlp_mid"] = mlp_mid;
  void *silu_mul_out;
  CUDA_CHECK(cudaMalloc(&silu_mul_out, 196608));
  all_tensors["silu_mul_out"] = silu_mul_out;
  void *mlp_out;
  CUDA_CHECK(cudaMalloc(&mlp_out, 65536));
  all_tensors["mlp_out"] = mlp_out;
  void *mlp_final;
  CUDA_CHECK(cudaMalloc(&mlp_final, 65536));
  all_tensors["mlp_final"] = mlp_final;
  void *argmax_in;
  CUDA_CHECK(cudaMalloc(&argmax_in, 2457600));
  all_tensors["argmax_in"] = argmax_in;
  void *argmax_part_value;
  CUDA_CHECK(cudaMalloc(&argmax_part_value, 2048));
  all_tensors["argmax_part_value"] = argmax_part_value;
  void *argmax_part_index;
  CUDA_CHECK(cudaMalloc(&argmax_part_index, 8192));
  all_tensors["argmax_part_index"] = argmax_part_index;
  char *output_token = static_cast<char*>(model_tensors.at("output_token"));
  all_tensors["output_token"] = output_token;
  char *embed_tokens = static_cast<char*>(model_tensors.at("embed_tokens"));
  all_tensors["embed_tokens"] = embed_tokens;
  char *layer_0_input_layernorm = static_cast<char*>(model_tensors.at("layer_0_input_layernorm"));
  all_tensors["layer_0_input_layernorm"] = layer_0_input_layernorm;
  char *layer_0_qkv_proj;
  CUDA_CHECK(cudaMalloc(&layer_0_qkv_proj, 50331648));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_0_qkv_proj + 0), 6291456, model_tensors.at("layer_0_q_proj"), 4194304, 4194304, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_0_qkv_proj + 4194304), 6291456, model_tensors.at("layer_0_k_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_0_qkv_proj + 5242880), 6291456, model_tensors.at("layer_0_v_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  all_tensors["layer_0_qkv_proj"] = layer_0_qkv_proj;
  char *layer_0_q_norm = static_cast<char*>(model_tensors.at("layer_0_q_norm"));
  all_tensors["layer_0_q_norm"] = layer_0_q_norm;
  char *layer_0_k_norm = static_cast<char*>(model_tensors.at("layer_0_k_norm"));
  all_tensors["layer_0_k_norm"] = layer_0_k_norm;
  char *layer_0_k_cache = static_cast<char*>(model_tensors.at("layer_0_k_cache"));
  all_tensors["layer_0_k_cache"] = layer_0_k_cache;
  char *layer_0_v_cache = static_cast<char*>(model_tensors.at("layer_0_v_cache"));
  all_tensors["layer_0_v_cache"] = layer_0_v_cache;
  char *layer_0_o_proj = static_cast<char*>(model_tensors.at("layer_0_o_proj"));
  all_tensors["layer_0_o_proj"] = layer_0_o_proj;
  char *layer_0_post_attn_layernorm = static_cast<char*>(model_tensors.at("layer_0_post_attn_layernorm"));
  all_tensors["layer_0_post_attn_layernorm"] = layer_0_post_attn_layernorm;
  char *layer_0_gatedup_proj;
  CUDA_CHECK(cudaMalloc(&layer_0_gatedup_proj, 201326592));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_0_gatedup_proj + 0), 4194304, model_tensors.at("layer_0_gate_proj"), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_0_gatedup_proj + 2097152), 4194304, model_tensors.at("layer_0_up_proj"), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice));
  all_tensors["layer_0_gatedup_proj"] = layer_0_gatedup_proj;
  char *layer_0_down_proj = static_cast<char*>(model_tensors.at("layer_0_down_proj"));
  all_tensors["layer_0_down_proj"] = layer_0_down_proj;
  char *layer_1_input_layernorm = static_cast<char*>(model_tensors.at("layer_1_input_layernorm"));
  all_tensors["layer_1_input_layernorm"] = layer_1_input_layernorm;
  char *layer_1_qkv_proj;
  CUDA_CHECK(cudaMalloc(&layer_1_qkv_proj, 50331648));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_1_qkv_proj + 0), 6291456, model_tensors.at("layer_1_q_proj"), 4194304, 4194304, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_1_qkv_proj + 4194304), 6291456, model_tensors.at("layer_1_k_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_1_qkv_proj + 5242880), 6291456, model_tensors.at("layer_1_v_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  all_tensors["layer_1_qkv_proj"] = layer_1_qkv_proj;
  char *layer_1_q_norm = static_cast<char*>(model_tensors.at("layer_1_q_norm"));
  all_tensors["layer_1_q_norm"] = layer_1_q_norm;
  char *layer_1_k_norm = static_cast<char*>(model_tensors.at("layer_1_k_norm"));
  all_tensors["layer_1_k_norm"] = layer_1_k_norm;
  char *layer_1_k_cache = static_cast<char*>(model_tensors.at("layer_1_k_cache"));
  all_tensors["layer_1_k_cache"] = layer_1_k_cache;
  char *layer_1_v_cache = static_cast<char*>(model_tensors.at("layer_1_v_cache"));
  all_tensors["layer_1_v_cache"] = layer_1_v_cache;
  char *layer_1_o_proj = static_cast<char*>(model_tensors.at("layer_1_o_proj"));
  all_tensors["layer_1_o_proj"] = layer_1_o_proj;
  char *layer_1_post_attn_layernorm = static_cast<char*>(model_tensors.at("layer_1_post_attn_layernorm"));
  all_tensors["layer_1_post_attn_layernorm"] = layer_1_post_attn_layernorm;
  char *layer_1_gatedup_proj;
  CUDA_CHECK(cudaMalloc(&layer_1_gatedup_proj, 201326592));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_1_gatedup_proj + 0), 4194304, model_tensors.at("layer_1_gate_proj"), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_1_gatedup_proj + 2097152), 4194304, model_tensors.at("layer_1_up_proj"), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice));
  all_tensors["layer_1_gatedup_proj"] = layer_1_gatedup_proj;
  char *layer_1_down_proj = static_cast<char*>(model_tensors.at("layer_1_down_proj"));
  all_tensors["layer_1_down_proj"] = layer_1_down_proj;
  char *layer_2_input_layernorm = static_cast<char*>(model_tensors.at("layer_2_input_layernorm"));
  all_tensors["layer_2_input_layernorm"] = layer_2_input_layernorm;
  char *layer_2_qkv_proj;
  CUDA_CHECK(cudaMalloc(&layer_2_qkv_proj, 50331648));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_2_qkv_proj + 0), 6291456, model_tensors.at("layer_2_q_proj"), 4194304, 4194304, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_2_qkv_proj + 4194304), 6291456, model_tensors.at("layer_2_k_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_2_qkv_proj + 5242880), 6291456, model_tensors.at("layer_2_v_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  all_tensors["layer_2_qkv_proj"] = layer_2_qkv_proj;
  char *layer_2_q_norm = static_cast<char*>(model_tensors.at("layer_2_q_norm"));
  all_tensors["layer_2_q_norm"] = layer_2_q_norm;
  char *layer_2_k_norm = static_cast<char*>(model_tensors.at("layer_2_k_norm"));
  all_tensors["layer_2_k_norm"] = layer_2_k_norm;
  char *layer_2_k_cache = static_cast<char*>(model_tensors.at("layer_2_k_cache"));
  all_tensors["layer_2_k_cache"] = layer_2_k_cache;
  char *layer_2_v_cache = static_cast<char*>(model_tensors.at("layer_2_v_cache"));
  all_tensors["layer_2_v_cache"] = layer_2_v_cache;
  char *layer_2_o_proj = static_cast<char*>(model_tensors.at("layer_2_o_proj"));
  all_tensors["layer_2_o_proj"] = layer_2_o_proj;
  char *layer_2_post_attn_layernorm = static_cast<char*>(model_tensors.at("layer_2_post_attn_layernorm"));
  all_tensors["layer_2_post_attn_layernorm"] = layer_2_post_attn_layernorm;
  char *layer_2_gatedup_proj;
  CUDA_CHECK(cudaMalloc(&layer_2_gatedup_proj, 201326592));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_2_gatedup_proj + 0), 4194304, model_tensors.at("layer_2_gate_proj"), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_2_gatedup_proj + 2097152), 4194304, model_tensors.at("layer_2_up_proj"), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice));
  all_tensors["layer_2_gatedup_proj"] = layer_2_gatedup_proj;
  char *layer_2_down_proj = static_cast<char*>(model_tensors.at("layer_2_down_proj"));
  all_tensors["layer_2_down_proj"] = layer_2_down_proj;
  char *layer_3_input_layernorm = static_cast<char*>(model_tensors.at("layer_3_input_layernorm"));
  all_tensors["layer_3_input_layernorm"] = layer_3_input_layernorm;
  char *layer_3_qkv_proj;
  CUDA_CHECK(cudaMalloc(&layer_3_qkv_proj, 50331648));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_3_qkv_proj + 0), 6291456, model_tensors.at("layer_3_q_proj"), 4194304, 4194304, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_3_qkv_proj + 4194304), 6291456, model_tensors.at("layer_3_k_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_3_qkv_proj + 5242880), 6291456, model_tensors.at("layer_3_v_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  all_tensors["layer_3_qkv_proj"] = layer_3_qkv_proj;
  char *layer_3_q_norm = static_cast<char*>(model_tensors.at("layer_3_q_norm"));
  all_tensors["layer_3_q_norm"] = layer_3_q_norm;
  char *layer_3_k_norm = static_cast<char*>(model_tensors.at("layer_3_k_norm"));
  all_tensors["layer_3_k_norm"] = layer_3_k_norm;
  char *layer_3_k_cache = static_cast<char*>(model_tensors.at("layer_3_k_cache"));
  all_tensors["layer_3_k_cache"] = layer_3_k_cache;
  char *layer_3_v_cache = static_cast<char*>(model_tensors.at("layer_3_v_cache"));
  all_tensors["layer_3_v_cache"] = layer_3_v_cache;
  char *layer_3_o_proj = static_cast<char*>(model_tensors.at("layer_3_o_proj"));
  all_tensors["layer_3_o_proj"] = layer_3_o_proj;
  char *layer_3_post_attn_layernorm = static_cast<char*>(model_tensors.at("layer_3_post_attn_layernorm"));
  all_tensors["layer_3_post_attn_layernorm"] = layer_3_post_attn_layernorm;
  char *layer_3_gatedup_proj;
  CUDA_CHECK(cudaMalloc(&layer_3_gatedup_proj, 201326592));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_3_gatedup_proj + 0), 4194304, model_tensors.at("layer_3_gate_proj"), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_3_gatedup_proj + 2097152), 4194304, model_tensors.at("layer_3_up_proj"), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice));
  all_tensors["layer_3_gatedup_proj"] = layer_3_gatedup_proj;
  char *layer_3_down_proj = static_cast<char*>(model_tensors.at("layer_3_down_proj"));
  all_tensors["layer_3_down_proj"] = layer_3_down_proj;
  char *layer_4_input_layernorm = static_cast<char*>(model_tensors.at("layer_4_input_layernorm"));
  all_tensors["layer_4_input_layernorm"] = layer_4_input_layernorm;
  char *layer_4_qkv_proj;
  CUDA_CHECK(cudaMalloc(&layer_4_qkv_proj, 50331648));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_4_qkv_proj + 0), 6291456, model_tensors.at("layer_4_q_proj"), 4194304, 4194304, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_4_qkv_proj + 4194304), 6291456, model_tensors.at("layer_4_k_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_4_qkv_proj + 5242880), 6291456, model_tensors.at("layer_4_v_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  all_tensors["layer_4_qkv_proj"] = layer_4_qkv_proj;
  char *layer_4_q_norm = static_cast<char*>(model_tensors.at("layer_4_q_norm"));
  all_tensors["layer_4_q_norm"] = layer_4_q_norm;
  char *layer_4_k_norm = static_cast<char*>(model_tensors.at("layer_4_k_norm"));
  all_tensors["layer_4_k_norm"] = layer_4_k_norm;
  char *layer_4_k_cache = static_cast<char*>(model_tensors.at("layer_4_k_cache"));
  all_tensors["layer_4_k_cache"] = layer_4_k_cache;
  char *layer_4_v_cache = static_cast<char*>(model_tensors.at("layer_4_v_cache"));
  all_tensors["layer_4_v_cache"] = layer_4_v_cache;
  char *layer_4_o_proj = static_cast<char*>(model_tensors.at("layer_4_o_proj"));
  all_tensors["layer_4_o_proj"] = layer_4_o_proj;
  char *layer_4_post_attn_layernorm = static_cast<char*>(model_tensors.at("layer_4_post_attn_layernorm"));
  all_tensors["layer_4_post_attn_layernorm"] = layer_4_post_attn_layernorm;
  char *layer_4_gatedup_proj;
  CUDA_CHECK(cudaMalloc(&layer_4_gatedup_proj, 201326592));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_4_gatedup_proj + 0), 4194304, model_tensors.at("layer_4_gate_proj"), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_4_gatedup_proj + 2097152), 4194304, model_tensors.at("layer_4_up_proj"), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice));
  all_tensors["layer_4_gatedup_proj"] = layer_4_gatedup_proj;
  char *layer_4_down_proj = static_cast<char*>(model_tensors.at("layer_4_down_proj"));
  all_tensors["layer_4_down_proj"] = layer_4_down_proj;
  char *layer_5_input_layernorm = static_cast<char*>(model_tensors.at("layer_5_input_layernorm"));
  all_tensors["layer_5_input_layernorm"] = layer_5_input_layernorm;
  char *layer_5_qkv_proj;
  CUDA_CHECK(cudaMalloc(&layer_5_qkv_proj, 50331648));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_5_qkv_proj + 0), 6291456, model_tensors.at("layer_5_q_proj"), 4194304, 4194304, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_5_qkv_proj + 4194304), 6291456, model_tensors.at("layer_5_k_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_5_qkv_proj + 5242880), 6291456, model_tensors.at("layer_5_v_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  all_tensors["layer_5_qkv_proj"] = layer_5_qkv_proj;
  char *layer_5_q_norm = static_cast<char*>(model_tensors.at("layer_5_q_norm"));
  all_tensors["layer_5_q_norm"] = layer_5_q_norm;
  char *layer_5_k_norm = static_cast<char*>(model_tensors.at("layer_5_k_norm"));
  all_tensors["layer_5_k_norm"] = layer_5_k_norm;
  char *layer_5_k_cache = static_cast<char*>(model_tensors.at("layer_5_k_cache"));
  all_tensors["layer_5_k_cache"] = layer_5_k_cache;
  char *layer_5_v_cache = static_cast<char*>(model_tensors.at("layer_5_v_cache"));
  all_tensors["layer_5_v_cache"] = layer_5_v_cache;
  char *layer_5_o_proj = static_cast<char*>(model_tensors.at("layer_5_o_proj"));
  all_tensors["layer_5_o_proj"] = layer_5_o_proj;
  char *layer_5_post_attn_layernorm = static_cast<char*>(model_tensors.at("layer_5_post_attn_layernorm"));
  all_tensors["layer_5_post_attn_layernorm"] = layer_5_post_attn_layernorm;
  char *layer_5_gatedup_proj;
  CUDA_CHECK(cudaMalloc(&layer_5_gatedup_proj, 201326592));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_5_gatedup_proj + 0), 4194304, model_tensors.at("layer_5_gate_proj"), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_5_gatedup_proj + 2097152), 4194304, model_tensors.at("layer_5_up_proj"), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice));
  all_tensors["layer_5_gatedup_proj"] = layer_5_gatedup_proj;
  char *layer_5_down_proj = static_cast<char*>(model_tensors.at("layer_5_down_proj"));
  all_tensors["layer_5_down_proj"] = layer_5_down_proj;
  char *layer_6_input_layernorm = static_cast<char*>(model_tensors.at("layer_6_input_layernorm"));
  all_tensors["layer_6_input_layernorm"] = layer_6_input_layernorm;
  char *layer_6_qkv_proj;
  CUDA_CHECK(cudaMalloc(&layer_6_qkv_proj, 50331648));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_6_qkv_proj + 0), 6291456, model_tensors.at("layer_6_q_proj"), 4194304, 4194304, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_6_qkv_proj + 4194304), 6291456, model_tensors.at("layer_6_k_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_6_qkv_proj + 5242880), 6291456, model_tensors.at("layer_6_v_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  all_tensors["layer_6_qkv_proj"] = layer_6_qkv_proj;
  char *layer_6_q_norm = static_cast<char*>(model_tensors.at("layer_6_q_norm"));
  all_tensors["layer_6_q_norm"] = layer_6_q_norm;
  char *layer_6_k_norm = static_cast<char*>(model_tensors.at("layer_6_k_norm"));
  all_tensors["layer_6_k_norm"] = layer_6_k_norm;
  char *layer_6_k_cache = static_cast<char*>(model_tensors.at("layer_6_k_cache"));
  all_tensors["layer_6_k_cache"] = layer_6_k_cache;
  char *layer_6_v_cache = static_cast<char*>(model_tensors.at("layer_6_v_cache"));
  all_tensors["layer_6_v_cache"] = layer_6_v_cache;
  char *layer_6_o_proj = static_cast<char*>(model_tensors.at("layer_6_o_proj"));
  all_tensors["layer_6_o_proj"] = layer_6_o_proj;
  char *layer_6_post_attn_layernorm = static_cast<char*>(model_tensors.at("layer_6_post_attn_layernorm"));
  all_tensors["layer_6_post_attn_layernorm"] = layer_6_post_attn_layernorm;
  char *layer_6_gatedup_proj;
  CUDA_CHECK(cudaMalloc(&layer_6_gatedup_proj, 201326592));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_6_gatedup_proj + 0), 4194304, model_tensors.at("layer_6_gate_proj"), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_6_gatedup_proj + 2097152), 4194304, model_tensors.at("layer_6_up_proj"), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice));
  all_tensors["layer_6_gatedup_proj"] = layer_6_gatedup_proj;
  char *layer_6_down_proj = static_cast<char*>(model_tensors.at("layer_6_down_proj"));
  all_tensors["layer_6_down_proj"] = layer_6_down_proj;
  char *layer_7_input_layernorm = static_cast<char*>(model_tensors.at("layer_7_input_layernorm"));
  all_tensors["layer_7_input_layernorm"] = layer_7_input_layernorm;
  char *layer_7_qkv_proj;
  CUDA_CHECK(cudaMalloc(&layer_7_qkv_proj, 50331648));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_7_qkv_proj + 0), 6291456, model_tensors.at("layer_7_q_proj"), 4194304, 4194304, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_7_qkv_proj + 4194304), 6291456, model_tensors.at("layer_7_k_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_7_qkv_proj + 5242880), 6291456, model_tensors.at("layer_7_v_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  all_tensors["layer_7_qkv_proj"] = layer_7_qkv_proj;
  char *layer_7_q_norm = static_cast<char*>(model_tensors.at("layer_7_q_norm"));
  all_tensors["layer_7_q_norm"] = layer_7_q_norm;
  char *layer_7_k_norm = static_cast<char*>(model_tensors.at("layer_7_k_norm"));
  all_tensors["layer_7_k_norm"] = layer_7_k_norm;
  char *layer_7_k_cache = static_cast<char*>(model_tensors.at("layer_7_k_cache"));
  all_tensors["layer_7_k_cache"] = layer_7_k_cache;
  char *layer_7_v_cache = static_cast<char*>(model_tensors.at("layer_7_v_cache"));
  all_tensors["layer_7_v_cache"] = layer_7_v_cache;
  char *layer_7_o_proj = static_cast<char*>(model_tensors.at("layer_7_o_proj"));
  all_tensors["layer_7_o_proj"] = layer_7_o_proj;
  char *layer_7_post_attn_layernorm = static_cast<char*>(model_tensors.at("layer_7_post_attn_layernorm"));
  all_tensors["layer_7_post_attn_layernorm"] = layer_7_post_attn_layernorm;
  char *layer_7_gatedup_proj;
  CUDA_CHECK(cudaMalloc(&layer_7_gatedup_proj, 201326592));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_7_gatedup_proj + 0), 4194304, model_tensors.at("layer_7_gate_proj"), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_7_gatedup_proj + 2097152), 4194304, model_tensors.at("layer_7_up_proj"), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice));
  all_tensors["layer_7_gatedup_proj"] = layer_7_gatedup_proj;
  char *layer_7_down_proj = static_cast<char*>(model_tensors.at("layer_7_down_proj"));
  all_tensors["layer_7_down_proj"] = layer_7_down_proj;
  char *layer_8_input_layernorm = static_cast<char*>(model_tensors.at("layer_8_input_layernorm"));
  all_tensors["layer_8_input_layernorm"] = layer_8_input_layernorm;
  char *layer_8_qkv_proj;
  CUDA_CHECK(cudaMalloc(&layer_8_qkv_proj, 50331648));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_8_qkv_proj + 0), 6291456, model_tensors.at("layer_8_q_proj"), 4194304, 4194304, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_8_qkv_proj + 4194304), 6291456, model_tensors.at("layer_8_k_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_8_qkv_proj + 5242880), 6291456, model_tensors.at("layer_8_v_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  all_tensors["layer_8_qkv_proj"] = layer_8_qkv_proj;
  char *layer_8_q_norm = static_cast<char*>(model_tensors.at("layer_8_q_norm"));
  all_tensors["layer_8_q_norm"] = layer_8_q_norm;
  char *layer_8_k_norm = static_cast<char*>(model_tensors.at("layer_8_k_norm"));
  all_tensors["layer_8_k_norm"] = layer_8_k_norm;
  char *layer_8_k_cache = static_cast<char*>(model_tensors.at("layer_8_k_cache"));
  all_tensors["layer_8_k_cache"] = layer_8_k_cache;
  char *layer_8_v_cache = static_cast<char*>(model_tensors.at("layer_8_v_cache"));
  all_tensors["layer_8_v_cache"] = layer_8_v_cache;
  char *layer_8_o_proj = static_cast<char*>(model_tensors.at("layer_8_o_proj"));
  all_tensors["layer_8_o_proj"] = layer_8_o_proj;
  char *layer_8_post_attn_layernorm = static_cast<char*>(model_tensors.at("layer_8_post_attn_layernorm"));
  all_tensors["layer_8_post_attn_layernorm"] = layer_8_post_attn_layernorm;
  char *layer_8_gatedup_proj;
  CUDA_CHECK(cudaMalloc(&layer_8_gatedup_proj, 201326592));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_8_gatedup_proj + 0), 4194304, model_tensors.at("layer_8_gate_proj"), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_8_gatedup_proj + 2097152), 4194304, model_tensors.at("layer_8_up_proj"), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice));
  all_tensors["layer_8_gatedup_proj"] = layer_8_gatedup_proj;
  char *layer_8_down_proj = static_cast<char*>(model_tensors.at("layer_8_down_proj"));
  all_tensors["layer_8_down_proj"] = layer_8_down_proj;
  char *layer_9_input_layernorm = static_cast<char*>(model_tensors.at("layer_9_input_layernorm"));
  all_tensors["layer_9_input_layernorm"] = layer_9_input_layernorm;
  char *layer_9_qkv_proj;
  CUDA_CHECK(cudaMalloc(&layer_9_qkv_proj, 50331648));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_9_qkv_proj + 0), 6291456, model_tensors.at("layer_9_q_proj"), 4194304, 4194304, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_9_qkv_proj + 4194304), 6291456, model_tensors.at("layer_9_k_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_9_qkv_proj + 5242880), 6291456, model_tensors.at("layer_9_v_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  all_tensors["layer_9_qkv_proj"] = layer_9_qkv_proj;
  char *layer_9_q_norm = static_cast<char*>(model_tensors.at("layer_9_q_norm"));
  all_tensors["layer_9_q_norm"] = layer_9_q_norm;
  char *layer_9_k_norm = static_cast<char*>(model_tensors.at("layer_9_k_norm"));
  all_tensors["layer_9_k_norm"] = layer_9_k_norm;
  char *layer_9_k_cache = static_cast<char*>(model_tensors.at("layer_9_k_cache"));
  all_tensors["layer_9_k_cache"] = layer_9_k_cache;
  char *layer_9_v_cache = static_cast<char*>(model_tensors.at("layer_9_v_cache"));
  all_tensors["layer_9_v_cache"] = layer_9_v_cache;
  char *layer_9_o_proj = static_cast<char*>(model_tensors.at("layer_9_o_proj"));
  all_tensors["layer_9_o_proj"] = layer_9_o_proj;
  char *layer_9_post_attn_layernorm = static_cast<char*>(model_tensors.at("layer_9_post_attn_layernorm"));
  all_tensors["layer_9_post_attn_layernorm"] = layer_9_post_attn_layernorm;
  char *layer_9_gatedup_proj;
  CUDA_CHECK(cudaMalloc(&layer_9_gatedup_proj, 201326592));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_9_gatedup_proj + 0), 4194304, model_tensors.at("layer_9_gate_proj"), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_9_gatedup_proj + 2097152), 4194304, model_tensors.at("layer_9_up_proj"), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice));
  all_tensors["layer_9_gatedup_proj"] = layer_9_gatedup_proj;
  char *layer_9_down_proj = static_cast<char*>(model_tensors.at("layer_9_down_proj"));
  all_tensors["layer_9_down_proj"] = layer_9_down_proj;
  char *layer_10_input_layernorm = static_cast<char*>(model_tensors.at("layer_10_input_layernorm"));
  all_tensors["layer_10_input_layernorm"] = layer_10_input_layernorm;
  char *layer_10_qkv_proj;
  CUDA_CHECK(cudaMalloc(&layer_10_qkv_proj, 50331648));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_10_qkv_proj + 0), 6291456, model_tensors.at("layer_10_q_proj"), 4194304, 4194304, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_10_qkv_proj + 4194304), 6291456, model_tensors.at("layer_10_k_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_10_qkv_proj + 5242880), 6291456, model_tensors.at("layer_10_v_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  all_tensors["layer_10_qkv_proj"] = layer_10_qkv_proj;
  char *layer_10_q_norm = static_cast<char*>(model_tensors.at("layer_10_q_norm"));
  all_tensors["layer_10_q_norm"] = layer_10_q_norm;
  char *layer_10_k_norm = static_cast<char*>(model_tensors.at("layer_10_k_norm"));
  all_tensors["layer_10_k_norm"] = layer_10_k_norm;
  char *layer_10_k_cache = static_cast<char*>(model_tensors.at("layer_10_k_cache"));
  all_tensors["layer_10_k_cache"] = layer_10_k_cache;
  char *layer_10_v_cache = static_cast<char*>(model_tensors.at("layer_10_v_cache"));
  all_tensors["layer_10_v_cache"] = layer_10_v_cache;
  char *layer_10_o_proj = static_cast<char*>(model_tensors.at("layer_10_o_proj"));
  all_tensors["layer_10_o_proj"] = layer_10_o_proj;
  char *layer_10_post_attn_layernorm = static_cast<char*>(model_tensors.at("layer_10_post_attn_layernorm"));
  all_tensors["layer_10_post_attn_layernorm"] = layer_10_post_attn_layernorm;
  char *layer_10_gatedup_proj;
  CUDA_CHECK(cudaMalloc(&layer_10_gatedup_proj, 201326592));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_10_gatedup_proj + 0), 4194304, model_tensors.at("layer_10_gate_proj"), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_10_gatedup_proj + 2097152), 4194304, model_tensors.at("layer_10_up_proj"), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice));
  all_tensors["layer_10_gatedup_proj"] = layer_10_gatedup_proj;
  char *layer_10_down_proj = static_cast<char*>(model_tensors.at("layer_10_down_proj"));
  all_tensors["layer_10_down_proj"] = layer_10_down_proj;
  char *layer_11_input_layernorm = static_cast<char*>(model_tensors.at("layer_11_input_layernorm"));
  all_tensors["layer_11_input_layernorm"] = layer_11_input_layernorm;
  char *layer_11_qkv_proj;
  CUDA_CHECK(cudaMalloc(&layer_11_qkv_proj, 50331648));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_11_qkv_proj + 0), 6291456, model_tensors.at("layer_11_q_proj"), 4194304, 4194304, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_11_qkv_proj + 4194304), 6291456, model_tensors.at("layer_11_k_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_11_qkv_proj + 5242880), 6291456, model_tensors.at("layer_11_v_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  all_tensors["layer_11_qkv_proj"] = layer_11_qkv_proj;
  char *layer_11_q_norm = static_cast<char*>(model_tensors.at("layer_11_q_norm"));
  all_tensors["layer_11_q_norm"] = layer_11_q_norm;
  char *layer_11_k_norm = static_cast<char*>(model_tensors.at("layer_11_k_norm"));
  all_tensors["layer_11_k_norm"] = layer_11_k_norm;
  char *layer_11_k_cache = static_cast<char*>(model_tensors.at("layer_11_k_cache"));
  all_tensors["layer_11_k_cache"] = layer_11_k_cache;
  char *layer_11_v_cache = static_cast<char*>(model_tensors.at("layer_11_v_cache"));
  all_tensors["layer_11_v_cache"] = layer_11_v_cache;
  char *layer_11_o_proj = static_cast<char*>(model_tensors.at("layer_11_o_proj"));
  all_tensors["layer_11_o_proj"] = layer_11_o_proj;
  char *layer_11_post_attn_layernorm = static_cast<char*>(model_tensors.at("layer_11_post_attn_layernorm"));
  all_tensors["layer_11_post_attn_layernorm"] = layer_11_post_attn_layernorm;
  char *layer_11_gatedup_proj;
  CUDA_CHECK(cudaMalloc(&layer_11_gatedup_proj, 201326592));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_11_gatedup_proj + 0), 4194304, model_tensors.at("layer_11_gate_proj"), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_11_gatedup_proj + 2097152), 4194304, model_tensors.at("layer_11_up_proj"), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice));
  all_tensors["layer_11_gatedup_proj"] = layer_11_gatedup_proj;
  char *layer_11_down_proj = static_cast<char*>(model_tensors.at("layer_11_down_proj"));
  all_tensors["layer_11_down_proj"] = layer_11_down_proj;
  char *layer_12_input_layernorm = static_cast<char*>(model_tensors.at("layer_12_input_layernorm"));
  all_tensors["layer_12_input_layernorm"] = layer_12_input_layernorm;
  char *layer_12_qkv_proj;
  CUDA_CHECK(cudaMalloc(&layer_12_qkv_proj, 50331648));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_12_qkv_proj + 0), 6291456, model_tensors.at("layer_12_q_proj"), 4194304, 4194304, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_12_qkv_proj + 4194304), 6291456, model_tensors.at("layer_12_k_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_12_qkv_proj + 5242880), 6291456, model_tensors.at("layer_12_v_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  all_tensors["layer_12_qkv_proj"] = layer_12_qkv_proj;
  char *layer_12_q_norm = static_cast<char*>(model_tensors.at("layer_12_q_norm"));
  all_tensors["layer_12_q_norm"] = layer_12_q_norm;
  char *layer_12_k_norm = static_cast<char*>(model_tensors.at("layer_12_k_norm"));
  all_tensors["layer_12_k_norm"] = layer_12_k_norm;
  char *layer_12_k_cache = static_cast<char*>(model_tensors.at("layer_12_k_cache"));
  all_tensors["layer_12_k_cache"] = layer_12_k_cache;
  char *layer_12_v_cache = static_cast<char*>(model_tensors.at("layer_12_v_cache"));
  all_tensors["layer_12_v_cache"] = layer_12_v_cache;
  char *layer_12_o_proj = static_cast<char*>(model_tensors.at("layer_12_o_proj"));
  all_tensors["layer_12_o_proj"] = layer_12_o_proj;
  char *layer_12_post_attn_layernorm = static_cast<char*>(model_tensors.at("layer_12_post_attn_layernorm"));
  all_tensors["layer_12_post_attn_layernorm"] = layer_12_post_attn_layernorm;
  char *layer_12_gatedup_proj;
  CUDA_CHECK(cudaMalloc(&layer_12_gatedup_proj, 201326592));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_12_gatedup_proj + 0), 4194304, model_tensors.at("layer_12_gate_proj"), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_12_gatedup_proj + 2097152), 4194304, model_tensors.at("layer_12_up_proj"), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice));
  all_tensors["layer_12_gatedup_proj"] = layer_12_gatedup_proj;
  char *layer_12_down_proj = static_cast<char*>(model_tensors.at("layer_12_down_proj"));
  all_tensors["layer_12_down_proj"] = layer_12_down_proj;
  char *layer_13_input_layernorm = static_cast<char*>(model_tensors.at("layer_13_input_layernorm"));
  all_tensors["layer_13_input_layernorm"] = layer_13_input_layernorm;
  char *layer_13_qkv_proj;
  CUDA_CHECK(cudaMalloc(&layer_13_qkv_proj, 50331648));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_13_qkv_proj + 0), 6291456, model_tensors.at("layer_13_q_proj"), 4194304, 4194304, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_13_qkv_proj + 4194304), 6291456, model_tensors.at("layer_13_k_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_13_qkv_proj + 5242880), 6291456, model_tensors.at("layer_13_v_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  all_tensors["layer_13_qkv_proj"] = layer_13_qkv_proj;
  char *layer_13_q_norm = static_cast<char*>(model_tensors.at("layer_13_q_norm"));
  all_tensors["layer_13_q_norm"] = layer_13_q_norm;
  char *layer_13_k_norm = static_cast<char*>(model_tensors.at("layer_13_k_norm"));
  all_tensors["layer_13_k_norm"] = layer_13_k_norm;
  char *layer_13_k_cache = static_cast<char*>(model_tensors.at("layer_13_k_cache"));
  all_tensors["layer_13_k_cache"] = layer_13_k_cache;
  char *layer_13_v_cache = static_cast<char*>(model_tensors.at("layer_13_v_cache"));
  all_tensors["layer_13_v_cache"] = layer_13_v_cache;
  char *layer_13_o_proj = static_cast<char*>(model_tensors.at("layer_13_o_proj"));
  all_tensors["layer_13_o_proj"] = layer_13_o_proj;
  char *layer_13_post_attn_layernorm = static_cast<char*>(model_tensors.at("layer_13_post_attn_layernorm"));
  all_tensors["layer_13_post_attn_layernorm"] = layer_13_post_attn_layernorm;
  char *layer_13_gatedup_proj;
  CUDA_CHECK(cudaMalloc(&layer_13_gatedup_proj, 201326592));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_13_gatedup_proj + 0), 4194304, model_tensors.at("layer_13_gate_proj"), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_13_gatedup_proj + 2097152), 4194304, model_tensors.at("layer_13_up_proj"), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice));
  all_tensors["layer_13_gatedup_proj"] = layer_13_gatedup_proj;
  char *layer_13_down_proj = static_cast<char*>(model_tensors.at("layer_13_down_proj"));
  all_tensors["layer_13_down_proj"] = layer_13_down_proj;
  char *layer_14_input_layernorm = static_cast<char*>(model_tensors.at("layer_14_input_layernorm"));
  all_tensors["layer_14_input_layernorm"] = layer_14_input_layernorm;
  char *layer_14_qkv_proj;
  CUDA_CHECK(cudaMalloc(&layer_14_qkv_proj, 50331648));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_14_qkv_proj + 0), 6291456, model_tensors.at("layer_14_q_proj"), 4194304, 4194304, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_14_qkv_proj + 4194304), 6291456, model_tensors.at("layer_14_k_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_14_qkv_proj + 5242880), 6291456, model_tensors.at("layer_14_v_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  all_tensors["layer_14_qkv_proj"] = layer_14_qkv_proj;
  char *layer_14_q_norm = static_cast<char*>(model_tensors.at("layer_14_q_norm"));
  all_tensors["layer_14_q_norm"] = layer_14_q_norm;
  char *layer_14_k_norm = static_cast<char*>(model_tensors.at("layer_14_k_norm"));
  all_tensors["layer_14_k_norm"] = layer_14_k_norm;
  char *layer_14_k_cache = static_cast<char*>(model_tensors.at("layer_14_k_cache"));
  all_tensors["layer_14_k_cache"] = layer_14_k_cache;
  char *layer_14_v_cache = static_cast<char*>(model_tensors.at("layer_14_v_cache"));
  all_tensors["layer_14_v_cache"] = layer_14_v_cache;
  char *layer_14_o_proj = static_cast<char*>(model_tensors.at("layer_14_o_proj"));
  all_tensors["layer_14_o_proj"] = layer_14_o_proj;
  char *layer_14_post_attn_layernorm = static_cast<char*>(model_tensors.at("layer_14_post_attn_layernorm"));
  all_tensors["layer_14_post_attn_layernorm"] = layer_14_post_attn_layernorm;
  char *layer_14_gatedup_proj;
  CUDA_CHECK(cudaMalloc(&layer_14_gatedup_proj, 201326592));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_14_gatedup_proj + 0), 4194304, model_tensors.at("layer_14_gate_proj"), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_14_gatedup_proj + 2097152), 4194304, model_tensors.at("layer_14_up_proj"), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice));
  all_tensors["layer_14_gatedup_proj"] = layer_14_gatedup_proj;
  char *layer_14_down_proj = static_cast<char*>(model_tensors.at("layer_14_down_proj"));
  all_tensors["layer_14_down_proj"] = layer_14_down_proj;
  char *layer_15_input_layernorm = static_cast<char*>(model_tensors.at("layer_15_input_layernorm"));
  all_tensors["layer_15_input_layernorm"] = layer_15_input_layernorm;
  char *layer_15_qkv_proj;
  CUDA_CHECK(cudaMalloc(&layer_15_qkv_proj, 50331648));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_15_qkv_proj + 0), 6291456, model_tensors.at("layer_15_q_proj"), 4194304, 4194304, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_15_qkv_proj + 4194304), 6291456, model_tensors.at("layer_15_k_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_15_qkv_proj + 5242880), 6291456, model_tensors.at("layer_15_v_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  all_tensors["layer_15_qkv_proj"] = layer_15_qkv_proj;
  char *layer_15_q_norm = static_cast<char*>(model_tensors.at("layer_15_q_norm"));
  all_tensors["layer_15_q_norm"] = layer_15_q_norm;
  char *layer_15_k_norm = static_cast<char*>(model_tensors.at("layer_15_k_norm"));
  all_tensors["layer_15_k_norm"] = layer_15_k_norm;
  char *layer_15_k_cache = static_cast<char*>(model_tensors.at("layer_15_k_cache"));
  all_tensors["layer_15_k_cache"] = layer_15_k_cache;
  char *layer_15_v_cache = static_cast<char*>(model_tensors.at("layer_15_v_cache"));
  all_tensors["layer_15_v_cache"] = layer_15_v_cache;
  char *layer_15_o_proj = static_cast<char*>(model_tensors.at("layer_15_o_proj"));
  all_tensors["layer_15_o_proj"] = layer_15_o_proj;
  char *layer_15_post_attn_layernorm = static_cast<char*>(model_tensors.at("layer_15_post_attn_layernorm"));
  all_tensors["layer_15_post_attn_layernorm"] = layer_15_post_attn_layernorm;
  char *layer_15_gatedup_proj;
  CUDA_CHECK(cudaMalloc(&layer_15_gatedup_proj, 201326592));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_15_gatedup_proj + 0), 4194304, model_tensors.at("layer_15_gate_proj"), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_15_gatedup_proj + 2097152), 4194304, model_tensors.at("layer_15_up_proj"), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice));
  all_tensors["layer_15_gatedup_proj"] = layer_15_gatedup_proj;
  char *layer_15_down_proj = static_cast<char*>(model_tensors.at("layer_15_down_proj"));
  all_tensors["layer_15_down_proj"] = layer_15_down_proj;
  char *layer_16_input_layernorm = static_cast<char*>(model_tensors.at("layer_16_input_layernorm"));
  all_tensors["layer_16_input_layernorm"] = layer_16_input_layernorm;
  char *layer_16_qkv_proj;
  CUDA_CHECK(cudaMalloc(&layer_16_qkv_proj, 50331648));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_16_qkv_proj + 0), 6291456, model_tensors.at("layer_16_q_proj"), 4194304, 4194304, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_16_qkv_proj + 4194304), 6291456, model_tensors.at("layer_16_k_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_16_qkv_proj + 5242880), 6291456, model_tensors.at("layer_16_v_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  all_tensors["layer_16_qkv_proj"] = layer_16_qkv_proj;
  char *layer_16_q_norm = static_cast<char*>(model_tensors.at("layer_16_q_norm"));
  all_tensors["layer_16_q_norm"] = layer_16_q_norm;
  char *layer_16_k_norm = static_cast<char*>(model_tensors.at("layer_16_k_norm"));
  all_tensors["layer_16_k_norm"] = layer_16_k_norm;
  char *layer_16_k_cache = static_cast<char*>(model_tensors.at("layer_16_k_cache"));
  all_tensors["layer_16_k_cache"] = layer_16_k_cache;
  char *layer_16_v_cache = static_cast<char*>(model_tensors.at("layer_16_v_cache"));
  all_tensors["layer_16_v_cache"] = layer_16_v_cache;
  char *layer_16_o_proj = static_cast<char*>(model_tensors.at("layer_16_o_proj"));
  all_tensors["layer_16_o_proj"] = layer_16_o_proj;
  char *layer_16_post_attn_layernorm = static_cast<char*>(model_tensors.at("layer_16_post_attn_layernorm"));
  all_tensors["layer_16_post_attn_layernorm"] = layer_16_post_attn_layernorm;
  char *layer_16_gatedup_proj;
  CUDA_CHECK(cudaMalloc(&layer_16_gatedup_proj, 201326592));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_16_gatedup_proj + 0), 4194304, model_tensors.at("layer_16_gate_proj"), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_16_gatedup_proj + 2097152), 4194304, model_tensors.at("layer_16_up_proj"), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice));
  all_tensors["layer_16_gatedup_proj"] = layer_16_gatedup_proj;
  char *layer_16_down_proj = static_cast<char*>(model_tensors.at("layer_16_down_proj"));
  all_tensors["layer_16_down_proj"] = layer_16_down_proj;
  char *layer_17_input_layernorm = static_cast<char*>(model_tensors.at("layer_17_input_layernorm"));
  all_tensors["layer_17_input_layernorm"] = layer_17_input_layernorm;
  char *layer_17_qkv_proj;
  CUDA_CHECK(cudaMalloc(&layer_17_qkv_proj, 50331648));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_17_qkv_proj + 0), 6291456, model_tensors.at("layer_17_q_proj"), 4194304, 4194304, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_17_qkv_proj + 4194304), 6291456, model_tensors.at("layer_17_k_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_17_qkv_proj + 5242880), 6291456, model_tensors.at("layer_17_v_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  all_tensors["layer_17_qkv_proj"] = layer_17_qkv_proj;
  char *layer_17_q_norm = static_cast<char*>(model_tensors.at("layer_17_q_norm"));
  all_tensors["layer_17_q_norm"] = layer_17_q_norm;
  char *layer_17_k_norm = static_cast<char*>(model_tensors.at("layer_17_k_norm"));
  all_tensors["layer_17_k_norm"] = layer_17_k_norm;
  char *layer_17_k_cache = static_cast<char*>(model_tensors.at("layer_17_k_cache"));
  all_tensors["layer_17_k_cache"] = layer_17_k_cache;
  char *layer_17_v_cache = static_cast<char*>(model_tensors.at("layer_17_v_cache"));
  all_tensors["layer_17_v_cache"] = layer_17_v_cache;
  char *layer_17_o_proj = static_cast<char*>(model_tensors.at("layer_17_o_proj"));
  all_tensors["layer_17_o_proj"] = layer_17_o_proj;
  char *layer_17_post_attn_layernorm = static_cast<char*>(model_tensors.at("layer_17_post_attn_layernorm"));
  all_tensors["layer_17_post_attn_layernorm"] = layer_17_post_attn_layernorm;
  char *layer_17_gatedup_proj;
  CUDA_CHECK(cudaMalloc(&layer_17_gatedup_proj, 201326592));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_17_gatedup_proj + 0), 4194304, model_tensors.at("layer_17_gate_proj"), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_17_gatedup_proj + 2097152), 4194304, model_tensors.at("layer_17_up_proj"), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice));
  all_tensors["layer_17_gatedup_proj"] = layer_17_gatedup_proj;
  char *layer_17_down_proj = static_cast<char*>(model_tensors.at("layer_17_down_proj"));
  all_tensors["layer_17_down_proj"] = layer_17_down_proj;
  char *layer_18_input_layernorm = static_cast<char*>(model_tensors.at("layer_18_input_layernorm"));
  all_tensors["layer_18_input_layernorm"] = layer_18_input_layernorm;
  char *layer_18_qkv_proj;
  CUDA_CHECK(cudaMalloc(&layer_18_qkv_proj, 50331648));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_18_qkv_proj + 0), 6291456, model_tensors.at("layer_18_q_proj"), 4194304, 4194304, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_18_qkv_proj + 4194304), 6291456, model_tensors.at("layer_18_k_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_18_qkv_proj + 5242880), 6291456, model_tensors.at("layer_18_v_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  all_tensors["layer_18_qkv_proj"] = layer_18_qkv_proj;
  char *layer_18_q_norm = static_cast<char*>(model_tensors.at("layer_18_q_norm"));
  all_tensors["layer_18_q_norm"] = layer_18_q_norm;
  char *layer_18_k_norm = static_cast<char*>(model_tensors.at("layer_18_k_norm"));
  all_tensors["layer_18_k_norm"] = layer_18_k_norm;
  char *layer_18_k_cache = static_cast<char*>(model_tensors.at("layer_18_k_cache"));
  all_tensors["layer_18_k_cache"] = layer_18_k_cache;
  char *layer_18_v_cache = static_cast<char*>(model_tensors.at("layer_18_v_cache"));
  all_tensors["layer_18_v_cache"] = layer_18_v_cache;
  char *layer_18_o_proj = static_cast<char*>(model_tensors.at("layer_18_o_proj"));
  all_tensors["layer_18_o_proj"] = layer_18_o_proj;
  char *layer_18_post_attn_layernorm = static_cast<char*>(model_tensors.at("layer_18_post_attn_layernorm"));
  all_tensors["layer_18_post_attn_layernorm"] = layer_18_post_attn_layernorm;
  char *layer_18_gatedup_proj;
  CUDA_CHECK(cudaMalloc(&layer_18_gatedup_proj, 201326592));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_18_gatedup_proj + 0), 4194304, model_tensors.at("layer_18_gate_proj"), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_18_gatedup_proj + 2097152), 4194304, model_tensors.at("layer_18_up_proj"), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice));
  all_tensors["layer_18_gatedup_proj"] = layer_18_gatedup_proj;
  char *layer_18_down_proj = static_cast<char*>(model_tensors.at("layer_18_down_proj"));
  all_tensors["layer_18_down_proj"] = layer_18_down_proj;
  char *layer_19_input_layernorm = static_cast<char*>(model_tensors.at("layer_19_input_layernorm"));
  all_tensors["layer_19_input_layernorm"] = layer_19_input_layernorm;
  char *layer_19_qkv_proj;
  CUDA_CHECK(cudaMalloc(&layer_19_qkv_proj, 50331648));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_19_qkv_proj + 0), 6291456, model_tensors.at("layer_19_q_proj"), 4194304, 4194304, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_19_qkv_proj + 4194304), 6291456, model_tensors.at("layer_19_k_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_19_qkv_proj + 5242880), 6291456, model_tensors.at("layer_19_v_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  all_tensors["layer_19_qkv_proj"] = layer_19_qkv_proj;
  char *layer_19_q_norm = static_cast<char*>(model_tensors.at("layer_19_q_norm"));
  all_tensors["layer_19_q_norm"] = layer_19_q_norm;
  char *layer_19_k_norm = static_cast<char*>(model_tensors.at("layer_19_k_norm"));
  all_tensors["layer_19_k_norm"] = layer_19_k_norm;
  char *layer_19_k_cache = static_cast<char*>(model_tensors.at("layer_19_k_cache"));
  all_tensors["layer_19_k_cache"] = layer_19_k_cache;
  char *layer_19_v_cache = static_cast<char*>(model_tensors.at("layer_19_v_cache"));
  all_tensors["layer_19_v_cache"] = layer_19_v_cache;
  char *layer_19_o_proj = static_cast<char*>(model_tensors.at("layer_19_o_proj"));
  all_tensors["layer_19_o_proj"] = layer_19_o_proj;
  char *layer_19_post_attn_layernorm = static_cast<char*>(model_tensors.at("layer_19_post_attn_layernorm"));
  all_tensors["layer_19_post_attn_layernorm"] = layer_19_post_attn_layernorm;
  char *layer_19_gatedup_proj;
  CUDA_CHECK(cudaMalloc(&layer_19_gatedup_proj, 201326592));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_19_gatedup_proj + 0), 4194304, model_tensors.at("layer_19_gate_proj"), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_19_gatedup_proj + 2097152), 4194304, model_tensors.at("layer_19_up_proj"), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice));
  all_tensors["layer_19_gatedup_proj"] = layer_19_gatedup_proj;
  char *layer_19_down_proj = static_cast<char*>(model_tensors.at("layer_19_down_proj"));
  all_tensors["layer_19_down_proj"] = layer_19_down_proj;
  char *layer_20_input_layernorm = static_cast<char*>(model_tensors.at("layer_20_input_layernorm"));
  all_tensors["layer_20_input_layernorm"] = layer_20_input_layernorm;
  char *layer_20_qkv_proj;
  CUDA_CHECK(cudaMalloc(&layer_20_qkv_proj, 50331648));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_20_qkv_proj + 0), 6291456, model_tensors.at("layer_20_q_proj"), 4194304, 4194304, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_20_qkv_proj + 4194304), 6291456, model_tensors.at("layer_20_k_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_20_qkv_proj + 5242880), 6291456, model_tensors.at("layer_20_v_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  all_tensors["layer_20_qkv_proj"] = layer_20_qkv_proj;
  char *layer_20_q_norm = static_cast<char*>(model_tensors.at("layer_20_q_norm"));
  all_tensors["layer_20_q_norm"] = layer_20_q_norm;
  char *layer_20_k_norm = static_cast<char*>(model_tensors.at("layer_20_k_norm"));
  all_tensors["layer_20_k_norm"] = layer_20_k_norm;
  char *layer_20_k_cache = static_cast<char*>(model_tensors.at("layer_20_k_cache"));
  all_tensors["layer_20_k_cache"] = layer_20_k_cache;
  char *layer_20_v_cache = static_cast<char*>(model_tensors.at("layer_20_v_cache"));
  all_tensors["layer_20_v_cache"] = layer_20_v_cache;
  char *layer_20_o_proj = static_cast<char*>(model_tensors.at("layer_20_o_proj"));
  all_tensors["layer_20_o_proj"] = layer_20_o_proj;
  char *layer_20_post_attn_layernorm = static_cast<char*>(model_tensors.at("layer_20_post_attn_layernorm"));
  all_tensors["layer_20_post_attn_layernorm"] = layer_20_post_attn_layernorm;
  char *layer_20_gatedup_proj;
  CUDA_CHECK(cudaMalloc(&layer_20_gatedup_proj, 201326592));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_20_gatedup_proj + 0), 4194304, model_tensors.at("layer_20_gate_proj"), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_20_gatedup_proj + 2097152), 4194304, model_tensors.at("layer_20_up_proj"), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice));
  all_tensors["layer_20_gatedup_proj"] = layer_20_gatedup_proj;
  char *layer_20_down_proj = static_cast<char*>(model_tensors.at("layer_20_down_proj"));
  all_tensors["layer_20_down_proj"] = layer_20_down_proj;
  char *layer_21_input_layernorm = static_cast<char*>(model_tensors.at("layer_21_input_layernorm"));
  all_tensors["layer_21_input_layernorm"] = layer_21_input_layernorm;
  char *layer_21_qkv_proj;
  CUDA_CHECK(cudaMalloc(&layer_21_qkv_proj, 50331648));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_21_qkv_proj + 0), 6291456, model_tensors.at("layer_21_q_proj"), 4194304, 4194304, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_21_qkv_proj + 4194304), 6291456, model_tensors.at("layer_21_k_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_21_qkv_proj + 5242880), 6291456, model_tensors.at("layer_21_v_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  all_tensors["layer_21_qkv_proj"] = layer_21_qkv_proj;
  char *layer_21_q_norm = static_cast<char*>(model_tensors.at("layer_21_q_norm"));
  all_tensors["layer_21_q_norm"] = layer_21_q_norm;
  char *layer_21_k_norm = static_cast<char*>(model_tensors.at("layer_21_k_norm"));
  all_tensors["layer_21_k_norm"] = layer_21_k_norm;
  char *layer_21_k_cache = static_cast<char*>(model_tensors.at("layer_21_k_cache"));
  all_tensors["layer_21_k_cache"] = layer_21_k_cache;
  char *layer_21_v_cache = static_cast<char*>(model_tensors.at("layer_21_v_cache"));
  all_tensors["layer_21_v_cache"] = layer_21_v_cache;
  char *layer_21_o_proj = static_cast<char*>(model_tensors.at("layer_21_o_proj"));
  all_tensors["layer_21_o_proj"] = layer_21_o_proj;
  char *layer_21_post_attn_layernorm = static_cast<char*>(model_tensors.at("layer_21_post_attn_layernorm"));
  all_tensors["layer_21_post_attn_layernorm"] = layer_21_post_attn_layernorm;
  char *layer_21_gatedup_proj;
  CUDA_CHECK(cudaMalloc(&layer_21_gatedup_proj, 201326592));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_21_gatedup_proj + 0), 4194304, model_tensors.at("layer_21_gate_proj"), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_21_gatedup_proj + 2097152), 4194304, model_tensors.at("layer_21_up_proj"), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice));
  all_tensors["layer_21_gatedup_proj"] = layer_21_gatedup_proj;
  char *layer_21_down_proj = static_cast<char*>(model_tensors.at("layer_21_down_proj"));
  all_tensors["layer_21_down_proj"] = layer_21_down_proj;
  char *layer_22_input_layernorm = static_cast<char*>(model_tensors.at("layer_22_input_layernorm"));
  all_tensors["layer_22_input_layernorm"] = layer_22_input_layernorm;
  char *layer_22_qkv_proj;
  CUDA_CHECK(cudaMalloc(&layer_22_qkv_proj, 50331648));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_22_qkv_proj + 0), 6291456, model_tensors.at("layer_22_q_proj"), 4194304, 4194304, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_22_qkv_proj + 4194304), 6291456, model_tensors.at("layer_22_k_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_22_qkv_proj + 5242880), 6291456, model_tensors.at("layer_22_v_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  all_tensors["layer_22_qkv_proj"] = layer_22_qkv_proj;
  char *layer_22_q_norm = static_cast<char*>(model_tensors.at("layer_22_q_norm"));
  all_tensors["layer_22_q_norm"] = layer_22_q_norm;
  char *layer_22_k_norm = static_cast<char*>(model_tensors.at("layer_22_k_norm"));
  all_tensors["layer_22_k_norm"] = layer_22_k_norm;
  char *layer_22_k_cache = static_cast<char*>(model_tensors.at("layer_22_k_cache"));
  all_tensors["layer_22_k_cache"] = layer_22_k_cache;
  char *layer_22_v_cache = static_cast<char*>(model_tensors.at("layer_22_v_cache"));
  all_tensors["layer_22_v_cache"] = layer_22_v_cache;
  char *layer_22_o_proj = static_cast<char*>(model_tensors.at("layer_22_o_proj"));
  all_tensors["layer_22_o_proj"] = layer_22_o_proj;
  char *layer_22_post_attn_layernorm = static_cast<char*>(model_tensors.at("layer_22_post_attn_layernorm"));
  all_tensors["layer_22_post_attn_layernorm"] = layer_22_post_attn_layernorm;
  char *layer_22_gatedup_proj;
  CUDA_CHECK(cudaMalloc(&layer_22_gatedup_proj, 201326592));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_22_gatedup_proj + 0), 4194304, model_tensors.at("layer_22_gate_proj"), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_22_gatedup_proj + 2097152), 4194304, model_tensors.at("layer_22_up_proj"), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice));
  all_tensors["layer_22_gatedup_proj"] = layer_22_gatedup_proj;
  char *layer_22_down_proj = static_cast<char*>(model_tensors.at("layer_22_down_proj"));
  all_tensors["layer_22_down_proj"] = layer_22_down_proj;
  char *layer_23_input_layernorm = static_cast<char*>(model_tensors.at("layer_23_input_layernorm"));
  all_tensors["layer_23_input_layernorm"] = layer_23_input_layernorm;
  char *layer_23_qkv_proj;
  CUDA_CHECK(cudaMalloc(&layer_23_qkv_proj, 50331648));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_23_qkv_proj + 0), 6291456, model_tensors.at("layer_23_q_proj"), 4194304, 4194304, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_23_qkv_proj + 4194304), 6291456, model_tensors.at("layer_23_k_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_23_qkv_proj + 5242880), 6291456, model_tensors.at("layer_23_v_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  all_tensors["layer_23_qkv_proj"] = layer_23_qkv_proj;
  char *layer_23_q_norm = static_cast<char*>(model_tensors.at("layer_23_q_norm"));
  all_tensors["layer_23_q_norm"] = layer_23_q_norm;
  char *layer_23_k_norm = static_cast<char*>(model_tensors.at("layer_23_k_norm"));
  all_tensors["layer_23_k_norm"] = layer_23_k_norm;
  char *layer_23_k_cache = static_cast<char*>(model_tensors.at("layer_23_k_cache"));
  all_tensors["layer_23_k_cache"] = layer_23_k_cache;
  char *layer_23_v_cache = static_cast<char*>(model_tensors.at("layer_23_v_cache"));
  all_tensors["layer_23_v_cache"] = layer_23_v_cache;
  char *layer_23_o_proj = static_cast<char*>(model_tensors.at("layer_23_o_proj"));
  all_tensors["layer_23_o_proj"] = layer_23_o_proj;
  char *layer_23_post_attn_layernorm = static_cast<char*>(model_tensors.at("layer_23_post_attn_layernorm"));
  all_tensors["layer_23_post_attn_layernorm"] = layer_23_post_attn_layernorm;
  char *layer_23_gatedup_proj;
  CUDA_CHECK(cudaMalloc(&layer_23_gatedup_proj, 201326592));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_23_gatedup_proj + 0), 4194304, model_tensors.at("layer_23_gate_proj"), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_23_gatedup_proj + 2097152), 4194304, model_tensors.at("layer_23_up_proj"), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice));
  all_tensors["layer_23_gatedup_proj"] = layer_23_gatedup_proj;
  char *layer_23_down_proj = static_cast<char*>(model_tensors.at("layer_23_down_proj"));
  all_tensors["layer_23_down_proj"] = layer_23_down_proj;
  char *layer_24_input_layernorm = static_cast<char*>(model_tensors.at("layer_24_input_layernorm"));
  all_tensors["layer_24_input_layernorm"] = layer_24_input_layernorm;
  char *layer_24_qkv_proj;
  CUDA_CHECK(cudaMalloc(&layer_24_qkv_proj, 50331648));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_24_qkv_proj + 0), 6291456, model_tensors.at("layer_24_q_proj"), 4194304, 4194304, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_24_qkv_proj + 4194304), 6291456, model_tensors.at("layer_24_k_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_24_qkv_proj + 5242880), 6291456, model_tensors.at("layer_24_v_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  all_tensors["layer_24_qkv_proj"] = layer_24_qkv_proj;
  char *layer_24_q_norm = static_cast<char*>(model_tensors.at("layer_24_q_norm"));
  all_tensors["layer_24_q_norm"] = layer_24_q_norm;
  char *layer_24_k_norm = static_cast<char*>(model_tensors.at("layer_24_k_norm"));
  all_tensors["layer_24_k_norm"] = layer_24_k_norm;
  char *layer_24_k_cache = static_cast<char*>(model_tensors.at("layer_24_k_cache"));
  all_tensors["layer_24_k_cache"] = layer_24_k_cache;
  char *layer_24_v_cache = static_cast<char*>(model_tensors.at("layer_24_v_cache"));
  all_tensors["layer_24_v_cache"] = layer_24_v_cache;
  char *layer_24_o_proj = static_cast<char*>(model_tensors.at("layer_24_o_proj"));
  all_tensors["layer_24_o_proj"] = layer_24_o_proj;
  char *layer_24_post_attn_layernorm = static_cast<char*>(model_tensors.at("layer_24_post_attn_layernorm"));
  all_tensors["layer_24_post_attn_layernorm"] = layer_24_post_attn_layernorm;
  char *layer_24_gatedup_proj;
  CUDA_CHECK(cudaMalloc(&layer_24_gatedup_proj, 201326592));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_24_gatedup_proj + 0), 4194304, model_tensors.at("layer_24_gate_proj"), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_24_gatedup_proj + 2097152), 4194304, model_tensors.at("layer_24_up_proj"), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice));
  all_tensors["layer_24_gatedup_proj"] = layer_24_gatedup_proj;
  char *layer_24_down_proj = static_cast<char*>(model_tensors.at("layer_24_down_proj"));
  all_tensors["layer_24_down_proj"] = layer_24_down_proj;
  char *layer_25_input_layernorm = static_cast<char*>(model_tensors.at("layer_25_input_layernorm"));
  all_tensors["layer_25_input_layernorm"] = layer_25_input_layernorm;
  char *layer_25_qkv_proj;
  CUDA_CHECK(cudaMalloc(&layer_25_qkv_proj, 50331648));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_25_qkv_proj + 0), 6291456, model_tensors.at("layer_25_q_proj"), 4194304, 4194304, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_25_qkv_proj + 4194304), 6291456, model_tensors.at("layer_25_k_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_25_qkv_proj + 5242880), 6291456, model_tensors.at("layer_25_v_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  all_tensors["layer_25_qkv_proj"] = layer_25_qkv_proj;
  char *layer_25_q_norm = static_cast<char*>(model_tensors.at("layer_25_q_norm"));
  all_tensors["layer_25_q_norm"] = layer_25_q_norm;
  char *layer_25_k_norm = static_cast<char*>(model_tensors.at("layer_25_k_norm"));
  all_tensors["layer_25_k_norm"] = layer_25_k_norm;
  char *layer_25_k_cache = static_cast<char*>(model_tensors.at("layer_25_k_cache"));
  all_tensors["layer_25_k_cache"] = layer_25_k_cache;
  char *layer_25_v_cache = static_cast<char*>(model_tensors.at("layer_25_v_cache"));
  all_tensors["layer_25_v_cache"] = layer_25_v_cache;
  char *layer_25_o_proj = static_cast<char*>(model_tensors.at("layer_25_o_proj"));
  all_tensors["layer_25_o_proj"] = layer_25_o_proj;
  char *layer_25_post_attn_layernorm = static_cast<char*>(model_tensors.at("layer_25_post_attn_layernorm"));
  all_tensors["layer_25_post_attn_layernorm"] = layer_25_post_attn_layernorm;
  char *layer_25_gatedup_proj;
  CUDA_CHECK(cudaMalloc(&layer_25_gatedup_proj, 201326592));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_25_gatedup_proj + 0), 4194304, model_tensors.at("layer_25_gate_proj"), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_25_gatedup_proj + 2097152), 4194304, model_tensors.at("layer_25_up_proj"), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice));
  all_tensors["layer_25_gatedup_proj"] = layer_25_gatedup_proj;
  char *layer_25_down_proj = static_cast<char*>(model_tensors.at("layer_25_down_proj"));
  all_tensors["layer_25_down_proj"] = layer_25_down_proj;
  char *layer_26_input_layernorm = static_cast<char*>(model_tensors.at("layer_26_input_layernorm"));
  all_tensors["layer_26_input_layernorm"] = layer_26_input_layernorm;
  char *layer_26_qkv_proj;
  CUDA_CHECK(cudaMalloc(&layer_26_qkv_proj, 50331648));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_26_qkv_proj + 0), 6291456, model_tensors.at("layer_26_q_proj"), 4194304, 4194304, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_26_qkv_proj + 4194304), 6291456, model_tensors.at("layer_26_k_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_26_qkv_proj + 5242880), 6291456, model_tensors.at("layer_26_v_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  all_tensors["layer_26_qkv_proj"] = layer_26_qkv_proj;
  char *layer_26_q_norm = static_cast<char*>(model_tensors.at("layer_26_q_norm"));
  all_tensors["layer_26_q_norm"] = layer_26_q_norm;
  char *layer_26_k_norm = static_cast<char*>(model_tensors.at("layer_26_k_norm"));
  all_tensors["layer_26_k_norm"] = layer_26_k_norm;
  char *layer_26_k_cache = static_cast<char*>(model_tensors.at("layer_26_k_cache"));
  all_tensors["layer_26_k_cache"] = layer_26_k_cache;
  char *layer_26_v_cache = static_cast<char*>(model_tensors.at("layer_26_v_cache"));
  all_tensors["layer_26_v_cache"] = layer_26_v_cache;
  char *layer_26_o_proj = static_cast<char*>(model_tensors.at("layer_26_o_proj"));
  all_tensors["layer_26_o_proj"] = layer_26_o_proj;
  char *layer_26_post_attn_layernorm = static_cast<char*>(model_tensors.at("layer_26_post_attn_layernorm"));
  all_tensors["layer_26_post_attn_layernorm"] = layer_26_post_attn_layernorm;
  char *layer_26_gatedup_proj;
  CUDA_CHECK(cudaMalloc(&layer_26_gatedup_proj, 201326592));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_26_gatedup_proj + 0), 4194304, model_tensors.at("layer_26_gate_proj"), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_26_gatedup_proj + 2097152), 4194304, model_tensors.at("layer_26_up_proj"), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice));
  all_tensors["layer_26_gatedup_proj"] = layer_26_gatedup_proj;
  char *layer_26_down_proj = static_cast<char*>(model_tensors.at("layer_26_down_proj"));
  all_tensors["layer_26_down_proj"] = layer_26_down_proj;
  char *layer_27_input_layernorm = static_cast<char*>(model_tensors.at("layer_27_input_layernorm"));
  all_tensors["layer_27_input_layernorm"] = layer_27_input_layernorm;
  char *layer_27_qkv_proj;
  CUDA_CHECK(cudaMalloc(&layer_27_qkv_proj, 50331648));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_27_qkv_proj + 0), 6291456, model_tensors.at("layer_27_q_proj"), 4194304, 4194304, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_27_qkv_proj + 4194304), 6291456, model_tensors.at("layer_27_k_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_27_qkv_proj + 5242880), 6291456, model_tensors.at("layer_27_v_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  all_tensors["layer_27_qkv_proj"] = layer_27_qkv_proj;
  char *layer_27_q_norm = static_cast<char*>(model_tensors.at("layer_27_q_norm"));
  all_tensors["layer_27_q_norm"] = layer_27_q_norm;
  char *layer_27_k_norm = static_cast<char*>(model_tensors.at("layer_27_k_norm"));
  all_tensors["layer_27_k_norm"] = layer_27_k_norm;
  char *layer_27_k_cache = static_cast<char*>(model_tensors.at("layer_27_k_cache"));
  all_tensors["layer_27_k_cache"] = layer_27_k_cache;
  char *layer_27_v_cache = static_cast<char*>(model_tensors.at("layer_27_v_cache"));
  all_tensors["layer_27_v_cache"] = layer_27_v_cache;
  char *layer_27_o_proj = static_cast<char*>(model_tensors.at("layer_27_o_proj"));
  all_tensors["layer_27_o_proj"] = layer_27_o_proj;
  char *layer_27_post_attn_layernorm = static_cast<char*>(model_tensors.at("layer_27_post_attn_layernorm"));
  all_tensors["layer_27_post_attn_layernorm"] = layer_27_post_attn_layernorm;
  char *layer_27_gatedup_proj;
  CUDA_CHECK(cudaMalloc(&layer_27_gatedup_proj, 201326592));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_27_gatedup_proj + 0), 4194304, model_tensors.at("layer_27_gate_proj"), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_27_gatedup_proj + 2097152), 4194304, model_tensors.at("layer_27_up_proj"), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice));
  all_tensors["layer_27_gatedup_proj"] = layer_27_gatedup_proj;
  char *layer_27_down_proj = static_cast<char*>(model_tensors.at("layer_27_down_proj"));
  all_tensors["layer_27_down_proj"] = layer_27_down_proj;
  char *layer_28_input_layernorm = static_cast<char*>(model_tensors.at("layer_28_input_layernorm"));
  all_tensors["layer_28_input_layernorm"] = layer_28_input_layernorm;
  char *layer_28_qkv_proj;
  CUDA_CHECK(cudaMalloc(&layer_28_qkv_proj, 50331648));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_28_qkv_proj + 0), 6291456, model_tensors.at("layer_28_q_proj"), 4194304, 4194304, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_28_qkv_proj + 4194304), 6291456, model_tensors.at("layer_28_k_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_28_qkv_proj + 5242880), 6291456, model_tensors.at("layer_28_v_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  all_tensors["layer_28_qkv_proj"] = layer_28_qkv_proj;
  char *layer_28_q_norm = static_cast<char*>(model_tensors.at("layer_28_q_norm"));
  all_tensors["layer_28_q_norm"] = layer_28_q_norm;
  char *layer_28_k_norm = static_cast<char*>(model_tensors.at("layer_28_k_norm"));
  all_tensors["layer_28_k_norm"] = layer_28_k_norm;
  char *layer_28_k_cache = static_cast<char*>(model_tensors.at("layer_28_k_cache"));
  all_tensors["layer_28_k_cache"] = layer_28_k_cache;
  char *layer_28_v_cache = static_cast<char*>(model_tensors.at("layer_28_v_cache"));
  all_tensors["layer_28_v_cache"] = layer_28_v_cache;
  char *layer_28_o_proj = static_cast<char*>(model_tensors.at("layer_28_o_proj"));
  all_tensors["layer_28_o_proj"] = layer_28_o_proj;
  char *layer_28_post_attn_layernorm = static_cast<char*>(model_tensors.at("layer_28_post_attn_layernorm"));
  all_tensors["layer_28_post_attn_layernorm"] = layer_28_post_attn_layernorm;
  char *layer_28_gatedup_proj;
  CUDA_CHECK(cudaMalloc(&layer_28_gatedup_proj, 201326592));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_28_gatedup_proj + 0), 4194304, model_tensors.at("layer_28_gate_proj"), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_28_gatedup_proj + 2097152), 4194304, model_tensors.at("layer_28_up_proj"), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice));
  all_tensors["layer_28_gatedup_proj"] = layer_28_gatedup_proj;
  char *layer_28_down_proj = static_cast<char*>(model_tensors.at("layer_28_down_proj"));
  all_tensors["layer_28_down_proj"] = layer_28_down_proj;
  char *layer_29_input_layernorm = static_cast<char*>(model_tensors.at("layer_29_input_layernorm"));
  all_tensors["layer_29_input_layernorm"] = layer_29_input_layernorm;
  char *layer_29_qkv_proj;
  CUDA_CHECK(cudaMalloc(&layer_29_qkv_proj, 50331648));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_29_qkv_proj + 0), 6291456, model_tensors.at("layer_29_q_proj"), 4194304, 4194304, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_29_qkv_proj + 4194304), 6291456, model_tensors.at("layer_29_k_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_29_qkv_proj + 5242880), 6291456, model_tensors.at("layer_29_v_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  all_tensors["layer_29_qkv_proj"] = layer_29_qkv_proj;
  char *layer_29_q_norm = static_cast<char*>(model_tensors.at("layer_29_q_norm"));
  all_tensors["layer_29_q_norm"] = layer_29_q_norm;
  char *layer_29_k_norm = static_cast<char*>(model_tensors.at("layer_29_k_norm"));
  all_tensors["layer_29_k_norm"] = layer_29_k_norm;
  char *layer_29_k_cache = static_cast<char*>(model_tensors.at("layer_29_k_cache"));
  all_tensors["layer_29_k_cache"] = layer_29_k_cache;
  char *layer_29_v_cache = static_cast<char*>(model_tensors.at("layer_29_v_cache"));
  all_tensors["layer_29_v_cache"] = layer_29_v_cache;
  char *layer_29_o_proj = static_cast<char*>(model_tensors.at("layer_29_o_proj"));
  all_tensors["layer_29_o_proj"] = layer_29_o_proj;
  char *layer_29_post_attn_layernorm = static_cast<char*>(model_tensors.at("layer_29_post_attn_layernorm"));
  all_tensors["layer_29_post_attn_layernorm"] = layer_29_post_attn_layernorm;
  char *layer_29_gatedup_proj;
  CUDA_CHECK(cudaMalloc(&layer_29_gatedup_proj, 201326592));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_29_gatedup_proj + 0), 4194304, model_tensors.at("layer_29_gate_proj"), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_29_gatedup_proj + 2097152), 4194304, model_tensors.at("layer_29_up_proj"), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice));
  all_tensors["layer_29_gatedup_proj"] = layer_29_gatedup_proj;
  char *layer_29_down_proj = static_cast<char*>(model_tensors.at("layer_29_down_proj"));
  all_tensors["layer_29_down_proj"] = layer_29_down_proj;
  char *layer_30_input_layernorm = static_cast<char*>(model_tensors.at("layer_30_input_layernorm"));
  all_tensors["layer_30_input_layernorm"] = layer_30_input_layernorm;
  char *layer_30_qkv_proj;
  CUDA_CHECK(cudaMalloc(&layer_30_qkv_proj, 50331648));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_30_qkv_proj + 0), 6291456, model_tensors.at("layer_30_q_proj"), 4194304, 4194304, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_30_qkv_proj + 4194304), 6291456, model_tensors.at("layer_30_k_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_30_qkv_proj + 5242880), 6291456, model_tensors.at("layer_30_v_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  all_tensors["layer_30_qkv_proj"] = layer_30_qkv_proj;
  char *layer_30_q_norm = static_cast<char*>(model_tensors.at("layer_30_q_norm"));
  all_tensors["layer_30_q_norm"] = layer_30_q_norm;
  char *layer_30_k_norm = static_cast<char*>(model_tensors.at("layer_30_k_norm"));
  all_tensors["layer_30_k_norm"] = layer_30_k_norm;
  char *layer_30_k_cache = static_cast<char*>(model_tensors.at("layer_30_k_cache"));
  all_tensors["layer_30_k_cache"] = layer_30_k_cache;
  char *layer_30_v_cache = static_cast<char*>(model_tensors.at("layer_30_v_cache"));
  all_tensors["layer_30_v_cache"] = layer_30_v_cache;
  char *layer_30_o_proj = static_cast<char*>(model_tensors.at("layer_30_o_proj"));
  all_tensors["layer_30_o_proj"] = layer_30_o_proj;
  char *layer_30_post_attn_layernorm = static_cast<char*>(model_tensors.at("layer_30_post_attn_layernorm"));
  all_tensors["layer_30_post_attn_layernorm"] = layer_30_post_attn_layernorm;
  char *layer_30_gatedup_proj;
  CUDA_CHECK(cudaMalloc(&layer_30_gatedup_proj, 201326592));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_30_gatedup_proj + 0), 4194304, model_tensors.at("layer_30_gate_proj"), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_30_gatedup_proj + 2097152), 4194304, model_tensors.at("layer_30_up_proj"), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice));
  all_tensors["layer_30_gatedup_proj"] = layer_30_gatedup_proj;
  char *layer_30_down_proj = static_cast<char*>(model_tensors.at("layer_30_down_proj"));
  all_tensors["layer_30_down_proj"] = layer_30_down_proj;
  char *layer_31_input_layernorm = static_cast<char*>(model_tensors.at("layer_31_input_layernorm"));
  all_tensors["layer_31_input_layernorm"] = layer_31_input_layernorm;
  char *layer_31_qkv_proj;
  CUDA_CHECK(cudaMalloc(&layer_31_qkv_proj, 50331648));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_31_qkv_proj + 0), 6291456, model_tensors.at("layer_31_q_proj"), 4194304, 4194304, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_31_qkv_proj + 4194304), 6291456, model_tensors.at("layer_31_k_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_31_qkv_proj + 5242880), 6291456, model_tensors.at("layer_31_v_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  all_tensors["layer_31_qkv_proj"] = layer_31_qkv_proj;
  char *layer_31_q_norm = static_cast<char*>(model_tensors.at("layer_31_q_norm"));
  all_tensors["layer_31_q_norm"] = layer_31_q_norm;
  char *layer_31_k_norm = static_cast<char*>(model_tensors.at("layer_31_k_norm"));
  all_tensors["layer_31_k_norm"] = layer_31_k_norm;
  char *layer_31_k_cache = static_cast<char*>(model_tensors.at("layer_31_k_cache"));
  all_tensors["layer_31_k_cache"] = layer_31_k_cache;
  char *layer_31_v_cache = static_cast<char*>(model_tensors.at("layer_31_v_cache"));
  all_tensors["layer_31_v_cache"] = layer_31_v_cache;
  char *layer_31_o_proj = static_cast<char*>(model_tensors.at("layer_31_o_proj"));
  all_tensors["layer_31_o_proj"] = layer_31_o_proj;
  char *layer_31_post_attn_layernorm = static_cast<char*>(model_tensors.at("layer_31_post_attn_layernorm"));
  all_tensors["layer_31_post_attn_layernorm"] = layer_31_post_attn_layernorm;
  char *layer_31_gatedup_proj;
  CUDA_CHECK(cudaMalloc(&layer_31_gatedup_proj, 201326592));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_31_gatedup_proj + 0), 4194304, model_tensors.at("layer_31_gate_proj"), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_31_gatedup_proj + 2097152), 4194304, model_tensors.at("layer_31_up_proj"), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice));
  all_tensors["layer_31_gatedup_proj"] = layer_31_gatedup_proj;
  char *layer_31_down_proj = static_cast<char*>(model_tensors.at("layer_31_down_proj"));
  all_tensors["layer_31_down_proj"] = layer_31_down_proj;
  char *layer_32_input_layernorm = static_cast<char*>(model_tensors.at("layer_32_input_layernorm"));
  all_tensors["layer_32_input_layernorm"] = layer_32_input_layernorm;
  char *layer_32_qkv_proj;
  CUDA_CHECK(cudaMalloc(&layer_32_qkv_proj, 50331648));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_32_qkv_proj + 0), 6291456, model_tensors.at("layer_32_q_proj"), 4194304, 4194304, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_32_qkv_proj + 4194304), 6291456, model_tensors.at("layer_32_k_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_32_qkv_proj + 5242880), 6291456, model_tensors.at("layer_32_v_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  all_tensors["layer_32_qkv_proj"] = layer_32_qkv_proj;
  char *layer_32_q_norm = static_cast<char*>(model_tensors.at("layer_32_q_norm"));
  all_tensors["layer_32_q_norm"] = layer_32_q_norm;
  char *layer_32_k_norm = static_cast<char*>(model_tensors.at("layer_32_k_norm"));
  all_tensors["layer_32_k_norm"] = layer_32_k_norm;
  char *layer_32_k_cache = static_cast<char*>(model_tensors.at("layer_32_k_cache"));
  all_tensors["layer_32_k_cache"] = layer_32_k_cache;
  char *layer_32_v_cache = static_cast<char*>(model_tensors.at("layer_32_v_cache"));
  all_tensors["layer_32_v_cache"] = layer_32_v_cache;
  char *layer_32_o_proj = static_cast<char*>(model_tensors.at("layer_32_o_proj"));
  all_tensors["layer_32_o_proj"] = layer_32_o_proj;
  char *layer_32_post_attn_layernorm = static_cast<char*>(model_tensors.at("layer_32_post_attn_layernorm"));
  all_tensors["layer_32_post_attn_layernorm"] = layer_32_post_attn_layernorm;
  char *layer_32_gatedup_proj;
  CUDA_CHECK(cudaMalloc(&layer_32_gatedup_proj, 201326592));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_32_gatedup_proj + 0), 4194304, model_tensors.at("layer_32_gate_proj"), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_32_gatedup_proj + 2097152), 4194304, model_tensors.at("layer_32_up_proj"), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice));
  all_tensors["layer_32_gatedup_proj"] = layer_32_gatedup_proj;
  char *layer_32_down_proj = static_cast<char*>(model_tensors.at("layer_32_down_proj"));
  all_tensors["layer_32_down_proj"] = layer_32_down_proj;
  char *layer_33_input_layernorm = static_cast<char*>(model_tensors.at("layer_33_input_layernorm"));
  all_tensors["layer_33_input_layernorm"] = layer_33_input_layernorm;
  char *layer_33_qkv_proj;
  CUDA_CHECK(cudaMalloc(&layer_33_qkv_proj, 50331648));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_33_qkv_proj + 0), 6291456, model_tensors.at("layer_33_q_proj"), 4194304, 4194304, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_33_qkv_proj + 4194304), 6291456, model_tensors.at("layer_33_k_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_33_qkv_proj + 5242880), 6291456, model_tensors.at("layer_33_v_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  all_tensors["layer_33_qkv_proj"] = layer_33_qkv_proj;
  char *layer_33_q_norm = static_cast<char*>(model_tensors.at("layer_33_q_norm"));
  all_tensors["layer_33_q_norm"] = layer_33_q_norm;
  char *layer_33_k_norm = static_cast<char*>(model_tensors.at("layer_33_k_norm"));
  all_tensors["layer_33_k_norm"] = layer_33_k_norm;
  char *layer_33_k_cache = static_cast<char*>(model_tensors.at("layer_33_k_cache"));
  all_tensors["layer_33_k_cache"] = layer_33_k_cache;
  char *layer_33_v_cache = static_cast<char*>(model_tensors.at("layer_33_v_cache"));
  all_tensors["layer_33_v_cache"] = layer_33_v_cache;
  char *layer_33_o_proj = static_cast<char*>(model_tensors.at("layer_33_o_proj"));
  all_tensors["layer_33_o_proj"] = layer_33_o_proj;
  char *layer_33_post_attn_layernorm = static_cast<char*>(model_tensors.at("layer_33_post_attn_layernorm"));
  all_tensors["layer_33_post_attn_layernorm"] = layer_33_post_attn_layernorm;
  char *layer_33_gatedup_proj;
  CUDA_CHECK(cudaMalloc(&layer_33_gatedup_proj, 201326592));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_33_gatedup_proj + 0), 4194304, model_tensors.at("layer_33_gate_proj"), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_33_gatedup_proj + 2097152), 4194304, model_tensors.at("layer_33_up_proj"), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice));
  all_tensors["layer_33_gatedup_proj"] = layer_33_gatedup_proj;
  char *layer_33_down_proj = static_cast<char*>(model_tensors.at("layer_33_down_proj"));
  all_tensors["layer_33_down_proj"] = layer_33_down_proj;
  char *layer_34_input_layernorm = static_cast<char*>(model_tensors.at("layer_34_input_layernorm"));
  all_tensors["layer_34_input_layernorm"] = layer_34_input_layernorm;
  char *layer_34_qkv_proj;
  CUDA_CHECK(cudaMalloc(&layer_34_qkv_proj, 50331648));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_34_qkv_proj + 0), 6291456, model_tensors.at("layer_34_q_proj"), 4194304, 4194304, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_34_qkv_proj + 4194304), 6291456, model_tensors.at("layer_34_k_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_34_qkv_proj + 5242880), 6291456, model_tensors.at("layer_34_v_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  all_tensors["layer_34_qkv_proj"] = layer_34_qkv_proj;
  char *layer_34_q_norm = static_cast<char*>(model_tensors.at("layer_34_q_norm"));
  all_tensors["layer_34_q_norm"] = layer_34_q_norm;
  char *layer_34_k_norm = static_cast<char*>(model_tensors.at("layer_34_k_norm"));
  all_tensors["layer_34_k_norm"] = layer_34_k_norm;
  char *layer_34_k_cache = static_cast<char*>(model_tensors.at("layer_34_k_cache"));
  all_tensors["layer_34_k_cache"] = layer_34_k_cache;
  char *layer_34_v_cache = static_cast<char*>(model_tensors.at("layer_34_v_cache"));
  all_tensors["layer_34_v_cache"] = layer_34_v_cache;
  char *layer_34_o_proj = static_cast<char*>(model_tensors.at("layer_34_o_proj"));
  all_tensors["layer_34_o_proj"] = layer_34_o_proj;
  char *layer_34_post_attn_layernorm = static_cast<char*>(model_tensors.at("layer_34_post_attn_layernorm"));
  all_tensors["layer_34_post_attn_layernorm"] = layer_34_post_attn_layernorm;
  char *layer_34_gatedup_proj;
  CUDA_CHECK(cudaMalloc(&layer_34_gatedup_proj, 201326592));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_34_gatedup_proj + 0), 4194304, model_tensors.at("layer_34_gate_proj"), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_34_gatedup_proj + 2097152), 4194304, model_tensors.at("layer_34_up_proj"), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice));
  all_tensors["layer_34_gatedup_proj"] = layer_34_gatedup_proj;
  char *layer_34_down_proj = static_cast<char*>(model_tensors.at("layer_34_down_proj"));
  all_tensors["layer_34_down_proj"] = layer_34_down_proj;
  char *layer_35_input_layernorm = static_cast<char*>(model_tensors.at("layer_35_input_layernorm"));
  all_tensors["layer_35_input_layernorm"] = layer_35_input_layernorm;
  char *layer_35_qkv_proj;
  CUDA_CHECK(cudaMalloc(&layer_35_qkv_proj, 50331648));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_35_qkv_proj + 0), 6291456, model_tensors.at("layer_35_q_proj"), 4194304, 4194304, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_35_qkv_proj + 4194304), 6291456, model_tensors.at("layer_35_k_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_35_qkv_proj + 5242880), 6291456, model_tensors.at("layer_35_v_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  all_tensors["layer_35_qkv_proj"] = layer_35_qkv_proj;
  char *layer_35_q_norm = static_cast<char*>(model_tensors.at("layer_35_q_norm"));
  all_tensors["layer_35_q_norm"] = layer_35_q_norm;
  char *layer_35_k_norm = static_cast<char*>(model_tensors.at("layer_35_k_norm"));
  all_tensors["layer_35_k_norm"] = layer_35_k_norm;
  char *layer_35_k_cache = static_cast<char*>(model_tensors.at("layer_35_k_cache"));
  all_tensors["layer_35_k_cache"] = layer_35_k_cache;
  char *layer_35_v_cache = static_cast<char*>(model_tensors.at("layer_35_v_cache"));
  all_tensors["layer_35_v_cache"] = layer_35_v_cache;
  char *layer_35_o_proj = static_cast<char*>(model_tensors.at("layer_35_o_proj"));
  all_tensors["layer_35_o_proj"] = layer_35_o_proj;
  char *layer_35_post_attn_layernorm = static_cast<char*>(model_tensors.at("layer_35_post_attn_layernorm"));
  all_tensors["layer_35_post_attn_layernorm"] = layer_35_post_attn_layernorm;
  char *layer_35_gatedup_proj;
  CUDA_CHECK(cudaMalloc(&layer_35_gatedup_proj, 201326592));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_35_gatedup_proj + 0), 4194304, model_tensors.at("layer_35_gate_proj"), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_35_gatedup_proj + 2097152), 4194304, model_tensors.at("layer_35_up_proj"), 2097152, 2097152, 48, cudaMemcpyDeviceToDevice));
  all_tensors["layer_35_gatedup_proj"] = layer_35_gatedup_proj;
  char *layer_35_down_proj = static_cast<char*>(model_tensors.at("layer_35_down_proj"));
  all_tensors["layer_35_down_proj"] = layer_35_down_proj;
  char *model_norm_weight = static_cast<char*>(model_tensors.at("model_norm_weight"));
  all_tensors["model_norm_weight"] = model_norm_weight;
  char *lm_head = static_cast<char*>(model_tensors.at("lm_head"));
  all_tensors["lm_head"] = lm_head;
  all_tensors["nullptr"] = nullptr;
  construct_task_graph(num_gpus, my_gpu_id, all_tasks, all_events, first_tasks, all_tensors);
  cudaDeviceSynchronize();
}

__device__ __forceinline__
void _execute_task(TaskDesc const* task_desc,
                   RuntimeConfig const &runtime_config) {
  if (task_desc->task_type == TASK_EMBEDDING && task_desc->variant_id == 0) {
      kernel::embedding_kernel<bfloat16, 8, 4096, 4096>(
      task_desc->input_ptrs[0],
      task_desc->input_ptrs[1],
      task_desc->output_ptrs[0]);

  }
  else if (task_desc->task_type == TASK_SILU_MUL && task_desc->variant_id == 0) {
      kernel::silu_mul_task_impl<bfloat16, 8, 256, 24576, 12288>(
      task_desc->input_ptrs[0],
      task_desc->output_ptrs[0],
      runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS]);

  }
  else if (task_desc->task_type == TASK_RMS_NORM_HOPPER && task_desc->variant_id == 0) {
      kernel::rms_norm_hopper_impl<bfloat16, 1, 4096>(
      task_desc->input_ptrs[0],
      task_desc->input_ptrs[1],
      task_desc->output_ptrs[0],
      1e-6f);

  }
  else if (task_desc->task_type == TASK_LINEAR_WITH_RESIDUAL_SM100 && task_desc->variant_id == 0) {
      using TMA_A = kernel::tma::tma_2d<cute::bfloat16_t, 3, 3, 3, 64, 4096, 128, 64, 4096, 1, 1, 1, 8192, true>;
  using TMA_B = kernel::tma::tma_2d<cute::bfloat16_t, 3, 3, 3, 8, 4096, 16, 64, 4096, 1, 1, 1, 1024, true>;
  using TMA_OUT = kernel::tma::tma_2d<cute::bfloat16_t, 0, 3, 3, 8, 64, 16, 128, 4096, 1, 1, 1, 2048, true>;
    TMA_A tma_a(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[1][0]));
    TMA_B tma_b(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[0][0]));
    TMA_OUT tma_out(static_cast<CUtensorMap*>(task_desc->output_tma_desc_ptrs[0][0]));
    cute::Layout layout_Bias = cute::make_layout(cute::make_shape(8, 64), cute::make_stride(4096, cute::Int<1>{}));
    cute::Tensor mBias = cute::make_tensor(cute::make_gmem_ptr(static_cast<cute::bfloat16_t*>(task_desc->input_ptrs[2])), layout_Bias);
    kernel::linear_sm100_mpk_task_impl<cute::bfloat16_t, TMA_A, TMA_B, decltype(mBias), TMA_OUT, 128, 16, 8, 64, 4096, false, false, 8, 2, 4>(
        tma_a,
        tma_b,
        mBias,
        tma_out); 

  }
  else if (task_desc->task_type == TASK_LINEAR_WITH_RESIDUAL_SM100 && task_desc->variant_id == 1) {
      using TMA_A = kernel::tma::tma_2d<cute::bfloat16_t, 3, 3, 3, 64, 12288, 128, 64, 12288, 1, 1, 1, 8192, true>;
  using TMA_B = kernel::tma::tma_2d<cute::bfloat16_t, 3, 3, 3, 8, 12288, 16, 64, 12288, 1, 1, 1, 1024, true>;
  using TMA_OUT = kernel::tma::tma_2d<cute::bfloat16_t, 0, 3, 3, 8, 64, 16, 128, 4096, 1, 1, 1, 2048, true>;
    TMA_A tma_a(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[1][0]));
    TMA_B tma_b(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[0][0]));
    TMA_OUT tma_out(static_cast<CUtensorMap*>(task_desc->output_tma_desc_ptrs[0][0]));
    cute::Layout layout_Bias = cute::make_layout(cute::make_shape(8, 64), cute::make_stride(4096, cute::Int<1>{}));
    cute::Tensor mBias = cute::make_tensor(cute::make_gmem_ptr(static_cast<cute::bfloat16_t*>(task_desc->input_ptrs[2])), layout_Bias);
    kernel::linear_sm100_mpk_task_impl<cute::bfloat16_t, TMA_A, TMA_B, decltype(mBias), TMA_OUT, 128, 16, 8, 64, 12288, false, false, 8, 2, 4>(
        tma_a,
        tma_b,
        mBias,
        tma_out); 

  }
  else if (task_desc->task_type == TASK_LINEAR_SM100 && task_desc->variant_id == 0) {
      using TMA_A = kernel::tma::tma_2d<cute::bfloat16_t, 3, 3, 3, 64, 4096, 128, 64, 4096, 1, 1, 1, 8192, true>;
  using TMA_B = kernel::tma::tma_2d<cute::bfloat16_t, 3, 3, 3, 8, 4096, 16, 64, 4096, 1, 1, 1, 1024, true>;
  using TMA_OUT = kernel::tma::tma_2d<cute::bfloat16_t, 0, 3, 3, 8, 64, 16, 128, 6144, 1, 1, 1, 2048, true>;
    TMA_A tma_a(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[1][0]));
    TMA_B tma_b(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[0][0]));
    TMA_OUT tma_out(static_cast<CUtensorMap*>(task_desc->output_tma_desc_ptrs[0][0]));
    cute::Layout layout_Bias = cute::make_layout(cute::make_shape(8, 64), cute::make_stride(6144, cute::Int<1>{}));
    cute::Tensor mBias = cute::make_tensor(cute::make_gmem_ptr(static_cast<cute::bfloat16_t*>(nullptr)), layout_Bias);
    kernel::linear_sm100_mpk_task_impl<cute::bfloat16_t, TMA_A, TMA_B, decltype(mBias), TMA_OUT, 128, 16, 8, 64, 4096, true, false, 8, 2, 4>(
        tma_a,
        tma_b,
        mBias,
        tma_out); 

  }
  else if (task_desc->task_type == TASK_LINEAR_SM100 && task_desc->variant_id == 1) {
      using TMA_A = kernel::tma::tma_2d<cute::bfloat16_t, 3, 3, 3, 256, 4096, 128, 64, 4096, 1, 1, 1, 8192, true>;
  using TMA_B = kernel::tma::tma_2d<cute::bfloat16_t, 3, 3, 3, 8, 4096, 16, 64, 4096, 1, 1, 1, 1024, true>;
  using TMA_OUT = kernel::tma::tma_2d<cute::bfloat16_t, 0, 3, 3, 8, 256, 16, 128, 24576, 1, 1, 1, 2048, true>;
    TMA_A tma_a(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[1][0]));
    TMA_B tma_b(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[0][0]));
    TMA_OUT tma_out(static_cast<CUtensorMap*>(task_desc->output_tma_desc_ptrs[0][0]));
    cute::Layout layout_Bias = cute::make_layout(cute::make_shape(8, 256), cute::make_stride(24576, cute::Int<1>{}));
    cute::Tensor mBias = cute::make_tensor(cute::make_gmem_ptr(static_cast<cute::bfloat16_t*>(nullptr)), layout_Bias);
    kernel::linear_sm100_mpk_task_impl<cute::bfloat16_t, TMA_A, TMA_B, decltype(mBias), TMA_OUT, 128, 16, 8, 256, 4096, true, false, 8, 2, 4>(
        tma_a,
        tma_b,
        mBias,
        tma_out); 

  }
  else if (task_desc->task_type == TASK_LINEAR_SM100 && task_desc->variant_id == 2) {
      using TMA_A = kernel::tma::tma_2d<cute::bfloat16_t, 3, 3, 3, 1200, 4096, 128, 64, 4096, 1, 1, 1, 8192, true>;
  using TMA_B = kernel::tma::tma_2d<cute::bfloat16_t, 3, 3, 3, 8, 4096, 16, 64, 4096, 1, 1, 1, 1024, true>;
  using TMA_OUT = kernel::tma::tma_2d<cute::bfloat16_t, 0, 3, 3, 8, 1200, 16, 128, 153600, 1, 1, 1, 2048, true>;
    TMA_A tma_a(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[1][0]));
    TMA_B tma_b(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[0][0]));
    TMA_OUT tma_out(static_cast<CUtensorMap*>(task_desc->output_tma_desc_ptrs[0][0]));
    cute::Layout layout_Bias = cute::make_layout(cute::make_shape(8, 1200), cute::make_stride(153600, cute::Int<1>{}));
    cute::Tensor mBias = cute::make_tensor(cute::make_gmem_ptr(static_cast<cute::bfloat16_t*>(nullptr)), layout_Bias);
    kernel::linear_sm100_mpk_task_impl<cute::bfloat16_t, TMA_A, TMA_B, decltype(mBias), TMA_OUT, 128, 16, 8, 1200, 4096, true, false, 8, 2, 4>(
        tma_a,
        tma_b,
        mBias,
        tma_out); 

  }
  else if (task_desc->task_type == TASK_ATTN_SM100 && task_desc->variant_id == 0) {
      kernel::multitoken_paged_attention_sm100_task_impl<bfloat16, 4, 1, 1024, 6144, 4096, 128, 512, 4096>(
      task_desc->input_ptrs[0],
      task_desc->input_ptrs[1],
      task_desc->input_ptrs[2],
      task_desc->output_ptrs[0],
      runtime_config.qo_indptr_buffer,
      runtime_config.paged_kv_indptr_buffer,
      runtime_config.paged_kv_indices_buffer,
      runtime_config.paged_kv_last_page_len_buffer,
      task_desc->task_metadata.request_id,
      true,
      true,
      task_desc->input_ptrs[3],
      task_desc->input_ptrs[4],
      task_desc->input_ptrs[5],
      task_desc->input_ptrs[6],
      1e-6f,
      1e-6f);

  }
  else if (task_desc->task_type == TASK_ARGMAX_REDUCE_SM100 && task_desc->variant_id == 0) {
      kernel::argmax_reduce_sm100_kernel<bfloat16, 8, 1200, 128>(
      task_desc->input_ptrs[0],
      task_desc->input_ptrs[1],
      task_desc->output_ptrs[0],
      runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS]);

  }
  else if (task_desc->task_type == TASK_ARGMAX_PARTIAL_SM100 && task_desc->variant_id == 0) {
      kernel::argmax_partial_sm100_kernel<bfloat16, 8, 1200, 128>(
      task_desc->input_ptrs[0],
      task_desc->output_ptrs[0],
      task_desc->output_ptrs[1],
      runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS]);

  }
}

#include <Python.h>
#include <cuda_runtime.h>
#include <string>
#include <vector>

extern std::string g_task_graph_json_path;

static PyObject *init_func(PyObject *self, PyObject *args) {
  PyObject *meta_list, *py_profiler_buffer, *tensor_names_list, *tensor_ptrs_list, *py_json_path;
  std::vector<void*> meta_tensors;
  std::vector<std::string> model_tensor_names;
  std::vector<void*> model_tensor_ptrs;
  int my_mpi_rank, num_workers, num_local_schedulers, num_remote_schedulers, max_seq_length, total_num_requests;
  long long eos_token_id;
  void *profiler_buffer;

  if (!PyArg_ParseTuple(args, "OOiiiiiiLOOO", &meta_list, &py_profiler_buffer, &my_mpi_rank, &num_workers, &num_local_schedulers, &num_remote_schedulers, &max_seq_length, &total_num_requests, &eos_token_id, &tensor_names_list, &tensor_ptrs_list, &py_json_path)) {
    PyErr_SetString(PyExc_TypeError, "Invalid parameters");
    return NULL;
  }

  if(!PyList_Check(meta_list)) {
    PyErr_SetString(PyExc_TypeError, "arg1 must be a list.");
    return NULL;
  }
  if(!PyList_Check(tensor_names_list)) {
    PyErr_SetString(PyExc_TypeError, "tensor_names must be a list.");
    return NULL;
  }
  if(!PyList_Check(tensor_ptrs_list)) {
    PyErr_SetString(PyExc_TypeError, "tensor_ptrs must be a list.");
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

  Py_ssize_t num_tensors = PyList_Size(tensor_names_list);
  for(Py_ssize_t i = 0; i < num_tensors; i++) {
    PyObject *name_item = PyList_GetItem(tensor_names_list, i);
    PyObject *ptr_item = PyList_GetItem(tensor_ptrs_list, i);
    
    const char *name_str = PyUnicode_AsUTF8(name_item);
    if (!name_str) {
      PyErr_Format(PyExc_TypeError, "Failed to convert tensor name %d to string", i);
      return NULL;
    }
    model_tensor_names.push_back(std::string(name_str));
    
    void *ptr = PyLong_AsVoidPtr(ptr_item);
    model_tensor_ptrs.push_back(ptr);
  }

  if (PyUnicode_Check(py_json_path)) {
    const char *json_path = PyUnicode_AsUTF8(py_json_path);
    if (json_path && strlen(json_path) > 0) {
      g_task_graph_json_path = std::string(json_path);
    }
  }

  init_persistent_kernel(meta_tensors, profiler_buffer, my_mpi_rank, num_workers, num_local_schedulers, num_remote_schedulers, max_seq_length, total_num_requests, eos_token_id, model_tensor_names, model_tensor_ptrs);

  Py_RETURN_NONE;
}

static PyObject *init_request_func(PyObject *self, PyObject *args) {
  Py_BEGIN_ALLOW_THREADS
  init_request_resources();
  Py_END_ALLOW_THREADS
  Py_RETURN_NONE;
}

static PyObject *launch_func(PyObject *self, PyObject *args) {
  PyObject *py_stream;
  cudaStream_t stream;
  if (!PyArg_ParseTuple(args, "O", &py_stream)) {
    PyErr_SetString(PyExc_TypeError, "Invalid parameters");
    return NULL;
  }
  stream = (cudaStream_t)PyLong_AsVoidPtr(py_stream);
  launch_persistent_kernel(stream);

  Py_RETURN_NONE;
}

static PyObject *finalize_func(PyObject *self, PyObject *args) {
  finalize_persistent_kernel();

  Py_RETURN_NONE;
}

static PyMethodDef ModuleMethods[] = {
  {"init_func", init_func, METH_VARARGS, "initialize persistent kernel"},
  {"init_request_func", init_request_func, METH_VARARGS, "initialize request resources"},
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
