#include "persistent_kernel.cuh"

static void _init_persistent_kernel(std::vector<TaskDesc> &all_tasks,
                                    std::vector<EventDesc> &all_events,
                                    std::vector<TaskId> &first_tasks,
                                    int num_gpus,
                                    int my_gpu_id) {}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::vector<void const *> input_tensors;
  std::vector<void *> output_tensors;
  void *buf;
  init_persistent_kernel(input_tensors, output_tensors, buf, 106, 6, 2);
  MPI_Finalize();
}
