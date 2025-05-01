import mirage
import os
import threading
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world_size = comm.Get_size()

print(f"MPI: rank({rank}) world_size({world_size})")
print(f"process id: {os.getpid()} thread id: {threading.get_ident()}")

kernel = mirage.PersistentKernel(file_path="/home/ubuntu/mirage_cpp/debug_build/test.cu", mpi_rank=rank, num_workers=106, num_local_schedulers=6, num_remote_schedulers=2, use_nvshmem=True)

kernel()
