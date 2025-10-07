This project aims to compile a model into a megakernel, a single GPU kernel that encapsulates all computation and communication.
There is exactly 1 thread block per Streaming Multiprocessor (SM). Tasks are device functions that are called by a single thread block. 
Shared memory is dynamically allocated by each task, with the kernel being launched with the maximum shared memory size possible.