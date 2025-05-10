
namespace mirage {
    namespace runtime {

 
 __device__ static inline void mma_m16n16k16_bf16bf16bf32(float* C, uint32_t* A,
    uint32_t* B, float *D) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
            "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));

    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(D[4]), "=f"(D[5]), "=f"(D[6]), "=f"(D[7])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]),
            "f"(C[4]), "f"(C[5]), "f"(C[6]), "f"(C[7]));

    // asm volatile(
    //     "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
    //     "{%0,  %1,  %2,  %3},"
    //     "{%4,  %5,  %6,  %7},"
    //     "{%8,  %9},"
    //     "{%10, %11, %12, %13};\n"
    //     : "=f"(C[4]), "=f"(C[5]), "=f"(C[6]), "=f"(C[7])
    //     : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "f"(0.f), "f"(0.f),
    //         "f"(0.f), "f"(0.f));
 }
    }
}