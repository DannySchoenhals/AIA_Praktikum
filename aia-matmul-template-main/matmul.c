/*
* Prakitikum 1 – Matrix Multiplication on CPU
 * ============================================
 * AI Accelerators (AIA) – Lab Assignment
 *
 * Your task is to progressively optimize this naive C implementation
 * of matrix multiplication (C = A * B) through the steps below.
 * Read README.md carefully before you start!
 *
 * Build:  make
 * Run:    ./matmul <size>      (e.g. ./matmul 512)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tgmath.h>
#include <time.h>
#include <omp.h>
#include <pthread.h>

#define MODE 4 // 1: naive, 2: loop order, 3: tiling, 4: multithreading
#define NUM_THREADS 8
const int num_iterations = 4;
#define JB 32 //Tile size divides matrix size

// ============================================================================
// IMPLEMENTATION 1: NAIVE MATRIX MULTIPLICATION
// ============================================================================
void matmul_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// ============================================================================
// IMPLEMENTATION 2: -03 -ffmastmath, does loop unrolling and vectorization, reorder k,j
// ============================================================================
void matmul_looporder(const float* A, const float* B, float* C, int M, int N, int K) {
    
    for (int i = 0; i < M; i++) {
        memset(&C[i * N], 0, N * sizeof(float));
        for (int k = 0; k < K; k++) {
            float a_ik = A[i * K + k];
            for (int j = 0; j < N; j++) {
                C[i * N + j] += a_ik * B[k * N + j];
            }
        }
    }
}

// ============================================================================
// IMPLEMENTATION 3: Tiling
// ============================================================================

void matmul_looptiling(const float* A, const float* B, float* C, int M, int N, int K) {
    memset(C, 0, M * N * sizeof(float));
    for (int ii = 0; ii < M; ii += JB) {
        for (int i = ii; i < ii + JB; i++) {
            for (int kk = 0; kk < K; kk += JB) {
                for (int k = kk; k < kk + JB; k++) {
                    float a_ik = A[i * K + k];
                    for (int jj = 0; jj < N; jj += JB) {
                        for (int j = jj; j < jj + JB; j++) {
                            C[i * N + j] += a_ik * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}


// ============================================================================
// IMPLEMENTATION 4: Multithreading
// ============================================================================

// Previous OMP-based approach (kept for reference):
// void matmul_parallel_ikj(const float* A, const float* B, float* C,
//                          int M, int N, int K) {
//     memset(C, 0, M * N * sizeof(float));
//     #pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
//     for (int i = 0; i < M; i++) {
//         for (int k = 0; k < K; k++) {
//             float a_ik = A[i * K + k];
//             for (int j = 0; j < N; j++) {
//                 C[i * N + j] += a_ik * B[k * N + j];
//             }
//         }
//     }
// }

typedef struct {
    const float* A;
    const float* B;
    float* C;
    int M, N, K;
    int start_row;
    int end_row;
} ThreadArgs;

void* thread_worker(void* arg) {
    ThreadArgs* t = (ThreadArgs*)arg;
    for (int i = t->start_row; i < t->end_row; i++) {
        for (int k = 0; k < t->K; k++) {
            float a_ik = t->A[i * t->K + k];
            for (int j = 0; j < t->N; j++) {
                t->C[i * t->N + j] += a_ik * t->B[k * t->N + j];
            }
        }
    }
    return NULL;
}

void matmul_parallel_ikj(const float* A, const float* B, float* C,
                         int M, int N, int K) {
    memset(C, 0, M * N * sizeof(float));

    pthread_t threads[NUM_THREADS];
    ThreadArgs args[NUM_THREADS];
    int rows_per_thread = M / NUM_THREADS;

    for (int t = 0; t < NUM_THREADS; t++) {
        args[t].A = A; args[t].B = B; args[t].C = C;
        args[t].M = M; args[t].N = N; args[t].K = K;
        args[t].start_row = t * rows_per_thread;
        args[t].end_row = (t == NUM_THREADS - 1) ? M : (t + 1) * rows_per_thread;
        pthread_create(&threads[t], NULL, thread_worker, &args[t]);
    }
    for (int t = 0; t < NUM_THREADS; t++) {
        pthread_join(threads[t], NULL);
    }
}

// ============================================================================
// Utility functions: Init Matrix, Benchmarking, Calculate Gflops
// ============================================================================
void initialize_matrix(float *matrix, int rows, int cols){
    for (int i = 0; i < rows * cols; i++){
        matrix[i] = rand() % 100;
    }
}

double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

double calculate_gflops(int M, int N, int K, double total_time) {
    double flops = 2.0 * M * N * K;
    double gflops = (flops / ((total_time) / 1000.0)) / 1e9;
    return gflops;
}

int verify_result(const float* C_ref, const float* C_test, int M, int N, float tolerance) {
    for (int i = 0; i < M * N; i++) {
        if (fabs(C_ref[i] - C_test[i]) > tolerance) {
            printf("Mismatch at index %d: ref=%f, test=%f\n", i, C_ref[i], C_test[i]);
            return 0;
        }
    }
    return 1;
}

typedef void (*matmul_fn)(const float* A, const float* B, float* C, int M, int N, int K);

float benchmark(matmul_fn matmul, const float* A, const float *B, float *C, int M, int N, int K)
{
    matmul(A, B, C, M, N, K); //Warmup
    double total_time = 0.0;
    for (int i = 0; i < num_iterations; i++) {
        double start = get_time_ms();
        matmul(A, B, C, M, N, K);
        __asm__ __volatile__("" : "+m" (C[0]) : : "memory");
        double end = get_time_ms();
        total_time += end - start;
    }

    return total_time/num_iterations;
}

// ============================================================================
// Main: Verify results and performance benchmakrk
// ============================================================================
int main(int argc, char *argv[]) {
    srand(42);
    printf("MatMul Benchmark: Square Matrix\n");

    int sizes[] = {2048}; // Adjust sizes as needed   
    int n = sizeof(sizes) / sizeof(sizes[0]);

    printf("%-8s %-15s %-15s %-15s %-15s\n", "Size", "Naive", "Reordered", "Tiled", "Parallel");
    printf("%-8s %-15s %-15s %-15s %-15s\n", "----", "-----", "---------", "-----", "--------");

    for (int i = 0; i < n; i++) {
        int M = sizes[i], N = M, K = M;

        float *A = (float *)malloc(M * K * sizeof(float));
        float *B = (float *)malloc(K * N * sizeof(float));
        float *C = (float *)malloc(M * N * sizeof(float));

        initialize_matrix(A, M, K);
        initialize_matrix(B, K, N);

        double g_naive = 0.0, g_reorder = 0.0, g_blocking = 0.0, g_parallel = 0.0;

        // --- 1. Naive ---
        if (MODE == 1) {
            memset(C, 0, M * N * sizeof(float));
            float t_naive = benchmark(matmul_naive, A, B, C, M, N, K);
            g_naive = calculate_gflops(M, N, K, t_naive);
        }

        // --- 2. Reordered ---
        if (MODE == 2) {
            memset(C, 0, M * N * sizeof(float));
            float t_reorder = benchmark(matmul_looporder, A, B, C, M, N, K);
            g_reorder = calculate_gflops(M, N, K, t_reorder);
        }

        // --- 3. Tiled ---
        if (MODE == 3) {
            memset(C, 0, M * N * sizeof(float));
            float t_blocking = benchmark(matmul_looptiling, A, B, C, M, N, K);
            g_blocking = calculate_gflops(M, N, K, t_blocking);
        }

        // --- 4. Parallel ---
        if (MODE == 4) {
            memset(C, 0, M * N * sizeof(float));
            float t_parallel = benchmark(matmul_parallel_ikj, A, B, C, M, N, K);
            g_parallel = calculate_gflops(M, N, K, t_parallel);
        }

        printf("%d\t%.2f GFLOPS\t%.2f GFLOPS\t%.2f GFLOPS\t%.2f GFLOPS\n",
               M, g_naive, g_reorder, g_blocking, g_parallel);

        free(A); free(B); free(C);
    }

    return 0;
}