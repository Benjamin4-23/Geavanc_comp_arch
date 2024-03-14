#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

#define MIN_SIZE 16 // Minimum matrix size
#define MAX_SIZE 88 // Maximum matrix size
#define SIZE_INCREMENT 2 // Increment size for matrix size
#define MIN_BLOCK_SIZE 32// Minimum block size
#define MAX_BLOCK_SIZE 33 // Maximum block size
#define BLOCK_SIZE_INCREMENT 8 // Increment size for block size

#define TILE_SIZE 32

__global__ void matrixMultiplicationGlobalOnly(const int *A, const int *B, int *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        int sum = 0;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void matrixMultiplicationGlobalAndShared(const int *A, const int *B, int *C, int N) {
    __shared__ int As[TILE_SIZE][TILE_SIZE];
    __shared__ int Bs[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int sum = 0;
    for (int m = 0; m < (N + TILE_SIZE - 1) / TILE_SIZE; ++m) {
        if (row < N && m * TILE_SIZE + tx < N) {
            As[ty][tx] = A[row * N + m * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0;
        }
        if (col < N && m * TILE_SIZE + ty < N) {
            Bs[ty][tx] = B[(m * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0;
        }
        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// A and B are constants to run global and global_and_shared with bigger sizes then 88x88, comment these 2 constants and the kernel below
__constant__ int const_B[MAX_SIZE*MAX_SIZE]; 
__constant__ int const_A[MAX_SIZE*MAX_SIZE];

__global__ void matrixMultiplicationGlobalAndConstant(const int *A, int *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        int sum = 0;
        for (int k = 0; k < N; ++k) {
            sum += const_A[row * N + k] * const_B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void initializeMatrix(int *matrix, int size) {
    for (int i = 0; i < size * size; ++i) {
        matrix[i] = rand() % 10; // Random values between 0 and 9
    }
}

int main() {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    for (int size = MIN_SIZE; size <= MAX_SIZE; size += SIZE_INCREMENT) {
        for (int block_size = MIN_BLOCK_SIZE; block_size <= MAX_BLOCK_SIZE; block_size += BLOCK_SIZE_INCREMENT) {
            dim3 block_dim(block_size, block_size);
            dim3 grid_dim((size + block_size - 1) / block_size, (size + block_size - 1) / block_size);

            int *h_A, *h_B, *h_C;
            int *d_A, *d_B, *d_C;
            size_t matrix_size = size * size * sizeof(int);

            h_A = (int*)malloc(matrix_size);
            h_B = (int*)malloc(matrix_size);
            h_C = (int*)malloc(matrix_size);

            cudaMalloc(&d_A, matrix_size);
            cudaMalloc(&d_B, matrix_size);
            cudaMalloc(&d_C, matrix_size);

            initializeMatrix(h_A, size);
            initializeMatrix(h_B, size);

            
            cudaMemcpy(d_A, h_A, matrix_size, cudaMemcpyHostToDevice);
            cudaMemcpy(d_B, h_B, matrix_size, cudaMemcpyHostToDevice);

            
            cudaEventRecord(start, 0);

            // Launch the kernel for each memory configuration
            //matrixMultiplicationGlobalOnly<<<grid_dim, block_dim>>>(d_A, d_B, d_C, size);
            matrixMultiplicationGlobalAndShared<<<grid_dim, block_dim>>>(d_A, d_B, d_C, size);
            //cudaMemcpyToSymbol(const_B, d_B, matrix_size); // Copy matrix B to constant memory
            //matrixMultiplicationGlobalAndConstant<<<grid_dim, block_dim>>>(d_A, d_C, size);

            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);

            float elapsed_time;
            cudaEventElapsedTime(&elapsed_time, start, stop);

            std::cout << elapsed_time << std::endl;

            free(h_A);
            free(h_B);
            free(h_C);
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_C);
        }
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
