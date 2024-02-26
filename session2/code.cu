#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <cuda_runtime.h>

#define NUM_ELEMENTS 8

int cpu_max(const int* arr, int size) {
    int max = arr[0];
    for (int i = 1; i < size; ++i) {
        if (arr[i] > max) {
            max = arr[i];
        }
    }
    return max;
}

__global__ void gpu_max_atomic(const int* arr, int* result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    atomicMax(result, arr[idx]);
}

__global__ void gpu_max_reduction(int* arr, int* result) {
    unsigned int tid = threadIdx.x;
    unsigned int offset = blockIdx.x * blockDim.x;
    int nuttige_threads = NUM_ELEMENTS /2;

    // How many steps
    int waarde = 1;
    int count = 0;
    while (waarde < NUM_ELEMENTS) {
        waarde = waarde*2;
        count++;
    }

    for (int step_number = 1; step_number <= count; step_number++) {
        if (tid < nuttige_threads ) {
            if (arr[offset+tid] < arr[(offset+(nuttige_threads*2)-1)-tid]) {
                arr[offset+tid] = arr[(offset+(nuttige_threads*2)-1)-tid];
            }
        }
        nuttige_threads = nuttige_threads/2;
        __syncthreads();
    }
    
    if (tid == 0) {
        result[0] = arr[offset+tid];
    }
}



int main() {
    int* h_arr = new int[NUM_ELEMENTS];

    srand(time(nullptr));
    for (int i = 0; i < NUM_ELEMENTS; ++i) {
        h_arr[i] = rand() % 1000;
    }

    // CPU
    auto start_cpu = std::chrono::steady_clock::now();
    int cpu_result = cpu_max(h_arr, NUM_ELEMENTS);
    auto end_cpu = std::chrono::steady_clock::now();
    std::chrono::duration<double> cpu_time = end_cpu - start_cpu;
    std::cout << "CPU max: " << cpu_result << ", Time: " << cpu_time.count()*1000 << " ms\n";








    int* d_arr;
    cudaMalloc(&d_arr, NUM_ELEMENTS * sizeof(int));
    cudaMemcpy(d_arr, h_arr, NUM_ELEMENTS * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);



    // Atomic
    int* d_result;
    cudaMalloc(&d_result, sizeof(int));
    cudaEventRecord(start, 0);
    gpu_max_atomic<<<1, NUM_ELEMENTS>>>(d_arr, d_result);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float atomic_time;
    cudaEventElapsedTime(&atomic_time, start, stop);
    int gpu_result_atomic;
    cudaMemcpy(&gpu_result_atomic, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "GPU max (atomic): " << gpu_result_atomic << ", Time: " << atomic_time << " ms\n";
    cudaFree(d_result);




    // Reduction
    cudaMalloc(&d_result, sizeof(int));
    cudaEventRecord(start, 0);
    gpu_max_reduction<<<(NUM_ELEMENTS+255)/256,256>>>(d_arr, d_result);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float reduction_time;
    cudaEventElapsedTime(&reduction_time, start, stop);
    int gpu_result_reduction;
    cudaMemcpy(&gpu_result_reduction, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "GPU max (reduction): " << gpu_result_reduction << ", Time: " << reduction_time << " ms\n";



    delete[] h_arr;
    cudaFree(d_arr);
    cudaFree(d_result);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
