#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <cuda_runtime.h>

#define NUM_ELEMENTS 4

__global__ void gpu_reduction(int* arr, int elements, int* result, int operation) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned usefull_threads = (elements+1)/2;

    // Shared memory for storing partial results
    extern __shared__ int sdata[];
    if (tid < usefull_threads) {
        sdata[tid*2] = arr[tid * 2];
        if ((tid*2 + 2) <= elements) sdata[(tid*2) + 1] = arr[(tid * 2) + 1];
    }
    __syncthreads();

    // How many steps
    int waarde = 1;
    int count = 0;
    while (waarde < elements) {
        waarde = waarde*2;
        count++;
    }


    if (operation == 0) { // Sum
        for (int i = 0; i < count; i++) {
            if (tid < usefull_threads) {
                sdata[tid] = ((tid*2 + 2) <= elements) ? sdata[tid*2] + sdata[tid*2+1] : sdata[tid*2];
                usefull_threads = (usefull_threads+1)/2;
            }
            __syncthreads();
        }

        if (tid == 0) {
            result[0] = sdata[tid];
        }
    } else if (operation == 1) { // Product
        for (int i = 0; i < count; i++) {
            if (tid < usefull_threads) {
                sdata[tid] = ((tid*2 + 2) <= elements) ? sdata[tid*2] * sdata[tid*2+1] : sdata[tid*2];
                usefull_threads = (usefull_threads+1)/2;
            }
            __syncthreads();
        }

        if (tid == 0) {
            result[0] = sdata[tid];
        }
        result[1] = 14;
    } else if (operation == 2) { // Minimum
        for (int i = 0; i < count; i++) {
            if (tid < usefull_threads) {
                sdata[tid] = ((tid*2 + 2) <= elements) ? ((sdata[tid*2] < sdata[tid*2+1]) ? sdata[tid*2] : sdata[tid*2 +1]) : sdata[tid*2];
                usefull_threads = (usefull_threads+1)/2;
            }
            __syncthreads();
        }

        if (tid == 0) {
            result[0] = sdata[0];
        }
        
    } else if (operation == 3) { // Maximum
        for (int i = 0; i < count; i++) {
            if (tid < usefull_threads) {
                sdata[tid] = ((tid*2 + 2) <= elements) ? ((sdata[tid*2] > sdata[tid*2+1]) ? sdata[tid*2] : sdata[tid*2 +1]) : sdata[tid*2];
                usefull_threads = (usefull_threads+1)/2;
            }
            __syncthreads();
        }

        if (tid == 0) {
            result[0] = sdata[0];
        }
    }
}

int main() {
    double total = 0;
    for (int k = 0; k < 1000; k++) {
        // timing via cpu to time entire execution
        auto start_cpu = std::chrono::steady_clock::now();

        int* h_arr_1 = new int[NUM_ELEMENTS];
        int* h_arr_2 = new int[NUM_ELEMENTS];
        int* h_arr_3 = new int[NUM_ELEMENTS];
        int* h_arr_4 = new int[NUM_ELEMENTS];

        srand(time(nullptr) + k);
        for (int i = 0; i < NUM_ELEMENTS; ++i) {
            h_arr_1[i] = rand() % 10 +1;
        }
        for (int i = 0; i < NUM_ELEMENTS; ++i) {
            h_arr_2[i] = rand() % 10 +1;
        }
        for (int i = 0; i < NUM_ELEMENTS; ++i) {
            h_arr_3[i] = rand() % 10 +1;
        }
        for (int i = 0; i < NUM_ELEMENTS; ++i) {
            h_arr_4[i] = rand() % 10 +1;
        }

        // for (int i = 0; i < NUM_ELEMENTS; ++i) {
        //     std::cout << h_arr_1[i] << ",";
        // }
        // std::cout << "\n";
        // for (int i = 0; i < NUM_ELEMENTS; ++i) {
        //     std::cout << h_arr_2[i] << ",";
        // }
        // std::cout << "\n";
        // for (int i = 0; i < NUM_ELEMENTS; ++i) {
        //     std::cout << h_arr_3[i] << ",";
        // }
        // std::cout << "\n";
        // for (int i = 0; i < NUM_ELEMENTS; ++i) {
        //     std::cout << h_arr_4[i] << ",";
        // }
        // std::cout << "\n";


        int* d_arr[4];
        cudaMalloc(&d_arr[0], NUM_ELEMENTS * sizeof(int));
        cudaMalloc(&d_arr[1], NUM_ELEMENTS * sizeof(int));
        cudaMalloc(&d_arr[2], NUM_ELEMENTS * sizeof(int));
        cudaMalloc(&d_arr[3], NUM_ELEMENTS * sizeof(int));

        cudaMemcpy(d_arr[0], h_arr_1, NUM_ELEMENTS * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_arr[1], h_arr_2, NUM_ELEMENTS * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_arr[2], h_arr_3, NUM_ELEMENTS * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_arr[3], h_arr_4, NUM_ELEMENTS * sizeof(int), cudaMemcpyHostToDevice);

        int* d_result;

        for (int i = 0; i < 4; ++i) {
            cudaMalloc(&d_result, sizeof(int));
            gpu_reduction<<<1, (NUM_ELEMENTS + 1) / 2, NUM_ELEMENTS * sizeof(int)>>>(d_arr[i], NUM_ELEMENTS, d_result, i);


            int gpu_result_reduction;
            cudaMemcpy(&gpu_result_reduction, d_result, sizeof(int), cudaMemcpyDeviceToHost);

            std::string operation;
            if (i == 0)
                operation = "sum";
            else if (i == 1)
                operation = "product";
            else if (i == 2)
                operation = "minimum";
            else
                operation = "maximum";

            std::cout << " GPU " << operation << " " << gpu_result_reduction << "\n";

            cudaFree(d_result);
        }

        auto end_cpu = std::chrono::steady_clock::now();
        std::chrono::duration<double> cpu_time = end_cpu - start_cpu;
        if (k >=  900) total+= cpu_time.count()*1000;
        std::cout << "execution time: " << cpu_time.count()*1000 << " ms\n";

        delete[] h_arr_1;
        delete[] h_arr_2;
        delete[] h_arr_3;
        delete[] h_arr_4;

        for (int i = 0; i < 4; ++i) {
            cudaFree(d_arr[i]);
        }
    }

    std::cout << "Mean execution time: " << total/100 << " ms\n";
    

    return 0;
}
