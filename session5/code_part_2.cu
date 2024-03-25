#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <ctime>
#include <chrono>

#define NUM_ELEMENTS 10000

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
    int randint = 23;
    double total = 0;
    for (int k = 0; k < 1000; k++) {
        //start timing
        auto start_cpu = std::chrono::steady_clock::now();

        // Create CUDA streams
        cudaStream_t stream[4];
        for (int i = 0; i < 4; ++i) {
            cudaStreamCreate(&stream[i]);
        }
        
        


        //1
        int* h_input = new int[NUM_ELEMENTS];
        int* h_output = new int[1];
        srand(time(nullptr) + randint);
        randint++;
        for (int i = 0; i < NUM_ELEMENTS; ++i) {
            h_input[i] = rand() % 10 + 1;
            //std::cout << h_input[i] << ",";
        }
        //std::cout << std::endl;
        int* d_input;
        int* d_result;
        cudaMalloc(&d_input, NUM_ELEMENTS * sizeof(int));
        cudaMalloc(&d_result, sizeof(int));
        cudaMemcpyAsync(d_input, h_input, NUM_ELEMENTS * sizeof(int), cudaMemcpyHostToDevice, stream[0]);
        gpu_reduction<<<1, (NUM_ELEMENTS + 1) / 2, NUM_ELEMENTS * sizeof(int), stream[0]>>>(d_input, NUM_ELEMENTS, d_result, 0);
        cudaMemcpyAsync(h_output, d_result, sizeof(int), cudaMemcpyDeviceToHost, stream[0]);


        //2
        int* h_input_2 = new int[NUM_ELEMENTS];
        int* h_output_2 = new int[1];
        srand(time(nullptr) + randint + 1);
        for (int i = 0; i < NUM_ELEMENTS; ++i) {
            h_input_2[i] = rand() % 10 + 1;
            //std::cout << h_input_2[i] << ",";
        }
        //std::cout << std::endl;
        int* d_input_2;
        int* d_result_2;
        cudaMalloc(&d_input_2, NUM_ELEMENTS * sizeof(int));
        cudaMalloc(&d_result_2, sizeof(int));
        cudaMemcpyAsync(d_input_2, h_input_2, NUM_ELEMENTS * sizeof(int), cudaMemcpyHostToDevice, stream[1]);
        gpu_reduction<<<1, (NUM_ELEMENTS + 1) / 2, NUM_ELEMENTS * sizeof(int), stream[1]>>>(d_input_2, NUM_ELEMENTS, d_result_2, 1);
        cudaMemcpyAsync(h_output_2, d_result_2, sizeof(int), cudaMemcpyDeviceToHost, stream[1]);


        //3
        int* h_input_3 = new int[NUM_ELEMENTS];
        int* h_output_3 = new int[1];
        srand(time(nullptr) + randint + 2);
        for (int i = 0; i < NUM_ELEMENTS; ++i) {
            h_input_3[i] = rand() % 10 + 1;
            //std::cout << h_input_3[i] << ",";
        }
        //std::cout << std::endl;
        int* d_input_3;
        int* d_result_3;
        cudaMalloc(&d_input_3, NUM_ELEMENTS * sizeof(int));
        cudaMalloc(&d_result_3, sizeof(int));
        cudaMemcpyAsync(d_input_3, h_input_3, NUM_ELEMENTS * sizeof(int), cudaMemcpyHostToDevice, stream[2]);
        gpu_reduction<<<1, (NUM_ELEMENTS + 1) / 2, NUM_ELEMENTS * sizeof(int), stream[2]>>>(d_input_3, NUM_ELEMENTS, d_result_3, 2);
        cudaMemcpyAsync(h_output_3, d_result_3, sizeof(int), cudaMemcpyDeviceToHost, stream[2]);

        //4
        int* h_input_4 = new int[NUM_ELEMENTS];
        int* h_output_4 = new int[1];
        srand(time(nullptr) + randint + 3);
        for (int i = 0; i < NUM_ELEMENTS; ++i) {
            h_input_4[i] = rand() % 10 + 1;
            //std::cout << h_input_4[i] << ",";
        }
        //std::cout << std::endl;
        int* d_input_4;
        int* d_result_4;
        cudaMalloc(&d_input_4, NUM_ELEMENTS * sizeof(int));
        cudaMalloc(&d_result_4, sizeof(int));
        cudaMemcpyAsync(d_input_4, h_input_4, NUM_ELEMENTS * sizeof(int), cudaMemcpyHostToDevice, stream[3]);
        gpu_reduction<<<1, (NUM_ELEMENTS + 1) / 2, NUM_ELEMENTS * sizeof(int), stream[3]>>>(d_input_4, NUM_ELEMENTS, d_result_4, 3);
        cudaMemcpyAsync(h_output_4, d_result_4, sizeof(int), cudaMemcpyDeviceToHost, stream[3]);




        // Print the output array
        cudaStreamSynchronize(stream[0]);
        cudaStreamSynchronize(stream[1]);
        cudaStreamSynchronize(stream[2]);
        cudaStreamSynchronize(stream[3]);
        //std::cout << "Sum " << h_output[0] << "  Product " << h_output_2[0] << "  Minimum " << h_output_3[0] << "  Maximum " << h_output_4[0];
        //std::cout << std::endl;



        //end timing
        auto end_cpu = std::chrono::steady_clock::now();
        std::chrono::duration<double> cpu_time = end_cpu - start_cpu;
        if (k >=  900) total+= cpu_time.count()*1000;
        //std::cout << "execution time: " << cpu_time.count()*1000 << " ms\n";




        cudaFree(d_input);
        cudaFree(d_result);

        cudaFree(d_input_2);
        cudaFree(d_result_2);

        cudaFree(d_input_3);
        cudaFree(d_result_3);

        cudaFree(d_input_4);
        cudaFree(d_result_4);

        // Free host memory
        delete[] h_input;
        delete[] h_output;

        delete[] h_input_2;
        delete[] h_output_2;

        delete[] h_input_3;
        delete[] h_output_3;

        delete[] h_input_4;
        delete[] h_output_4;



        // Destroy CUDA streams
        for (int i = 0; i < 4; ++i) {
            cudaStreamDestroy(stream[i]);
        }
    }
    
    std::cout << "Mean execution time: " << total/100 << " ms\n"; // 55ms met 4, 1.28ms met 10k

    return 0;
}
