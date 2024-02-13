#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <chrono>

using namespace std::chrono;


__global__ void arrayFlipFunction(int *arr) {
    // thread id is aantal blokken * aantal threads per blok + threadid binnen block
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 999 / 2) {
        int temp = arr[idx];
        arr[idx] = arr[999 - idx - 1];
        arr[999 - idx - 1] = temp;
    }
}

int main() {
    // Array op CPU aanmaken
    int *h_arr = new int[999];
    for (int i = 0; i < 999; ++i) {
        h_arr[i] = i;
    }

    // Array op GPU aanmaken
    int *d_arr;
    cudaMalloc(&d_arr, 999 * sizeof(int));
    cudaMemcpy(d_arr, h_arr, 999 * sizeof(int), cudaMemcpyHostToDevice);

    // Uitvoeren
    auto start = high_resolution_clock::now();
    arrayFlipFunction<<<1, 1000>>>(d_arr);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << "Time taken by GPU: "
         << duration.count() << " microseconds" << std::endl;

    // Data terug naar CPU memory kopieren
    cudaMemcpy(h_arr, d_arr, 999 * sizeof(int), cudaMemcpyDeviceToHost);


    // Zelfde op CPU uitvoeren
    int *c_arr = new int[999];
    for (int i = 0; i < 999; ++i) {
        c_arr[i] = i;
    }

    start = high_resolution_clock::now();
    for (int i = 0; i < 999; ++i) {
        if (i < 999 / 2) {
            int temp = c_arr[i];
            c_arr[i] = c_arr[999 - i - 1];
            c_arr[999 - i - 1] = temp;
        }
    }
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    std::cout << "Time taken by CPU: "
         << duration.count() << " microseconds" << std::endl;

    // Print the flipped array
    //std::cout << "Array na bewerking:\n";
    //for (int i = 0; i < 999; ++i) {
    //    std::cout << h_arr[i] << " ";
    //}
    //std::cout << std::endl;

    // Geheugen terug vrijmaken
    delete[] h_arr;
    cudaFree(d_arr);

    return 0;
}
