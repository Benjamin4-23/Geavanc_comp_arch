#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// im-import
#include <cstdint>      // Data types
#include <iostream>     // File operations
// #define M 512       // Lenna width
// #define N 512       // Lenna height
#define M 941       // VR width
#define N 704       // VR height
#define C 3         // Colors
#define OFFSET 15   // Header length

uint8_t* get_image_array(void){
    /*
     * Get the data of an (RGB) image as a 1D array.
     * 
     * Returns: Flattened image array.
     * 
     * Noets:
     *  - Images data is flattened per color, column, row.
     *  - The first 3 data elements are the RGB components
     *  - The first 3*M data elements represent the firts row of the image
     *  - For example, r_{0,0}, g_{0,0}, b_{0,0}, ..., b_{0,M}, r_{1,0}, ..., b_{b,M}, ..., b_{N,M}
     * 
     */        
    // Try opening the file
    FILE *imageFile;
    imageFile=fopen("./input_image.ppm","rb");
    if(imageFile==NULL){
        perror("ERROR: Cannot open output file");
        exit(EXIT_FAILURE);
    }
    
    // Initialize empty image array
    uint8_t* image_array = (uint8_t*)malloc(M*N*C*sizeof(uint8_t)+OFFSET);
    
    // Read the image
    fread(image_array, sizeof(uint8_t), M*N*C*sizeof(uint8_t)+OFFSET, imageFile);
    
    // Close the file
    fclose(imageFile);
        
    // Move the starting pointer and return the flattened image array
    return image_array + OFFSET;
}

void save_image_array(uint8_t* image_array){
    /*
     * Save the data of an (RGB) image as a pixel map.
     * 
     * Parameters:
     *  - param1: The data of an (RGB) image as a 1D array
     * 
     */            
    // Try opening the file
    FILE *imageFile;
    imageFile=fopen("./output_image.ppm","wb");
    if(imageFile==NULL){
        perror("ERROR: Cannot open output file");
        exit(EXIT_FAILURE);
    }
    
    // Configure the file
    fprintf(imageFile,"P6\n");               // P6 filetype
    fprintf(imageFile,"%d %d\n", M, N);      // dimensions
    fprintf(imageFile,"255\n");              // Max pixel
    
    // Write the image
    fwrite(image_array, 1, M*N*C, imageFile);
    
    // Close the file
    fclose(imageFile);
}







__global__ void averageGrayscaleCoalesced(uint8_t* image, int width, int height, uint8_t* new_image) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    for (int j = 0; j <3; j++) {
        sum += image[(idx*3)+j];
    }
    int avarage = sum/3;
    for (int j = 0; j <3; j++) {
        new_image[(idx*3)+j] = avarage;
    }

}

__global__ void averageGrayscaleUncoalesced(uint8_t* image, int width, int height, uint8_t* new_image) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate the grayscale value of the pixel
    int sum = 0;
    for (int pixel_index = idx; pixel_index < height*width*3; pixel_index+=width*height) {
        sum += image[pixel_index];
    }
    int average = sum / 3;

    // Set the grayscale value for all channels of the pixel in the new image
    for (int j = 0; j < 3; j++) {
        new_image[idx*3 + j] = average;
    }
}











int main (void) {
    // Part 1


    // timing events
    cudaEvent_t start, stop;
    uint8_t* h_image_array_coalesced = get_image_array();

    // [RR...RGG...GBB...B] array maken
    uint8_t* h_image_array_uncoalesced = (uint8_t*)malloc(M*N*C*sizeof(uint8_t));
    int index = 0;
    for (int i = 0; i < M*N*C; i+=3) {
        h_image_array_uncoalesced[index] = h_image_array_coalesced[i];
        h_image_array_uncoalesced[index+(M*N)] = h_image_array_coalesced[i+1];
        h_image_array_uncoalesced[index+(M*N*2)] = h_image_array_coalesced[i+2];
        index += 1;
    }


    uint8_t* d_image_array;
    //cudaMalloc(&d_image_array, M*N*C*sizeof(uint8_t));
    //cudaMemcpy(d_image_array, h_image_array_coalesced, M*N*C*sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMalloc(&d_image_array, M*N*C*sizeof(uint8_t));
    cudaMemcpy(d_image_array, h_image_array_uncoalesced, M*N*C*sizeof(uint8_t), cudaMemcpyHostToDevice);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    uint8_t* h_new_image_array = (uint8_t*)malloc(M*N*C*sizeof(uint8_t));
    uint8_t* d_new_image_array;
    cudaMalloc(&d_new_image_array, M*N*C*sizeof(uint8_t));

    for (int i = 1; i < 200; i++) {
        float mean = 0;
        for (int h = 0; h < 10; h++) {
            int blockSize = i;
            int numBlocks = ((M * N)+blockSize - 1) / blockSize;
            cudaEventRecord(start, 0);
            //averageGrayscaleCoalesced<<<numBlocks, blockSize>>>(d_image_array, M, N, d_new_image_array);
            averageGrayscaleUncoalesced<<<numBlocks, blockSize>>>(d_image_array, M,N, d_new_image_array);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float time;
            cudaEventElapsedTime(&time, start, stop);
            mean += time;
        }
        std::cout << (mean/10) << "\n";
    }
    cudaMemcpy(h_new_image_array, d_new_image_array, M*N*C*sizeof(uint8_t), cudaMemcpyDeviceToHost);
    save_image_array(h_new_image_array);
    cudaFree(&d_new_image_array);
    cudaFree(&d_image_array);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_new_image_array);

    // Part 2




    return 0;
}


