#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <limits>
#include <algorithm>
#include <cuda_runtime.h>

const int BLOCK_SIZE = 1024;

struct OHLCData {
  std::string timestamp;
  double open;
  double high;
  double low;
  double close;
  int volume;
};

struct StockData {
  std::string name;
  std::string file_path;
};

void readCSV(const std::string& filename, std::vector<OHLCData>& data) {
    std::ifstream file(filename);
    std::string line;
  
    while (std::getline(file, line)) {
        size_t spacePos = line.find(' ');
        if (spacePos != std::string::npos) {
            line[spacePos] = '_';  // Replace space with '_'
        }
        std::istringstream iss(line);
        OHLCData entry;
        char comma;
        std::getline(iss, entry.timestamp, ',');
        if (!(iss >> entry.open >> comma >> entry.high >> comma >> entry.low >> comma >> entry.close >> comma >> entry.volume)) {
            std::cerr << "Failed to parse line: " << line << std::endl;
            continue;
        }
        data.push_back(entry);
    }
    file.close();
}




__global__ void calculateReturns(double* open, double* close, double* returns, int numRecords) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx < numRecords) {
        double returnVal = (close[idx] - open[idx]) / open[idx];
        returns[idx] = returnVal*100; 
    }
}

__global__ void calculateMeans(double* returns1, double* returns2, int numReturns, int numBlocks, double* intermediate_values, double* means) { 
    int tid = threadIdx.x;
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    extern __shared__ double shared_data[];
    // Each thread initializes its own shared memory value to 0
    shared_data[tid] = 0.0;
    shared_data[tid + BLOCK_SIZE] = 0.0;

    if (tid == 0) {
        if (idx == 0) intermediate_values[0] = 0.0;
        else intermediate_values[(idx/BLOCK_SIZE)] = 0.0;
        if (idx == 0) intermediate_values[numBlocks] = 0.0;
        intermediate_values[(idx/BLOCK_SIZE) + numBlocks] = 0.0;
    }

    double mean_x = 0.0;
    double mean_y = 0.0;
    double total_sum_x = 0.0;
    double total_sum_y = 0.0;

    __syncthreads();
    
    if (idx < numReturns) {
        // Each thread loads 2 returns (at the threads index) into shared memory
        double x = returns1[idx];
        double y = returns2[idx];
        shared_data[tid] = x;
        shared_data[tid + BLOCK_SIZE] = y;

        // Summarize partial sums across threads within a block
        for (int stride = BLOCK_SIZE / 2; stride >= 1; stride /= 2) {
            __syncthreads(); 
            if (tid < stride) {
                shared_data[tid] += shared_data[tid + stride];
                shared_data[tid + BLOCK_SIZE] += shared_data[tid + stride + BLOCK_SIZE];
            }
        }
       
        // The first thread of each block writes the partial sum to global memory
        __syncthreads(); 
        if (tid == 0) {
            intermediate_values[(idx/BLOCK_SIZE)] = shared_data[tid];
            intermediate_values[(idx/BLOCK_SIZE) + numBlocks] = shared_data[tid + BLOCK_SIZE];
        }
        
        // The first thread of the first block sums the partial sums from all blocks
        __syncthreads(); 
        if (idx == 0) {
            for (int i = 0; i < numBlocks; i++) {
                total_sum_x += intermediate_values[i];
                total_sum_y += intermediate_values[i + numBlocks];
            }
            
            mean_x = total_sum_x/numReturns;
            mean_y = total_sum_y/numReturns;
            means[0] = mean_x; 
            means[1] = mean_y; 
        }
    }
}


__global__ void calculateCorrelation(double* returns1, double* returns2, int numReturns, int numBlocks, double* intermediate_values, double* correlation, double* means) { 
    int tid = threadIdx.x;
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    extern __shared__ double shared_data[];
    shared_data[tid] = 0.0;
    shared_data[tid+BLOCK_SIZE] = 0.0;
    shared_data[tid+ (2*BLOCK_SIZE)] = 0.0;

    if (tid == 0) {
        if (idx == 0) intermediate_values[0] = 0.0;
        else intermediate_values[(idx/BLOCK_SIZE)] = 0.0;
        if (idx == 0) intermediate_values[numBlocks] = 0.0;
        intermediate_values[(idx/BLOCK_SIZE) + numBlocks] = 0.0;
        if (idx == 0) intermediate_values[2*numBlocks] = 0.0;
        intermediate_values[(idx/BLOCK_SIZE) + (2*numBlocks)] = 0.0;
    }
    double total_numerator = 0.0;
    double total_denom1 = 0.0;
    double total_denom2 = 0.0;

    __syncthreads();
    
    if (idx < numReturns) {
        double x = returns1[idx];
        double y = returns2[idx];

        // Thread-local storage for numerator and denominators
        double thread_numerator = (x-means[0])*(y-means[1]);
        double thread_denom1 = (x-means[0])*(x-means[0]); 
        double thread_denom2 = (y-means[1])*(y-means[1]);
    
        
        // Place thread-local values in shared memory one after the other 
        shared_data[tid] = thread_numerator;
        shared_data[tid + BLOCK_SIZE] = thread_denom1;
        shared_data[tid + (2 * BLOCK_SIZE)] = thread_denom2;

        // Reduce partial sums across threads within a block
        for (int stride = BLOCK_SIZE / 2; stride >= 1; stride /= 2) {
            __syncthreads(); // Ensure all threads have updated memory
            if (tid < stride) {
                shared_data[tid] += shared_data[tid + stride];
                shared_data[tid + BLOCK_SIZE] += shared_data[tid + stride + BLOCK_SIZE];
                shared_data[tid + (2 * BLOCK_SIZE)] += shared_data[tid + stride + (2 * BLOCK_SIZE)];
            }
        }

        __syncthreads(); 
        // write partial sums to global memory
        if (tid == 0) {
            intermediate_values[(idx/BLOCK_SIZE)] = shared_data[0];
            intermediate_values[(idx/BLOCK_SIZE) + numBlocks] = shared_data[BLOCK_SIZE];
            intermediate_values[(idx/BLOCK_SIZE) + (2 * numBlocks)] = shared_data[2*BLOCK_SIZE];
        }
        __syncthreads(); 
        // The first thread of the first block sums the partial sums from all blocks
        if (idx == 0) {
            for (int i = 0; i < numBlocks; i++) {
                total_numerator += intermediate_values[i];
                total_denom1 += intermediate_values[i + numBlocks];
                total_denom2 += intermediate_values[i + (2*numBlocks)];
            }
            correlation[0] = total_numerator;
            correlation[1] = total_denom1;
            correlation[2] = total_denom2;
        }
    }
}












 

int main() {
    std::vector<StockData> stock_data = {
        {"brent crude oil", "correlation data\\commodities\\brent crude oil\\BRENTCMDUSD_H1.csv"},
        {"gold", "correlation data\\commodities\\gold\\XAUUSD_H1.csv"},
        {"silver", "correlation data\\commodities\\silver\\XAGUSD_H1.csv"},
        {"bitcoin", "correlation data\\crypto\\btc\\BTCUSD_H1.csv"},
        {"cardano", "correlation data\\crypto\\cardano\\ADAUSDT_H1.csv"},
        {"ethereum", "correlation data\\crypto\\eth\\ETHUSD_H1.csv"},
        {"litecoin", "correlation data\\crypto\\litecoin\\LTCUSDT_H1.csv"},
        {"ripple", "correlation data\\crypto\\xrp\\XRPUSDT_H1.csv"},
        {"EUR/USD", "correlation data\\forex\\eurusd\\EURUSD_H1.csv"},
        {"GBP/USD", "correlation data\\forex\\gbpusd\\GBPUSD_H1.csv"},
        {"USD/CAD", "correlation data\\forex\\usdcad\\USDCAD_H1.csv"},
        {"USD/CHF", "correlation data\\forex\\usdchf\\USDCHF_H1.csv"},
        {"USD/JPY", "correlation data\\forex\\usdjpy\\USDJPY_H1.csv"},
        {"AUD/CAD", "correlation data\\forex minors\\audcad\\AUDCAD_H1.csv"},
        {"AUD/CHF", "correlation data\\forex minors\\audchf\\AUDCHF_H1.csv"},
        {"AUD/JPY", "correlation data\\forex minors\\audjpy\\AUDJPY_H1.csv"},
        {"AUD/NZD", "correlation data\\forex minors\\audnzd\\AUDNZD_H1.csv"},
        {"AUD/USD", "correlation data\\forex minors\\audusd\\AUDUSD_H1.csv"},
        {"CAD/CHF", "correlation data\\forex minors\\cadchf\\CADCHF_H1.csv"},
        {"CAD/JPY", "correlation data\\forex minors\\cadjpy\\CADJPY_H1.csv"},
        {"CHF/JPY", "correlation data\\forex minors\\chfjpy\\CHFJPY_H1.csv"},
        {"EUR/AUD", "correlation data\\forex minors\\euraud\\EURAUD_H1.csv"},
        {"EUR/CAD", "correlation data\\forex minors\\eurcad\\EURCAD_H1.csv"},
        {"EUR/CHF", "correlation data\\forex minors\\eurchf\\EURCHF_H1.csv"},
        {"EUR/GBP", "correlation data\\forex minors\\eurgbp\\EURGBP_H1.csv"},
        {"EUR/JPY", "correlation data\\forex minors\\eurjpy\\EURJPY_H1.csv"},
        {"EUR/NZD", "correlation data\\forex minors\\eurnzd\\EURNZD_H1.csv"},
        {"GBP/AUD", "correlation data\\forex minors\\gbpaud\\GBPAUD_H1.csv"},
        {"GBP/CAD", "correlation data\\forex minors\\gbpcad\\GBPCAD_H1.csv"},
        {"GBP/CHF", "correlation data\\forex minors\\gbpchf\\GBPCHF_H1.csv"},
        {"GBP/JPY", "correlation data\\forex minors\\gbpjpy\\GBPJPY_H1.csv"},
        {"GBP/NZD", "correlation data\\forex minors\\gbpznd\\GBPNZD_H1.csv"},
        {"NZD/CAD", "correlation data\\forex minors\\nzdcad\\NZDCAD_H1.csv"},
        {"NZD/CHF", "correlation data\\forex minors\\nzdchf\\NZDCHF_H1.csv"},
        {"NZD/JPY", "correlation data\\forex minors\\nzdjpy\\NZDJPY_H1.csv"},
        {"NZD/USD", "correlation data\\forex minors\\nzdusd\\NZDUSD_H1.csv"},
        {"US Tech Index", "correlation data\\indices\\america\\US100\\USATECHIDXUSD_H1.csv"},
        {"US Dow Index", "correlation data\\indices\\america\\US30\\USA30IDXUSD_H1.csv"},
        {"US S&P 500 Index", "correlation data\\indices\\america\\US500\\USA500IDXUSD_H1.csv"},
        {"Germany DAX Index", "correlation data\\indices\\germany\\DAX30\\DEUIDXEUR_H1.csv"},
        {"UK FTSE Index", "correlation data\\indices\\UK\\FTSY\\GBRIDXGBP_H1.csv"},
        {"Apple", "correlation data\\US Stocks\\apple\\AAPLUSUSD_H1.csv"},
        {"Netflix", "correlation data\\US Stocks\\netflix\\NFLXUSUSD_H1.csv"},
        {"Tesla", "correlation data\\US Stocks\\tesla\\TSLAUSUSD_H1.csv"}
    };
    for (int i = 10; i <= 10000; i*=10) {
        std::cout << "Started calculation with " << i << " elements..." << std::endl;
        double total_execution_time = 0;
        int iterations_count = 0;
        for (const auto& stock1 : stock_data) {
            std::vector<OHLCData> stock1_data;
            readCSV(stock1.file_path, stock1_data);

            for (const auto& stock2 : stock_data) {
                int numRecords = i;
                float current_execution_time = 0;
                iterations_count++;
                cudaEvent_t start, stop;
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
    
    
                //                                       -------------------------------- Read data
                std::vector<OHLCData> stock2_data;
                readCSV(stock2.file_path, stock2_data);
                
    
                //                                         --------------------------- Find common timestamps to compare the two stocks
                std::vector<std::string> timestamps1, timestamps2;
                for (const auto& entry : stock1_data) {
                    timestamps1.push_back(entry.timestamp);
                }
                for (const auto& entry : stock2_data) {
                    timestamps2.push_back(entry.timestamp);
                }
                std::vector<std::string> commonTimestamps;
                std::set_intersection(timestamps1.begin(), timestamps1.end(),
                                    timestamps2.begin(), timestamps2.end(),
                                    std::back_inserter(commonTimestamps));

                // Go through data and if it's a common timestamp, add the open and close prices to separate vectors
                std::vector<double> open1_all, close1_all, open2_all, close2_all;
                for (const auto& entry : stock1_data) {
                    if (std::binary_search(commonTimestamps.begin(), commonTimestamps.end(), entry.timestamp)) {
                        open1_all.push_back(entry.open);
                        close1_all.push_back(entry.close);
                    }
                }
                for (const auto& entry : stock2_data) {
                    if (std::binary_search(commonTimestamps.begin(), commonTimestamps.end(), entry.timestamp)) {
                        open2_all.push_back(entry.open);
                        close2_all.push_back(entry.close);
                    }
                    
                }
                
                //                                       -------------------------------- More data preperation
                // for testing purposes, we only take the first numRecords records
                std::vector<double> open1, close1, open2, close2;
                auto copy_first_n = [](const std::vector<double>& src, std::vector<double>& dest, size_t n) {
                    dest.insert(dest.end(), src.begin(), src.begin() + std::min(n, src.size()));
                };
                if (numRecords > commonTimestamps.size()) {
                    std::cout << "Not enough data for " << stock1.name << " and " << stock2.name << ". Only " << commonTimestamps.size() << " common data points available." << std::endl;
                }   
                numRecords = std::min<int>(static_cast<int>(commonTimestamps.size()), numRecords);
                copy_first_n(open1_all, open1, numRecords);
                copy_first_n(close1_all, close1, numRecords);
                copy_first_n(open2_all, open2, numRecords);
                copy_first_n(close2_all, close2, numRecords);
    
                // Allocate device memory for GPU processing
                double* d_open1, *d_close1, *d_open2, *d_close2, *d_returns1, *d_returns2, *d_correlation, *d_means,* d_intermediate_storage;
                cudaMalloc((void**)&d_open1, numRecords * sizeof(double));
                cudaMalloc((void**)&d_close1, numRecords * sizeof(double));
                cudaMalloc((void**)&d_open2, numRecords * sizeof(double));
                cudaMalloc((void**)&d_close2, numRecords * sizeof(double));
                cudaMalloc((void**)&d_returns1, numRecords * sizeof(double));
                cudaMalloc((void**)&d_returns2, numRecords * sizeof(double));
                cudaMalloc((void**)&d_correlation, 3 * sizeof(double));
                cudaMalloc((void**)&d_means, 2 * sizeof(double));
    
    
    
                //                                       ---------------------------------------------------returns 1
                // Copy data from host to device
                cudaMemcpy(d_open1, open1.data(), numRecords * sizeof(double), cudaMemcpyHostToDevice);
                cudaMemcpy(d_close1, close1.data(), numRecords * sizeof(double), cudaMemcpyHostToDevice);
                // Launch CUDA kernel to calculate returns for stock 1
                int numBlocks = (numRecords + BLOCK_SIZE - 1) / BLOCK_SIZE;
                cudaEventRecord(start, 0);
                calculateReturns<<<numBlocks, BLOCK_SIZE>>>(d_open1, d_close1, d_returns1, numRecords);
                cudaEventRecord(stop, 0);
                cudaEventSynchronize(stop);
                float time;
                cudaEventElapsedTime(&time, start, stop);
                current_execution_time += time;
    
    
    
                //                                       --------------------------------------------------- Returns 2
                cudaMemcpy(d_open2, open2.data(), numRecords * sizeof(double), cudaMemcpyHostToDevice);
                cudaMemcpy(d_close2, close2.data(), numRecords * sizeof(double), cudaMemcpyHostToDevice);
                cudaEventRecord(start, 0);
                calculateReturns<<<numBlocks, BLOCK_SIZE>>>(d_open2, d_close2, d_returns2, numRecords);
                cudaEventRecord(stop, 0);
                cudaEventSynchronize(stop);
                time = 0;
                cudaEventElapsedTime(&time, start, stop);
                current_execution_time += time;
                
    
                //                                     ------------------------------------- Calculate means
                size_t global_mem_intermediate_storage_size = 3 * ((numRecords + BLOCK_SIZE - 1)/BLOCK_SIZE) * sizeof(double); // per block 1 intermediate value 
                cudaMalloc((void**)&d_intermediate_storage, global_mem_intermediate_storage_size);
                size_t shared_mem_size = 2*BLOCK_SIZE*sizeof(double);
                numBlocks = (numRecords + BLOCK_SIZE - 1) / BLOCK_SIZE;
                
                cudaEventRecord(start, 0);
                calculateMeans<<<numBlocks, BLOCK_SIZE,shared_mem_size>>>(d_returns1,d_returns2, numRecords, numBlocks, d_intermediate_storage, d_means);
                cudaEventRecord(stop, 0);
                cudaEventSynchronize(stop);
                time = 0;
                cudaEventElapsedTime(&time, start, stop);
                current_execution_time += time;
    
    
    
                //                                                     ---------------- Calculate correlation
                shared_mem_size = 3*BLOCK_SIZE*sizeof(double);
                cudaEventRecord(start, 0);
                calculateCorrelation<<<numBlocks, BLOCK_SIZE, shared_mem_size>>>(d_returns1, d_returns2, numRecords, numBlocks, d_intermediate_storage, d_correlation, d_means);
                cudaEventRecord(stop, 0);
                cudaEventSynchronize(stop);
                time = 0;
                cudaEventElapsedTime(&time, start, stop);
                current_execution_time += time;
    
                
    
                // Copy correlation components back to host
                double h_correlation[3] = {0.0}; 
                cudaMemcpy(h_correlation, d_correlation, 3 * sizeof(double), cudaMemcpyDeviceToHost);
                // Calculate correlation coefficient (Pearson's r)
                double numerator = h_correlation[0];
                double denom1 = h_correlation[1];
                double denom2 = h_correlation[2];
                //std::cout << "numerator,denom1,denom2: " << numerator << "," << denom1 << "," << denom2 << std::endl;
                double correlation = numerator / (sqrt(denom1) * sqrt(denom2));
                std::cout << stock1.name << "," << stock2.name << "," << correlation << std::endl;
                std::cout << " Exectution time: " << current_execution_time << std::endl;
                total_execution_time += current_execution_time;
    
                cudaFree(d_open1);
                cudaFree(d_close1);
                cudaFree(d_open2);
                cudaFree(d_close2);
                cudaFree(d_returns1);
                cudaFree(d_returns2);
                cudaFree(d_correlation);
                cudaFree(d_intermediate_storage);
                cudaEventDestroy(start);
                cudaFree(d_means);
                cudaEventDestroy(stop);
            }
        }
    
        std::cout << "Aantal values: " << i << ", Average execution time: " << (total_execution_time/iterations_count) << std::endl;
    }
    


    
    return 0;
}