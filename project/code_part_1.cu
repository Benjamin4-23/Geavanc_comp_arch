#include <iostream>
#include <numeric>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <limits>
#include <ctime>
#include <chrono>
#include <cmath>
#include <algorithm>
using namespace std;

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
        // Replace spaces in the timestamp with a placeholder (e.g., '_')
        size_t spacePos = line.find(' ');
        if (spacePos != std::string::npos) {
            line[spacePos] = '_';  // Replace space with '_'
        }

        std::istringstream iss(line);
        OHLCData entry;
        char comma;

        // Read timestamp (including spaces) as a single string
        std::getline(iss, entry.timestamp, ','); // Read until the next comma

        // Read other fields using comma as delimiter
        if (!(iss >> entry.open >> comma >> entry.high >> comma >> entry.low >> comma >> entry.close >> comma >> entry.volume)) {
            std::cerr << "Failed to parse line: " << line << std::endl;
            continue; // Skip invalid entries
        }

        data.push_back(entry);
    }
    file.close();
}




double calculate_correlation_CPU(const double* open1, const double* close1,
    const double* open2, const double* close2, int window_size) {
    // Calculate returns for both stocks
    double* returns1 = new double[window_size];
    double* returns2 = new double[window_size];
    
    for (int i = 0; i < window_size; i++) {
        returns1[i] = (close1[i] - open1[i]) / open1[i];
        returns2[i] = (close2[i] - open2[i]) / open2[i];
    }

    

    // Calculate mean of returns for both stocks
    double mean_return1 = accumulate(returns1, returns1 + window_size, 0.0) / window_size;
    double mean_return2 = accumulate(returns2, returns2 + window_size, 0.0) / window_size;

    // Calculate covariance and standard deviations
    double numerator = 0.0;
    double denom1 = 0.0, denom2 = 0.0;
    for (int i = 0; i < window_size; ++i) {
        numerator += (returns1[i] - mean_return1) * (returns2[i] - mean_return2);
        denom1 += pow(returns1[i] - mean_return1, 2.0);
        denom2 += pow(returns2[i] - mean_return2, 2.0);
    }

    // Calculate Pearson correlation coefficient (handle potential division by zero)
    double correlation = 0.0;
    if (denom1 * denom2 != 0.0) {
        correlation = numerator / sqrt(denom1 * denom2);
    }

    // Clean up memory
    delete[] returns1;
    delete[] returns2;

    return correlation;
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
                double current_execution_time = 0;
                iterations_count++;
                //                                       -------------------------------- data inlezen
                std::vector<OHLCData> stock2_data;
                readCSV(stock2.file_path, stock2_data);
                // Find common timestamps to determine valid records
                std::vector<std::string> timestamps1, timestamps2;
                for (const auto& entry : stock1_data) {
                    timestamps1.push_back(entry.timestamp);
                }
                for (const auto& entry : stock2_data) {
                    timestamps2.push_back(entry.timestamp);
                }
                
                // Determine common timestamps (intersection) --------------------------- timestamps filteren
                std::vector<std::string> commonTimestamps;
                std::set_intersection(timestamps1.begin(), timestamps1.end(),
                                    timestamps2.begin(), timestamps2.end(),
                                    std::back_inserter(commonTimestamps));
                // Filter data based on common timestamps
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
                // Check if there is enough data for the specified number of records    
                //if (numRecords > commonTimestamps.size()) {
                    //std::cout << "Not enough data for " << stock1.name << " and " << stock2.name << ". Only " << commonTimestamps.size() << " common data points available." << std::endl;
                //}   
                //                               --------------------------------data klaarmaken voor gebruik
                numRecords = std::min<int>(static_cast<int>(commonTimestamps.size()), numRecords);
                
                std::vector<double> open1, close1, open2, close2;
                // Function to copy first n elements from src to dest
                auto copy_first_n = [](const std::vector<double>& src, std::vector<double>& dest, size_t n) {
                    dest.insert(dest.end(), src.begin(), src.begin() + std::min(n, src.size()));
                };
                copy_first_n(open1_all, open1, numRecords);
                copy_first_n(close1_all, close1, numRecords);
                copy_first_n(open2_all, open2, numRecords);
                copy_first_n(close2_all, close2, numRecords);
                
                double* open1_double_array, *open2_double_array,*close1_double_array,*close2_double_array;
                open1_double_array= new double[open1.size()];
                open2_double_array= new double[open2.size()];
                close1_double_array= new double[close1.size()];
                close2_double_array= new double[close2.size()];
                copy(open1.begin(), open1.end(), open1_double_array);
                copy(open2.begin(), open2.end(), open2_double_array);
                copy(close1.begin(), close1.end(),close1_double_array);
                copy(close2.begin(), close2.end(), close2_double_array);


                auto start_cpu = std::chrono::steady_clock::now();
                double correlation = calculate_correlation_CPU(open1_double_array, close1_double_array, open2_double_array, close2_double_array, numRecords);
                auto end_cpu = std::chrono::steady_clock::now();
                std::chrono::duration<double> cpu_time = end_cpu - start_cpu;
                current_execution_time = cpu_time.count()*1000;
                total_execution_time += current_execution_time;
                
                std::cout << "" << stock1.name << "," << stock2.name << "," << correlation << std::endl;
                //std::cout << "Exectution time: " << current_execution_time << std::endl;
            }
        }

        std::cout << "total_execution_time execution time: " << (total_execution_time/iterations_count) << std::endl;

    }
    
    return 0;
}