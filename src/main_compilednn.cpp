#include <CompiledNN/Model.h>
#include <CompiledNN/CompiledNN.h>
#include "metrics.h"
#include <chrono>
#include <iostream>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: ./compilednn_runner model.h5 test_data.csv\n";
        return 1;
    }

    NeuralNetwork::Model model;
    model.load(argv[1]);

    NeuralNetwork::CompiledNN nn;
    nn.compile(model);

    // TODO: Load test_data.csv into test_data and test_labels
    std::vector<std::vector<float>> test_data;
    std::vector<int> test_labels;

    std::vector<float> predictions;
    auto start = std::chrono::high_resolution_clock::now();

    for (const auto& input : test_data) {
        std::copy(input.begin(), input.end(), nn.input(0).begin());
        nn.apply();
        predictions.push_back(nn.output(0)[0]);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> inference_time = end - start;
    std::cout << "CompiledNN Inference Time: " << inference_time.count() << " sec\n";

    log_metrics(predictions, test_labels);
    return 0;
}
