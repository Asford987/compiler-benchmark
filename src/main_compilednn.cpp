#include <CompiledNN/Model.h>
#include <CompiledNN/CompiledNN.h>
#include <chrono>
#include <iostream>

// Function to compute evaluation metrics
void compute_metrics(const std::vector<float>& predictions, const std::vector<float>& labels) {
    // Implement accuracy, precision, recall, F1 score calculations
}

int main() {
    using namespace NeuralNetwork;

    Model model;
    model.load("model.h5");

    CompiledNN nn;
    nn.compile(model);

    // Load test dataset
    std::vector<std::vector<float>> test_data; // Populate with test inputs
    std::vector<float> test_labels; // Populate with corresponding labels

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<float> predictions;
    for (const auto& input : test_data) {
        // Assuming single input tensor
        std::copy(input.begin(), input.end(), nn.input(0).begin());
        nn.apply();
        predictions.push_back(nn.output(0)[0]); // Adjust based on output dimensions
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> inference_time = end - start;

    std::cout << "Inference Time: " << inference_time.count() << " seconds" << std::endl;

    compute_metrics(predictions, test_labels);

    return 0;
}
