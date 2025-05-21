#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>
#include "metrics.h"
#include <chrono>
#include <iostream>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: ./tensorflow_runner model.h5 test_data.csv\n";
        return 1;
    }

    tensorflow::Session* session;
    tensorflow::Status status = tensorflow::NewSession(tensorflow::SessionOptions(), &session);
    if (!status.ok()) {
        std::cerr << status.ToString() << "\n";
        return 1;
    }

    tensorflow::GraphDef graph_def;
    status = ReadBinaryProto(tensorflow::Env::Default(), "model/saved_model.pb", &graph_def);
    if (!status.ok()) {
        std::cerr << status.ToString() << "\n";
        return 1;
    }

    status = session->Create(graph_def);
    if (!status.ok()) {
        std::cerr << status.ToString() << "\n";
        return 1;
    }

    std::vector<std::vector<float>> test_data;
    std::vector<int> test_labels;
    std::vector<float> predictions;

    auto start = std::chrono::high_resolution_clock::now();

    for (const auto& input : test_data) {
        tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, input.size()}));
        std::copy(input.begin(), input.end(), input_tensor.flat<float>().data());

        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {"input_tensor", input_tensor}
        };

        std::vector<tensorflow::Tensor> outputs;
        status = session->Run(inputs, {"output_tensor"}, {}, &outputs);
        if (!status.ok()) {
            std::cerr << status.ToString() << "\n";
            return 1;
        }

        predictions.push_back(outputs[0].flat<float>()(0));
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> inference_time = end - start;
    std::cout << "TensorFlow Inference Time: " << inference_time.count() << " sec\n";

    log_metrics(predictions, test_labels);

    session->Close();
    return 0;
}
