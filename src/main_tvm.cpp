#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include "metrics.h"
#include <chrono>
#include <iostream>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: ./tvm_runner model.so test_data.csv\n";
        return 1;
    }

    tvm::runtime::Module mod_factory = tvm::runtime::Module::LoadFromFile(argv[1]);
    tvm::runtime::PackedFunc create = mod_factory.GetFunction("default");
    tvm::runtime::Module mod = create();

    tvm::runtime::PackedFunc set_input = mod.GetFunction("set_input");
    tvm::runtime::PackedFunc run = mod.GetFunction("run");
    tvm::runtime::PackedFunc get_output = mod.GetFunction("get_output");

    std::vector<std::vector<float>> test_data;
    std::vector<int> test_labels;
    std::vector<float> predictions;

    auto start = std::chrono::high_resolution_clock::now();

    for (const auto& input : test_data) {
        // Assume you've wrapped TVM's NDArray or DLTensor for input/output
        // This is a placeholder — actual buffer setup will vary.
        // set_input("input", tvm_input_tensor);
        run();
        // get_output(0, tvm_output_tensor);
        predictions.push_back(0.5f); // dummy value
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> inference_time = end - start;
    std::cout << "TVM Inference Time: " << inference_time.count() << " sec\n";

    log_metrics(predictions, test_labels);
    return 0;
}
