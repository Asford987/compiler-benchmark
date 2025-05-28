#include <CompiledNN/Model.h>
#include <CompiledNN/CompiledNN.h>
#include "metrics.h"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: ./compilednn_runner model.h5 image.png [label]\n";
        return 1;
    }

    /* ---------- load & compile ---------- */
    NeuralNetwork::Model       model;
    model.load(argv[1]);
    NeuralNetwork::CompiledNN  nn;
    nn.compile(model);

    /* ---------- read image -------------- */
    cv::Mat img = cv::imread(argv[2], cv::IMREAD_COLOR);   // always 3-ch
    if (img.empty()) {
        std::cerr << "Cannot read " << argv[2] << '\n';
        return 1;
    }

    /* ---------- resize to network size -- */
    const int H  = 32;                 // <- change if your net isn’t 32×32
    const int W  = 32;
    cv::resize(img, img, cv::Size(W, H));
    img.convertTo(img, CV_32F, 1.0/255.0);

    /* ---------- find how many floats the NN expects -------- */
    const size_t expected = nn.input(0).size();           // e.g. 3072
    const int    channels = static_cast<int>(expected / (H*W));

    if (expected != channels * H * W || (channels != 1 && channels != 3)) {
        std::cerr << "Unexpected input tensor size: " << expected << '\n';
        return 1;
    }

    /* ---------- pack image -> CHW float vector ------------- */
    std::vector<float> input;
    input.reserve(expected);

    if (channels == 1) {                       // grayscale network
        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        input.assign(gray.begin<float>(), gray.end<float>());
    } else {                                   // RGB network
        std::vector<cv::Mat> split;
        cv::split(img, split);                 // B,G,R  → planes
        for (const auto& ch : split)
            input.insert(input.end(), ch.begin<float>(), ch.end<float>());
    }

    /* ---------- sanity check ------------ */
    if (input.size() != expected) {
        std::cerr << "Input size after packing = " << input.size()
                  << ", but NN expects "         << expected << '\n';
        return 1;
    }

    /* ---------- inference --------------- */
    auto t0 = std::chrono::high_resolution_clock::now();
    std::copy(input.begin(), input.end(), nn.input(0).begin());
    nn.apply();
    auto t1 = std::chrono::high_resolution_clock::now();

    auto output = nn.output(0);
    auto max_it = std::max_element(output.begin(), output.end());
    int pred_class = std::distance(output.begin(), max_it);

    std::cout << "Prediction: " << pred_class << '\n';
    std::cout << "Inference: "
              << std::chrono::duration<double>(t1 - t0).count()
              << " s\n";

    /* ---------- optional metrics -------- */
    if (argc > 3) {
        std::vector<float> preds  = { pred_class };
        std::vector<int>   labels = { std::stoi(argv[3]) };
        log_metrics(preds, labels);
    }
    return 0;
}
