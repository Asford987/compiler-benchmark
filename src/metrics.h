// metrics.h
#pragma once

#include <vector>
#include <iostream>
#include <cmath>
#include <fstream>
#include <string>
#include <sys/resource.h>

inline double get_memory_usage_mb() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss / 1024.0; // kilobytes to MB
}

inline float compute_accuracy(const std::vector<int>& preds, const std::vector<int>& labels) {
    int correct = 0;
    for (size_t i = 0; i < preds.size(); ++i) {
        if (preds[i] == labels[i]) ++correct;
    }
    return static_cast<float>(correct) / preds.size();
}

inline float compute_precision(const std::vector<int>& preds, const std::vector<int>& labels) {
    int tp = 0, fp = 0;
    for (size_t i = 0; i < preds.size(); ++i) {
        if (preds[i] == 1 && labels[i] == 1) ++tp;
        else if (preds[i] == 1 && labels[i] == 0) ++fp;
    }
    return tp + fp == 0 ? 0 : static_cast<float>(tp) / (tp + fp);
}

inline float compute_recall(const std::vector<int>& preds, const std::vector<int>& labels) {
    int tp = 0, fn = 0;
    for (size_t i = 0; i < preds.size(); ++i) {
        if (preds[i] == 1 && labels[i] == 1) ++tp;
        else if (preds[i] == 0 && labels[i] == 1) ++fn;
    }
    return tp + fn == 0 ? 0 : static_cast<float>(tp) / (tp + fn);
}

inline float compute_f1(float precision, float recall) {
    return (precision + recall) == 0 ? 0 : 2 * (precision * recall) / (precision + recall);
}

inline void log_metrics(const std::vector<float>& raw_preds, const std::vector<int>& labels) {
    std::vector<int> preds;
    for (float p : raw_preds) {
        preds.push_back(p >= 0.5f ? 1 : 0); // binary classification
    }

    float acc = compute_accuracy(preds, labels);
    float prec = compute_precision(preds, labels);
    float rec = compute_recall(preds, labels);
    float f1 = compute_f1(prec, rec);
    double mem = get_memory_usage_mb();

    std::cout << "Accuracy:  " << acc << "\n";
    std::cout << "Precision: " << prec << "\n";
    std::cout << "Recall:    " << rec << "\n";
    std::cout << "F1 Score:  " << f1 << "\n";
    std::cout << "Memory:    " << mem << " MB\n";
}
