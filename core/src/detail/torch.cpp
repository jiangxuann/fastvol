#ifdef FASTVOL_NEURAL_ENABLED

#include "fastvol/detail/torch.hpp"
#include <filesystem>
#include <iostream>

namespace fastvol::detail::torch_utils
{

void ModelManager::load_model(const std::string& model_path, const Config& config)
{
    try {
        // Check if file exists
        if (!std::filesystem::exists(model_path)) {
            throw std::runtime_error("Model file not found: " + model_path);
        }

        // Load model
        model_ = torch::jit::load(model_path);
        model_.to(config.device);
        model_.eval();

        // Configure threading
        if (config.num_threads > 0) {
            at::set_num_threads(config.num_threads);
        }

        config_ = config;
        loaded_ = true;

    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to load neural model '" + model_path + "': " + e.what());
    }
}

torch::Tensor ModelManager::forward(const std::vector<torch::Tensor>& inputs)
{
    if (!loaded_) {
        throw std::runtime_error("Model not loaded");
    }

    try {
        torch::NoGradGuard no_grad;

        std::vector<torch::jit::IValue> jit_inputs;
        jit_inputs.reserve(inputs.size());

        for (const auto& tensor : inputs) {
            jit_inputs.emplace_back(tensor);
        }

        return model_.forward(jit_inputs).toTensor();

    } catch (const std::exception& e) {
        throw std::runtime_error("Neural network forward pass failed: " + std::string(e.what()));
    }
}

std::string get_default_model_path(const std::string& model_type)
{
    // Try common locations
    std::vector<std::string> search_paths = {
        "./",
        "./models/",
        "../models/",
        "../../python/fastvol/neural/checkpoints/",
        std::string(std::getenv("HOME") ? std::getenv("HOME") : "") + "/.fastvol/models/"
    };

    std::string filename = model_type + ".pt";

    for (const auto& path : search_paths) {
        std::string full_path = path + filename;
        if (std::filesystem::exists(full_path)) {
            return full_path;
        }
    }

    // Fallback to filename only (let user specify path)
    return filename;
}

void initialize_torch(int num_threads)
{
    if (num_threads > 0) {
        at::set_num_threads(num_threads);
    }

    // Disable gradient computation globally for inference
    torch::NoGradGuard no_grad;
}

} // namespace fastvol::detail::torch_utils

#endif // FASTVOL_NEURAL_ENABLED
