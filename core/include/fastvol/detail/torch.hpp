/**
 * @file torch.hpp
 * @brief LibTorch utility functions and helpers.
 *
 * @author Valerio Galanti
 * @date 2025
 * @version 0.1.1
 * @license MIT License
 *
 * This file provides utility functions for LibTorch integration, including
 * tensor conversion, model loading, and error handling.
 */

#pragma once

#ifdef FASTVOL_NEURAL_ENABLED

#include <torch/script.h>
#include <torch/torch.h>
#include <string>
#include <memory>
#include <stdexcept>

namespace fastvol::detail::torch_utils
{

/**
 * @brief Configuration for neural network inference
 */
struct Config {
    torch::Device device = torch::kCPU;
    torch::ScalarType dtype = torch::kFloat64;
    int num_threads = -1;  // -1 = auto-detect
};

/**
 * @brief RAII wrapper for LibTorch model management
 */
class ModelManager {
private:
    torch::jit::script::Module model_;
    Config config_;
    bool loaded_ = false;

public:
    ModelManager() = default;
    ~ModelManager() = default;

    // Non-copyable, movable
    ModelManager(const ModelManager&) = delete;
    ModelManager& operator=(const ModelManager&) = delete;
    ModelManager(ModelManager&&) = default;
    ModelManager& operator=(ModelManager&&) = default;

    void load_model(const std::string& model_path, const Config& config = Config{});
    torch::Tensor forward(const std::vector<torch::Tensor>& inputs);
    bool is_loaded() const { return loaded_; }
    const Config& get_config() const { return config_; }
};

/**
 * @brief Get default model paths for different model types
 */
std::string get_default_model_path(const std::string& model_type);

/**
 * @brief Initialize LibTorch threading and configuration
 */
void initialize_torch(int num_threads = -1);

} // namespace fastvol::detail::torch_utils

#endif // FASTVOL_NEURAL_ENABLED
