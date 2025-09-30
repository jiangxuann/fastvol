#ifdef FASTVOL_NEURAL_ENABLED

#include "fastvol/american/neural.hpp"
#include "fastvol/detail/torch.hpp"
#include <torch/torch.h>
#include <memory>
#include <unordered_map>
#include <mutex>

namespace fastvol::american::neural
{

// Thread-safe model cache
static std::unordered_map<std::string, std::unique_ptr<fastvol::detail::torch_utils::ModelManager>> model_cache_;
static std::mutex cache_mutex_;

// Get or create model manager
static fastvol::detail::torch_utils::ModelManager* get_model_manager(const std::string& model_path,
                                                                     torch::ScalarType dtype)
{
    std::lock_guard<std::mutex> lock(cache_mutex_);

    std::string cache_key = model_path + "_" + (dtype == torch::kFloat32 ? "fp32" : "fp64");

    auto it = model_cache_.find(cache_key);
    if (it != model_cache_.end()) {
        return it->second.get();
    }

    // Create new model manager
    auto manager = std::make_unique<fastvol::detail::torch_utils::ModelManager>();
    fastvol::detail::torch_utils::Config config;
    config.dtype = dtype;
    config.device = torch::kCPU;  // CPU only for now

    manager->load_model(model_path, config);

    auto* ptr = manager.get();
    model_cache_[cache_key] = std::move(manager);
    return ptr;
}

/* template implementations ======================================================================*/
template <typename T>
T price(T S, T K, char cp_flag, T ttm, T iv, T r, T q, const std::string& model_path)
{
    // Determine model path
    std::string path = model_path.empty() ?
        fastvol::detail::torch_utils::get_default_model_path("american_vnet") : model_path;

    // Get appropriate dtype
    torch::ScalarType dtype = std::is_same_v<T, float> ? torch::kFloat32 : torch::kFloat64;

    // Get model manager
    auto* manager = get_model_manager(path, dtype);

    // Create input tensors
    auto opts = torch::TensorOptions().dtype(dtype).device(torch::kCPU);

    auto spot_tensor = torch::tensor({static_cast<double>(S)}, opts);
    auto strike_tensor = torch::tensor({static_cast<double>(K)}, opts);
    auto cp_tensor = torch::tensor({static_cast<double>(cp_flag & 1)}, opts);  // Convert to 0/1
    auto ttm_tensor = torch::tensor({static_cast<double>(ttm)}, opts);
    auto iv_tensor = torch::tensor({static_cast<double>(iv)}, opts);
    auto r_tensor = torch::tensor({static_cast<double>(r)}, opts);
    auto q_tensor = torch::tensor({static_cast<double>(q)}, opts);

    // Run inference
    std::vector<torch::Tensor> inputs = {
        spot_tensor, strike_tensor, cp_tensor, ttm_tensor, iv_tensor, r_tensor, q_tensor
    };

    torch::Tensor result = manager->forward(inputs);

    // Extract scalar result
    return static_cast<T>(result.item<double>());
}

template <typename T>
T iv(T P, T S, T K, char cp_flag, T ttm, T r, T q, const std::string& model_path)
{
    // Determine model path
    std::string path = model_path.empty() ?
        fastvol::detail::torch_utils::get_default_model_path("american_ivnet") : model_path;

    // Get appropriate dtype
    torch::ScalarType dtype = std::is_same_v<T, float> ? torch::kFloat32 : torch::kFloat64;

    // Get model manager
    auto* manager = get_model_manager(path, dtype);

    // Create input tensors
    auto opts = torch::TensorOptions().dtype(dtype).device(torch::kCPU);

    auto price_tensor = torch::tensor({static_cast<double>(P)}, opts);
    auto spot_tensor = torch::tensor({static_cast<double>(S)}, opts);
    auto strike_tensor = torch::tensor({static_cast<double>(K)}, opts);
    auto cp_tensor = torch::tensor({static_cast<double>(cp_flag & 1)}, opts);
    auto ttm_tensor = torch::tensor({static_cast<double>(ttm)}, opts);
    auto r_tensor = torch::tensor({static_cast<double>(r)}, opts);
    auto q_tensor = torch::tensor({static_cast<double>(q)}, opts);

    // Run inference
    std::vector<torch::Tensor> inputs = {
        price_tensor, spot_tensor, strike_tensor, cp_tensor, ttm_tensor, r_tensor, q_tensor
    };

    torch::Tensor result = manager->forward(inputs);

    // Extract scalar result
    T iv_result = static_cast<T>(result.item<double>());

    // Handle invalid IV (negative values indicate no solution)
    return iv_result < T(0) ? T(-1) : iv_result;
}

template <typename T>
void price_batch(const T *__restrict__ S,
                 const T *__restrict__ K,
                 const char *__restrict__ cp_flag,
                 const T *__restrict__ ttm,
                 const T *__restrict__ iv,
                 const T *__restrict__ r,
                 const T *__restrict__ q,
                 size_t n_options,
                 T *__restrict__ results,
                 const std::string& model_path)
{
    // Determine model path
    std::string path = model_path.empty() ?
        fastvol::detail::torch_utils::get_default_model_path("american_vnet") : model_path;

    // Get appropriate dtype
    torch::ScalarType dtype = std::is_same_v<T, float> ? torch::kFloat32 : torch::kFloat64;

    // Get model manager
    auto* manager = get_model_manager(path, dtype);

    // Create input tensors
    auto opts = torch::TensorOptions().dtype(dtype).device(torch::kCPU);

    // Convert arrays to tensors
    std::vector<double> spot_vec(S, S + n_options);
    std::vector<double> strike_vec(K, K + n_options);
    std::vector<double> cp_vec(n_options);
    std::vector<double> ttm_vec(ttm, ttm + n_options);
    std::vector<double> iv_vec(iv, iv + n_options);
    std::vector<double> r_vec(r, r + n_options);
    std::vector<double> q_vec(q, q + n_options);

    // Convert cp_flag to 0/1
    for (size_t i = 0; i < n_options; ++i) {
        cp_vec[i] = static_cast<double>(cp_flag[i] & 1);
    }

    auto spot_tensor = torch::from_blob(spot_vec.data(), {static_cast<long>(n_options)}, torch::kFloat64).to(opts);
    auto strike_tensor = torch::from_blob(strike_vec.data(), {static_cast<long>(n_options)}, torch::kFloat64).to(opts);
    auto cp_tensor = torch::from_blob(cp_vec.data(), {static_cast<long>(n_options)}, torch::kFloat64).to(opts);
    auto ttm_tensor = torch::from_blob(ttm_vec.data(), {static_cast<long>(n_options)}, torch::kFloat64).to(opts);
    auto iv_tensor = torch::from_blob(iv_vec.data(), {static_cast<long>(n_options)}, torch::kFloat64).to(opts);
    auto r_tensor = torch::from_blob(r_vec.data(), {static_cast<long>(n_options)}, torch::kFloat64).to(opts);
    auto q_tensor = torch::from_blob(q_vec.data(), {static_cast<long>(n_options)}, torch::kFloat64).to(opts);

    // Run inference
    std::vector<torch::Tensor> inputs = {
        spot_tensor, strike_tensor, cp_tensor, ttm_tensor, iv_tensor, r_tensor, q_tensor
    };

    torch::Tensor result = manager->forward(inputs);

    // Copy results back
    result = result.to(torch::kCPU).to(torch::kFloat64);
    auto result_ptr = result.data_ptr<double>();

    for (size_t i = 0; i < n_options; ++i) {
        results[i] = static_cast<T>(result_ptr[i]);
    }
}

template <typename T>
void iv_batch(const T *__restrict__ P,
              const T *__restrict__ S,
              const T *__restrict__ K,
              const char *__restrict__ cp_flag,
              const T *__restrict__ ttm,
              const T *__restrict__ r,
              const T *__restrict__ q,
              size_t n_options,
              T *__restrict__ results,
              const std::string& model_path)
{
    // Similar implementation to price_batch but for IV model
    std::string path = model_path.empty() ?
        fastvol::detail::torch_utils::get_default_model_path("american_ivnet") : model_path;

    torch::ScalarType dtype = std::is_same_v<T, float> ? torch::kFloat32 : torch::kFloat64;
    auto* manager = get_model_manager(path, dtype);

    auto opts = torch::TensorOptions().dtype(dtype).device(torch::kCPU);

    // Convert arrays to tensors
    std::vector<double> price_vec(P, P + n_options);
    std::vector<double> spot_vec(S, S + n_options);
    std::vector<double> strike_vec(K, K + n_options);
    std::vector<double> cp_vec(n_options);
    std::vector<double> ttm_vec(ttm, ttm + n_options);
    std::vector<double> r_vec(r, r + n_options);
    std::vector<double> q_vec(q, q + n_options);

    for (size_t i = 0; i < n_options; ++i) {
        cp_vec[i] = static_cast<double>(cp_flag[i] & 1);
    }

    auto price_tensor = torch::from_blob(price_vec.data(), {static_cast<long>(n_options)}, torch::kFloat64).to(opts);
    auto spot_tensor = torch::from_blob(spot_vec.data(), {static_cast<long>(n_options)}, torch::kFloat64).to(opts);
    auto strike_tensor = torch::from_blob(strike_vec.data(), {static_cast<long>(n_options)}, torch::kFloat64).to(opts);
    auto cp_tensor = torch::from_blob(cp_vec.data(), {static_cast<long>(n_options)}, torch::kFloat64).to(opts);
    auto ttm_tensor = torch::from_blob(ttm_vec.data(), {static_cast<long>(n_options)}, torch::kFloat64).to(opts);
    auto r_tensor = torch::from_blob(r_vec.data(), {static_cast<long>(n_options)}, torch::kFloat64).to(opts);
    auto q_tensor = torch::from_blob(q_vec.data(), {static_cast<long>(n_options)}, torch::kFloat64).to(opts);

    std::vector<torch::Tensor> inputs = {
        price_tensor, spot_tensor, strike_tensor, cp_tensor, ttm_tensor, r_tensor, q_tensor
    };

    torch::Tensor result = manager->forward(inputs);
    result = result.to(torch::kCPU).to(torch::kFloat64);
    auto result_ptr = result.data_ptr<double>();

    for (size_t i = 0; i < n_options; ++i) {
        T iv_result = static_cast<T>(result_ptr[i]);
        results[i] = iv_result < T(0) ? T(-1) : iv_result;
    }
}

// Explicit template instantiations
template double price<double>(double, double, char, double, double, double, double, const std::string&);
template float price<float>(float, float, char, float, float, float, float, const std::string&);
template double iv<double>(double, double, double, char, double, double, double, const std::string&);
template float iv<float>(float, float, float, char, float, float, float, const std::string&);

template void price_batch<double>(const double*, const double*, const char*, const double*,
                                 const double*, const double*, const double*, size_t, double*, const std::string&);
template void price_batch<float>(const float*, const float*, const char*, const float*,
                                const float*, const float*, const float*, size_t, float*, const std::string&);

template void iv_batch<double>(const double*, const double*, const double*, const char*, const double*,
                              const double*, const double*, size_t, double*, const std::string&);
template void iv_batch<float>(const float*, const float*, const float*, const char*, const float*,
                             const float*, const float*, size_t, float*, const std::string&);

} // namespace fastvol::american::neural

#endif // FASTVOL_NEURAL_ENABLED
