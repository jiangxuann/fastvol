/**
 * @file neural.hpp
 * @brief Neural network surrogates for American options via LibTorch.
 *
 * @author Valerio Galanti
 * @date 2025
 * @version 0.1.1
 * @license MIT License
 *
 * This file contains neural network implementations for pricing American options
 * and calculating implied volatility using pre-trained LibTorch models.
 * It follows the same template-based design as other fastvol pricing methods.
 */

#pragma once

#ifdef FASTVOL_NEURAL_ENABLED

#include "fastvol/detail/torch.hpp"
#include <cstddef>
#include <string>

namespace fastvol::american::neural
{

// Default model paths (can be overridden)
inline const std::string DEFAULT_PRICE_MODEL = "american_vnet.pt";
inline const std::string DEFAULT_IV_MODEL = "american_ivnet.pt";

/* templates =====================================================================================*/
/* price -----------------------------------------------------------------------------------------*/
/**
 * @brief Calculates the price of an American option using a neural network surrogate.
 * @tparam T The floating-point type (float or double).
 * @param S The current price of the underlying asset.
 * @param K The strike price of the option.
 * @param cp_flag The option type flag. Accepted values are 1, 'c', or 'C' for a call option, and 0,
 * 'p', or 'P' for a put option.
 * @param ttm The time to maturity of the option, in years.
 * @param iv The implied volatility of the underlying asset.
 * @param r The risk-free interest rate.
 * @param q The dividend yield of the underlying asset.
 * @param model_path Path to the TorchScript model file. If empty, uses default.
 * @return The calculated price of the option.
 */
template <typename T>
T price(T S, T K, char cp_flag, T ttm, T iv, T r, T q,
        const std::string& model_path = "");

/**
 * @brief Calculates the implied volatility of an American option using a neural network surrogate.
 * @tparam T The floating-point type (float or double).
 * @param P The market price of the option.
 * @param S The current price of the underlying asset.
 * @param K The strike price of the option.
 * @param cp_flag The option type flag.
 * @param ttm The time to maturity of the option, in years.
 * @param r The risk-free interest rate.
 * @param q The dividend yield of the underlying asset.
 * @param model_path Path to the TorchScript model file. If empty, uses default.
 * @return The calculated implied volatility.
 */
template <typename T>
T iv(T P, T S, T K, char cp_flag, T ttm, T r, T q,
     const std::string& model_path = "");

/* batch operations ------------------------------------------------------------------------------*/
/**
 * @brief Calculates the prices of a batch of American options using neural networks.
 * @tparam T The floating-point type (float or double).
 * @param S An array of underlying asset prices.
 * @param K An array of strike prices.
 * @param cp_flag An array of option type flags.
 * @param ttm An array of times to maturity.
 * @param iv An array of implied volatilities.
 * @param r An array of risk-free interest rates.
 * @param q An array of dividend yields.
 * @param n_options The number of options in the batch.
 * @param[out] results A pre-allocated array to store the calculated option prices.
 * @param model_path Path to the TorchScript model file. If empty, uses default.
 */
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
                 const std::string& model_path = "");

/**
 * @brief Calculates the implied volatilities for a batch of American options using neural networks.
 */
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
              const std::string& model_path = "");

/* instantiations ================================================================================*/
/* fp64 ------------------------------------------------------------------------------------------*/
inline double price_fp64(double S, double K, char cp_flag, double ttm, double iv, double r, double q,
                         const char* model_path = nullptr)
{
    std::string path = model_path ? std::string(model_path) : "";
    return price<double>(S, K, cp_flag, ttm, iv, r, q, path);
}

inline double iv_fp64(double P, double S, double K, char cp_flag, double ttm, double r, double q,
                      const char* model_path = nullptr)
{
    std::string path = model_path ? std::string(model_path) : "";
    return iv<double>(P, S, K, cp_flag, ttm, r, q, path);
}

inline void price_fp64_batch(const double *__restrict__ S,
                             const double *__restrict__ K,
                             const char *__restrict__ cp_flag,
                             const double *__restrict__ ttm,
                             const double *__restrict__ iv,
                             const double *__restrict__ r,
                             const double *__restrict__ q,
                             size_t n_options,
                             double *__restrict__ results,
                             const char* model_path = nullptr)
{
    std::string path = model_path ? std::string(model_path) : "";
    price_batch<double>(S, K, cp_flag, ttm, iv, r, q, n_options, results, path);
}

inline void iv_fp64_batch(const double *__restrict__ P,
                          const double *__restrict__ S,
                          const double *__restrict__ K,
                          const char *__restrict__ cp_flag,
                          const double *__restrict__ ttm,
                          const double *__restrict__ r,
                          const double *__restrict__ q,
                          size_t n_options,
                          double *__restrict__ results,
                          const char* model_path = nullptr)
{
    std::string path = model_path ? std::string(model_path) : "";
    iv_batch<double>(P, S, K, cp_flag, ttm, r, q, n_options, results, path);
}

/* fp32 ------------------------------------------------------------------------------------------*/
inline float price_fp32(float S, float K, char cp_flag, float ttm, float iv, float r, float q,
                        const char* model_path = nullptr)
{
    std::string path = model_path ? std::string(model_path) : "";
    return price<float>(S, K, cp_flag, ttm, iv, r, q, path);
}

inline float iv_fp32(float P, float S, float K, char cp_flag, float ttm, float r, float q,
                     const char* model_path = nullptr)
{
    std::string path = model_path ? std::string(model_path) : "";
    return iv<float>(P, S, K, cp_flag, ttm, r, q, path);
}

inline void price_fp32_batch(const float *__restrict__ S,
                             const float *__restrict__ K,
                             const char *__restrict__ cp_flag,
                             const float *__restrict__ ttm,
                             const float *__restrict__ iv,
                             const float *__restrict__ r,
                             const float *__restrict__ q,
                             size_t n_options,
                             float *__restrict__ results,
                             const char* model_path = nullptr)
{
    std::string path = model_path ? std::string(model_path) : "";
    price_batch<float>(S, K, cp_flag, ttm, iv, r, q, n_options, results, path);
}

inline void iv_fp32_batch(const float *__restrict__ P,
                          const float *__restrict__ S,
                          const float *__restrict__ K,
                          const char *__restrict__ cp_flag,
                          const float *__restrict__ ttm,
                          const float *__restrict__ r,
                          const float *__restrict__ q,
                          size_t n_options,
                          float *__restrict__ results,
                          const char* model_path = nullptr)
{
    std::string path = model_path ? std::string(model_path) : "";
    iv_batch<float>(P, S, K, cp_flag, ttm, r, q, n_options, results, path);
}

} // namespace fastvol::american::neural

#endif // FASTVOL_NEURAL_ENABLED
