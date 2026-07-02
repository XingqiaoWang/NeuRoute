// torch_scorer.cpp
//
// Implementation of TorchScorer (Scheme B).
//
// Build note (example, adapt to your environment):
//   g++ -O3 -std=c++17 -fopenmp \
//       torch_scorer.cpp \
//       -I${TORCH_INCLUDE} -L${TORCH_LIB} -ltorch -ltorch_cpu -lc10 \
//       -Wl,-rpath,${TORCH_LIB} \
//       -o your_binary
//
// In CMake, prefer find_package(Torch REQUIRED).

#include "torch_scorer.h"

#include <cassert>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <utility>

#include <torch/script.h>   // torch::jit::load, torch::jit::Module
#include <torch/torch.h>    // torch::from_blob, NoGradGuard, set_num_threads

#ifdef _OPENMP
#include <omp.h>
#endif

namespace torchpipe {

struct TorchScorer::Impl {
  torch::jit::Module mod;
  bool threads_applied = false;

  // Reusable conversion buffer for u8 -> f32 (size Q*in_dim)
  std::vector<float> u8_f32_buf;
};

static inline void throw_runtime(const std::string& msg) {
  throw std::runtime_error(msg);
}

TorchScorer::TorchScorer(const std::string& model_path, const TorchScorerConfig& cfg)
    : cfg_(cfg), model_path_(model_path) {
  impl_ = new Impl();

  try {
    impl_->mod = torch::jit::load(model_path_);
  } catch (const c10::Error& e) {
    throw_runtime(std::string("Failed to load TorchScript model: ") + e.what());
  }

  if (cfg_.eval_mode) {
    impl_->mod.eval();
  }

  // Apply thread settings early.
  // Note: set_num_interop_threads can only be called once per process in PyTorch.
  apply_thread_settings_once_();
}

void TorchScorer::apply_thread_settings_once_() {
  if (impl_->threads_applied) return;

  if (cfg_.torch_threads > 0) {
    torch::set_num_threads(cfg_.torch_threads);
  }

  if (cfg_.try_set_interop) {
    // IMPORTANT:
    // torch::set_num_interop_threads can only be set ONCE.
    // If some other part of your process already set it, this will throw.
    try {
      if (cfg_.interop_threads > 0) {
        torch::set_num_interop_threads(cfg_.interop_threads);
      }
    } catch (const c10::Error& e) {
      // Do not hard fail: just print a warning and continue.
      std::cerr << "[warn] set_num_interop_threads failed (already set?): " << e.what() << "\n";
    }
  }

  impl_->threads_applied = true;
}

void TorchScorer::enforce_output_shape_(int64_t Q, int64_t D) const {
  (void)Q;
  if (!(D >= 1 && D <= 32)) {
    throw_runtime("Invalid model output D=" + std::to_string(D) + " (must be 1..32)");
  }
}

void TorchScorer::score_f32(const float* x, int64_t Q, int64_t in_dim, std::vector<float>& logits) {
  if (!x) throw_runtime("score_f32: x is null");
  if (Q <= 0) throw_runtime("score_f32: Q must be > 0");
  if (in_dim <= 0) throw_runtime("score_f32: in_dim must be > 0");

  apply_thread_settings_once_();

  torch::NoGradGuard nograd;

  // Wrap caller memory as a Tensor WITHOUT copy.
  // Shape: [Q, in_dim], contiguous row-major.
  auto x_tensor = torch::from_blob(
      (void*)x,
      {Q, in_dim},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));

  // Ensure contiguous (from_blob should already be contiguous given our shape/stride).
  x_tensor = x_tensor.contiguous();

  // Forward
  torch::Tensor y;
  try {
    y = impl_->mod.forward({x_tensor}).toTensor();
  } catch (const c10::Error& e) {
    throw_runtime(std::string("Torch forward() failed: ") + e.what());
  }

  if (!y.defined()) throw_runtime("Model output is undefined");
  if (y.dim() != 2) throw_runtime("Model output must be 2D [Q, D], got dim=" + std::to_string(y.dim()));
  if (y.size(0) != Q) {
    throw_runtime("Model output Q mismatch: got " + std::to_string(y.size(0)) + ", expect " + std::to_string(Q));
  }

  const int64_t D = y.size(1);
  enforce_output_shape_(Q, D);

  if (out_dim_ < 0) out_dim_ = (int)D;
  if (D != out_dim_) {
    throw_runtime("Output dim changed: got D=" + std::to_string(D) + ", cached=" + std::to_string(out_dim_));
  }

  y = y.contiguous();
  if (y.scalar_type() != torch::kFloat32) {
    throw_runtime("Model output dtype must be float32 for this scorer");
  }

  logits.resize((size_t)Q * (size_t)D);
  std::memcpy(logits.data(), y.data_ptr<float>(), sizeof(float) * (size_t)Q * (size_t)D);
}

void TorchScorer::score_u8(const uint8_t* x, int64_t Q, int64_t in_dim, std::vector<float>& logits) {
  if (!x) throw_runtime("score_u8: x is null");
  if (Q <= 0) throw_runtime("score_u8: Q must be > 0");
  if (in_dim <= 0) throw_runtime("score_u8: in_dim must be > 0");

  apply_thread_settings_once_();

  // Convert u8 -> f32 into a reusable buffer.
  const size_t n = (size_t)Q * (size_t)in_dim;
  impl_->u8_f32_buf.resize(n);

  const float scale = cfg_.u8_scale;
  const float zp = cfg_.u8_zero_point;

#ifdef _OPENMP
  if (cfg_.convert_threads > 1) {
#pragma omp parallel for schedule(static) num_threads(cfg_.convert_threads)
    for (int64_t i = 0; i < (int64_t)n; i++) {
      impl_->u8_f32_buf[(size_t)i] = (float(x[(size_t)i]) - zp) * scale;
    }
  } else
#endif
  {
    for (size_t i = 0; i < n; i++) {
      impl_->u8_f32_buf[i] = (float(x[i]) - zp) * scale;
    }
  }

  // Now score as f32.
  score_f32(impl_->u8_f32_buf.data(), Q, in_dim, logits);
}

void TorchScorer::warmup_f32(const float* x, int64_t Q, int64_t in_dim, int n) {
  if (n <= 0) return;
  apply_thread_settings_once_();

  torch::NoGradGuard nograd;

  auto x_tensor = torch::from_blob(
      (void*)x,
      {Q, in_dim},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)).contiguous();

  for (int i = 0; i < n; i++) {
    try {
      auto y = impl_->mod.forward({x_tensor}).toTensor();
      (void)y;
    } catch (const c10::Error& e) {
      throw_runtime(std::string("Warmup forward() failed: ") + e.what());
    }
  }
}

} // namespace torchpipe
