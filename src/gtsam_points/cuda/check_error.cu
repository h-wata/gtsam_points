// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_points/cuda/check_error.cuh>

namespace gtsam_points {

void CUDACheckError::operator<<(cudaError_t error) const {
  if (error == cudaSuccess) return;
  std::cerr << "warning: " << cudaGetErrorName(error) << "\n"
            << "         : " << cudaGetErrorString(error) << "\n";
  if (file_) {
    std::cerr << "  (from " << file_ << ":" << line_ << ")\n";
  }
}

CUDACheckError check_error;

}  // namespace gtsam_points
