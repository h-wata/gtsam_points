// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <string>
#include <iostream>

#include <cuda_runtime.h>

namespace gtsam_points {

class CUDACheckError {
public:
  CUDACheckError(const char* file = nullptr, int line = -1) : file_(file), line_(line) {}
  void operator<<(cudaError_t error) const;

private:
  const char* file_;
  int line_;
};

extern CUDACheckError check_error;

}  // namespace gtsam_points
