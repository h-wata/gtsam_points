// SPDX-License-Identifier: MIT
// Copyright (c) 2025  Kenji Koide (k.koide@aist.go.jp)
#include <gtsam_points/registration/ransac.hpp>

#include <atomic>
#include <unordered_set>
#include <gtsam_points/config.hpp>
#include <gtsam_points/util/fast_floor.hpp>
#include <gtsam_points/util/vector3i_hash.hpp>
#include <gtsam_points/util/parallelism.hpp>
#include <gtsam_points/types/frame_traits.hpp>
#include <gtsam_points/registration/alignment.hpp>

#ifdef GTSAM_POINTS_USE_TBB
#include <tbb/parallel_for.h>
#endif

namespace gtsam_points {

template <typename PointCloud, typename Features>
RegistrationResult estimate_pose_ransac(
  const PointCloud& target,
  const PointCloud& source,
  const Features& target_features,
  const Features& source_features,
  const NearestNeighborSearch& target_tree,
  const NearestNeighborSearch& target_features_tree,
  const RANSACParams& params) {
  //
  const double inv_resolution = 1.0 / params.inlier_voxel_resolution;
  std::unordered_set<Eigen::Vector3i, Vector3iHash> target_voxels;
  for (size_t i = 0; i < frame::size(target); i++) {
    const Eigen::Array4i coord = fast_floor(frame::point(target, i) * inv_resolution);
    target_voxels.emplace(coord.head<3>());
  }

  const auto sample_indices = [&](auto& samples, std::mt19937& mt) {
    std::uniform_int_distribution<> udist(0, frame::size(source) - 1);
    for (auto it = std::begin(samples); it != std::end(samples); it++) {
      *it = udist(mt);
      if (std::find(std::begin(samples), it, *it) != it) {
        it--;
      }
    }
  };

  const auto find_target_indices = [&](const auto& source_indices, auto& target_indices) {
    auto target_itr = std::begin(target_indices);

    for (auto source_itr = std::begin(source_indices); source_itr != std::end(source_indices); source_itr++, target_itr++) {
      double sq_dist;
      const auto& source_f = source_features[*source_itr];
      if (!target_features_tree.knn_search(source_f.data(), 1, &(*target_itr), &sq_dist)) {
        std::cerr << "warning: knn_search failed" << std::endl;
        *target_itr = 0;
      }
    }
  };

  const auto poly_error = [&](const auto& source_indices, const auto& target_indices) {
    double max_error = 0.0;

    auto target_itr = std::begin(target_indices);
    for (auto source_itr = std::begin(source_indices); source_itr != std::end(source_indices); source_itr++, target_itr++) {
      const auto source_next = source_itr == std::end(source_indices) - 1 ? std::begin(source_indices) : source_itr + 1;
      const auto target_next = target_itr == std::end(target_indices) - 1 ? std::begin(target_indices) : target_itr + 1;

      const double dt = (frame::point(target, *target_itr) - frame::point(target, *target_next)).norm();
      const double ds = (frame::point(source, *source_itr) - frame::point(source, *source_next)).norm();
      const double error = std::abs(dt - ds) / std::max(std::max(dt, ds), 1e-6);
      max_error = std::max(max_error, error);
    }

    return max_error;
  };

  const auto count_inliers = [&](const Eigen::Isometry3d& T_target_source) {
    size_t inliers = 0;
    for (size_t i = 0; i < frame::size(source); i++) {
      const Eigen::Vector4d transformed = T_target_source * frame::point(source, i);
      const Eigen::Array4i coord = fast_floor(transformed * inv_resolution);
      if (target_voxels.count(coord.head<3>())) {
        inliers++;
      }
    }
    return inliers;
  };

  const auto calc_T_target_source = [&](const auto& source_indices, const auto& target_indices) {
    switch (params.dof) {
      case 6:
        return align_points(
          frame::point(target, target_indices[0]),
          frame::point(target, target_indices[1]),
          frame::point(target, target_indices[2]),
          frame::point(source, source_indices[0]),
          frame::point(source, source_indices[1]),
          frame::point(source, source_indices[2]));
      case 4:
        return align_points_4dof(
          frame::point(target, target_indices[0]),
          frame::point(target, target_indices[1]),
          frame::point(source, source_indices[0]),
          frame::point(source, source_indices[1]));
      default:
        std::cerr << "error: invalid dof " << params.dof << std::endl;
        abort();
    }
    return Eigen::Isometry3d::Identity();
  };

  const int num_samples = params.dof == 6 ? 3 : 2;

  std::mutex mutex;
  std::atomic_uint64_t seed = params.seed;
  std::atomic_bool early_stop = false;
  std::atomic_uint64_t best_inliers = 0;
  Eigen::Isometry3d best_T_target_source = Eigen::Isometry3d::Identity();

  const auto perpoint_task = [&] {
    if (early_stop) {
      return;
    }

    std::mt19937 mt(seed);
    seed += mt() + 2347891;

    std::vector<size_t> source_indices(num_samples);
    sample_indices(source_indices, mt);

    std::vector<size_t> target_indices(num_samples);
    find_target_indices(source_indices, target_indices);

    const double error = poly_error(source_indices, target_indices);
    if (error > params.poly_error_thresh) {
      return;
    }

    const Eigen::Isometry3d T_target_source = calc_T_target_source(source_indices, target_indices);

    const size_t inliers = count_inliers(T_target_source);
    if (inliers < best_inliers) {
      return;
    }

    early_stop = inliers > frame::size(source) * params.early_stop_inlier_rate;

    std::lock_guard<std::mutex> lock(mutex);
    if (best_inliers < inliers) {
      best_inliers = inliers;
      best_T_target_source = T_target_source;
    }
  };

  if (is_omp_default() || params.num_threads == 1) {
#pragma omp parallel for num_threads(params.num_threads) schedule(guided, 4)
    for (size_t k = 0; k < params.max_iterations; k++) {
      perpoint_task();
    }
  } else {
#ifdef GTSAM_POINTS_USE_TBB
    tbb::parallel_for(tbb::blocked_range<int>(0, params.max_iterations, 4), [&](const tbb::blocked_range<int>& range) {
      for (int k = range.begin(); k < range.end(); k++) {
        perpoint_task();
      }
    });
#else
    std::cerr << "error: TBB is not available" << std::endl;
    abort();
#endif
  }

  return RegistrationResult{best_inliers / static_cast<double>(frame::size(target)), best_T_target_source};
}

}  // namespace gtsam_points
