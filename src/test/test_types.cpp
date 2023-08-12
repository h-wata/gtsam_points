#include <vector>
#include <iostream>
#include <Eigen/Core>
#include <boost/filesystem.hpp>

#include <gtest/gtest.h>
#include <gtsam_ext/types/point_cloud_cpu.hpp>
#include <gtsam_ext/types/point_cloud_gpu.hpp>
#include <gtsam_ext/types/gaussian_voxelmap_cpu.hpp>
#include <gtsam_ext/types/gaussian_voxelmap_gpu.hpp>

template <typename T, int D>
struct RandomSet {
  RandomSet()
  : num_points(128),
    points(num_points),
    normals(num_points),
    covs(num_points),
    intensities(num_points),
    times(num_points),
    aux1(num_points),
    aux2(num_points) {
    //
    for (int i = 0; i < num_points; i++) {
      points[i].setOnes();
      points[i].template head<3>() = Eigen::Matrix<T, 3, 1>::Random();
      normals[i].setZero();
      normals[i].template head<3>() = Eigen::Matrix<T, 3, 1>::Random().normalized();
      covs[i].setZero();
      covs[i].template block<3, 3>(0, 0) = Eigen::Matrix<T, 3, 3>::Random();
      covs[i] = (covs[i] * covs[i].transpose()).eval();
      intensities[i] = Eigen::Vector2d::Random()[0];
      times[i] = Eigen::Vector2d::Random()[0];

      aux1[i] = Eigen::Matrix<T, D, 1>::Random();
      aux2[i] = Eigen::Matrix<T, D, D>::Random();
    }
  }

  const int num_points;
  std::vector<Eigen::Matrix<T, D, 1>, Eigen::aligned_allocator<Eigen::Matrix<T, D, 1>>> points;
  std::vector<Eigen::Matrix<T, D, 1>, Eigen::aligned_allocator<Eigen::Matrix<T, D, 1>>> normals;
  std::vector<Eigen::Matrix<T, D, D>, Eigen::aligned_allocator<Eigen::Matrix<T, D, D>>> covs;
  std::vector<T> intensities;
  std::vector<T> times;

  std::vector<Eigen::Matrix<T, D, 1>, Eigen::aligned_allocator<Eigen::Matrix<T, D, 1>>> aux1;
  std::vector<Eigen::Matrix<T, D, D>, Eigen::aligned_allocator<Eigen::Matrix<T, D, D>>> aux2;
};

void compare_frames(const gtsam_ext::PointCloud::ConstPtr& frame1, const gtsam_ext::PointCloud::ConstPtr& frame2, const std::string& label = "") {
  ASSERT_NE(frame1, nullptr) << label;
  ASSERT_NE(frame2, nullptr) << label;

  EXPECT_EQ(frame1->size(), frame2->size()) << "frame size mismatch";

  if (frame1->points) {
    EXPECT_NE(frame2->points, nullptr);
    for (int i = 0; i < frame1->size(); i++) {
      EXPECT_LT((frame1->points[i] - frame2->points[i]).norm(), 1e-6) << label;
    }
  } else {
    EXPECT_EQ(frame1->points, frame2->points);
  }

  if (frame1->times) {
    EXPECT_NE(frame1->times, nullptr);
    for (int i = 0; i < frame1->size(); i++) {
      EXPECT_LT(abs(frame1->times[i] - frame2->times[i]), 1e-6) << label;
    }
  } else {
    EXPECT_EQ(frame1->times, frame2->times);
  }

  if (frame1->normals) {
    EXPECT_NE(frame1->normals, nullptr);
    for (int i = 0; i < frame1->size(); i++) {
      EXPECT_LT((frame1->normals[i] - frame2->normals[i]).norm(), 1e-6) << label;
    }
  } else {
    EXPECT_EQ(frame1->normals, frame2->normals);
  }

  if (frame1->covs) {
    EXPECT_NE(frame1->covs, nullptr);
    for (int i = 0; i < frame1->size(); i++) {
      EXPECT_LT((frame1->covs[i] - frame2->covs[i]).norm(), 1e-6) << label;
    }
  } else {
    EXPECT_EQ(frame1->covs, frame2->covs);
  }

  if (frame1->intensities) {
    EXPECT_NE(frame1->intensities, nullptr);
    for (int i = 0; i < frame1->size(); i++) {
      EXPECT_LT(abs(frame1->intensities[i] - frame2->intensities[i]), 1e-6) << label;
    }
  } else {
    EXPECT_EQ(frame1->intensities, frame2->intensities);
  }

  for (const auto& aux : frame1->aux_attributes) {
    const auto& name = aux.first;
    const size_t aux_size = aux.second.first;
    const char* aux1_ptr = reinterpret_cast<const char*>(aux.second.second);

    EXPECT_TRUE(frame2->aux_attributes.count(name));
    const auto found = frame2->aux_attributes.find(name);
    if (found == frame2->aux_attributes.end()) {
      continue;
    }

    EXPECT_EQ(found->second.first, aux_size);
    if (found->second.first != aux_size) {
      continue;
    }

    const char* aux2_ptr = reinterpret_cast<const char*>(found->second.second);
    EXPECT_TRUE(std::equal(aux1_ptr, aux1_ptr + aux_size * frame1->size(), aux2_ptr)) << label;
  }
}

template <typename T, int D>
void creation_test() {
  RandomSet<T, D> randomset;
  const int num_points = randomset.num_points;
  const auto& points = randomset.points;
  const auto& normals = randomset.normals;
  const auto& covs = randomset.covs;
  const auto& intensities = randomset.intensities;
  const auto& times = randomset.times;
  const auto& aux1 = randomset.aux1;
  const auto& aux2 = randomset.aux2;

  auto frame = std::make_shared<gtsam_ext::PointCloudCPU>();

  EXPECT_FALSE(frame->has_points());
  frame->add_points(points);

  EXPECT_TRUE(frame->has_points() && frame->check_points());
  compare_frames(frame, std::make_shared<gtsam_ext::PointCloudCPU>(points));
  compare_frames(frame, gtsam_ext::PointCloudCPU::clone(*frame));

  for (int i = 0; i < num_points; i++) {
    const double diff = (frame->points[i].template head<D>() - points[i].template cast<double>()).squaredNorm();
    EXPECT_LT(diff, std::numeric_limits<double>::epsilon()) << "point copy failure";
    EXPECT_DOUBLE_EQ(frame->points[i][3], 1.0) << "point copy failure";
  }

  EXPECT_FALSE(frame->has_times());
  EXPECT_FALSE(frame->has_normals());
  EXPECT_FALSE(frame->has_covs());
  EXPECT_FALSE(frame->has_intensities());

  frame->add_times(times);
  compare_frames(frame, gtsam_ext::PointCloudCPU::clone(*frame));
  EXPECT_TRUE(frame->has_times() && frame->check_times());

  frame->add_covs(covs);
  compare_frames(frame, gtsam_ext::PointCloudCPU::clone(*frame));
  EXPECT_TRUE(frame->has_covs() && frame->check_covs());

  frame->add_normals(normals);
  compare_frames(frame, gtsam_ext::PointCloudCPU::clone(*frame));
  EXPECT_TRUE(frame->has_normals() && frame->check_normals());

  frame->add_intensities(intensities);
  compare_frames(frame, gtsam_ext::PointCloudCPU::clone(*frame));
  EXPECT_TRUE(frame->has_intensities() && frame->check_intensities());

  frame->add_aux_attribute("aux1", aux1);
  compare_frames(frame, gtsam_ext::PointCloudCPU::clone(*frame));
  frame->add_aux_attribute("aux2", aux2);
  compare_frames(frame, gtsam_ext::PointCloudCPU::clone(*frame));

  for (int i = 0; i < num_points; i++) {
    const double diff_t = std::pow(frame->times[i] - times[i], 2);
    const double diff_n = (frame->normals[i].template head<D>() - normals[i].template cast<double>()).squaredNorm();
    const double diff_c = (frame->covs[i].template block<D, D>(0, 0) - covs[i].template cast<double>()).squaredNorm();
    EXPECT_LT(diff_t, std::numeric_limits<double>::epsilon()) << "time copy failure";
    EXPECT_LT(diff_n, std::numeric_limits<double>::epsilon()) << "normal copy failure";
    EXPECT_LT(diff_c, std::numeric_limits<double>::epsilon()) << "cov copy failure";

    EXPECT_DOUBLE_EQ(frame->normals[i][3], 0.0) << "normal copy failure";
    for (int j = 0; j < 4; j++) {
      EXPECT_DOUBLE_EQ(frame->covs[i](3, j), 0.0) << "cov copy failure";
      EXPECT_DOUBLE_EQ(frame->covs[i](j, 3), 0.0) << "cov copy failure";
    }
  }

  boost::filesystem::create_directories("/tmp/frame_test");
  frame->save("/tmp/frame_test");
  compare_frames(frame, gtsam_ext::PointCloudCPU::load("/tmp/frame_test"));

  boost::filesystem::create_directories("/tmp/frame_test_compact");
  frame->save_compact("/tmp/frame_test_compact");
  compare_frames(frame, gtsam_ext::PointCloudCPU::load("/tmp/frame_test_compact"));
}

TEST(TestTypes, TestPointCloudCPU) {
  creation_test<float, 3>();
  creation_test<float, 4>();
  creation_test<double, 3>();
  creation_test<double, 4>();
}

#ifdef BUILD_GTSAM_EXT_GPU

template <typename T, int D>
void creation_test_gpu() {
  RandomSet<T, D> randomset;
  const int num_points = randomset.num_points;
  const auto& points = randomset.points;
  const auto& normals = randomset.normals;
  const auto& covs = randomset.covs;
  const auto& times = randomset.times;

  auto frame = std::make_shared<gtsam_ext::PointCloudCPU>();
  frame->add_points(points);

  auto frame_gpu = std::make_shared<gtsam_ext::PointCloudGPU>();

  // add_points
  frame_gpu->add_points_gpu(points);
  ASSERT_EQ(frame_gpu->points, nullptr);
  ASSERT_NE(frame_gpu->points_gpu, nullptr);

  const auto points_gpu = gtsam_ext::download_points_gpu(*frame_gpu);
  ASSERT_EQ(points_gpu.size(), num_points);
  for (int i = 0; i < num_points; i++) {
    EXPECT_LT((points_gpu[i].template cast<double>() - frame->points[i].template head<3>()).norm(), 1e-6);
  }

  frame_gpu->add_points(points);
  compare_frames(frame, frame_gpu);
  compare_frames(frame, gtsam_ext::PointCloudGPU::clone(*frame));

  // add_covs
  frame->add_covs(covs);
  frame_gpu->add_covs_gpu(covs);
  ASSERT_EQ(frame_gpu->covs, nullptr);
  ASSERT_NE(frame_gpu->covs_gpu, nullptr);

  const auto covs_gpu = gtsam_ext::download_covs_gpu(*frame_gpu);
  ASSERT_EQ(covs_gpu.size(), num_points);
  for (int i = 0; i < num_points; i++) {
    EXPECT_LT((covs_gpu[i].template cast<double>() - frame->covs[i].template block<3, 3>(0, 0)).norm(), 1e-6);
  }

  frame_gpu->add_covs(covs);
  compare_frames(frame, frame_gpu);
  compare_frames(frame, gtsam_ext::PointCloudGPU::clone(*frame));
}

TEST(TestTypes, TestPointCloudGPU) {
  creation_test_gpu<float, 3>();
  creation_test_gpu<float, 4>();
  creation_test_gpu<double, 3>();
  creation_test_gpu<double, 4>();
}

#endif

TEST(TestTypes, TestPointCloudCPUFuncs) {
  RandomSet<double, 4> randomset;
  auto frame = std::make_shared<gtsam_ext::PointCloudCPU>();
  frame->add_points(randomset.points);
  frame->add_normals(randomset.normals);
  frame->add_covs(randomset.covs);
  frame->add_intensities(randomset.intensities);
  frame->add_times(randomset.times);
  frame->add_aux_attribute("aux1", randomset.aux1);
  frame->add_aux_attribute("aux2", randomset.aux2);

  const auto validate_samples = [](const gtsam_ext::PointCloud::ConstPtr& frame, bool test_aux = true) {
    ASSERT_TRUE(frame->points);
    ASSERT_TRUE(frame->normals);
    ASSERT_TRUE(frame->covs);
    ASSERT_TRUE(frame->intensities);
    ASSERT_TRUE(frame->times);

    for (int i = 0; i < frame->size(); i++) {
      EXPECT_TRUE(frame->points[i].array().isFinite().all());
      EXPECT_TRUE(frame->normals[i].array().isFinite().all());
      EXPECT_TRUE(frame->covs[i].array().isFinite().all());
      EXPECT_TRUE(std::isfinite(frame->intensities[i]));
      EXPECT_TRUE(std::isfinite(frame->times[i]));
    }

    if (!test_aux) {
      return;
    }

    const Eigen::Vector4d* aux1 = frame->aux_attribute<Eigen::Vector4d>("aux1");
    const Eigen::Matrix4d* aux2 = frame->aux_attribute<Eigen::Matrix4d>("aux2");
    ASSERT_TRUE(aux1);
    ASSERT_TRUE(aux2);
    for (int i = 0; i < frame->size(); i++) {
      EXPECT_TRUE(aux1[i].array().isFinite().all());
      EXPECT_TRUE(aux2[i].array().isFinite().all());
    }
  };

  // Test for gtsam_ext::sample()
  std::mt19937 mt;
  std::vector<int> indices(frame->size());
  std::iota(indices.begin(), indices.end(), 0);

  const int num_samples = frame->size() * 0.5;
  std::vector<int> samples(num_samples);
  std::sample(indices.begin(), indices.end(), samples.begin(), num_samples, mt);

  auto sampled = gtsam_ext::sample(frame, samples);
  ASSERT_EQ(sampled->size(), num_samples);
  validate_samples(sampled);

  const Eigen::Vector4d* aux1_ = frame->aux_attribute<Eigen::Vector4d>("aux1");
  const Eigen::Matrix4d* aux2_ = frame->aux_attribute<Eigen::Matrix4d>("aux2");
  const Eigen::Vector4d* aux1 = sampled->aux_attribute<Eigen::Vector4d>("aux1");
  const Eigen::Matrix4d* aux2 = sampled->aux_attribute<Eigen::Matrix4d>("aux2");
  for (int i = 0; i < samples.size(); i++) {
    const int idx = samples[i];
    EXPECT_LT((frame->points[idx] - sampled->points[i]).norm(), 1e-6);
    EXPECT_LT((frame->normals[idx] - sampled->normals[i]).norm(), 1e-6);
    EXPECT_LT((frame->covs[idx] - sampled->covs[i]).norm(), 1e-6);
    EXPECT_DOUBLE_EQ(frame->intensities[idx], sampled->intensities[i]);
    EXPECT_DOUBLE_EQ(frame->times[idx], sampled->times[i]);
    EXPECT_LT((aux1_[idx] - aux1[i]).norm(), 1e-6);
    EXPECT_LT((aux2_[idx] - aux2[i]).norm(), 1e-6);
  }

  // Test for random_sampling, voxelgrid_sampling, and randomgrid_sampling
  sampled = gtsam_ext::random_sampling(frame, 0.5, mt);
  EXPECT_DOUBLE_EQ(static_cast<double>(sampled->size()) / frame->size(), 0.5);
  validate_samples(sampled);

  sampled = gtsam_ext::voxelgrid_sampling(frame, 0.1);
  EXPECT_LE(sampled->size(), frame->size());
  validate_samples(sampled, false);

  sampled = gtsam_ext::randomgrid_sampling(frame, 0.1, 0.5, mt);
  EXPECT_LE(sampled->size(), frame->size());
  validate_samples(sampled);

  // Test for filter
  auto filtered1 = gtsam_ext::filter(frame, [](const Eigen::Vector4d& pt) { return pt.x() < 0.0; });
  auto filtered2 = gtsam_ext::filter(frame, [](const Eigen::Vector4d& pt) { return pt.x() >= 0.0; });

  validate_samples(filtered1);
  validate_samples(filtered2);
  EXPECT_EQ(filtered1->size() + filtered2->size(), frame->size());
  EXPECT_TRUE(std::all_of(filtered1->points, filtered1->points + filtered1->size(), [](const Eigen::Vector4d& pt) { return pt.x() < 0.0; }));
  EXPECT_TRUE(std::all_of(filtered2->points, filtered2->points + filtered2->size(), [](const Eigen::Vector4d& pt) { return pt.x() >= 0.0; }));

  // Test for filter_by_index
  filtered1 = gtsam_ext::filter_by_index(frame, [&](int i) { return frame->points[i].x() < 0.0; });
  filtered2 = gtsam_ext::filter_by_index(frame, [&](int i) { return frame->points[i].x() >= 0.0; });

  validate_samples(filtered1);
  validate_samples(filtered2);
  EXPECT_EQ(filtered1->size() + filtered2->size(), frame->size());
  EXPECT_TRUE(std::all_of(filtered1->points, filtered1->points + filtered1->size(), [](const Eigen::Vector4d& pt) { return pt.x() < 0.0; }));
  EXPECT_TRUE(std::all_of(filtered2->points, filtered2->points + filtered2->size(), [](const Eigen::Vector4d& pt) { return pt.x() >= 0.0; }));

  // Test for sort
  auto sorted = gtsam_ext::sort(frame, [&](int lhs, int rhs) { return frame->points[lhs].x() < frame->points[rhs].x(); });
  validate_samples(sorted);
  EXPECT_EQ(sorted->size(), frame->size());
  EXPECT_TRUE(std::is_sorted(sorted->points, sorted->points + sorted->size(), [](const auto& lhs, const auto& rhs) { return lhs.x() < rhs.x(); }));

  // Test for sort_by_time
  sorted = gtsam_ext::sort_by_time(frame);
  validate_samples(sorted);
  EXPECT_EQ(sorted->size(), frame->size());
  EXPECT_TRUE(std::is_sorted(sorted->times, sorted->times + sorted->size()));

  // Test for transform
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.linear() = Eigen::AngleAxisd(0.5, Eigen::Vector3d::Random().normalized()).toRotationMatrix();
  T.translation() = Eigen::Vector3d::Random();

  auto transformed = gtsam_ext::transform(frame, T);
  validate_samples(transformed);
  ASSERT_EQ(transformed->size(), frame->size());
  for (int i = 0; i < frame->size(); i++) {
    EXPECT_LT((T * frame->points[i] - transformed->points[i]).norm(), 1e-6);
    EXPECT_LT((T.linear() * frame->normals[i].head<3>() - transformed->normals[i].head<3>()).norm(), 1e-6);
    EXPECT_LT((T.linear() * frame->covs[i].topLeftCorner<3, 3>() * T.linear().transpose() - transformed->covs[i].topLeftCorner<3, 3>()).norm(), 1e-6);
  }

  auto transformed2 = gtsam_ext::transform(frame, T.cast<float>());
  compare_frames(transformed, transformed2, "transform<Isometry3f>");

  // Test for transform_inplace
  transformed2 = gtsam_ext::PointCloudCPU::clone(*frame);
  gtsam_ext::transform_inplace(transformed2, T);
  compare_frames(transformed, transformed2, "transform_inplace<Isometry3d>");

  transformed2 = gtsam_ext::PointCloudCPU::clone(*frame);
  gtsam_ext::transform_inplace(transformed2, T.cast<float>());
  compare_frames(transformed, transformed2, "transform_inplace<Isometry3f>");
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}