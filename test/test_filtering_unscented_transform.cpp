#include <gtest/gtest.h>

#include "filtering/unscented_transform.hpp"

#include <Eigen/Dense>

#include <vector>

namespace {

auto make_simplex_params() -> filtering::UnscentedTransformParameters {
    filtering::UnscentedTransformParameters params;
    params.sigma_point_scheme =
        filtering::UnscentedSigmaPointScheme::JulierSphericalSimplex;
    params.simplex_weight_0 = 0.5;
    return params;
}

auto make_scaled_symmetric_params() -> filtering::UnscentedTransformParameters {
    filtering::UnscentedTransformParameters params;
    params.sigma_point_scheme =
        filtering::UnscentedSigmaPointScheme::ScaledSymmetric;
    params.alpha = 0.3;
    params.beta = 2.0;
    params.kappa = 0.0;
    return params;
}

} // namespace

TEST(UnscentedTransformTest, JulierSphericalSimplexWeightsAreNormalized) {
    const auto weights = filtering::UnscentedTransform::compute_weights(
        6,
        make_simplex_params()
    );

    EXPECT_EQ(weights.mean.size(), 8);
    EXPECT_EQ(weights.covariance.size(), 8);
    EXPECT_NEAR(weights.mean.sum(), 1.0, 1e-12);
}

TEST(UnscentedTransformTest, ScaledSymmetricWeightsRemainAvailable) {
    const auto weights = filtering::UnscentedTransform::compute_weights(
        6,
        make_scaled_symmetric_params()
    );

    EXPECT_EQ(weights.mean.size(), 13);
    EXPECT_EQ(weights.covariance.size(), 13);
    EXPECT_NEAR(weights.mean.sum(), 1.0, 1e-12);
}

TEST(UnscentedTransformTest, SigmaPointsContainMeanAndCorrectCount) {
    Eigen::VectorXd mean(3);
    mean << 10.0, -2.0, 4.0;

    Eigen::MatrixXd covariance = Eigen::MatrixXd::Identity(3, 3);
    covariance(0, 0) = 4.0;
    covariance(1, 1) = 9.0;
    covariance(2, 2) = 1.0;

    const auto sigma_points = filtering::UnscentedTransform::generate_sigma_points(
        mean,
        covariance,
        make_simplex_params()
    );

    ASSERT_EQ(sigma_points.size(), 5);
    EXPECT_TRUE(sigma_points.front().isApprox(mean, 1e-12));
}

TEST(UnscentedTransformTest, RecoverGaussianPreservesOriginalMoments) {
    Eigen::VectorXd mean(3);
    mean << 5.0, -3.0, 2.0;

    Eigen::MatrixXd covariance = Eigen::MatrixXd::Zero(3, 3);
    covariance << 4.0, 0.8, 0.2,
                  0.8, 3.0, 0.5,
                  0.2, 0.5, 2.0;

    const auto params = make_simplex_params();
    const auto weights = filtering::UnscentedTransform::compute_weights(3, params);
    const auto sigma_points =
        filtering::UnscentedTransform::generate_sigma_points(mean, covariance, params);
    const auto moments =
        filtering::UnscentedTransform::recover_gaussian(sigma_points, weights);

    EXPECT_TRUE(moments.mean.isApprox(mean, 1e-9));
    EXPECT_TRUE(moments.covariance.isApprox(covariance, 1e-8));
}

TEST(UnscentedTransformTest, TransformDistributionRunsFullRoundInOneCall) {
    Eigen::VectorXd mean_x(3);
    mean_x << 2.0, -1.0, 4.0;

    Eigen::MatrixXd cov_x = Eigen::MatrixXd::Zero(3, 3);
    cov_x << 5.0, 0.2, 0.1,
             0.2, 2.0, 0.4,
             0.1, 0.4, 3.0;

    Eigen::MatrixXd A(2, 3);
    A << 1.0, -0.5, 2.0,
         0.3, 0.8, -1.2;
    Eigen::VectorXd b(2);
    b << 1.5, -0.7;

    const auto result = filtering::UnscentedTransform::transform_distribution(
        mean_x,
        cov_x,
        [&A, &b](const Eigen::VectorXd& x) {
            return A * x + b;
        },
        Eigen::MatrixXd(),
        make_simplex_params()
    );

    const Eigen::VectorXd expected_mean_y = A * mean_x + b;
    const Eigen::MatrixXd expected_cov_y = A * cov_x * A.transpose();

    EXPECT_EQ(result.source_sigma_points.size(), 5);
    EXPECT_EQ(result.transformed_sigma_points.size(), 5);
    EXPECT_EQ(result.weights.mean.size(), 5);
    EXPECT_EQ(result.weights.covariance.size(), 5);
    EXPECT_TRUE(result.moments.mean.isApprox(expected_mean_y, 1e-9));
    EXPECT_TRUE(result.moments.covariance.isApprox(expected_cov_y, 1e-8));
}

TEST(UnscentedTransformTest, CrossCovarianceMatchesLinearMapping) {
    Eigen::VectorXd mean_x(3);
    mean_x << 2.0, -1.0, 4.0;

    Eigen::MatrixXd cov_x = Eigen::MatrixXd::Zero(3, 3);
    cov_x << 5.0, 0.2, 0.1,
             0.2, 2.0, 0.4,
             0.1, 0.4, 3.0;

    Eigen::MatrixXd A(2, 3);
    A << 1.0, -0.5, 2.0,
         0.3, 0.8, -1.2;
    Eigen::VectorXd b(2);
    b << 1.5, -0.7;

    const auto params = make_simplex_params();
    const auto weights = filtering::UnscentedTransform::compute_weights(3, params);
    const auto sigma_x =
        filtering::UnscentedTransform::generate_sigma_points(mean_x, cov_x, params);

    std::vector<Eigen::VectorXd> sigma_y(sigma_x.size());
    for (std::size_t i = 0; i < sigma_x.size(); ++i) {
        sigma_y[i] = A * sigma_x[i] + b;
    }

    const auto moments_y =
        filtering::UnscentedTransform::recover_gaussian(sigma_y, weights);

    const Eigen::VectorXd expected_mean_y = A * mean_x + b;
    const Eigen::MatrixXd expected_cov_y = A * cov_x * A.transpose();
    const Eigen::MatrixXd expected_cross_cov = cov_x * A.transpose();

    const Eigen::MatrixXd cross_cov = filtering::UnscentedTransform::compute_cross_covariance(
        sigma_x,
        mean_x,
        sigma_y,
        moments_y.mean,
        weights
    );

    EXPECT_TRUE(moments_y.mean.isApprox(expected_mean_y, 1e-9));
    EXPECT_TRUE(moments_y.covariance.isApprox(expected_cov_y, 1e-8));
    EXPECT_TRUE(cross_cov.isApprox(expected_cross_cov, 1e-8));
}
