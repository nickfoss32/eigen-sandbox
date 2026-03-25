#include <gtest/gtest.h>

#include <estimation/rts_smoother.hpp>

#include <stdexcept>
#include <vector>

namespace {

using estimation::PredictedEstimate;
using estimation::RTSSmoother;
using estimation::SmootherEstimate;

TEST(RTSSmootherTest, ComputesExpectedBackwardPassForScalarHistory) {
    const std::vector<SmootherEstimate> filtered_estimates{
        {0.0, Eigen::VectorXd::Constant(1, 0.0), Eigen::MatrixXd::Constant(1, 1, 1.0)},
        {1.0, Eigen::VectorXd::Constant(1, 0.9), Eigen::MatrixXd::Constant(1, 1, 1.0)},
        {2.0, Eigen::VectorXd::Constant(1, 2.4), Eigen::MatrixXd::Constant(1, 1, 1.0)}
    };
    const std::vector<PredictedEstimate> predicted_estimates{
        {0.0, 1.0, Eigen::VectorXd::Constant(1, 1.0), Eigen::MatrixXd::Constant(1, 1, 2.0),
         Eigen::MatrixXd::Constant(1, 1, 1.0)},
        {1.0, 2.0, Eigen::VectorXd::Constant(1, 1.9), Eigen::MatrixXd::Constant(1, 1, 2.0),
         Eigen::MatrixXd::Constant(1, 1, 1.0)}
    };

    RTSSmoother smoother;
    const auto smoothed = smoother.smooth(filtered_estimates, predicted_estimates);

    ASSERT_EQ(smoothed.size(), filtered_estimates.size());
    EXPECT_NEAR(smoothed[0].state(0), 0.075, 1e-9);
    EXPECT_NEAR(smoothed[1].state(0), 1.15, 1e-9);
    EXPECT_NEAR(smoothed[2].state(0), 2.4, 1e-9);
    EXPECT_NEAR(smoothed[0].covariance(0, 0), 0.6875, 1e-9);
    EXPECT_NEAR(smoothed[1].covariance(0, 0), 0.75, 1e-9);
    EXPECT_NEAR(smoothed[2].covariance(0, 0), 1.0, 1e-9);
}

TEST(RTSSmootherTest, RejectsMismatchedHistoryLengths) {
    const std::vector<SmootherEstimate> filtered_estimates{
        {0.0, Eigen::VectorXd::Zero(2), Eigen::MatrixXd::Identity(2, 2)},
        {1.0, Eigen::VectorXd::Zero(2), Eigen::MatrixXd::Identity(2, 2)}
    };

    RTSSmoother smoother;
    EXPECT_THROW(smoother.smooth(filtered_estimates, {}), std::invalid_argument);
}

TEST(RTSSmootherTest, ReturnsSingleEstimateUnchanged) {
    const std::vector<SmootherEstimate> filtered_estimates{
        {3.0, Eigen::VectorXd::LinSpaced(2, 1.0, 2.0), Eigen::MatrixXd::Identity(2, 2)}
    };

    RTSSmoother smoother;
    const auto smoothed = smoother.smooth(filtered_estimates, {});

    ASSERT_EQ(smoothed.size(), 1U);
    EXPECT_DOUBLE_EQ(smoothed.front().time_seconds, 3.0);
    EXPECT_TRUE(smoothed.front().state.isApprox(filtered_estimates.front().state));
    EXPECT_TRUE(smoothed.front().covariance.isApprox(filtered_estimates.front().covariance));
}

} // namespace
