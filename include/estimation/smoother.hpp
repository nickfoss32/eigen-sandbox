#pragma once

#include <Eigen/Dense>

#include <vector>

namespace estimation {

/// @brief Smoothed or filtered estimate at a specific time.
struct SmootherEstimate {
    double time_seconds = 0.0;
    Eigen::VectorXd state;
    Eigen::MatrixXd covariance;
};

/// @brief Predicted transition from one filtered epoch to the next.
struct PredictedEstimate {
    double start_time_seconds = 0.0;
    double end_time_seconds = 0.0;
    Eigen::VectorXd predicted_state;
    Eigen::MatrixXd predicted_covariance;
    Eigen::MatrixXd transition_jacobian;
};

/// @brief Interface for fixed-interval smoothers.
class ISmoother {
public:
    virtual ~ISmoother() = default;

    /// @brief Smooth a filtered state history using one-step prediction records.
    virtual auto smooth(
        const std::vector<SmootherEstimate>& filtered_estimates,
        const std::vector<PredictedEstimate>& predicted_estimates
    ) const -> std::vector<SmootherEstimate> = 0;
};

} // namespace estimation
