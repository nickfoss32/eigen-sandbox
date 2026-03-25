#include "estimation/rts_smoother.hpp"

#include <Eigen/Cholesky>

#include <cmath>
#include <stdexcept>

namespace estimation {

namespace {

void validate_estimate_dimensions(
    const std::vector<SmootherEstimate>& filtered_estimates,
    const std::vector<PredictedEstimate>& predicted_estimates
) {
    if (filtered_estimates.empty()) {
        return;
    }

    const Eigen::Index state_dim = filtered_estimates.front().state.size();
    if (state_dim <= 0) {
        throw std::invalid_argument("RTSSmoother::smooth: state dimension must be positive");
    }

    for (std::size_t i = 0; i < filtered_estimates.size(); ++i) {
        const auto& estimate = filtered_estimates[i];
        if (estimate.state.size() != state_dim ||
            estimate.covariance.rows() != state_dim ||
            estimate.covariance.cols() != state_dim) {
            throw std::invalid_argument(
                "RTSSmoother::smooth: filtered estimate dimensions must be consistent"
            );
        }

        if (!std::isfinite(estimate.time_seconds)) {
            throw std::invalid_argument(
                "RTSSmoother::smooth: filtered estimate times must be finite"
            );
        }
    }

    if (filtered_estimates.size() < 2) {
        return;
    }

    if (predicted_estimates.size() + 1 != filtered_estimates.size()) {
        throw std::invalid_argument(
            "RTSSmoother::smooth: predicted estimate count must equal filtered count minus one"
        );
    }

    for (std::size_t i = 0; i < predicted_estimates.size(); ++i) {
        const auto& predicted = predicted_estimates[i];
        if (predicted.predicted_state.size() != state_dim ||
            predicted.predicted_covariance.rows() != state_dim ||
            predicted.predicted_covariance.cols() != state_dim ||
            predicted.transition_jacobian.rows() != state_dim ||
            predicted.transition_jacobian.cols() != state_dim) {
            throw std::invalid_argument(
                "RTSSmoother::smooth: predicted estimate dimensions must match filtered state dimension"
            );
        }

        if (!std::isfinite(predicted.start_time_seconds) ||
            !std::isfinite(predicted.end_time_seconds)) {
            throw std::invalid_argument(
                "RTSSmoother::smooth: predicted estimate times must be finite"
            );
        }

        const double start_time = filtered_estimates[i].time_seconds;
        const double end_time = filtered_estimates[i + 1].time_seconds;
        if (std::abs(predicted.start_time_seconds - start_time) > 1e-9 ||
            std::abs(predicted.end_time_seconds - end_time) > 1e-9) {
            throw std::invalid_argument(
                "RTSSmoother::smooth: predicted estimate times must align with filtered history"
            );
        }
    }
}

} // namespace

auto RTSSmoother::smooth(
    const std::vector<SmootherEstimate>& filtered_estimates,
    const std::vector<PredictedEstimate>& predicted_estimates
) const -> std::vector<SmootherEstimate> {
    if (filtered_estimates.empty()) {
        return {};
    }

    validate_estimate_dimensions(filtered_estimates, predicted_estimates);
    if (filtered_estimates.size() < 2) {
        return filtered_estimates;
    }

    std::vector<SmootherEstimate> smoothed_estimates = filtered_estimates;
    for (int k = static_cast<int>(filtered_estimates.size()) - 2; k >= 0; --k) {
        const auto& filtered = filtered_estimates[static_cast<std::size_t>(k)];
        const auto& predicted = predicted_estimates[static_cast<std::size_t>(k)];

        Eigen::MatrixXd predicted_covariance =
            0.5 * (predicted.predicted_covariance + predicted.predicted_covariance.transpose());
        predicted_covariance.diagonal().array() += 1e-9;

        Eigen::LDLT<Eigen::MatrixXd> ldlt(predicted_covariance);
        if (ldlt.info() != Eigen::Success) {
            continue;
        }

        const Eigen::MatrixXd cross_covariance =
            filtered.covariance * predicted.transition_jacobian.transpose();
        const Eigen::MatrixXd smoother_gain_transpose = ldlt.solve(cross_covariance.transpose());
        if (ldlt.info() != Eigen::Success || !smoother_gain_transpose.allFinite()) {
            continue;
        }

        const Eigen::MatrixXd smoother_gain = smoother_gain_transpose.transpose();
        const auto& next_smoothed = smoothed_estimates[static_cast<std::size_t>(k + 1)];

        smoothed_estimates[static_cast<std::size_t>(k)].state =
            filtered.state +
            smoother_gain * (next_smoothed.state - predicted.predicted_state);
        smoothed_estimates[static_cast<std::size_t>(k)].covariance =
            filtered.covariance +
            smoother_gain *
                (next_smoothed.covariance - predicted.predicted_covariance) *
                smoother_gain.transpose();
        smoothed_estimates[static_cast<std::size_t>(k)].covariance =
            0.5 * (
                smoothed_estimates[static_cast<std::size_t>(k)].covariance +
                smoothed_estimates[static_cast<std::size_t>(k)].covariance.transpose()
            );
    }

    return smoothed_estimates;
}

} // namespace estimation
