#include "filtering/unscented_kalman_filter.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace filtering {

UnscentedKalmanFilter::UnscentedKalmanFilter(
    const Eigen::VectorXd& initial_state,
    const Eigen::MatrixXd& initial_covariance,
    std::shared_ptr<propagator::IPropagator> propagator,
    std::shared_ptr<sensor::ISensorModel> sensor_model,
    ProcessNoiseFunction Q_func,
    double initial_time,
    UnscentedTransformParameters ut_parameters
) : x_(initial_state),
    P_(initial_covariance),
    propagator_(std::move(propagator)),
    sensor_model_(std::move(sensor_model)),
    Q_func_(std::move(Q_func)),
    current_time_(initial_time),
    ut_parameters_(ut_parameters) {
    const int n = x_.size();
    if (n <= 0) {
        throw std::invalid_argument("UKF: initial state cannot be empty");
    }
    if (P_.rows() != n || P_.cols() != n) {
        throw std::invalid_argument(
            "UKF: covariance matrix dimensions must match state dimension"
        );
    }
    if (!propagator_) {
        throw std::invalid_argument("UKF: propagator cannot be null");
    }
    if (!sensor_model_) {
        throw std::invalid_argument("UKF: sensor model cannot be null");
    }
    if (!Q_func_) {
        throw std::invalid_argument("UKF: process noise function cannot be empty");
    }

    // Validate UT parameters early.
    (void)UnscentedTransform::compute_weights(n, ut_parameters_);
}

void UnscentedKalmanFilter::predict(double dt) {
    if (dt <= 0.0) {
        throw std::invalid_argument("UKF: time step must be positive");
    }

    const int n = x_.size();
    const double t_end = current_time_ + dt;

    const Eigen::MatrixXd Q = Q_func_(dt);
    if (Q.rows() != n || Q.cols() != n) {
        throw std::invalid_argument(
            "UKF: process noise covariance dimensions must match state dimension"
        );
    }

    const auto predicted = UnscentedTransform::transform_distribution(
        x_,
        P_,
        [this, t_end, n](const Eigen::VectorXd& sigma_point) {
            const auto trajectory = propagator_->propagate(current_time_, sigma_point, t_end);
            if (trajectory.empty()) {
                throw std::runtime_error("UKF: propagator returned empty trajectory");
            }

            const Eigen::VectorXd propagated_state = trajectory.back().second;
            if (propagated_state.size() != n) {
                throw std::runtime_error(
                    "UKF: propagated sigma point has inconsistent state dimension"
                );
            }
            return propagated_state;
        },
        Q,
        ut_parameters_
    );

    x_ = predicted.moments.mean;
    P_ = predicted.moments.covariance;
    predicted_sigma_points_ = predicted.transformed_sigma_points;
    predicted_sigma_weights_ = predicted.weights;
    has_predicted_sigma_points_ = true;
    current_time_ = t_end;
}

auto UnscentedKalmanFilter::get_state_sigma_point_set() const -> StateSigmaPointSet {
    if (has_predicted_sigma_points_) {
        return StateSigmaPointSet{predicted_sigma_points_, predicted_sigma_weights_};
    }

    StateSigmaPointSet sigma_point_set;
    sigma_point_set.weights = UnscentedTransform::compute_weights(x_.size(), ut_parameters_);
    sigma_point_set.sigma_points = UnscentedTransform::generate_sigma_points(
        x_,
        P_,
        ut_parameters_
    );
    return sigma_point_set;
}

auto UnscentedKalmanFilter::predict_measurement_distribution(
    const common::Measurement& measurement
) const -> MeasurementPrediction {
    const StateSigmaPointSet sigma_point_set = get_state_sigma_point_set();

    MeasurementPrediction prediction;
    prediction.sigma_states = sigma_point_set.sigma_points;
    prediction.sigma_measurements.reserve(prediction.sigma_states.size());

    for (const Eigen::VectorXd& sigma_state : prediction.sigma_states) {
        sensor::SensorContext ctx;
        ctx.state = sigma_state;
        ctx.time = measurement.time;
        ctx.sensor_position = measurement.sensor_position;
        ctx.sensor_orientation = measurement.sensor_orientation;
        prediction.sigma_measurements.push_back(sensor_model_->compute_measurement(ctx));
    }

    const auto measurement_moments = UnscentedTransform::recover_gaussian(
        prediction.sigma_measurements,
        sigma_point_set.weights,
        measurement.R
    );

    if (measurement_moments.mean.size() != measurement.z.size()) {
        throw std::invalid_argument(
            "UKF: measurement dimension does not match sensor output dimension"
        );
    }

    prediction.measurement_mean = measurement_moments.mean;
    prediction.innovation_covariance = measurement_moments.covariance;
    prediction.state_measurement_cross_covariance = UnscentedTransform::compute_cross_covariance(
        prediction.sigma_states,
        x_,
        prediction.sigma_measurements,
        prediction.measurement_mean,
        sigma_point_set.weights
    );

    return prediction;
}

void UnscentedKalmanFilter::update(const common::Measurement& measurement) {
    const auto prediction = predict_measurement_distribution(measurement);

    const Eigen::VectorXd innovation = measurement.z - prediction.measurement_mean;

    Eigen::LDLT<Eigen::MatrixXd> ldlt(prediction.innovation_covariance);
    if (ldlt.info() != Eigen::Success) {
        throw std::runtime_error("UKF: innovation covariance decomposition failed");
    }

    const Eigen::MatrixXd rhs = prediction.state_measurement_cross_covariance.transpose();
    const Eigen::MatrixXd solved = ldlt.solve(rhs);
    if (ldlt.info() != Eigen::Success || !solved.allFinite()) {
        throw std::runtime_error("UKF: failed to solve for Kalman gain");
    }

    const Eigen::MatrixXd K = solved.transpose();

    x_ = x_ + K * innovation;
    P_ = P_ - K * prediction.innovation_covariance * K.transpose();
    P_ = 0.5 * (P_ + P_.transpose());

    current_time_ = measurement.time;
    predicted_sigma_points_.clear();
    predicted_sigma_weights_.mean.resize(0);
    predicted_sigma_weights_.covariance.resize(0);
    predicted_sigma_weights_.lambda = 0.0;
    has_predicted_sigma_points_ = false;
}

double UnscentedKalmanFilter::get_innovation_likelihood(
    const common::Measurement& measurement
) const {
    constexpr double kMinLikelihood = 1e-12;

    const auto prediction = predict_measurement_distribution(measurement);
    const Eigen::VectorXd innovation = measurement.z - prediction.measurement_mean;

    Eigen::LDLT<Eigen::MatrixXd> ldlt(prediction.innovation_covariance);
    if (ldlt.info() != Eigen::Success) {
        return kMinLikelihood;
    }

    const Eigen::VectorXd D = ldlt.vectorD();
    double log_det = 0.0;
    for (int i = 0; i < D.size(); ++i) {
        if (!(D(i) > 0.0) || !std::isfinite(D(i))) {
            return kMinLikelihood;
        }
        log_det += std::log(D(i));
    }

    const Eigen::VectorXd solved = ldlt.solve(innovation);
    if (ldlt.info() != Eigen::Success || !solved.allFinite()) {
        return kMinLikelihood;
    }

    const double mahalanobis = innovation.dot(solved);
    if (!std::isfinite(mahalanobis)) {
        return kMinLikelihood;
    }

    const double meas_dim = static_cast<double>(innovation.size());
    const double log_likelihood =
        -0.5 * (meas_dim * std::log(2.0 * std::numbers::pi) + log_det + mahalanobis);

    if (!std::isfinite(log_likelihood)) {
        return kMinLikelihood;
    }

    return std::max(std::exp(log_likelihood), kMinLikelihood);
}

} // namespace filtering
