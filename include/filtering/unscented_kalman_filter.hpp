#pragma once

#include "filtering/kalman_filter_base.hpp"
#include "filtering/unscented_transform.hpp"
#include "propagator/propagator.hpp"
#include "sensor/sensor_model.hpp"

#include <functional>
#include <memory>
#include <vector>

namespace filtering {

/// @brief Unscented Kalman Filter.
///
/// Implements nonlinear state estimation without Jacobian linearization by
/// propagating sigma points through process and measurement models.
///
/// Composition and usage are intentionally similar to ExtendedKalmanFilter:
/// - propagator for nonlinear prediction
/// - sensor model for nonlinear measurement mapping
/// - process noise callback for Q(dt)
class UnscentedKalmanFilter : public KalmanFilterBase {
public:
    /// @brief Callback type that returns process-noise covariance for a dt.
    using ProcessNoiseFunction = std::function<Eigen::MatrixXd(double)>;

    /// @brief Construct a UKF instance.
    /// @param initial_state Initial state estimate.
    /// @param initial_covariance Initial state covariance.
    /// @param propagator Propagator used for nonlinear state prediction.
    /// @param sensor_model Sensor model used for nonlinear measurement prediction.
    /// @param Q_func Process-noise covariance function Q(dt).
    /// @param initial_time Initial filter time in seconds.
    /// @param ut_parameters Unscented transform tuning parameters.
    UnscentedKalmanFilter(
        const Eigen::VectorXd& initial_state,
        const Eigen::MatrixXd& initial_covariance,
        std::shared_ptr<propagator::IPropagator> propagator,
        std::shared_ptr<sensor::ISensorModel> sensor_model,
        ProcessNoiseFunction Q_func,
        double initial_time = 0.0,
        UnscentedTransformParameters ut_parameters = {}
    );

    /// @brief Predict state/covariance forward by dt.
    /// @param dt Time step in seconds.
    void predict(double dt) override;

    /// @brief Incorporate a measurement.
    /// @param measurement Measurement container with z, R, and metadata.
    void update(const common::Measurement& measurement) override;

    /// @brief Compute Gaussian innovation likelihood for IMM mode updates.
    /// @param measurement Measurement to evaluate.
    /// @return Likelihood value (lower-bounded for numerical robustness).
    double get_innovation_likelihood(const common::Measurement& measurement) const override;

    /// @brief Get current state estimate.
    /// @return State vector.
    Eigen::VectorXd get_state() const override { return x_; }

    /// @brief Get current covariance estimate.
    /// @return Covariance matrix.
    Eigen::MatrixXd get_covariance() const override { return P_; }

    /// @brief Set state estimate.
    /// @param state New state vector.
    void set_state(const Eigen::VectorXd& state) override { x_ = state; }

    /// @brief Set covariance estimate.
    /// @param P New covariance matrix.
    void set_covariance(const Eigen::MatrixXd& P) override { P_ = P; }

    /// @brief Get current filter time.
    /// @return Time in seconds.
    double get_time() const override { return current_time_; }

    /// @brief Reset filter state, covariance, and time.
    /// @param initial_state New initial state.
    /// @param initial_covariance New initial covariance.
    /// @param initial_time New filter time.
    void reset(
        const Eigen::VectorXd& initial_state,
        const Eigen::MatrixXd& initial_covariance,
        double initial_time = 0.0
    ) {
        KalmanFilterBase::reset(initial_state, initial_covariance);
        current_time_ = initial_time;
    }

    /// @brief Get the active UT tuning parameters.
    /// @return Current UT parameter set.
    auto get_ut_parameters() const -> UnscentedTransformParameters { return ut_parameters_; }

private:
    /// @brief Cached measurement prediction terms from sigma points.
    struct MeasurementPrediction {
        /// @brief State-space sigma points used in this prediction.
        std::vector<Eigen::VectorXd> sigma_states;

        /// @brief Measurement-space sigma points mapped from sigma_states.
        std::vector<Eigen::VectorXd> sigma_measurements;

        /// @brief Predicted measurement mean.
        Eigen::VectorXd measurement_mean;

        /// @brief Innovation covariance S.
        Eigen::MatrixXd innovation_covariance;

        /// @brief Cross-covariance between state and measurement spaces.
        Eigen::MatrixXd state_measurement_cross_covariance;
    };

    /// @brief Predict measurement distribution for a given measurement context.
    /// @param measurement Measurement carrying time/sensor context/noise.
    /// @return Predicted measurement moments and cross-covariance terms.
    auto predict_measurement_distribution(
        const common::Measurement& measurement
    ) const -> MeasurementPrediction;

    /// @brief Current state estimate.
    Eigen::VectorXd x_;

    /// @brief Current state covariance.
    Eigen::MatrixXd P_;

    /// @brief Nonlinear process propagator.
    std::shared_ptr<propagator::IPropagator> propagator_;

    /// @brief Nonlinear measurement model.
    std::shared_ptr<sensor::ISensorModel> sensor_model_;

    /// @brief Process-noise covariance callback.
    ProcessNoiseFunction Q_func_;

    /// @brief Current filter time in seconds.
    double current_time_ = 0.0;

    /// @brief Unscented transform parameter set used by the filter.
    UnscentedTransformParameters ut_parameters_;
};

} // namespace filtering
