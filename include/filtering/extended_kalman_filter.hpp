#pragma once

#include "filtering/kalman_filter_base.hpp"
#include "propagator/propagator.hpp"
#include "sensor/sensor_model.hpp"

#include <functional>
#include <memory>

namespace filtering {

/// @brief Extended Kalman Filter
/// 
/// Implements nonlinear state estimation using linearization.
/// Uses propagator for state prediction and sensor model for updates.
class ExtendedKalmanFilter : public KalmanFilterBase {
public:
    using ProcessNoiseFunction = std::function<Eigen::MatrixXd(double)>;

    /// @brief Constructor
    /// @param initial_state Initial state estimate
    /// @param initial_covariance Initial covariance matrix
    /// @param propagator Propagator for state prediction
    /// @param sensor_model Sensor model for measurements
    /// @param Q_func Function to compute process noise covariance
    /// @param initial_time Initial time (default: 0.0)
    ExtendedKalmanFilter(
        const Eigen::VectorXd& initial_state,
        const Eigen::MatrixXd& initial_covariance,
        std::shared_ptr<propagator::IPropagator> propagator,
        std::shared_ptr<sensor::ISensorModel> sensor_model,
        ProcessNoiseFunction Q_func,
        double initial_time = 0.0
    );

    void predict(double dt) override;
    void update(const common::Measurement& measurement) override;
    
    Eigen::VectorXd get_state() const override { return x_; }
    Eigen::MatrixXd get_covariance() const override { return P_; }
    void set_state(const Eigen::VectorXd& state) override { x_ = state; }
    void set_covariance(const Eigen::MatrixXd& P) override { P_ = P; }
    double get_time() const override { return current_time_; }

    /// @brief Reset filter to new initial conditions
    /// @param initial_state New initial state
    /// @param initial_covariance New initial covariance
    /// @param initial_time New initial time
    void reset(
        const Eigen::VectorXd& initial_state,
        const Eigen::MatrixXd& initial_covariance,
        double initial_time = 0.0
    ) override {
        KalmanFilterBase::reset(initial_state, initial_covariance, initial_time);
        current_time_ = initial_time;
    }

private:
    Eigen::VectorXd x_;                                    ///< State estimate
    Eigen::MatrixXd P_;                                    ///< Covariance matrix
    std::shared_ptr<propagator::IPropagator> propagator_;  ///< State propagator
    std::shared_ptr<sensor::ISensorModel> sensor_model_;   ///< Sensor model (measurement model)
    ProcessNoiseFunction Q_func_;                          ///< Process noise covariance
    double current_time_;                                  ///< Current filter time
};

} // namespace filtering
