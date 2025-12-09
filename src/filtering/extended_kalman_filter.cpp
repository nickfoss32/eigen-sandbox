#include "filtering/extended_kalman_filter.hpp"
#include "sensor/sensor_model.hpp"

#include <stdexcept>

namespace filtering {

ExtendedKalmanFilter::ExtendedKalmanFilter(
    const Eigen::VectorXd& initial_state,
    const Eigen::MatrixXd& initial_covariance,
    std::shared_ptr<propagator::IPropagator> propagator,
    std::shared_ptr<sensor::ISensorModel> sensor_model,
    ProcessNoiseFunction Q_func,
    double initial_time
) : x_(initial_state),
    P_(initial_covariance),
    current_time_(initial_time),
    propagator_(std::move(propagator)),
    sensor_model_(std::move(sensor_model)),
    Q_func_(std::move(Q_func))
{
    // Validate dimensions
    int n = x_.size();
    if (P_.rows() != n || P_.cols() != n) {
        throw std::invalid_argument(
            "Covariance matrix dimensions must match state dimension"
        );
    }
    
    if (!propagator_) {
        throw std::invalid_argument("Propagator cannot be null");
    }
    
    if (!sensor_model_) {
        throw std::invalid_argument("Sensor model cannot be null");
    }
}

void ExtendedKalmanFilter::predict(double dt) {
    if (dt <= 0.0) {
        throw std::invalid_argument("Time step must be positive");
    }
    
    // Save initial state for Jacobian computation
    Eigen::VectorXd x_initial = x_;
    double t_initial = current_time_;
    
    // Propagate state
    double t_end = current_time_ + dt;
    auto trajectory = propagator_->propagate(current_time_, x_, t_end);
    
    if (trajectory.empty()) {
        throw std::runtime_error("Propagator returned empty trajectory");
    }
    
    // Update state to final point
    x_ = trajectory.back().second;
    
    // Get state transition Jacobian from propagator
    // Propagator uses dynamics->compute_jacobian() internally
    Eigen::MatrixXd F = propagator_->compute_transition_jacobian(
        t_initial, x_initial, dt
    );
    
    // Compute process noise
    Eigen::MatrixXd Q = Q_func_(dt);
    
    // Propagate covariance: P⁻ = F*P*Fᵀ + Q
    P_ = F * P_ * F.transpose() + Q;
    
    // Update time
    current_time_ = t_end;
}

void ExtendedKalmanFilter::update(const common::Measurement& measurement) {
    // Create sensor context
    sensor::SensorContext ctx;
    ctx.state = x_;
    ctx.time = measurement.time;
    
    // Predict measurement from current state
    Eigen::VectorXd z_pred = sensor_model_->compute_measurement(ctx);
    
    // Compute measurement Jacobian
    Eigen::MatrixXd H = sensor_model_->compute_jacobian(ctx);
    
    // Get measurement noise covariance
    Eigen::MatrixXd R = measurement.R;
    
    // Innovation (measurement residual)
    Eigen::VectorXd innovation = measurement.z - z_pred;
    
    // Innovation covariance: S = H*P*Hᵀ + R
    Eigen::MatrixXd S = H * P_ * H.transpose() + R;
    
    // Kalman gain: K = P*Hᵀ*S⁻¹
    Eigen::MatrixXd K = P_ * H.transpose() * S.inverse();
    
    // Update state: x⁺ = x⁻ + K*(z - h(x⁻))
    x_ = x_ + K * innovation;
    
    // Update covariance: P⁺ = (I - K*H)*P⁻
    int n = x_.size();
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n, n);
    P_ = (I - K * H) * P_;
    
    // Ensure symmetry (numerical stability)
    P_ = 0.5 * (P_ + P_.transpose());
    
    // Update time to measurement time
    current_time_ = measurement.time;
}

} // namespace filtering
