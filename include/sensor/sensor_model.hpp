#pragma once

#include <Eigen/Dense>

#include <memory>

namespace sensor {

/// @brief Context for sensor measurement computation
struct SensorContext {
    Eigen::VectorXd state;           ///< True state vector
    double time;                     ///< Measurement time
    Eigen::Vector3d sensor_position; ///< Sensor location
    Eigen::Quaterniond sensor_orientation; ///< Sensor orientation
};

/// @brief Abstract sensor model
class ISensorModel {
public:
    virtual ~ISensorModel() = default;

    /// @brief Compute ideal (noiseless) measurement
    /// @param ctx Measurement context
    /// @return Expected measurement z = h(x)
    virtual Eigen::VectorXd compute_measurement(const SensorContext& ctx) const = 0;

    /// @brief Get measurement noise covariance
    /// @return Measurement noise covariance R
    virtual Eigen::MatrixXd get_noise_covariance() const = 0;

    /// @brief Compute measurement Jacobian: H = ∂h/∂x
    /// @param ctx Sensor context
    /// @return Jacobian matrix
    virtual Eigen::MatrixXd compute_jacobian(const SensorContext& ctx) const = 0;

    /// @brief Get measurement dimension
    virtual int get_dimension() const = 0;

    // /// @brief Simulate a noisy measurement
    // /// @param ctx Measurement context
    // /// @return Noisy measurement z = h(x) + v
    // virtual Eigen::VectorXd simulate_measurement(const MeasurementContext& ctx) const;
};

} // namespace measurement
