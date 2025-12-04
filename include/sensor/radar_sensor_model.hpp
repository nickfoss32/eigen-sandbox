#pragma once

#include "sensor/sensor_model.hpp"

namespace sensor {

/// @brief Radar sensor model (coordinate frame agnostic)
///
/// Measurement: z = [range, azimuth, elevation]
/// Nonlinear measurement function h(x)
///
/// @note COORDINATE FRAME REQUIREMENT:
/// The radar_position and state vector in ctx.state must be in the SAME frame.
/// - If your filter tracks in ECI: provide radar position in ECI
/// - If your filter tracks in ECEF: provide radar position in ECEF
/// 
/// For ECI tracking: Remember that radar position changes as Earth rotates!
/// You may need to update it at each measurement time.
class RadarSensorModel : public ISensorModel {
public:
    /// @brief Constructor
    /// @param radar_position Radar location in world frame
    /// @param range_noise Range measurement noise std dev (m)
    /// @param angle_noise Angle measurement noise std dev (rad)
    RadarSensorModel(
        const Eigen::Vector3d& radar_position,
        double range_noise,
        double angle_noise
    );

    Eigen::VectorXd compute_measurement(const SensorContext& ctx) const override;
    Eigen::MatrixXd get_noise_covariance() const override;
    Eigen::MatrixXd compute_jacobian(const SensorContext& ctx) const override;
    int get_dimension() const override { return 3; }
    // std::string get_sensor_type() const override { return "radar"; }

    // /// @brief Set radar position
    // void set_position(const Eigen::Vector3d& pos) { radar_position_ = pos; }
    
    // /// @brief Get radar position
    // Eigen::Vector3d get_position() const { return radar_position_; }

private:
    Eigen::Vector3d radar_position_;
    double range_noise_;
    double angle_noise_;
    Eigen::Matrix3d R_;  ///< Measurement noise covariance
};

} // namespace sensor
