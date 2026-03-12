#pragma once

#include "sensor/sensor_model.hpp"

#include <Eigen/Dense>

namespace sensor {

/// @brief Space-based azimuth/elevation angle-only sensor model in ECI.
///
/// Measurement: z = [azimuth, elevation]
/// - azimuth   = atan2(y_s, x_s)
/// - elevation = atan2(z_s, sqrt(x_s^2 + y_s^2))
///
/// where [x_s, y_s, z_s] is the line-of-sight vector expressed in the
/// sensor's local frame. `ctx.sensor_orientation` is interpreted as the
/// rotation from the sensor frame into the world/state frame.
class SpaceBasedAzElSensorModel : public ISensorModel {
public:
    /// @brief Constructor.
    /// @param azimuth_noise Azimuth noise std dev [rad]
    /// @param elevation_noise Elevation noise std dev [rad]
    SpaceBasedAzElSensorModel(
        double azimuth_noise,
        double elevation_noise
    );

    Eigen::VectorXd compute_measurement(const SensorContext& ctx) const override;
    Eigen::MatrixXd get_noise_covariance() const override;
    Eigen::MatrixXd compute_jacobian(const SensorContext& ctx) const override;
    int get_dimension() const override { return 2; }

private:
    Eigen::Matrix2d R_;
};

} // namespace sensor
