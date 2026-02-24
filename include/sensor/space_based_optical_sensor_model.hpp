#pragma once

#include "sensor/sensor_model.hpp"

#include <Eigen/Dense>

namespace sensor {

/// @brief Space-based optical angle-only sensor model in ECI frame.
///
/// Measurement: z = [right_ascension, declination]
/// - right_ascension = atan2(y, x)
/// - declination = atan2(z, sqrt(x^2 + y^2))
///
/// where [x, y, z] is line-of-sight vector from sensor to target in ECI.
///
/// @note
/// - Target state position must be in ECI.
/// - Sensor position must be provided in ctx.sensor_position (ECI).
class SpaceBasedOpticalSensorModel : public ISensorModel {
public:
    /// @brief Constructor.
    /// @param ra_noise Right ascension noise std dev [rad]
    /// @param dec_noise Declination noise std dev [rad]
    SpaceBasedOpticalSensorModel(
        double ra_noise,
        double dec_noise
    );

    Eigen::VectorXd compute_measurement(const SensorContext& ctx) const override;
    Eigen::MatrixXd get_noise_covariance() const override;
    Eigen::MatrixXd compute_jacobian(const SensorContext& ctx) const override;
    int get_dimension() const override { return 2; }

private:
    Eigen::Matrix2d R_;
};

} // namespace sensor
