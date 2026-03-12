#include "sensor/space_based_azel_sensor_model.hpp"

#include <cmath>
#include <stdexcept>

namespace sensor {

namespace {

auto validate_orientation(const Eigen::Quaterniond& orientation) -> Eigen::Quaterniond {
    if (!orientation.coeffs().allFinite()) {
        throw std::invalid_argument(
            "SpaceBasedAzElSensorModel: ctx.sensor_orientation must be finite"
        );
    }

    const double norm = orientation.norm();
    if (!(norm > 0.0)) {
        throw std::invalid_argument(
            "SpaceBasedAzElSensorModel: ctx.sensor_orientation must be non-zero"
        );
    }

    return orientation.normalized();
}

} // namespace

SpaceBasedAzElSensorModel::SpaceBasedAzElSensorModel(
    double azimuth_noise,
    double elevation_noise
) : R_(Eigen::Matrix2d::Zero()) {
    if (azimuth_noise <= 0.0 || elevation_noise <= 0.0) {
        throw std::invalid_argument(
            "SpaceBasedAzElSensorModel: angle noise std dev must be positive"
        );
    }

    R_(0, 0) = azimuth_noise * azimuth_noise;
    R_(1, 1) = elevation_noise * elevation_noise;
}

Eigen::VectorXd SpaceBasedAzElSensorModel::compute_measurement(const SensorContext& ctx) const {
    if (ctx.state.size() < 3) {
        throw std::invalid_argument(
            "SpaceBasedAzElSensorModel: state must have at least 3 elements"
        );
    }
    if (!ctx.sensor_position.allFinite()) {
        throw std::invalid_argument(
            "SpaceBasedAzElSensorModel: ctx.sensor_position must be finite"
        );
    }

    const Eigen::Quaterniond q_sensor_to_world = validate_orientation(ctx.sensor_orientation);
    const Eigen::Vector3d los_world = ctx.state.head<3>() - ctx.sensor_position;
    const Eigen::Vector3d los_sensor = q_sensor_to_world.conjugate() * los_world;

    const double x = los_sensor.x();
    const double y = los_sensor.y();
    const double z = los_sensor.z();
    const double rho_xy = std::sqrt(x * x + y * y);

    Eigen::Vector2d z_meas;
    z_meas(0) = std::atan2(y, x);
    z_meas(1) = std::atan2(z, rho_xy);
    return z_meas;
}

Eigen::MatrixXd SpaceBasedAzElSensorModel::compute_jacobian(const SensorContext& ctx) const {
    if (ctx.state.size() < 3) {
        throw std::invalid_argument(
            "SpaceBasedAzElSensorModel: state must have at least 3 elements"
        );
    }
    if (!ctx.sensor_position.allFinite()) {
        throw std::invalid_argument(
            "SpaceBasedAzElSensorModel: ctx.sensor_position must be finite"
        );
    }

    const Eigen::Quaterniond q_sensor_to_world = validate_orientation(ctx.sensor_orientation);
    const Eigen::Matrix3d world_to_sensor =
        q_sensor_to_world.conjugate().toRotationMatrix();

    const Eigen::Vector3d los_world = ctx.state.head<3>() - ctx.sensor_position;
    const Eigen::Vector3d los_sensor = world_to_sensor * los_world;

    const double x = los_sensor.x();
    const double y = los_sensor.y();
    const double z = los_sensor.z();
    const double rho2 = los_sensor.squaredNorm();
    const double rho_xy2 = x * x + y * y;
    const double rho_xy = std::sqrt(rho_xy2);

    const int state_dim = ctx.state.size();
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2, state_dim);

    constexpr double kEps = 1e-12;
    if (rho2 < kEps || rho_xy2 < kEps || rho_xy < kEps) {
        return H;
    }

    Eigen::Matrix<double, 2, 3> H_sensor = Eigen::Matrix<double, 2, 3>::Zero();
    H_sensor(0, 0) = -y / rho_xy2;
    H_sensor(0, 1) =  x / rho_xy2;
    H_sensor(1, 0) = -x * z / (rho2 * rho_xy);
    H_sensor(1, 1) = -y * z / (rho2 * rho_xy);
    H_sensor(1, 2) =  rho_xy / rho2;

    H.block<2, 3>(0, 0) = H_sensor * world_to_sensor;
    return H;
}

Eigen::MatrixXd SpaceBasedAzElSensorModel::get_noise_covariance() const {
    return R_;
}

} // namespace sensor
