#include "sensor/space_based_optical_sensor_model.hpp"

#include <cmath>
#include <stdexcept>

namespace sensor {

SpaceBasedOpticalSensorModel::SpaceBasedOpticalSensorModel(
    double ra_noise,
    double dec_noise
) : R_(Eigen::Matrix2d::Zero())
{
    if (ra_noise <= 0.0 || dec_noise <= 0.0) {
        throw std::invalid_argument("SpaceBasedOpticalSensorModel: angle noise std dev must be positive");
    }

    R_(0, 0) = ra_noise * ra_noise;
    R_(1, 1) = dec_noise * dec_noise;
}

Eigen::VectorXd SpaceBasedOpticalSensorModel::compute_measurement(const SensorContext& ctx) const {
    if (ctx.state.size() < 3) {
        throw std::invalid_argument("SpaceBasedOpticalSensorModel: state must have at least 3 elements");
    }
    if (!ctx.sensor_position.allFinite()) {
        throw std::invalid_argument("SpaceBasedOpticalSensorModel: ctx.sensor_position must be finite");
    }

    const Eigen::Vector3d target_pos_eci = ctx.state.head<3>();
    const Eigen::Vector3d los = target_pos_eci - ctx.sensor_position;

    const double x = los.x();
    const double y = los.y();
    const double z = los.z();
    const double rho_xy = std::sqrt(x * x + y * y);

    Eigen::Vector2d z_meas;
    z_meas(0) = std::atan2(y, x);          // Right ascension
    z_meas(1) = std::atan2(z, rho_xy);     // Declination
    return z_meas;
}

Eigen::MatrixXd SpaceBasedOpticalSensorModel::compute_jacobian(const SensorContext& ctx) const {
    if (ctx.state.size() < 3) {
        throw std::invalid_argument("SpaceBasedOpticalSensorModel: state must have at least 3 elements");
    }
    if (!ctx.sensor_position.allFinite()) {
        throw std::invalid_argument("SpaceBasedOpticalSensorModel: ctx.sensor_position must be finite");
    }

    const Eigen::Vector3d target_pos_eci = ctx.state.head<3>();
    const Eigen::Vector3d los = target_pos_eci - ctx.sensor_position;

    const double x = los.x();
    const double y = los.y();
    const double z = los.z();
    const double rho2 = los.squaredNorm();
    const double rho_xy2 = x * x + y * y;
    const double rho_xy = std::sqrt(rho_xy2);

    const int state_dim = ctx.state.size();
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2, state_dim);

    constexpr double kEps = 1e-12;
    if (rho2 < kEps || rho_xy2 < kEps || rho_xy < kEps) {
        return H;
    }

    // d(ra)/d(position)
    H(0, 0) = -y / rho_xy2;
    H(0, 1) =  x / rho_xy2;
    H(0, 2) =  0.0;

    // d(dec)/d(position)
    H(1, 0) = -x * z / (rho2 * rho_xy);
    H(1, 1) = -y * z / (rho2 * rho_xy);
    H(1, 2) =  rho_xy / rho2;

    // d(angles)/d(velocity) = 0, remaining columns already zero.
    return H;
}

Eigen::MatrixXd SpaceBasedOpticalSensorModel::get_noise_covariance() const {
    return R_;
}

} // namespace sensor
