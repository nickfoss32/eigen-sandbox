#include "dynamics/forces/coordinated_turn_force.hpp"

namespace dynamics {

CoordinatedTurnForce::CoordinatedTurnForce(double turn_rate_rad_s)
    : omega_(turn_rate_rad_s) {}

auto CoordinatedTurnForce::compute_acceleration(const ForceContext& ctx) const
    -> Eigen::Vector3d {
    const double vx = ctx.velocity(0);
    const double vy = ctx.velocity(1);
    return Eigen::Vector3d(-omega_ * vy, omega_ * vx, 0.0);
}

auto CoordinatedTurnForce::compute_jacobian(const ForceContext& /*ctx*/) const
    -> std::pair<Eigen::Matrix3d, Eigen::Matrix3d> {
    Eigen::Matrix3d da_dv = Eigen::Matrix3d::Zero();
    da_dv(0, 1) = -omega_;
    da_dv(1, 0) = omega_;
    return {Eigen::Matrix3d::Zero(), da_dv};
}

} // namespace dynamics
