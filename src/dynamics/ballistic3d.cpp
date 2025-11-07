#include "ballistic3d.hpp"

namespace dynamics {
Ballistic3D::Ballistic3D(CoordinateFrame coordinateFrame, std::shared_ptr<IGravityModel> gravityModel)
    : IDynamics(coordinateFrame),
      gravityModel_(gravityModel)
{}

Eigen::VectorXd Ballistic3D::derivative(double t, const Eigen::VectorXd& state) const
{
    Eigen::VectorXd der(6);
    Eigen::Vector3d r = state.segment<3>(0); // [x, y, z]
    Eigen::Vector3d v = state.segment<3>(3); // [vx, vy, vz]

    // dx/dt = vx, dy/dt = vy, dz/dt = vz
    der.segment<3>(0) = v;

    // Gravitational acceleration from model
    Eigen::Vector3d a_total = gravityModel_->compute_force(r);
    // a_total += dragModel_->compute_force();
    // a_total += thrustModel_->compute_force();

    // Add fictitious forces for ECEF frame
    if (coordinateFrame_ == CoordinateFrame::ECEF) {
        // Earth's angular velocity (rad/s)
        Eigen::Vector3d omega(0.0, 0.0, 7.292115e-5);

        // Coriolis acceleration: -2 * omega x v
        Eigen::Vector3d a_coriolis = -2.0 * omega.cross(v);

        // Centrifugal acceleration: -omega x (omega x r)
        Eigen::Vector3d a_centrifugal = -omega.cross(omega.cross(r));

        a_total += a_coriolis + a_centrifugal;
    }

    // dv/dt
    der.segment<3>(3) = a_total;

    return der;
}
} // namespace dynamics
