#pragma once

#include "dynamics/dynamics.hpp"
#include "dynamics/gravity.hpp"
#include <Eigen/Dense>
#include <memory>

/// @brief Dynamics model for a 3D ballistic trajectory under constant gravity with no other forces.
class Ballistic3D : public Dynamics {
public:
    /// @brief Constructs a 3D ballistic dynamics model with specified gravity.
    /// @param coordinateFrame Coordinate frame the dynamics object is configured to use.
    /// @param gravityModel gravity model to use for this system.
    Ballistic3D(CoordinateFrame coordinateFrame, std::shared_ptr<GravityModel> gravityModel)
     : Dynamics(coordinateFrame),
       gravityModel_(gravityModel)
    {}

    /// @brief Computes the time derivative of the state vector for a 3D ballistic trajectory.
    /// @param t Current time (in seconds).
    /// @param state 6D state vector [x, y, z, vx, vy, vz] representing position and velocity.
    /// @return 6D derivative vector [dx/dt, dy/dt, dz/dt, d2x/dt, d2y/dt, d2z/dt] representing velocities and accelerations.
    Eigen::VectorXd derivative(double t, const Eigen::VectorXd& state) const override
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

private:
    /// @brief gravity model for this system's dynamics
    std::shared_ptr<GravityModel> gravityModel_;
    // std::shared_ptr<ThrustModel> thrustModel_;
    // std::shared_ptr<DragModel> dragModel_;
};
