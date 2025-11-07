#pragma once

#include <dynamics/dynamics.hpp>
#include <dynamics/ballistic3d.hpp>
#include <integrator/rk4.hpp>
#include <transforms/coord_transforms.hpp>
#include <vector>
#include <memory>

namespace propagator {
class Propagator {
public:
    Propagator(
        std::shared_ptr<dynamics::IDynamics> dynamics,
        std::shared_ptr<integrator::Integrator> integrator,
        double timestep,
        dynamics::CoordinateFrame frame,
        std::shared_ptr<transforms::CoordTransforms> transforms
    ) : dynamics_(dynamics), integrator_(integrator), timestep_(timestep)
    {}

    /// @brief propagate state to specific time
    /// @todo update this to take a termination predicate so user can decide when prop is complete
    ///
    /// @param t0 initial time
    /// @param initial_state initial state of the system before propagation
    /// @param tf time to propagate to
    ///
    /// @return propagated states
    std::vector<std::pair<double, Eigen::VectorXd>> propagate(double t0, const Eigen::VectorXd& initial_state, double tf) const
    {
        std::vector<std::pair<double, Eigen::VectorXd>> trajectory;
        double t = t0;
        Eigen::VectorXd state = initial_state;
        trajectory.emplace_back(t, state);

        while (t < tf) {
            double step_dt = std::min(timestep_, tf - t);
            state = integrator_->step(t, state, step_dt, *dynamics_);
            t += step_dt;
            trajectory.emplace_back(t, state);
        }

        return trajectory;
    }

    /// @brief propagate state to ground impact
    /// @todo update this to take a termination predicate so user can decide when prop is complete
    ///
    /// @param t0 initial time
    /// @param initial_state initial state of the system before propagation
    ///
    /// @return propagated states
    std::vector<std::pair<double, Eigen::VectorXd>> propagate_to_impact(double t0, const Eigen::VectorXd& initial_state) const
    {
        std::vector<std::pair<double, Eigen::VectorXd>> trajectory;
        double t = t0;
        Eigen::VectorXd state = initial_state;
        trajectory.emplace_back(t, state);

        while (true) {
            // Convert state to ECEF if in ECI frame
            Eigen::VectorXd state_ecef = (coordinateFrame_ == dynamics::CoordinateFrame::ECI) ? coordTransforms_->eci_to_ecef(state, t) : state;

            // Check altitude using ECEF-to-LLA conversion
            Eigen::Vector3d lla = coordTransforms_->ecef_to_lla(state_ecef.head(3));
            double altitude = lla(2); // Altitude in meters

            if (altitude <= 0.0) {
                break; // Ground impact detected
            }

            double step_dt = timestep_;
            state = integrator_->step(t, state, step_dt, *dynamics_);
            t += step_dt;
            trajectory.emplace_back(t, state);
        }

        return trajectory;
    }

private:
    /// @brief underlying system dynamics
    std::shared_ptr<dynamics::IDynamics> dynamics_;
    /// @brief underlying integrator to use for propagation
    std::shared_ptr<integrator::Integrator> integrator_;
    /// @brief timestep to use for propagation
    double timestep_;
    /// @brief Coordinate frame to use
    dynamics::CoordinateFrame coordinateFrame_;
    /// @brief coordinate transforms
    std::shared_ptr<transforms::CoordTransforms> coordTransforms_;
};
} // namespace propagator
