#pragma once

#include <Eigen/Dense>
#include "integrator.hpp"

namespace integrator {
/// @brief Runge-Kutta 4 integrator implementation
class RK4Integrator : public Integrator {
public:
    /// @brief Computes the next state by integrating the state derivative over a time step based on RK4 implementation.
    /// @param t Current time (in seconds).
    /// @param state Current state vector (e.g., position and velocity components).
    /// @param dt Time step for integration (in seconds).
    /// @param dyn Dynamics model providing the state derivative.
    /// @return The updated state vector after one integration step.
    Eigen::VectorXd step(double t, const Eigen::VectorXd& state, double dt, const dynamics::IDynamics& dyn) const override;
};
} // namespace integrator
