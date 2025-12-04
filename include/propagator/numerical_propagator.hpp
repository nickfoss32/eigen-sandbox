#pragma once

#include "propagator/propagator.hpp"
#include "dynamics/dynamics.hpp"
#include "integrator/integrator.hpp"

#include <Eigen/Dense>

#include <vector>
#include <memory>
#include <utility>

namespace propagator {

/// @brief Propagator class for simulating system dynamics over time
class NumericalPropagator : public IPropagator {
public:
    /// @brief Constructor
    /// @param dynamics Dynamics model to use for propagation
    /// @param integrator Integrator to use for propagation
    /// @param timestep Timestep to use for propagation
    NumericalPropagator(
        std::shared_ptr<dynamics::IDynamics> dynamics,
        std::shared_ptr<integrator::IIntegrator> integrator,
        double timestep
    );

    /// @brief propagate state to specific time
    ///
    /// @param t0 initial time
    /// @param initial_state initial state of the system before propagation
    /// @param tf time to propagate to
    ///
    /// @return propagated states
    auto propagate(double t0, const Eigen::VectorXd& initial_state, double tf) const -> std::vector<std::pair<double, Eigen::VectorXd>> override;

    /// @brief Compute state transition Jacobian using numerical differentiation
    /// @param t0 Initial time
    /// @param state Initial state
    /// @param dt Time step
    /// @return State transition matrix Φ(t0+dt, t0)
    auto compute_transition_jacobian(double t0, const Eigen::VectorXd& state, double dt) const -> Eigen::MatrixXd override;

private:
    /// @brief underlying system dynamics
    std::shared_ptr<dynamics::IDynamics> dynamics_;
    /// @brief underlying integrator to use for propagation
    std::shared_ptr<integrator::IIntegrator> integrator_;
    /// @brief timestep to use for propagation
    double timestep_;
};
} // namespace propagator
