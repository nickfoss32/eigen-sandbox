#pragma once

#include "propagator/propagator.hpp"
#include "dynamics/dynamics.hpp"

#include <Eigen/Dense>

#include <vector>
#include <memory>

namespace propagator {

/// @brief Propagator class for simulating system dynamics over time
class AnalyticalPropagator : public IPropagator {
public:
    /// @brief Constructor
    /// @param dynamics Dynamics model to use for propagation
    AnalyticalPropagator(std::shared_ptr<dynamics::IDynamics> dynamics);

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
    auto compute_transition_jacobian(double t0, const Eigen::VectorXd& state, double dt) const -> Eigen::MatrixXd override {
        throw std::runtime_error("AnalyticalPropagator does not implement compute_transition_jacobian");
    }

private:
    /// @brief underlying system dynamics
    std::shared_ptr<dynamics::IDynamics> dynamics_;
};
} // namespace propagator
