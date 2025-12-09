
#include "propagator/numerical_propagator.hpp"

#include <cmath>

namespace propagator {
NumericalPropagator::NumericalPropagator(
    std::shared_ptr<dynamics::IDynamics> dynamics,
    std::shared_ptr<integrator::IIntegrator> integrator,
    double timestep
) : dynamics_(dynamics), integrator_(integrator), timestep_(timestep)
{
    if (!dynamics_) {
        throw std::invalid_argument("Dynamics cannot be null");
    }
    if (!integrator_) {
        throw std::invalid_argument("Integrator cannot be null");
    }
    if (timestep_ <= 0.0) {
        throw std::invalid_argument("timestep must be positive");
    }
}

auto NumericalPropagator::propagate(double t0, const Eigen::VectorXd& initial_state, double tf) const -> std::vector<std::pair<double, Eigen::VectorXd>>
{
    std::vector<std::pair<double, Eigen::VectorXd>> trajectory;
    double t = t0;
    Eigen::VectorXd state = initial_state;
    trajectory.emplace_back(t, state);

    // Determine direction of propagation
    double direction = (tf >= t0) ? 1.0 : -1.0;
    double abs_timestep = std::abs(timestep_);
    
    while (direction * (tf - t) > 0) { // Continue while not reached tf
        double step_dt = direction * std::min(abs_timestep, std::abs(tf - t));
        state = integrator_->step(t, state, step_dt, *dynamics_);
        t += step_dt;
        trajectory.emplace_back(t, state);
    }

    return trajectory;
}

auto NumericalPropagator::compute_transition_jacobian(double t0, const Eigen::VectorXd& state, double dt) const -> Eigen::MatrixXd
{
    if (dt <= 0.0) {
        throw std::invalid_argument("Time step must be positive");
    }
    
    int n = state.size();
    Eigen::MatrixXd Phi = Eigen::MatrixXd::Zero(n, n);
    
    // Propagate nominal state
    auto traj_nominal = propagate(t0, state, t0 + dt);
    Eigen::VectorXd x_final = traj_nominal.back().second;
    
    // Compute each column of Jacobian via finite differences
    for (int i = 0; i < n; ++i) {
        // FIXED: Use adaptive epsilon based on state magnitude
        // For position (meters), use ~0.1m perturbation
        // For velocity (m/s), use ~0.1 m/s perturbation
        double state_magnitude = std::abs(state(i));
        double epsilon;
        
        if (state_magnitude > 1.0) {
            // For large values: use relative perturbation
            epsilon = state_magnitude * 1e-4;  // 0.01% of value
        } else {
            // For small values: use absolute perturbation
            epsilon = 1e-3;  // 1mm for position, 1mm/s for velocity
        }
        
        Eigen::VectorXd x_pert = state;
        x_pert(i) += epsilon;
        
        auto traj_pert = propagate(t0, x_pert, t0 + dt);
        Eigen::VectorXd x_pert_final = traj_pert.back().second;
        
        // ∂x_f/∂x_0[i] ≈ (x_pert_final - x_final) / ε
        Phi.col(i) = (x_pert_final - x_final) / epsilon;
    }
    
    return Phi;
}
} // namespace propagator
