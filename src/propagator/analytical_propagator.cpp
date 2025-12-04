#include "propagator/analytical_propagator.hpp"

namespace propagator {
AnalyticalPropagator::AnalyticalPropagator(std::shared_ptr<dynamics::IDynamics> dynamics)
: dynamics_(dynamics)
{}

auto 
AnalyticalPropagator::propagate(double t0, const Eigen::VectorXd& initial_state, double tf) const -> std::vector<std::pair<double, Eigen::VectorXd>>
{
    std::vector<std::pair<double, Eigen::VectorXd>> trajectory;
    
    // For analytical: just initial and final state (no intermediate steps needed)
    trajectory.emplace_back(t0, initial_state);
    
    // Use analytical solution to jump directly to final time
    // Eigen::VectorXd final_state = dynamics_->solve(t0, initial_state, tf);
    if (!dynamics_->has_analytical_solution()) {
        throw std::runtime_error("Dynamics model does not support analytical solution");
    }
    auto final_state_opt = dynamics_->solve_analytical(t0, initial_state, tf);
    if (!final_state_opt.has_value()) {
        throw std::runtime_error("Analytical solution failed to compute final state");
    }
    Eigen::VectorXd final_state = final_state_opt.value();
    trajectory.emplace_back(tf, final_state);
    
    return trajectory;
}
} // namespace propagator
