#include "propagator/factory.hpp"

#include <stdexcept>

namespace propagator {

auto PropagatorFactory::create_numerical(
    std::shared_ptr<dynamics::IDynamics> dynamics,
    std::shared_ptr<integrator::IIntegrator> integrator,
    double timestep
) -> std::unique_ptr<NumericalPropagator>
{
    if (!dynamics) {
        throw std::invalid_argument("Dynamics cannot be null");
    }
    if (!integrator) {
        throw std::invalid_argument("Integrator cannot be null");
    }
    if (timestep <= 0.0) {
        throw std::invalid_argument("Timestep must be positive");
    }
    
    return std::make_unique<NumericalPropagator>(dynamics, integrator, timestep);
}

auto PropagatorFactory::create_analytical(
    std::shared_ptr<dynamics::IDynamics> dynamics
) -> std::unique_ptr<AnalyticalPropagator>
{
    if (!dynamics) {
        throw std::invalid_argument("Dynamics cannot be null");
    }
    
    return std::make_unique<AnalyticalPropagator>(dynamics);
}

} // namespace propagator
