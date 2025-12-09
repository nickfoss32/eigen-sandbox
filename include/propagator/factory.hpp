#pragma once

#include "dynamics/dynamics.hpp"
#include "integrator/integrator.hpp"
#include "propagator/numerical_propagator.hpp"
#include "propagator/analytical_propagator.hpp"

#include <memory>

namespace propagator {
/// @brief Factory for creating propagators
class PropagatorFactory {
public:
    /// @brief Create a numerical propagator
    static auto create_numerical(
        std::shared_ptr<dynamics::IDynamics> dynamics,
        std::shared_ptr<integrator::IIntegrator> integrator,
        double timestep
    ) -> std::unique_ptr<NumericalPropagator>;

    /// @brief Create an analytical propagator
    static auto create_analytical(
        std::shared_ptr<dynamics::IDynamics> dynamics
    ) -> std::unique_ptr<AnalyticalPropagator>;
};
} // namespace propagator
