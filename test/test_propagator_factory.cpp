#include <gtest/gtest.h>


#include "propagator/factory.hpp"
#include "dynamics/dynamics.hpp"
#include "integrator/integrator.hpp"

#include <Eigen/Dense>

#include <memory>

using namespace propagator;
using namespace dynamics;
using namespace integrator;

// Mock numerical dynamics for testing
class MockNumericalDynamics : public IDynamics {
public:
    auto compute_dynamics(double t, const Eigen::VectorXd& state) const -> Eigen::VectorXd override {
        return Eigen::VectorXd::Zero(state.size());
    }

    auto compute_jacobian(double t, const Eigen::VectorXd& state) const -> Eigen::MatrixXd override {
        return Eigen::MatrixXd::Identity(state.size(), state.size());
    }

    auto get_state_dimension() const -> int override {
        return 3;
    }
};

// Mock analytical dynamics for testing
class MockAnalyticalDynamics : public IDynamics {
public:
    auto compute_dynamics(double t, const Eigen::VectorXd& state) const -> Eigen::VectorXd override {
        return Eigen::VectorXd::Zero(state.size());
    }

    auto compute_jacobian(double t, const Eigen::VectorXd& state) const -> Eigen::MatrixXd override {
        return Eigen::MatrixXd::Identity(state.size(), state.size());
    }

    auto get_state_dimension() const -> int override {
        return 3;
    }

    auto has_analytical_solution() const -> bool override {
        return true;
    }

    auto solve_analytical(double t0, const Eigen::VectorXd& state0, double tf) const -> std::optional<Eigen::VectorXd> override {
        return state0; // No change for testing
    }

    auto compute_analytical_stm(double t0, const Eigen::VectorXd& state0, double tf) const -> std::optional<Eigen::MatrixXd> override {
        return Eigen::MatrixXd::Identity(state0.size(), state0.size());
    }
};

// Mock integrator for testing
class MockIntegrator : public IIntegrator {
public:
    Eigen::VectorXd step(double t, const Eigen::VectorXd& state, double dt, const IDynamics& dyn) const override {
        return state;
    }
};

// Test fixture
class PropagatorFactoryTest : public ::testing::Test {
protected:
    std::shared_ptr<MockNumericalDynamics> numerical_dynamics_;
    std::shared_ptr<MockAnalyticalDynamics> analytical_dynamics_;
    std::shared_ptr<MockIntegrator> integrator_;
    
    void SetUp() override {
        numerical_dynamics_ = std::make_shared<MockNumericalDynamics>();
        analytical_dynamics_ = std::make_shared<MockAnalyticalDynamics>();
        integrator_ = std::make_shared<MockIntegrator>();
    }
};

// Test successful creation of numerical propagator
TEST_F(PropagatorFactoryTest, CreateNumericalPropagatorSuccess) {
    double timestep = 0.1;
    
    auto propagator = PropagatorFactory::create_numerical(
        numerical_dynamics_, 
        integrator_, 
        timestep
    );
    
    ASSERT_NE(propagator, nullptr);
    EXPECT_NE(dynamic_cast<NumericalPropagator*>(propagator.get()), nullptr);
}

// Test successful creation of analytical propagator
TEST_F(PropagatorFactoryTest, CreateAnalyticalPropagatorSuccess) {
    auto propagator = PropagatorFactory::create_analytical(analytical_dynamics_);
    
    ASSERT_NE(propagator, nullptr);
    EXPECT_NE(dynamic_cast<AnalyticalPropagator*>(propagator.get()), nullptr);
}

// Test numerical propagator with null dynamics
TEST_F(PropagatorFactoryTest, CreateNumericalWithNullDynamics) {
    double timestep = 0.1;
    
    EXPECT_THROW(
        PropagatorFactory::create_numerical(nullptr, integrator_, timestep),
        std::invalid_argument
    );
}

// Test numerical propagator with null integrator
TEST_F(PropagatorFactoryTest, CreateNumericalWithNullIntegrator) {
    double timestep = 0.1;
    
    EXPECT_THROW(
        PropagatorFactory::create_numerical(numerical_dynamics_, nullptr, timestep),
        std::invalid_argument
    );
}

// Test numerical propagator with zero timestep
TEST_F(PropagatorFactoryTest, CreateNumericalWithZeroTimestep) {
    EXPECT_THROW(
        PropagatorFactory::create_numerical(numerical_dynamics_, integrator_, 0.0),
        std::invalid_argument
    );
}

// Test numerical propagator with negative timestep
TEST_F(PropagatorFactoryTest, CreateNumericalWithNegativeTimestep) {
    EXPECT_THROW(
        PropagatorFactory::create_numerical(numerical_dynamics_, integrator_, -0.1),
        std::invalid_argument
    );
}

// Test analytical propagator with null dynamics
TEST_F(PropagatorFactoryTest, CreateAnalyticalWithNullDynamics) {
    EXPECT_THROW(
        PropagatorFactory::create_analytical(nullptr),
        std::invalid_argument
    );
}

// Test exception messages for null dynamics
TEST_F(PropagatorFactoryTest, NullDynamicsExceptionMessage) {
    try {
        PropagatorFactory::create_numerical(nullptr, integrator_, 0.1);
        FAIL() << "Expected std::invalid_argument";
    } catch (const std::invalid_argument& e) {
        EXPECT_STREQ(e.what(), "Dynamics cannot be null");
    }
}

// Test exception messages for null integrator
TEST_F(PropagatorFactoryTest, NullIntegratorExceptionMessage) {
    try {
        PropagatorFactory::create_numerical(numerical_dynamics_, nullptr, 0.1);
        FAIL() << "Expected std::invalid_argument";
    } catch (const std::invalid_argument& e) {
        EXPECT_STREQ(e.what(), "Integrator cannot be null");
    }
}

// Test exception messages for invalid timestep
TEST_F(PropagatorFactoryTest, InvalidTimestepExceptionMessage) {
    try {
        PropagatorFactory::create_numerical(numerical_dynamics_, integrator_, -0.1);
        FAIL() << "Expected std::invalid_argument";
    } catch (const std::invalid_argument& e) {
        EXPECT_STREQ(e.what(), "Timestep must be positive");
    }
}

// Test that created propagators are usable
TEST_F(PropagatorFactoryTest, NumericalPropagatorIsUsable) {
    double timestep = 0.1;
    auto propagator = PropagatorFactory::create_numerical(
        numerical_dynamics_, 
        integrator_, 
        timestep
    );
    
    Eigen::VectorXd initial_state(3);
    initial_state << 1.0, 2.0, 3.0;
    
    // Should not throw
    EXPECT_NO_THROW({
        auto trajectory = propagator->propagate(0.0, initial_state, 1.0);
    });
}

// Test that created analytical propagators are usable
TEST_F(PropagatorFactoryTest, AnalyticalPropagatorIsUsable) {
    auto propagator = PropagatorFactory::create_analytical(analytical_dynamics_);
    
    Eigen::VectorXd initial_state(3);
    initial_state << 1.0, 2.0, 3.0;
    
    // Should not throw
    EXPECT_NO_THROW({
        auto trajectory = propagator->propagate(0.0, initial_state, 1.0);
    });
}

// Test creation with very small positive timestep
TEST_F(PropagatorFactoryTest, CreateNumericalWithVerySmallTimestep) {
    double tiny_timestep = 1e-10;
    
    auto propagator = PropagatorFactory::create_numerical(
        numerical_dynamics_, 
        integrator_, 
        tiny_timestep
    );
    
    ASSERT_NE(propagator, nullptr);
}

// Test creation with large timestep
TEST_F(PropagatorFactoryTest, CreateNumericalWithLargeTimestep) {
    double large_timestep = 1000.0;
    
    auto propagator = PropagatorFactory::create_numerical(
        numerical_dynamics_, 
        integrator_, 
        large_timestep
    );
    
    ASSERT_NE(propagator, nullptr);
}

// Test that factory returns unique pointers (ownership transfer)
TEST_F(PropagatorFactoryTest, ReturnsUniquePointers) {
    auto propagator1 = PropagatorFactory::create_numerical(
        numerical_dynamics_, 
        integrator_, 
        0.1
    );
    
    auto propagator2 = PropagatorFactory::create_numerical(
        numerical_dynamics_, 
        integrator_, 
        0.1
    );
    
    // Each call should return a different object
    EXPECT_NE(propagator1.get(), propagator2.get());
}

// Test multiple null parameters
TEST_F(PropagatorFactoryTest, CreateNumericalWithMultipleNulls) {
    // All null
    EXPECT_THROW(
        PropagatorFactory::create_numerical(nullptr, nullptr, 0.1),
        std::invalid_argument
    );
    
    // Null dynamics and invalid timestep
    EXPECT_THROW(
        PropagatorFactory::create_numerical(nullptr, integrator_, -0.1),
        std::invalid_argument
    );
    
    // Null integrator and invalid timestep
    EXPECT_THROW(
        PropagatorFactory::create_numerical(numerical_dynamics_, nullptr, 0.0),
        std::invalid_argument
    );
}

// Test that factory methods are static (can be called without instance)
TEST_F(PropagatorFactoryTest, StaticMethodsWork) {
    // Should be able to call without creating a factory instance
    auto propagator = PropagatorFactory::create_numerical(
        numerical_dynamics_, 
        integrator_, 
        0.1
    );
    
    EXPECT_NE(propagator, nullptr);
    
    auto analytical = PropagatorFactory::create_analytical(analytical_dynamics_);
    EXPECT_NE(analytical, nullptr);
}
