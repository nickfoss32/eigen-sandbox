#include <gtest/gtest.h>

#include "dynamics/point_mass_dynamics.hpp"
#include "dynamics/force.hpp"

#include <Eigen/Dense>

#include <memory>
#include <cmath>

using namespace dynamics;

// Mock force: constant acceleration in x-direction
class ConstantForce : public IForce {
public:
    explicit ConstantForce(const Eigen::Vector3d& acceleration)
        : acceleration_(acceleration) {}
    
    auto compute_acceleration(const ForceContext& ctx) const -> Eigen::Vector3d override {
        return acceleration_;
    }

    auto compute_jacobian(const ForceContext& ctx) const -> std::pair<Eigen::Matrix3d, Eigen::Matrix3d> override {
        return {Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Zero()};
    }
    
private:
    Eigen::Vector3d acceleration_;
};

// Mock force: zero force
class ZeroForce : public IForce {
public:
    auto compute_acceleration(const ForceContext& ctx) const -> Eigen::Vector3d override {
        return Eigen::Vector3d::Zero();
    }

    auto compute_jacobian(const ForceContext& ctx) const -> std::pair<Eigen::Matrix3d, Eigen::Matrix3d> override {
        return {Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Zero()};
    }
};

// Mock force: linear damping (F = -k*v)
class DampingForce : public IForce {
public:
    explicit DampingForce(double coefficient) : k_(coefficient) {}
    
    auto compute_acceleration(const ForceContext& ctx) const -> Eigen::Vector3d override {
        return -k_ * ctx.velocity;
    }

    auto compute_jacobian(const ForceContext& ctx) const -> std::pair<Eigen::Matrix3d, Eigen::Matrix3d> override {
        return {Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Zero()};
    }
    
private:
    double k_;
};

// Mock force: position-dependent (F = -k*r, like spring)
class SpringForce : public IForce {
public:
    explicit SpringForce(double stiffness) : k_(stiffness) {}
    
    auto compute_acceleration(const ForceContext& ctx) const -> Eigen::Vector3d override {
        return -k_ * ctx.position;
    }

    auto compute_jacobian(const ForceContext& ctx) const -> std::pair<Eigen::Matrix3d, Eigen::Matrix3d> override {
        return {Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Zero()};
    }
    
private:
    double k_;
};

// Mock force: time-dependent sinusoidal force
class SinusoidalForce : public IForce {
public:
    explicit SinusoidalForce(double amplitude, double frequency)
        : amplitude_(amplitude), omega_(2.0 * M_PI * frequency) {}
    
    auto compute_acceleration(const ForceContext& ctx) const -> Eigen::Vector3d override {
        double magnitude = amplitude_ * std::sin(omega_ * ctx.t);
        return Eigen::Vector3d(magnitude, 0.0, 0.0);
    }

    auto compute_jacobian(const ForceContext& ctx) const -> std::pair<Eigen::Matrix3d, Eigen::Matrix3d> override {
        return {Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Zero()};
    }
    
private:
    double amplitude_;
    double omega_;
};

// Test fixture
class PointMassDynamicsTest : public ::testing::Test {
protected:
    // Helper to create a simple 6D state
    Eigen::VectorXd createState(const Eigen::Vector3d& pos, const Eigen::Vector3d& vel) {
        Eigen::VectorXd state(6);
        state << pos, vel;
        return state;
    }
};

// Test with no forces
TEST_F(PointMassDynamicsTest, NoForces) {
    std::vector<std::shared_ptr<IForce>> forces;
    PointMassDynamics dynamics(forces);
    
    Eigen::VectorXd state = createState(
        Eigen::Vector3d(1.0, 2.0, 3.0),
        Eigen::Vector3d(4.0, 5.0, 6.0)
    );
    
    Eigen::VectorXd deriv = dynamics.compute_dynamics(0.0, state);
    
    // Velocity components
    EXPECT_DOUBLE_EQ(deriv(0), 4.0);
    EXPECT_DOUBLE_EQ(deriv(1), 5.0);
    EXPECT_DOUBLE_EQ(deriv(2), 6.0);
    
    // Zero acceleration
    EXPECT_DOUBLE_EQ(deriv(3), 0.0);
    EXPECT_DOUBLE_EQ(deriv(4), 0.0);
    EXPECT_DOUBLE_EQ(deriv(5), 0.0);
}

// Test with single zero force
TEST_F(PointMassDynamicsTest, SingleZeroForce) {
    std::shared_ptr<IForce> zero_force = std::make_shared<ZeroForce>();
    PointMassDynamics dynamics(std::vector{zero_force});
    
    Eigen::VectorXd state = createState(
        Eigen::Vector3d(1.0, 2.0, 3.0),
        Eigen::Vector3d(10.0, 20.0, 30.0)
    );
    
    Eigen::VectorXd deriv = dynamics.compute_dynamics(0.0, state);
    
    // Velocity components
    EXPECT_DOUBLE_EQ(deriv(0), 10.0);
    EXPECT_DOUBLE_EQ(deriv(1), 20.0);
    EXPECT_DOUBLE_EQ(deriv(2), 30.0);
    
    // Zero acceleration
    EXPECT_DOUBLE_EQ(deriv(3), 0.0);
    EXPECT_DOUBLE_EQ(deriv(4), 0.0);
    EXPECT_DOUBLE_EQ(deriv(5), 0.0);
}

// Test with single constant force
TEST_F(PointMassDynamicsTest, SingleConstantForce) {
    Eigen::Vector3d accel(1.0, 2.0, 3.0);
    std::shared_ptr<IForce> constant_force = std::make_shared<ConstantForce>(accel);
    PointMassDynamics dynamics(std::vector{constant_force});
    
    Eigen::VectorXd state = createState(
        Eigen::Vector3d(0.0, 0.0, 0.0),
        Eigen::Vector3d(5.0, 6.0, 7.0)
    );
    
    Eigen::VectorXd deriv = dynamics.compute_dynamics(0.0, state);
    
    // Velocity components
    EXPECT_DOUBLE_EQ(deriv(0), 5.0);
    EXPECT_DOUBLE_EQ(deriv(1), 6.0);
    EXPECT_DOUBLE_EQ(deriv(2), 7.0);
    
    // Constant acceleration
    EXPECT_DOUBLE_EQ(deriv(3), 1.0);
    EXPECT_DOUBLE_EQ(deriv(4), 2.0);
    EXPECT_DOUBLE_EQ(deriv(5), 3.0);
}

// Test with multiple forces
TEST_F(PointMassDynamicsTest, MultipleForces) {
    std::shared_ptr<IForce> force1 = std::make_shared<ConstantForce>(Eigen::Vector3d(1.0, 0.0, 0.0));
    std::shared_ptr<IForce> force2 = std::make_shared<ConstantForce>(Eigen::Vector3d(0.0, 2.0, 0.0));
    std::shared_ptr<IForce> force3 = std::make_shared<ConstantForce>(Eigen::Vector3d(0.0, 0.0, 3.0));
    
    PointMassDynamics dynamics(std::vector{force1, force2, force3});
    
    Eigen::VectorXd state = createState(
        Eigen::Vector3d(0.0, 0.0, 0.0),
        Eigen::Vector3d(0.0, 0.0, 0.0)
    );
    
    Eigen::VectorXd deriv = dynamics.compute_dynamics(0.0, state);
    
    // Forces should sum
    EXPECT_DOUBLE_EQ(deriv(3), 1.0);
    EXPECT_DOUBLE_EQ(deriv(4), 2.0);
    EXPECT_DOUBLE_EQ(deriv(5), 3.0);
}

// Test with damping force (velocity-dependent)
TEST_F(PointMassDynamicsTest, DampingForce) {
    std::shared_ptr<IForce> damping = std::make_shared<DampingForce>(0.5);
    PointMassDynamics dynamics(std::vector{damping});
    
    Eigen::VectorXd state = createState(
        Eigen::Vector3d(0.0, 0.0, 0.0),
        Eigen::Vector3d(10.0, 20.0, 30.0)
    );
    
    Eigen::VectorXd deriv = dynamics.compute_dynamics(0.0, state);
    
    // Acceleration should be -0.5 * velocity
    EXPECT_DOUBLE_EQ(deriv(3), -5.0);
    EXPECT_DOUBLE_EQ(deriv(4), -10.0);
    EXPECT_DOUBLE_EQ(deriv(5), -15.0);
}

// Test with spring force (position-dependent)
TEST_F(PointMassDynamicsTest, SpringForce) {
    std::shared_ptr<IForce> spring = std::make_shared<SpringForce>(2.0);
    PointMassDynamics dynamics(std::vector{spring});
    
    Eigen::VectorXd state = createState(
        Eigen::Vector3d(1.0, 2.0, 3.0),
        Eigen::Vector3d(0.0, 0.0, 0.0)
    );
    
    Eigen::VectorXd deriv = dynamics.compute_dynamics(0.0, state);
    
    // Acceleration should be -2.0 * position
    EXPECT_DOUBLE_EQ(deriv(3), -2.0);
    EXPECT_DOUBLE_EQ(deriv(4), -4.0);
    EXPECT_DOUBLE_EQ(deriv(5), -6.0);
}

// Test with time-dependent force
TEST_F(PointMassDynamicsTest, TimeDependentForce) {
    std::shared_ptr<IForce> sinusoidal = std::make_shared<SinusoidalForce>(10.0, 1.0); // 10 m/s², 1 Hz
    PointMassDynamics dynamics(std::vector{sinusoidal});
    
    Eigen::VectorXd state = createState(
        Eigen::Vector3d(0.0, 0.0, 0.0),
        Eigen::Vector3d(0.0, 0.0, 0.0)
    );
    
    // At t=0, sin(0) = 0
    Eigen::VectorXd deriv0 = dynamics.compute_dynamics(0.0, state);
    EXPECT_NEAR(deriv0(3), 0.0, 1e-10);
    
    // At t=0.25, sin(2π*0.25) = sin(π/2) = 1
    Eigen::VectorXd deriv1 = dynamics.compute_dynamics(0.25, state);
    EXPECT_NEAR(deriv1(3), 10.0, 1e-10);
    
    // At t=0.5, sin(2π*0.5) = sin(π) = 0
    Eigen::VectorXd deriv2 = dynamics.compute_dynamics(0.5, state);
    EXPECT_NEAR(deriv2(3), 0.0, 1e-10);
}

// Test that context is correctly populated
TEST_F(PointMassDynamicsTest, ContextPopulation) {
    class ContextCheckerForce : public IForce {
    public:
        auto compute_acceleration(const ForceContext& ctx) const -> Eigen::Vector3d override {
            last_time = ctx.t;
            last_position = ctx.position;
            last_velocity = ctx.velocity;
            return Eigen::Vector3d::Zero();
        }

        auto compute_jacobian(const ForceContext& ctx) const -> std::pair<Eigen::Matrix3d, Eigen::Matrix3d> override {
            return {Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Zero()};
        }
        
        mutable double last_time;
        mutable Eigen::Vector3d last_position;
        mutable Eigen::Vector3d last_velocity;
    };
    
    auto checker = std::make_shared<ContextCheckerForce>();
    PointMassDynamics dynamics(std::vector<std::shared_ptr<IForce>>{checker});
    
    double t = 5.0;
    Eigen::VectorXd state = createState(
        Eigen::Vector3d(1.0, 2.0, 3.0),
        Eigen::Vector3d(4.0, 5.0, 6.0)
    );
    
    dynamics.compute_dynamics(t, state);
    
    EXPECT_DOUBLE_EQ(checker->last_time, 5.0);
    EXPECT_EQ(checker->last_position, Eigen::Vector3d(1.0, 2.0, 3.0));
    EXPECT_EQ(checker->last_velocity, Eigen::Vector3d(4.0, 5.0, 6.0));
}

// Test derivative vector size
TEST_F(PointMassDynamicsTest, DerivativeVectorSize) {
    PointMassDynamics dynamics(std::vector<std::shared_ptr<IForce>>{});
    
    Eigen::VectorXd state = createState(
        Eigen::Vector3d(0.0, 0.0, 0.0),
        Eigen::Vector3d(0.0, 0.0, 0.0)
    );
    
    Eigen::VectorXd deriv = dynamics.compute_dynamics(0.0, state);
    
    EXPECT_EQ(deriv.size(), 6);
}

// Test with opposing forces (cancellation)
TEST_F(PointMassDynamicsTest, OpposingForces) {
    std::shared_ptr<IForce> force1 = std::make_shared<ConstantForce>(Eigen::Vector3d(5.0, 0.0, 0.0));
    std::shared_ptr<IForce> force2 = std::make_shared<ConstantForce>(Eigen::Vector3d(-5.0, 0.0, 0.0));
    
    PointMassDynamics dynamics(std::vector{force1, force2});
    
    Eigen::VectorXd state = createState(
        Eigen::Vector3d(0.0, 0.0, 0.0),
        Eigen::Vector3d(0.0, 0.0, 0.0)
    );
    
    Eigen::VectorXd deriv = dynamics.compute_dynamics(0.0, state);
    
    // Forces should cancel
    EXPECT_DOUBLE_EQ(deriv(3), 0.0);
    EXPECT_DOUBLE_EQ(deriv(4), 0.0);
    EXPECT_DOUBLE_EQ(deriv(5), 0.0);
}

// Test with many forces
TEST_F(PointMassDynamicsTest, ManyForces) {
    std::vector<std::shared_ptr<IForce>> forces;
    for (int i = 0; i < 10; ++i) {
        forces.push_back(std::make_shared<ConstantForce>(Eigen::Vector3d(1.0, 0.0, 0.0)));
    }
    
    PointMassDynamics dynamics(forces);
    
    Eigen::VectorXd state = createState(
        Eigen::Vector3d(0.0, 0.0, 0.0),
        Eigen::Vector3d(0.0, 0.0, 0.0)
    );
    
    Eigen::VectorXd deriv = dynamics.compute_dynamics(0.0, state);
    
    // All forces should sum
    EXPECT_DOUBLE_EQ(deriv(3), 10.0);
}

// Test velocity derivative equals velocity (state continuity)
TEST_F(PointMassDynamicsTest, VelocityDerivativeEqualsVelocity) {
    std::shared_ptr<IForce> force = std::make_shared<ConstantForce>(Eigen::Vector3d(1.0, 2.0, 3.0));
    PointMassDynamics dynamics(std::vector{force});
    
    Eigen::Vector3d velocity(7.0, 8.0, 9.0);
    Eigen::VectorXd state = createState(
        Eigen::Vector3d(0.0, 0.0, 0.0),
        velocity
    );
    
    Eigen::VectorXd deriv = dynamics.compute_dynamics(0.0, state);
    
    // First three components of derivative should equal velocity
    EXPECT_DOUBLE_EQ(deriv(0), velocity(0));
    EXPECT_DOUBLE_EQ(deriv(1), velocity(1));
    EXPECT_DOUBLE_EQ(deriv(2), velocity(2));
}

// Test with zero velocity
TEST_F(PointMassDynamicsTest, ZeroVelocity) {
    std::shared_ptr<IForce> force = std::make_shared<ConstantForce>(Eigen::Vector3d(1.0, 2.0, 3.0));
    PointMassDynamics dynamics(std::vector{force});
    
    Eigen::VectorXd state = createState(
        Eigen::Vector3d(5.0, 6.0, 7.0),
        Eigen::Vector3d(0.0, 0.0, 0.0)
    );
    
    Eigen::VectorXd deriv = dynamics.compute_dynamics(0.0, state);
    
    // Position derivative should be zero
    EXPECT_DOUBLE_EQ(deriv(0), 0.0);
    EXPECT_DOUBLE_EQ(deriv(1), 0.0);
    EXPECT_DOUBLE_EQ(deriv(2), 0.0);
    
    // Still have acceleration
    EXPECT_DOUBLE_EQ(deriv(3), 1.0);
    EXPECT_DOUBLE_EQ(deriv(4), 2.0);
    EXPECT_DOUBLE_EQ(deriv(5), 3.0);
}

// Test at different positions
TEST_F(PointMassDynamicsTest, DifferentPositions) {
    std::shared_ptr<IForce> spring = std::make_shared<SpringForce>(1.0);
    PointMassDynamics dynamics(std::vector{spring});
    
    // Position A
    Eigen::VectorXd state_a = createState(
        Eigen::Vector3d(1.0, 0.0, 0.0),
        Eigen::Vector3d(0.0, 0.0, 0.0)
    );
    Eigen::VectorXd deriv_a = dynamics.compute_dynamics(0.0, state_a);
    EXPECT_DOUBLE_EQ(deriv_a(3), -1.0);
    
    // Position B
    Eigen::VectorXd state_b = createState(
        Eigen::Vector3d(2.0, 0.0, 0.0),
        Eigen::Vector3d(0.0, 0.0, 0.0)
    );
    Eigen::VectorXd deriv_b = dynamics.compute_dynamics(0.0, state_b);
    EXPECT_DOUBLE_EQ(deriv_b(3), -2.0);
}

// Test at different times
TEST_F(PointMassDynamicsTest, DifferentTimes) {
    std::shared_ptr<IForce> sinusoidal = std::make_shared<SinusoidalForce>(1.0, 1.0);
    PointMassDynamics dynamics(std::vector{sinusoidal});
    
    Eigen::VectorXd state = createState(
        Eigen::Vector3d(0.0, 0.0, 0.0),
        Eigen::Vector3d(0.0, 0.0, 0.0)
    );
    
    Eigen::VectorXd deriv_t0 = dynamics.compute_dynamics(0.0, state);
    Eigen::VectorXd deriv_t1 = dynamics.compute_dynamics(1.0, state);
    
    // Should be different due to time dependence
    EXPECT_NE(deriv_t0(3), deriv_t1(3));
}

// Test with large values
TEST_F(PointMassDynamicsTest, LargeValues) {
    std::shared_ptr<IForce> force = std::make_shared<ConstantForce>(Eigen::Vector3d(1e6, 1e6, 1e6));
    PointMassDynamics dynamics(std::vector{force});
    
    Eigen::VectorXd state = createState(
        Eigen::Vector3d(1e9, 1e9, 1e9),
        Eigen::Vector3d(1e6, 1e6, 1e6)
    );
    
    Eigen::VectorXd deriv = dynamics.compute_dynamics(0.0, state);
    
    EXPECT_DOUBLE_EQ(deriv(0), 1e6);
    EXPECT_DOUBLE_EQ(deriv(3), 1e6);
}

// Test with small values
TEST_F(PointMassDynamicsTest, SmallValues) {
    std::shared_ptr<IForce> force = std::make_shared<ConstantForce>(Eigen::Vector3d(1e-10, 1e-10, 1e-10));
    PointMassDynamics dynamics(std::vector{force});
    
    Eigen::VectorXd state = createState(
        Eigen::Vector3d(1e-15, 1e-15, 1e-15),
        Eigen::Vector3d(1e-12, 1e-12, 1e-12)
    );
    
    Eigen::VectorXd deriv = dynamics.compute_dynamics(0.0, state);
    
    EXPECT_DOUBLE_EQ(deriv(0), 1e-12);
    EXPECT_DOUBLE_EQ(deriv(3), 1e-10);
}

// Add these tests at the end of the file, before the closing brace

// ============================================================================
// STATE DIMENSION TESTS
// ============================================================================

TEST_F(PointMassDynamicsTest, StateDimension) {
    PointMassDynamics dynamics(std::vector<std::shared_ptr<IForce>>{});
    
    EXPECT_EQ(dynamics.get_state_dimension(), 6);
}

TEST_F(PointMassDynamicsTest, StateDimensionWithForces) {
    std::shared_ptr<IForce> force1 = std::make_shared<ConstantForce>(Eigen::Vector3d(1.0, 2.0, 3.0));
    std::shared_ptr<IForce> force2 = std::make_shared<DampingForce>(0.5);
    
    PointMassDynamics dynamics(std::vector{force1, force2});
    
    // State dimension should be 6 regardless of number of forces
    EXPECT_EQ(dynamics.get_state_dimension(), 6);
}

// ============================================================================
// JACOBIAN TESTS
// ============================================================================

// Mock force with known Jacobian
class LinearForceWithJacobian : public IForce {
public:
    explicit LinearForceWithJacobian(
        const Eigen::Matrix3d& position_jacobian,
        const Eigen::Matrix3d& velocity_jacobian)
        : da_dr_(position_jacobian), da_dv_(velocity_jacobian) {}
    
    auto compute_acceleration(const ForceContext& ctx) const -> Eigen::Vector3d override {
        return da_dr_ * ctx.position + da_dv_ * ctx.velocity;
    }

    auto compute_jacobian(const ForceContext& ctx) const -> std::pair<Eigen::Matrix3d, Eigen::Matrix3d> override {
        return {da_dr_, da_dv_};
    }
    
private:
    Eigen::Matrix3d da_dr_;
    Eigen::Matrix3d da_dv_;
};

TEST_F(PointMassDynamicsTest, JacobianSize) {
    PointMassDynamics dynamics(std::vector<std::shared_ptr<IForce>>{});
    
    Eigen::VectorXd state = createState(
        Eigen::Vector3d(1.0, 2.0, 3.0),
        Eigen::Vector3d(4.0, 5.0, 6.0)
    );
    
    Eigen::MatrixXd F = dynamics.compute_jacobian(0.0, state);
    
    EXPECT_EQ(F.rows(), 6);
    EXPECT_EQ(F.cols(), 6);
}

TEST_F(PointMassDynamicsTest, JacobianStructureNoForces) {
    PointMassDynamics dynamics(std::vector<std::shared_ptr<IForce>>{});
    
    Eigen::VectorXd state = createState(
        Eigen::Vector3d(1.0, 2.0, 3.0),
        Eigen::Vector3d(4.0, 5.0, 6.0)
    );
    
    Eigen::MatrixXd F = dynamics.compute_jacobian(0.0, state);
    
    // Expected structure:
    // F = [ 0   I ]
    //     [ 0   0 ]
    
    // Upper-left: ∂ṙ/∂r = 0
    EXPECT_TRUE((F.block<3,3>(0, 0).isZero()));
    
    // Upper-right: ∂ṙ/∂v = I
    EXPECT_TRUE((F.block<3,3>(0, 3).isIdentity()));
    
    // Lower-left: ∂v̇/∂r = 0
    EXPECT_TRUE((F.block<3,3>(3, 0).isZero()));
    
    // Lower-right: ∂v̇/∂v = 0
    EXPECT_TRUE((F.block<3,3>(3, 3).isZero()));
}

TEST_F(PointMassDynamicsTest, JacobianUpperRightAlwaysIdentity) {
    // The upper-right block should always be identity, regardless of forces
    Eigen::Matrix3d da_dr = Eigen::Matrix3d::Random();
    Eigen::Matrix3d da_dv = Eigen::Matrix3d::Random();
    
    auto force = std::make_shared<LinearForceWithJacobian>(da_dr, da_dv);
    PointMassDynamics dynamics(std::vector<std::shared_ptr<IForce>>{force});
    
    Eigen::VectorXd state = createState(
        Eigen::Vector3d(1.0, 2.0, 3.0),
        Eigen::Vector3d(4.0, 5.0, 6.0)
    );
    
    Eigen::MatrixXd F = dynamics.compute_jacobian(0.0, state);
    
    // ∂ṙ/∂v should always be identity
    EXPECT_TRUE((F.block<3,3>(0, 3).isIdentity()));
}

TEST_F(PointMassDynamicsTest, JacobianSingleForce) {
    // Create force with known Jacobian
    Eigen::Matrix3d da_dr;
    da_dr << 1.0, 0.0, 0.0,
             0.0, 2.0, 0.0,
             0.0, 0.0, 3.0;
    
    Eigen::Matrix3d da_dv;
    da_dv << -0.1, 0.0, 0.0,
             0.0, -0.2, 0.0,
             0.0, 0.0, -0.3;
    
    auto force = std::make_shared<LinearForceWithJacobian>(da_dr, da_dv);
    PointMassDynamics dynamics(std::vector<std::shared_ptr<IForce>>{force});
    
    Eigen::VectorXd state = createState(
        Eigen::Vector3d(1.0, 2.0, 3.0),
        Eigen::Vector3d(4.0, 5.0, 6.0)
    );
    
    Eigen::MatrixXd F = dynamics.compute_jacobian(0.0, state);
    
    // Check lower-left block: ∂v̇/∂r
    EXPECT_TRUE((F.block<3,3>(3, 0).isApprox(da_dr)));
    
    // Check lower-right block: ∂v̇/∂v
    EXPECT_TRUE((F.block<3,3>(3, 3).isApprox(da_dv)));
}

TEST_F(PointMassDynamicsTest, JacobianMultipleForcesSum) {
    // Create two forces with known Jacobians
    Eigen::Matrix3d da_dr1 = Eigen::Matrix3d::Identity() * 1.0;
    Eigen::Matrix3d da_dv1 = Eigen::Matrix3d::Identity() * -0.1;
    
    Eigen::Matrix3d da_dr2 = Eigen::Matrix3d::Identity() * 2.0;
    Eigen::Matrix3d da_dv2 = Eigen::Matrix3d::Identity() * -0.2;
    
    auto force1 = std::make_shared<LinearForceWithJacobian>(da_dr1, da_dv1);
    auto force2 = std::make_shared<LinearForceWithJacobian>(da_dr2, da_dv2);
    
    std::vector<std::shared_ptr<IForce>> forces = {force1, force2};
    PointMassDynamics dynamics(forces);
    
    Eigen::VectorXd state = createState(
        Eigen::Vector3d(1.0, 2.0, 3.0),
        Eigen::Vector3d(4.0, 5.0, 6.0)
    );
    
    Eigen::MatrixXd F = dynamics.compute_jacobian(0.0, state);
    
    // Jacobians should sum
    Eigen::Matrix3d expected_da_dr = da_dr1 + da_dr2;
    Eigen::Matrix3d expected_da_dv = da_dv1 + da_dv2;
    
    EXPECT_TRUE((F.block<3,3>(3, 0).isApprox(expected_da_dr)));
    EXPECT_TRUE((F.block<3,3>(3, 3).isApprox(expected_da_dv)));
}

TEST_F(PointMassDynamicsTest, JacobianNumericalValidation) {
    // Test Jacobian against numerical differentiation
    Eigen::Matrix3d da_dr;
    da_dr << -1.0, 0.5, 0.0,
             0.5, -2.0, 0.0,
             0.0, 0.0, -3.0;
    
    Eigen::Matrix3d da_dv;
    da_dv << -0.1, 0.0, 0.0,
             0.0, -0.2, 0.0,
             0.0, 0.0, -0.3;
    
    auto force = std::make_shared<LinearForceWithJacobian>(da_dr, da_dv);
    PointMassDynamics dynamics(std::vector<std::shared_ptr<IForce>>{force});
    
    Eigen::VectorXd state = createState(
        Eigen::Vector3d(1.0, 2.0, 3.0),
        Eigen::Vector3d(4.0, 5.0, 6.0)
    );
    
    // Analytical Jacobian
    Eigen::MatrixXd F = dynamics.compute_jacobian(0.0, state);
    
    // Numerical Jacobian
    double epsilon = 1e-6;
    Eigen::MatrixXd F_numerical = Eigen::MatrixXd::Zero(6, 6);
    
    for (int i = 0; i < 6; ++i) {
        Eigen::VectorXd state_plus = state;
        Eigen::VectorXd state_minus = state;
        
        state_plus(i) += epsilon;
        state_minus(i) -= epsilon;
        
        Eigen::VectorXd deriv_plus = dynamics.compute_dynamics(0.0, state_plus);
        Eigen::VectorXd deriv_minus = dynamics.compute_dynamics(0.0, state_minus);
        
        F_numerical.col(i) = (deriv_plus - deriv_minus) / (2.0 * epsilon);
    }
    
    // Compare
    EXPECT_TRUE(F.isApprox(F_numerical, 1e-5));
}

TEST_F(PointMassDynamicsTest, JacobianStateIndependent) {
    // For linear forces, Jacobian should not depend on state
    Eigen::Matrix3d da_dr = Eigen::Matrix3d::Identity() * 2.0;
    Eigen::Matrix3d da_dv = Eigen::Matrix3d::Identity() * -0.5;
    
    auto force = std::make_shared<LinearForceWithJacobian>(da_dr, da_dv);
    PointMassDynamics dynamics(std::vector<std::shared_ptr<IForce>>{force});
    
    Eigen::VectorXd state1 = createState(
        Eigen::Vector3d(1.0, 2.0, 3.0),
        Eigen::Vector3d(4.0, 5.0, 6.0)
    );
    
    Eigen::VectorXd state2 = createState(
        Eigen::Vector3d(10.0, 20.0, 30.0),
        Eigen::Vector3d(40.0, 50.0, 60.0)
    );
    
    Eigen::MatrixXd F1 = dynamics.compute_jacobian(0.0, state1);
    Eigen::MatrixXd F2 = dynamics.compute_jacobian(0.0, state2);
    
    // Should be identical for linear forces
    EXPECT_TRUE(F1.isApprox(F2));
}

TEST_F(PointMassDynamicsTest, JacobianTimeIndependent) {
    // Jacobian should not depend on time for time-independent forces
    Eigen::Matrix3d da_dr = Eigen::Matrix3d::Identity() * 2.0;
    Eigen::Matrix3d da_dv = Eigen::Matrix3d::Identity() * -0.5;
    
    auto force = std::make_shared<LinearForceWithJacobian>(da_dr, da_dv);
    PointMassDynamics dynamics(std::vector<std::shared_ptr<IForce>>{force});
    
    Eigen::VectorXd state = createState(
        Eigen::Vector3d(1.0, 2.0, 3.0),
        Eigen::Vector3d(4.0, 5.0, 6.0)
    );
    
    Eigen::MatrixXd F1 = dynamics.compute_jacobian(0.0, state);
    Eigen::MatrixXd F2 = dynamics.compute_jacobian(100.0, state);
    
    EXPECT_TRUE(F1.isApprox(F2));
}

TEST_F(PointMassDynamicsTest, JacobianUpperLeftAlwaysZero) {
    // ∂ṙ/∂r should always be zero (position rate doesn't depend on position)
    Eigen::Matrix3d da_dr = Eigen::Matrix3d::Random();
    Eigen::Matrix3d da_dv = Eigen::Matrix3d::Random();
    
    auto force = std::make_shared<LinearForceWithJacobian>(da_dr, da_dv);
    PointMassDynamics dynamics(std::vector<std::shared_ptr<IForce>>{force});
    
    Eigen::VectorXd state = createState(
        Eigen::Vector3d(1.0, 2.0, 3.0),
        Eigen::Vector3d(4.0, 5.0, 6.0)
    );
    
    Eigen::MatrixXd F = dynamics.compute_jacobian(0.0, state);
    
    EXPECT_TRUE((F.block<3,3>(0, 0).isZero()));
}

TEST_F(PointMassDynamicsTest, JacobianLinearConsistency) {
    // For linear dynamics, Taylor expansion should be exact
    Eigen::Matrix3d da_dr;
    da_dr << -1.0, 0.5, 0.0,
             0.5, -2.0, 0.0,
             0.0, 0.0, -3.0;
    
    Eigen::Matrix3d da_dv;
    da_dv << -0.1, 0.0, 0.0,
             0.0, -0.2, 0.0,
             0.0, 0.0, -0.3;
    
    auto force = std::make_shared<LinearForceWithJacobian>(da_dr, da_dv);
    PointMassDynamics dynamics(std::vector<std::shared_ptr<IForce>>{force});
    
    Eigen::VectorXd state = createState(
        Eigen::Vector3d(1.0, 2.0, 3.0),
        Eigen::Vector3d(4.0, 5.0, 6.0)
    );
    
    Eigen::MatrixXd F = dynamics.compute_jacobian(0.0, state);
    
    // Perturb state
    Eigen::VectorXd delta_state = Eigen::VectorXd::Random(6) * 0.1;
    Eigen::VectorXd state_perturbed = state + delta_state;
    
    // Compute derivatives
    Eigen::VectorXd deriv_base = dynamics.compute_dynamics(0.0, state);
    Eigen::VectorXd deriv_perturbed = dynamics.compute_dynamics(0.0, state_perturbed);
    
    // Taylor expansion: f(x + δx) ≈ f(x) + F·δx
    Eigen::VectorXd deriv_linear = deriv_base + F * delta_state;
    
    // For linear dynamics, should be exact
    EXPECT_TRUE(deriv_perturbed.isApprox(deriv_linear, 1e-10));
}

TEST_F(PointMassDynamicsTest, JacobianBlockStructure) {
    // Verify the 4-block structure of the Jacobian
    Eigen::Matrix3d da_dr;
    da_dr << 1.0, 2.0, 3.0,
             4.0, 5.0, 6.0,
             7.0, 8.0, 9.0;
    
    Eigen::Matrix3d da_dv;
    da_dv << -1.0, -2.0, -3.0,
             -4.0, -5.0, -6.0,
             -7.0, -8.0, -9.0;
    
    auto force = std::make_shared<LinearForceWithJacobian>(da_dr, da_dv);
    PointMassDynamics dynamics(std::vector<std::shared_ptr<IForce>>{force});
    
    Eigen::VectorXd state = createState(
        Eigen::Vector3d(1.0, 2.0, 3.0),
        Eigen::Vector3d(4.0, 5.0, 6.0)
    );
    
    Eigen::MatrixXd F = dynamics.compute_jacobian(0.0, state);
    
    // Expected structure:
    // F = [ 0      I    ]
    //     [ da_dr da_dv ]
    
    Eigen::MatrixXd expected = Eigen::MatrixXd::Zero(6, 6);
    expected.block<3,3>(0, 3) = Eigen::Matrix3d::Identity();
    expected.block<3,3>(3, 0) = da_dr;
    expected.block<3,3>(3, 3) = da_dv;
    
    EXPECT_TRUE(F.isApprox(expected));
}

TEST_F(PointMassDynamicsTest, JacobianZeroForces) {
    // With no forces, only kinematic coupling remains
    PointMassDynamics dynamics(std::vector<std::shared_ptr<IForce>>{});
    
    Eigen::VectorXd state = createState(
        Eigen::Vector3d(1.0, 2.0, 3.0),
        Eigen::Vector3d(4.0, 5.0, 6.0)
    );
    
    Eigen::MatrixXd F = dynamics.compute_jacobian(0.0, state);
    
    // Expected: only upper-right block is non-zero (identity)
    Eigen::MatrixXd expected = Eigen::MatrixXd::Zero(6, 6);
    expected.block<3,3>(0, 3) = Eigen::Matrix3d::Identity();
    
    EXPECT_TRUE(F.isApprox(expected));
}

TEST_F(PointMassDynamicsTest, JacobianManyForces) {
    // Test Jacobian with multiple forces
    std::vector<std::shared_ptr<IForce>> forces;
    Eigen::Matrix3d total_da_dr = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d total_da_dv = Eigen::Matrix3d::Zero();
    
    for (int i = 0; i < 5; ++i) {
        Eigen::Matrix3d da_dr = Eigen::Matrix3d::Identity() * (i + 1);
        Eigen::Matrix3d da_dv = Eigen::Matrix3d::Identity() * -(i + 1) * 0.1;
        
        forces.push_back(std::make_shared<LinearForceWithJacobian>(da_dr, da_dv));
        total_da_dr += da_dr;
        total_da_dv += da_dv;
    }
    
    PointMassDynamics dynamics(forces);
    
    Eigen::VectorXd state = createState(
        Eigen::Vector3d(1.0, 2.0, 3.0),
        Eigen::Vector3d(4.0, 5.0, 6.0)
    );
    
    Eigen::MatrixXd F = dynamics.compute_jacobian(0.0, state);
    
    // Check that Jacobians sum correctly
    EXPECT_TRUE((F.block<3,3>(3, 0).isApprox(total_da_dr)));
    EXPECT_TRUE((F.block<3,3>(3, 3).isApprox(total_da_dv)));
}

TEST_F(PointMassDynamicsTest, JacobianLargeValues) {
    // Test with large Jacobian values
    Eigen::Matrix3d da_dr = Eigen::Matrix3d::Identity() * 1e10;
    Eigen::Matrix3d da_dv = Eigen::Matrix3d::Identity() * -1e8;
    
    auto force = std::make_shared<LinearForceWithJacobian>(da_dr, da_dv);
    PointMassDynamics dynamics(std::vector<std::shared_ptr<IForce>>{force});
    
    Eigen::VectorXd state = createState(
        Eigen::Vector3d(1e6, 2e6, 3e6),
        Eigen::Vector3d(1e3, 2e3, 3e3)
    );
    
    Eigen::MatrixXd F = dynamics.compute_jacobian(0.0, state);
    
    EXPECT_TRUE((F.block<3,3>(3, 0).isApprox(da_dr)));
    EXPECT_TRUE((F.block<3,3>(3, 3).isApprox(da_dv)));
}

TEST_F(PointMassDynamicsTest, JacobianSmallValues) {
    // Test with small Jacobian values
    Eigen::Matrix3d da_dr = Eigen::Matrix3d::Identity() * 1e-10;
    Eigen::Matrix3d da_dv = Eigen::Matrix3d::Identity() * -1e-12;
    
    auto force = std::make_shared<LinearForceWithJacobian>(da_dr, da_dv);
    PointMassDynamics dynamics(std::vector<std::shared_ptr<IForce>>{force});
    
    Eigen::VectorXd state = createState(
        Eigen::Vector3d(1e-6, 2e-6, 3e-6),
        Eigen::Vector3d(1e-3, 2e-3, 3e-3)
    );
    
    Eigen::MatrixXd F = dynamics.compute_jacobian(0.0, state);
    
    EXPECT_TRUE((F.block<3,3>(3, 0).isApprox(da_dr)));
    EXPECT_TRUE((F.block<3,3>(3, 3).isApprox(da_dv)));
}
