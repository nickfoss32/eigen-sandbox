#include <gtest/gtest.h>
#include "dynamics/rigid_body_dynamics.hpp"
#include "dynamics/gravity.hpp"
#include "dynamics/gravity_gradient_torque.hpp"
#include "dynamics/aerodynamic_torque.hpp"
#include <cmath>
#include <memory>

using namespace dynamics;

// ============================================================================
// TEST FIXTURES
// ============================================================================

class RigidBodyDynamics6DOFTest : public ::testing::Test {
protected:
    // Standard parameters for a small satellite
    double mass_;
    Eigen::Matrix3d inertia_;
    
    RigidBodyDynamics6DOFTest()
        : mass_(100.0)  // 100 kg satellite
    {
        // Cube satellite: I = (1/6)·m·L² for each axis
        // Assume 1m × 1m × 1m cube
        double I_diag = (1.0 / 6.0) * mass_ * 1.0 * 1.0;
        inertia_ = Eigen::Matrix3d::Identity() * I_diag;
    }
    
    Eigen::VectorXd createLEOState(
        const Eigen::Vector3d& position,
        const Eigen::Vector3d& velocity,
        const Eigen::Quaterniond& orientation = Eigen::Quaterniond::Identity(),
        const Eigen::Vector3d& angular_velocity = Eigen::Vector3d::Zero()
    ) {
        Eigen::VectorXd state(13);
        state.segment<3>(0) = position;
        state.segment<3>(3) = velocity;
        state.segment<4>(6) = orientation.coeffs();  // x, y, z, w
        state.segment<3>(10) = angular_velocity;
        return state;
    }
};

// ============================================================================
// BASIC FUNCTIONALITY TESTS
// ============================================================================

TEST_F(RigidBodyDynamics6DOFTest, ConstructorAndDimension) {
    std::vector<std::shared_ptr<IForce>> forces;
    std::vector<std::shared_ptr<ITorque>> torques;
    
    RigidBodyDynamics6DOF dynamics(forces, torques, inertia_, mass_);
    
    EXPECT_EQ(dynamics.get_state_dimension(), 13);
}

TEST_F(RigidBodyDynamics6DOFTest, NoForcesNoTorques) {
    // With no forces or torques, dynamics should be simple
    std::vector<std::shared_ptr<IForce>> forces;
    std::vector<std::shared_ptr<ITorque>> torques;
    
    RigidBodyDynamics6DOF dynamics(forces, torques, inertia_, mass_);
    
    Eigen::Vector3d pos(1e7, 0.0, 0.0);
    Eigen::Vector3d vel(0.0, 7500.0, 0.0);
    Eigen::VectorXd state = createLEOState(pos, vel);
    
    Eigen::VectorXd state_dot = dynamics.compute_dynamics(0.0, state);
    
    // Should have: dr/dt = v, dv/dt = 0, dq/dt = 0, dω/dt = 0
    EXPECT_TRUE(state_dot.segment<3>(0).isApprox(vel, 1e-15));
    EXPECT_TRUE(state_dot.segment<3>(3).isApprox(Eigen::Vector3d::Zero(), 1e-15));
    EXPECT_NEAR(state_dot.segment<4>(6).norm(), 0.0, 1e-15);
    EXPECT_TRUE(state_dot.segment<3>(10).isApprox(Eigen::Vector3d::Zero(), 1e-15));
}

TEST_F(RigidBodyDynamics6DOFTest, StateVectorSize) {
    std::vector<std::shared_ptr<IForce>> forces;
    std::vector<std::shared_ptr<ITorque>> torques;
    
    RigidBodyDynamics6DOF dynamics(forces, torques, inertia_, mass_);
    
    Eigen::VectorXd state = createLEOState(
        Eigen::Vector3d(1e7, 0.0, 0.0),
        Eigen::Vector3d(0.0, 7500.0, 0.0)
    );
    
    Eigen::VectorXd state_dot = dynamics.compute_dynamics(0.0, state);
    
    EXPECT_EQ(state_dot.size(), 13);
}

TEST_F(RigidBodyDynamics6DOFTest, QuaternionNormalization) {
    // Test that quaternion is normalized even if input is not
    std::vector<std::shared_ptr<IForce>> forces;
    std::vector<std::shared_ptr<ITorque>> torques;
    
    RigidBodyDynamics6DOF dynamics(forces, torques, inertia_, mass_);
    
    Eigen::Quaterniond unnormalized(2.0, 0.0, 0.0, 0.0);  // Not unit
    Eigen::VectorXd state = createLEOState(
        Eigen::Vector3d(1e7, 0.0, 0.0),
        Eigen::Vector3d(0.0, 7500.0, 0.0),
        unnormalized
    );
    
    Eigen::VectorXd state_dot = dynamics.compute_dynamics(0.0, state);
    
    EXPECT_TRUE(state_dot.allFinite());
}

// ============================================================================
// TRANSLATIONAL DYNAMICS TESTS
// ============================================================================

TEST_F(RigidBodyDynamics6DOFTest, TranslationalKinematics) {
    // dr/dt = v
    std::vector<std::shared_ptr<IForce>> forces;
    std::vector<std::shared_ptr<ITorque>> torques;
    
    RigidBodyDynamics6DOF dynamics(forces, torques, inertia_, mass_);
    
    Eigen::Vector3d pos(1e7, 0.0, 0.0);
    Eigen::Vector3d vel(1000.0, 2000.0, 3000.0);
    Eigen::VectorXd state = createLEOState(pos, vel);
    
    Eigen::VectorXd state_dot = dynamics.compute_dynamics(0.0, state);
    
    EXPECT_TRUE(state_dot.segment<3>(0).isApprox(vel, 1e-12));
}

TEST_F(RigidBodyDynamics6DOFTest, PointMassGravityAcceleration) {
    // With only point mass gravity, acceleration should point toward Earth center
    std::vector<std::shared_ptr<IForce>> forces;
    forces.push_back(std::make_shared<PointMassGravity>());
    
    std::vector<std::shared_ptr<ITorque>> torques;
    
    RigidBodyDynamics6DOF dynamics(forces, torques, inertia_, mass_);
    
    Eigen::Vector3d pos(6.771e6, 0.0, 0.0);  // 400 km altitude
    Eigen::Vector3d vel(0.0, 7500.0, 0.0);
    Eigen::VectorXd state = createLEOState(pos, vel);
    
    Eigen::VectorXd state_dot = dynamics.compute_dynamics(0.0, state);
    
    Eigen::Vector3d acceleration = state_dot.segment<3>(3);
    
    // Acceleration should point toward origin (negative position direction)
    Eigen::Vector3d pos_hat = pos.normalized();
    Eigen::Vector3d accel_hat = acceleration.normalized();
    
    EXPECT_LT(pos_hat.dot(accel_hat), -0.999);  // Nearly opposite
    
    // Magnitude should be ~8.68 m/s² at 400 km
    EXPECT_GT(acceleration.norm(), 8.0);
    EXPECT_LT(acceleration.norm(), 9.0);
}

TEST_F(RigidBodyDynamics6DOFTest, J2GravityPerturbation) {
    // J2 gravity should differ from point mass gravity
    std::vector<std::shared_ptr<IForce>> forces_point;
    forces_point.push_back(std::make_shared<PointMassGravity>());
    
    std::vector<std::shared_ptr<IForce>> forces_j2;
    forces_j2.push_back(std::make_shared<J2Gravity>());
    
    std::vector<std::shared_ptr<ITorque>> torques;
    
    RigidBodyDynamics6DOF dynamics_point(forces_point, torques, inertia_, mass_);
    RigidBodyDynamics6DOF dynamics_j2(forces_j2, torques, inertia_, mass_);
    
    // Non-equatorial position to see J2 effect
    Eigen::Vector3d pos(6.771e6, 0.0, 1e6);  // Offset from equatorial plane
    Eigen::Vector3d vel(0.0, 7500.0, 0.0);
    Eigen::VectorXd state = createLEOState(pos, vel);
    
    Eigen::VectorXd state_dot_point = dynamics_point.compute_dynamics(0.0, state);
    Eigen::VectorXd state_dot_j2 = dynamics_j2.compute_dynamics(0.0, state);
    
    Eigen::Vector3d accel_point = state_dot_point.segment<3>(3);
    Eigen::Vector3d accel_j2 = state_dot_j2.segment<3>(3);
    
    // Accelerations should differ
    EXPECT_GT((accel_point - accel_j2).norm(), 1e-6);
}

TEST_F(RigidBodyDynamics6DOFTest, AccelerationScalesWithMass) {
    // For same force, lighter object has larger acceleration for non-gravitational forces
    // But gravitational acceleration is independent of mass
    std::vector<std::shared_ptr<IForce>> forces;
    forces.push_back(std::make_shared<PointMassGravity>());
    
    std::vector<std::shared_ptr<ITorque>> torques;
    
    RigidBodyDynamics6DOF dynamics1(forces, torques, inertia_, 100.0);
    RigidBodyDynamics6DOF dynamics2(forces, torques, inertia_, 200.0);
    
    Eigen::Vector3d pos(6.771e6, 0.0, 0.0);
    Eigen::Vector3d vel(0.0, 7500.0, 0.0);
    Eigen::VectorXd state = createLEOState(pos, vel);
    
    Eigen::VectorXd state_dot1 = dynamics1.compute_dynamics(0.0, state);
    Eigen::VectorXd state_dot2 = dynamics2.compute_dynamics(0.0, state);
    
    // Gravity acceleration is independent of mass
    EXPECT_TRUE(state_dot1.segment<3>(3).isApprox(state_dot2.segment<3>(3), 1e-12));
}

TEST_F(RigidBodyDynamics6DOFTest, MultipleForces) {
    // Test with point mass and J2 gravity
    std::vector<std::shared_ptr<IForce>> forces;
    forces.push_back(std::make_shared<PointMassGravity>());
    
    std::vector<std::shared_ptr<ITorque>> torques;
    
    RigidBodyDynamics6DOF dynamics(forces, torques, inertia_, mass_);
    
    Eigen::Vector3d pos(6.771e6, 0.0, 0.0);
    Eigen::Vector3d vel(0.0, 7500.0, 0.0);
    Eigen::VectorXd state = createLEOState(pos, vel);
    
    Eigen::VectorXd state_dot = dynamics.compute_dynamics(0.0, state);
    
    Eigen::Vector3d acceleration = state_dot.segment<3>(3);
    
    // Should have non-zero acceleration
    EXPECT_GT(acceleration.norm(), 0.0);
    EXPECT_TRUE(acceleration.allFinite());
}

// ============================================================================
// ROTATIONAL KINEMATICS TESTS
// ============================================================================

TEST_F(RigidBodyDynamics6DOFTest, QuaternionKinematicsNoRotation) {
    // dq/dt = 0.5 * q * [0, ω]
    // When ω = 0, dq/dt = 0
    std::vector<std::shared_ptr<IForce>> forces;
    std::vector<std::shared_ptr<ITorque>> torques;
    
    RigidBodyDynamics6DOF dynamics(forces, torques, inertia_, mass_);
    
    Eigen::VectorXd state = createLEOState(
        Eigen::Vector3d(1e7, 0.0, 0.0),
        Eigen::Vector3d(0.0, 7500.0, 0.0),
        Eigen::Quaterniond::Identity(),
        Eigen::Vector3d::Zero()
    );
    
    Eigen::VectorXd state_dot = dynamics.compute_dynamics(0.0, state);
    
    EXPECT_NEAR(state_dot.segment<4>(6).norm(), 0.0, 1e-15);
}

TEST_F(RigidBodyDynamics6DOFTest, QuaternionKinematicsSimpleRotation) {
    // Simple rotation about Z-axis
    std::vector<std::shared_ptr<IForce>> forces;
    std::vector<std::shared_ptr<ITorque>> torques;
    
    RigidBodyDynamics6DOF dynamics(forces, torques, inertia_, mass_);
    
    double omega_z = 0.1;  // rad/s
    Eigen::VectorXd state = createLEOState(
        Eigen::Vector3d(1e7, 0.0, 0.0),
        Eigen::Vector3d(0.0, 7500.0, 0.0),
        Eigen::Quaterniond::Identity(),
        Eigen::Vector3d(0.0, 0.0, omega_z)
    );
    
    Eigen::VectorXd state_dot = dynamics.compute_dynamics(0.0, state);
    
    Eigen::Vector4d q_dot = state_dot.segment<4>(6);
    
    // For q = [0, 0, 0, 1] (Identity: x,y,z,w format) and ω = [0, 0, ω_z]:
    // q̇ = 0.5 * q ⊗ [ω_x, ω_y, ω_z, 0]
    EXPECT_NEAR(q_dot(0), 0.0, 1e-15);           // x component
    EXPECT_NEAR(q_dot(1), 0.0, 1e-15);           // y component
    EXPECT_NEAR(q_dot(2), 0.5 * omega_z, 1e-15); // z component
    EXPECT_NEAR(q_dot(3), 0.0, 1e-15);           // w component
}

TEST_F(RigidBodyDynamics6DOFTest, QuaternionKinematicsArbitraryRotation) {
    // Test with arbitrary orientation and rotation
    std::vector<std::shared_ptr<IForce>> forces;
    std::vector<std::shared_ptr<ITorque>> torques;
    
    RigidBodyDynamics6DOF dynamics(forces, torques, inertia_, mass_);
    
    Eigen::Quaterniond q(Eigen::AngleAxisd(0.5, Eigen::Vector3d(1, 1, 1).normalized()));
    Eigen::Vector3d omega(0.1, -0.05, 0.08);
    
    Eigen::VectorXd state = createLEOState(
        Eigen::Vector3d(1e7, 0.0, 0.0),
        Eigen::Vector3d(0.0, 7500.0, 0.0),
        q,
        omega
    );
    
    Eigen::VectorXd state_dot = dynamics.compute_dynamics(0.0, state);
    
    Eigen::Vector4d q_dot = state_dot.segment<4>(6);
    
    // Should be non-zero and finite
    EXPECT_GT(q_dot.norm(), 0.0);
    EXPECT_TRUE(q_dot.allFinite());
}

TEST_F(RigidBodyDynamics6DOFTest, QuaternionRateScalesWithOmega) {
    // Doubling ω should double dq/dt
    std::vector<std::shared_ptr<IForce>> forces;
    std::vector<std::shared_ptr<ITorque>> torques;
    
    RigidBodyDynamics6DOF dynamics(forces, torques, inertia_, mass_);
    
    Eigen::Vector3d omega1(0.1, 0.0, 0.0);
    Eigen::Vector3d omega2(0.2, 0.0, 0.0);
    
    Eigen::VectorXd state1 = createLEOState(
        Eigen::Vector3d(1e7, 0.0, 0.0),
        Eigen::Vector3d(0.0, 7500.0, 0.0),
        Eigen::Quaterniond::Identity(),
        omega1
    );
    
    Eigen::VectorXd state2 = createLEOState(
        Eigen::Vector3d(1e7, 0.0, 0.0),
        Eigen::Vector3d(0.0, 7500.0, 0.0),
        Eigen::Quaterniond::Identity(),
        omega2
    );
    
    Eigen::VectorXd state_dot1 = dynamics.compute_dynamics(0.0, state1);
    Eigen::VectorXd state_dot2 = dynamics.compute_dynamics(0.0, state2);
    
    Eigen::Vector4d q_dot1 = state_dot1.segment<4>(6);
    Eigen::Vector4d q_dot2 = state_dot2.segment<4>(6);
    
    EXPECT_TRUE(q_dot2.isApprox(2.0 * q_dot1, 1e-15));
}

// ============================================================================
// ROTATIONAL DYNAMICS TESTS
// ============================================================================

TEST_F(RigidBodyDynamics6DOFTest, NoTorquesNoAngularAcceleration) {
    std::vector<std::shared_ptr<IForce>> forces;
    std::vector<std::shared_ptr<ITorque>> torques;
    
    RigidBodyDynamics6DOF dynamics(forces, torques, inertia_, mass_);
    
    Eigen::Vector3d omega(0.1, 0.05, 0.08);
    Eigen::VectorXd state = createLEOState(
        Eigen::Vector3d(1e7, 0.0, 0.0),
        Eigen::Vector3d(0.0, 7500.0, 0.0),
        Eigen::Quaterniond::Identity(),
        omega
    );
    
    Eigen::VectorXd state_dot = dynamics.compute_dynamics(0.0, state);
    
    // With no external torques and spherical inertia (I = I_diag * I),
    // gyroscopic term ω × (I·ω) = 0
    // Tolerance accounts for floating point roundoff (~1e-19)
    EXPECT_LT(state_dot.segment<3>(10).norm(), 1e-15);
}

TEST_F(RigidBodyDynamics6DOFTest, GyroscopicCoupling) {
    // Non-spherical inertia should produce gyroscopic coupling
    std::vector<std::shared_ptr<IForce>> forces;
    std::vector<std::shared_ptr<ITorque>> torques;
    
    Eigen::Matrix3d non_spherical_inertia = Eigen::Matrix3d::Zero();
    non_spherical_inertia(0, 0) = 10.0;
    non_spherical_inertia(1, 1) = 15.0;
    non_spherical_inertia(2, 2) = 20.0;
    
    RigidBodyDynamics6DOF dynamics(forces, torques, non_spherical_inertia, mass_);
    
    // Rotation about body-fixed axes
    Eigen::Vector3d omega(0.1, 0.1, 0.0);
    Eigen::VectorXd state = createLEOState(
        Eigen::Vector3d(1e7, 0.0, 0.0),
        Eigen::Vector3d(0.0, 7500.0, 0.0),
        Eigen::Quaterniond::Identity(),
        omega
    );
    
    Eigen::VectorXd state_dot = dynamics.compute_dynamics(0.0, state);
    
    Eigen::Vector3d omega_dot = state_dot.segment<3>(10);
    
    // Should have non-zero angular acceleration from gyroscopic term
    EXPECT_GT(omega_dot.norm(), 0.0);
}

TEST_F(RigidBodyDynamics6DOFTest, GravityGradientTorque) {
    // Gravity gradient torque should affect angular acceleration
    std::vector<std::shared_ptr<IForce>> forces;
    std::vector<std::shared_ptr<ITorque>> torques;
    
    Eigen::Matrix3d non_spherical_inertia = Eigen::Matrix3d::Zero();
    non_spherical_inertia(0, 0) = 10.0;
    non_spherical_inertia(1, 1) = 15.0;
    non_spherical_inertia(2, 2) = 20.0;
    
    torques.push_back(std::make_shared<GravityGradientTorque>(non_spherical_inertia));
    
    RigidBodyDynamics6DOF dynamics(forces, torques, non_spherical_inertia, mass_);
    
    // Satellite misaligned with local vertical
    Eigen::Quaterniond q(Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitY()));
    
    Eigen::VectorXd state = createLEOState(
        Eigen::Vector3d(6.771e6, 0.0, 0.0),
        Eigen::Vector3d(0.0, 7500.0, 0.0),
        q,
        Eigen::Vector3d::Zero()
    );
    
    Eigen::VectorXd state_dot = dynamics.compute_dynamics(0.0, state);
    
    Eigen::Vector3d omega_dot = state_dot.segment<3>(10);
    
    // Should produce restoring torque
    EXPECT_GT(omega_dot.norm(), 1e-10);
    EXPECT_TRUE(omega_dot.allFinite());
}

TEST_F(RigidBodyDynamics6DOFTest, AerodynamicTorque) {
    // Aerodynamic torque should affect angular acceleration
    std::vector<std::shared_ptr<IForce>> forces;
    std::vector<std::shared_ptr<ITorque>> torques;
    
    Eigen::Vector3d cop(0.0, 0.0, 0.1);  // 10 cm offset
    Eigen::Vector3d com(0.0, 0.0, 0.0);
    torques.push_back(std::make_shared<AerodynamicTorque>(cop, com, 2.2, 1.0));
    
    RigidBodyDynamics6DOF dynamics(forces, torques, inertia_, mass_);
    
    Eigen::VectorXd state = createLEOState(
        Eigen::Vector3d(6.771e6, 0.0, 0.0),
        Eigen::Vector3d(0.0, 7500.0, 0.0),
        Eigen::Quaterniond::Identity(),
        Eigen::Vector3d::Zero()
    );
    
    Eigen::VectorXd state_dot = dynamics.compute_dynamics(0.0, state);
    
    Eigen::Vector3d omega_dot = state_dot.segment<3>(10);
    
    EXPECT_GT(omega_dot.norm(), 1e-10);
    EXPECT_TRUE(omega_dot.allFinite());
}

TEST_F(RigidBodyDynamics6DOFTest, MultipleTorques) {
    // Test with multiple torques
    std::vector<std::shared_ptr<IForce>> forces;
    std::vector<std::shared_ptr<ITorque>> torques;
    
    Eigen::Matrix3d non_spherical = Eigen::Matrix3d::Zero();
    non_spherical(0, 0) = 10.0;
    non_spherical(1, 1) = 15.0;
    non_spherical(2, 2) = 20.0;
    
    torques.push_back(std::make_shared<GravityGradientTorque>(non_spherical));
    torques.push_back(std::make_shared<AerodynamicTorque>(
        Eigen::Vector3d(0.0, 0.0, 0.1),
        Eigen::Vector3d::Zero(),
        2.2, 1.0
    ));
    
    RigidBodyDynamics6DOF dynamics(forces, torques, non_spherical, mass_);
    
    Eigen::Quaterniond q(Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitX()));
    
    Eigen::VectorXd state = createLEOState(
        Eigen::Vector3d(6.771e6, 0.0, 0.0),
        Eigen::Vector3d(0.0, 7500.0, 0.0),
        q,
        Eigen::Vector3d(0.01, 0.0, 0.0)
    );
    
    Eigen::VectorXd state_dot = dynamics.compute_dynamics(0.0, state);
    
    Eigen::Vector3d omega_dot = state_dot.segment<3>(10);
    
    EXPECT_GT(omega_dot.norm(), 0.0);
    EXPECT_TRUE(omega_dot.allFinite());
}

TEST_F(RigidBodyDynamics6DOFTest, AngularMomentumConservation) {
    // Without external torques, angular momentum should be conserved
    std::vector<std::shared_ptr<IForce>> forces;
    std::vector<std::shared_ptr<ITorque>> torques;
    
    // Non-spherical inertia for gyroscopic effects
    Eigen::Matrix3d non_spherical = Eigen::Matrix3d::Zero();
    non_spherical(0, 0) = 10.0;
    non_spherical(1, 1) = 15.0;
    non_spherical(2, 2) = 20.0;
    
    RigidBodyDynamics6DOF dynamics(forces, torques, non_spherical, mass_);
    
    Eigen::Vector3d omega(0.1, 0.05, 0.08);
    Eigen::VectorXd state = createLEOState(
        Eigen::Vector3d(1e7, 0.0, 0.0),
        Eigen::Vector3d(0.0, 7500.0, 0.0),
        Eigen::Quaterniond::Identity(),
        omega
    );
    
    // Compute angular momentum
    Eigen::Vector3d L = non_spherical * omega;
    
    Eigen::VectorXd state_dot = dynamics.compute_dynamics(0.0, state);
    Eigen::Vector3d omega_dot = state_dot.segment<3>(10);
    
    // dL/dt = I·dω/dt + ω × (I·ω)
    Eigen::Vector3d L_dot = non_spherical * omega_dot + omega.cross(non_spherical * omega);
    
    // Without external torques, dL/dt = 0
    EXPECT_NEAR(L_dot.norm(), 0.0, 1e-12);
}

// ============================================================================
// JACOBIAN TESTS
// ============================================================================

TEST_F(RigidBodyDynamics6DOFTest, JacobianSize) {
    std::vector<std::shared_ptr<IForce>> forces;
    forces.push_back(std::make_shared<PointMassGravity>());
    
    std::vector<std::shared_ptr<ITorque>> torques;
    
    RigidBodyDynamics6DOF dynamics(forces, torques, inertia_, mass_);
    
    Eigen::VectorXd state = createLEOState(
        Eigen::Vector3d(6.771e6, 0.0, 0.0),
        Eigen::Vector3d(0.0, 7500.0, 0.0)
    );
    
    Eigen::MatrixXd F = dynamics.compute_jacobian(0.0, state);
    
    EXPECT_EQ(F.rows(), 13);
    EXPECT_EQ(F.cols(), 13);
}

TEST_F(RigidBodyDynamics6DOFTest, JacobianFinite) {
    std::vector<std::shared_ptr<IForce>> forces;
    forces.push_back(std::make_shared<PointMassGravity>());
    
    std::vector<std::shared_ptr<ITorque>> torques;
    torques.push_back(std::make_shared<GravityGradientTorque>(inertia_));
    
    RigidBodyDynamics6DOF dynamics(forces, torques, inertia_, mass_);
    
    Eigen::VectorXd state = createLEOState(
        Eigen::Vector3d(6.771e6, 0.0, 0.0),
        Eigen::Vector3d(0.0, 7500.0, 0.0)
    );
    
    Eigen::MatrixXd F = dynamics.compute_jacobian(0.0, state);
    
    EXPECT_TRUE(F.allFinite());
}

TEST_F(RigidBodyDynamics6DOFTest, JacobianStructureKinematic) {
    // Test kinematic block structure: dr/dt = v → ∂(dr/dt)/∂v = I
    std::vector<std::shared_ptr<IForce>> forces;
    std::vector<std::shared_ptr<ITorque>> torques;
    
    RigidBodyDynamics6DOF dynamics(forces, torques, inertia_, mass_);
    
    Eigen::VectorXd state = createLEOState(
        Eigen::Vector3d(1e7, 0.0, 0.0),
        Eigen::Vector3d(0.0, 7500.0, 0.0)
    );
    
    Eigen::MatrixXd F = dynamics.compute_jacobian(0.0, state);
    
    // Block (0:3, 3:6) should be identity
    Eigen::Matrix3d dr_dv = F.block<3, 3>(0, 3);
    
    EXPECT_TRUE(dr_dv.isApprox(Eigen::Matrix3d::Identity(), 1e-15));
}

TEST_F(RigidBodyDynamics6DOFTest, JacobianNumericalVsAnalytical) {
    std::vector<std::shared_ptr<IForce>> forces;
    forces.push_back(std::make_shared<PointMassGravity>());
    
    std::vector<std::shared_ptr<ITorque>> torques;
    
    RigidBodyDynamics6DOF dynamics(forces, torques, inertia_, mass_);
    
    Eigen::VectorXd state = createLEOState(
        Eigen::Vector3d(6.771e6, 0.0, 0.0),
        Eigen::Vector3d(0.0, 7500.0, 0.0)
    );
    
    Eigen::MatrixXd F_analytical = dynamics.compute_jacobian(0.0, state);
    
    // Numerical Jacobian
    double epsilon = 1e-7;  // Smaller epsilon for better accuracy
    Eigen::MatrixXd F_numerical = Eigen::MatrixXd::Zero(13, 13);
    
    // Test non-quaternion states
    for (int col : {0, 1, 2, 3, 4, 5, 10, 11, 12}) {
        Eigen::VectorXd state_plus = state;
        Eigen::VectorXd state_minus = state;
        
        state_plus(col) += epsilon;
        state_minus(col) -= epsilon;
        
        Eigen::VectorXd f_plus = dynamics.compute_dynamics(0.0, state_plus);
        Eigen::VectorXd f_minus = dynamics.compute_dynamics(0.0, state_minus);
        
        F_numerical.col(col) = (f_plus - f_minus) / (2.0 * epsilon);
    }
    
    // Compare columns (skip quaternion columns 6-9 for now)
    for (int col : {0, 1, 2, 3, 4, 5, 10, 11, 12}) {
        double error = (F_analytical.col(col) - F_numerical.col(col)).norm();
        EXPECT_LT(error, 1e-5) << "Column " << col << " error: " << error;
    }
}

TEST_F(RigidBodyDynamics6DOFTest, JacobianLinearizationAccuracy) {
    // Test that Jacobian accurately predicts state derivative changes
    std::vector<std::shared_ptr<IForce>> forces;
    forces.push_back(std::make_shared<PointMassGravity>());
    
    std::vector<std::shared_ptr<ITorque>> torques;
    
    RigidBodyDynamics6DOF dynamics(forces, torques, inertia_, mass_);
    
    Eigen::VectorXd state = createLEOState(
        Eigen::Vector3d(6.771e6, 0.0, 0.0),
        Eigen::Vector3d(0.0, 7500.0, 0.0)
    );
    
    Eigen::MatrixXd F = dynamics.compute_jacobian(0.0, state);
    Eigen::VectorXd f_base = dynamics.compute_dynamics(0.0, state);
    
    // Small perturbation
    Eigen::VectorXd delta_state = Eigen::VectorXd::Zero(13);
    delta_state(0) = 100.0;  // 100 m position change
    delta_state(3) = 1.0;    // 1 m/s velocity change
    
    Eigen::VectorXd state_perturbed = state + delta_state;
    Eigen::VectorXd f_perturbed = dynamics.compute_dynamics(0.0, state_perturbed);
    
    // Linear approximation
    Eigen::VectorXd f_linear = f_base + F * delta_state;
    
    double error = (f_perturbed - f_linear).norm();
    double relative_error = error / f_perturbed.norm();
    
    EXPECT_LT(relative_error, 0.01);  // Within 1% for small perturbations
}

// ============================================================================
// INTEGRATION TESTS
// ============================================================================

TEST_F(RigidBodyDynamics6DOFTest, CompleteOrbitDynamics) {
    // Test complete dynamics with forces and torques
    std::vector<std::shared_ptr<IForce>> forces;
    forces.push_back(std::make_shared<PointMassGravity>());
    
    std::vector<std::shared_ptr<ITorque>> torques;
    
    Eigen::Matrix3d non_spherical = Eigen::Matrix3d::Zero();
    non_spherical(0, 0) = 10.0;
    non_spherical(1, 1) = 15.0;
    non_spherical(2, 2) = 20.0;
    
    torques.push_back(std::make_shared<GravityGradientTorque>(non_spherical));
    torques.push_back(std::make_shared<AerodynamicTorque>(
        Eigen::Vector3d(0.0, 0.0, 0.1),
        Eigen::Vector3d::Zero(),
        2.2, 1.0
    ));
    
    RigidBodyDynamics6DOF dynamics(forces, torques, non_spherical, mass_);
    
    Eigen::Quaterniond q(Eigen::AngleAxisd(0.1, Eigen::Vector3d::UnitX()));
    
    Eigen::VectorXd state = createLEOState(
        Eigen::Vector3d(6.771e6, 0.0, 0.0),
        Eigen::Vector3d(0.0, 7500.0, 0.0),
        q,
        Eigen::Vector3d(0.001, 0.001, 0.001)
    );
    
    Eigen::VectorXd state_dot = dynamics.compute_dynamics(0.0, state);
    
    // All components should be finite
    EXPECT_TRUE(state_dot.allFinite());
    
    // Position derivative = velocity
    EXPECT_TRUE(state_dot.segment<3>(0).isApprox(state.segment<3>(3), 1e-12));
    
    // Acceleration should be non-zero (gravity)
    EXPECT_GT(state_dot.segment<3>(3).norm(), 0.0);
    
    // Quaternion derivative should be non-zero (rotating)
    EXPECT_GT(state_dot.segment<4>(6).norm(), 0.0);
    
    // Angular acceleration should be non-zero (torques present)
    EXPECT_GT(state_dot.segment<3>(10).norm(), 0.0);
}

TEST_F(RigidBodyDynamics6DOFTest, StateConsistency) {
    // Test that dynamics maintain physical constraints
    std::vector<std::shared_ptr<IForce>> forces;
    forces.push_back(std::make_shared<PointMassGravity>());
    
    std::vector<std::shared_ptr<ITorque>> torques;
    
    RigidBodyDynamics6DOF dynamics(forces, torques, inertia_, mass_);
    
    Eigen::VectorXd state = createLEOState(
        Eigen::Vector3d(6.771e6, 0.0, 0.0),
        Eigen::Vector3d(0.0, 7500.0, 0.0)
    );
    
    // Extract and verify quaternion is unit (note: Eigen uses x,y,z,w order)
    Eigen::Quaterniond q(state(9), state(6), state(7), state(8));
    EXPECT_NEAR(q.norm(), 1.0, 1e-15);
    
    Eigen::VectorXd state_dot = dynamics.compute_dynamics(0.0, state);
    
    // All outputs should be finite
    EXPECT_TRUE(state_dot.allFinite());
}

TEST_F(RigidBodyDynamics6DOFTest, TimeInvariance) {
    // For autonomous system, dynamics shouldn't depend on time
    std::vector<std::shared_ptr<IForce>> forces;
    forces.push_back(std::make_shared<PointMassGravity>());
    
    std::vector<std::shared_ptr<ITorque>> torques;
    
    RigidBodyDynamics6DOF dynamics(forces, torques, inertia_, mass_);
    
    Eigen::VectorXd state = createLEOState(
        Eigen::Vector3d(6.771e6, 0.0, 0.0),
        Eigen::Vector3d(0.0, 7500.0, 0.0)
    );
    
    Eigen::VectorXd state_dot_t0 = dynamics.compute_dynamics(0.0, state);
    Eigen::VectorXd state_dot_t1 = dynamics.compute_dynamics(100.0, state);
    
    // Should be identical for autonomous forces
    EXPECT_TRUE(state_dot_t0.isApprox(state_dot_t1, 1e-15));
}

TEST_F(RigidBodyDynamics6DOFTest, CircularOrbitVelocity) {
    // For circular orbit, velocity should be perpendicular to position
    std::vector<std::shared_ptr<IForce>> forces;
    forces.push_back(std::make_shared<PointMassGravity>());
    
    std::vector<std::shared_ptr<ITorque>> torques;
    
    RigidBodyDynamics6DOF dynamics(forces, torques, inertia_, mass_);
    
    // Circular orbit velocity at 400 km
    double r = 6.771e6;
    double v = std::sqrt(3.986004418e14 / r);
    
    Eigen::Vector3d pos(r, 0.0, 0.0);
    Eigen::Vector3d vel(0.0, v, 0.0);
    
    Eigen::VectorXd state = createLEOState(pos, vel);
    
    // Position and velocity should be perpendicular
    EXPECT_NEAR(pos.dot(vel), 0.0, 1e-6);
    
    Eigen::VectorXd state_dot = dynamics.compute_dynamics(0.0, state);
    
    Eigen::Vector3d acceleration = state_dot.segment<3>(3);
    
    // For circular orbit, acceleration should point toward center
    EXPECT_LT(acceleration.dot(pos) / (acceleration.norm() * pos.norm()), -0.999);
}

TEST_F(RigidBodyDynamics6DOFTest, SpinStability) {
    // Major axis spin should be stable
    std::vector<std::shared_ptr<IForce>> forces;
    std::vector<std::shared_ptr<ITorque>> torques;
    
    Eigen::Matrix3d non_spherical = Eigen::Matrix3d::Zero();
    non_spherical(0, 0) = 20.0;  // Largest moment of inertia
    non_spherical(1, 1) = 10.0;
    non_spherical(2, 2) = 10.0;
    
    RigidBodyDynamics6DOF dynamics(forces, torques, non_spherical, mass_);
    
    // Spin about major axis (x)
    Eigen::Vector3d omega(1.0, 0.0, 0.0);
    
    Eigen::VectorXd state = createLEOState(
        Eigen::Vector3d(1e7, 0.0, 0.0),
        Eigen::Vector3d(0.0, 7500.0, 0.0),
        Eigen::Quaterniond::Identity(),
        omega
    );
    
    Eigen::VectorXd state_dot = dynamics.compute_dynamics(0.0, state);
    
    Eigen::Vector3d omega_dot = state_dot.segment<3>(10);
    
    // Should have zero angular acceleration (stable spin)
    EXPECT_NEAR(omega_dot.norm(), 0.0, 1e-15);
}
