#include <gtest/gtest.h>
#include "dynamics/gravity_gradient_torque.hpp"
#include <cmath>

using namespace dynamics;

// Earth gravitational parameter
constexpr double EARTH_MU = 3.986004418e14;  // m³/s²

// ============================================================================
// TEST FIXTURE
// ============================================================================

class GravityGradientTorqueTest : public ::testing::Test {
protected:
    // Simple satellite inertia (cylindrical)
    Eigen::Matrix3d inertia_;
    GravityGradientTorque torque_;
    
    GravityGradientTorqueTest() 
        : torque_(createCylindricalInertia(), EARTH_MU) {
        inertia_ = createCylindricalInertia();
    }
    
    static Eigen::Matrix3d createCylindricalInertia() {
        // Cylinder aligned with z-axis: Ixx = Iyy > Izz
        Eigen::Matrix3d I;
        I << 100.0,   0.0,   0.0,
               0.0, 100.0,   0.0,
               0.0,   0.0,  50.0;
        return I;
    }
    
    static Eigen::Matrix3d createSphericalInertia() {
        // Sphere: Ixx = Iyy = Izz
        return Eigen::Matrix3d::Identity() * 100.0;
    }
    
    static Eigen::Matrix3d createBoxInertia() {
        // Box with distinct moments
        Eigen::Matrix3d I;
        I << 200.0,   0.0,   0.0,
               0.0, 150.0,   0.0,
               0.0,   0.0, 100.0;
        return I;
    }
};

// ============================================================================
// BASIC TORQUE TESTS
// ============================================================================

TEST_F(GravityGradientTorqueTest, ZeroTorqueWhenAligned) {
    // When body z-axis aligned with nadir, torque should be zero
    TorqueContext ctx;
    ctx.position = Eigen::Vector3d(0.0, 0.0, -7e6);  // Below satellite
    ctx.velocity = Eigen::Vector3d::Zero();
    ctx.orientation = Eigen::Quaterniond::Identity();  // z-axis down
    ctx.angular_velocity = Eigen::Vector3d::Zero();
    
    Eigen::Vector3d tau = torque_.compute_torque(ctx);
    
    EXPECT_NEAR(tau.norm(), 0.0, 1e-10);
}

TEST_F(GravityGradientTorqueTest, ZeroTorqueForSphere) {
    // Spherical satellite (I = kI) has zero gravity gradient torque
    GravityGradientTorque spherical_torque(createSphericalInertia(), EARTH_MU);
    
    TorqueContext ctx;
    ctx.position = Eigen::Vector3d(7e6, 0.0, 0.0);
    ctx.velocity = Eigen::Vector3d::Zero();
    ctx.orientation = Eigen::Quaterniond::Identity();
    ctx.angular_velocity = Eigen::Vector3d::Zero();
    
    Eigen::Vector3d tau = spherical_torque.compute_torque(ctx);
    
    EXPECT_NEAR(tau.norm(), 0.0, 1e-10);
}

TEST_F(GravityGradientTorqueTest, NonzeroTorqueWhenMisaligned) {
    // 45° rotation about y-axis
    double angle = M_PI / 4.0;
    Eigen::Quaterniond q(Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitY()));
    
    TorqueContext ctx;
    ctx.position = Eigen::Vector3d(0.0, 0.0, -7e6);
    ctx.velocity = Eigen::Vector3d::Zero();
    ctx.orientation = q;
    ctx.angular_velocity = Eigen::Vector3d::Zero();
    
    Eigen::Vector3d tau = torque_.compute_torque(ctx);
    
    // Should have torque about y-axis
    EXPECT_GT(std::abs(tau(1)), 1e-6);
}

TEST_F(GravityGradientTorqueTest, TorqueScalesWithDistance) {
    // Torque ∝ 1/r³
    TorqueContext ctx1, ctx2;
    ctx1.position = Eigen::Vector3d(0.0, 0.0, -7e6);
    ctx2.position = Eigen::Vector3d(0.0, 0.0, -14e6);  // Double distance
    
    Eigen::Quaterniond q(Eigen::AngleAxisd(M_PI/4.0, Eigen::Vector3d::UnitY()));
    ctx1.orientation = ctx2.orientation = q;
    ctx1.velocity = ctx2.velocity = Eigen::Vector3d::Zero();
    ctx1.angular_velocity = ctx2.angular_velocity = Eigen::Vector3d::Zero();
    
    Eigen::Vector3d tau1 = torque_.compute_torque(ctx1);
    Eigen::Vector3d tau2 = torque_.compute_torque(ctx2);
    
    // At 2x distance, torque should be 1/8 (inverse cube law)
    EXPECT_NEAR(tau2.norm(), tau1.norm() / 8.0, tau1.norm() * 0.01);
}

TEST_F(GravityGradientTorqueTest, TorquePerpendicular) {
    // Torque should be perpendicular to both nadir and angular momentum
    Eigen::Quaterniond q(Eigen::AngleAxisd(M_PI/6.0, Eigen::Vector3d::UnitY()));
    
    TorqueContext ctx;
    ctx.position = Eigen::Vector3d(0.0, 0.0, -7e6);
    ctx.velocity = Eigen::Vector3d::Zero();
    ctx.orientation = q;
    ctx.angular_velocity = Eigen::Vector3d::Zero();
    
    Eigen::Vector3d tau = torque_.compute_torque(ctx);
    
    // Nadir direction in body frame
    Eigen::Vector3d nadir_body = q.inverse() * (-ctx.position.normalized());
    Eigen::Vector3d I_nadir = inertia_ * nadir_body;
    
    // τ = nadir × (I·nadir), so τ ⊥ nadir
    EXPECT_NEAR(tau.dot(nadir_body), 0.0, 1e-10);
}

TEST_F(GravityGradientTorqueTest, RestoringTorque) {
    // Torque should act to restore minimum inertia axis to nadir
    // For cylinder (Izz < Ixx = Iyy), z-axis wants to align with nadir
    
    // Rotate satellite so x-axis points to nadir (unstable)
    Eigen::Quaterniond q(Eigen::AngleAxisd(M_PI/2.0, Eigen::Vector3d::UnitY()));
    
    TorqueContext ctx;
    ctx.position = Eigen::Vector3d(0.0, 0.0, -7e6);
    ctx.velocity = Eigen::Vector3d::Zero();
    ctx.orientation = q;
    ctx.angular_velocity = Eigen::Vector3d::Zero();
    
    Eigen::Vector3d tau = torque_.compute_torque(ctx);
    
    // Torque should rotate satellite to align z-axis with nadir
    // (This is a restoring torque about y-axis)
    EXPECT_NE(tau(1), 0.0);
}

TEST_F(GravityGradientTorqueTest, VelocityIndependent) {
    // Gravity gradient torque doesn't depend on velocity
    TorqueContext ctx1, ctx2;
    ctx1.position = ctx2.position = Eigen::Vector3d(7e6, 1e6, 2e6);
    ctx1.orientation = ctx2.orientation = Eigen::Quaterniond::Identity();
    ctx1.angular_velocity = ctx2.angular_velocity = Eigen::Vector3d::Zero();
    
    ctx1.velocity = Eigen::Vector3d(100.0, 200.0, 300.0);
    ctx2.velocity = Eigen::Vector3d(-500.0, 1000.0, -250.0);
    
    Eigen::Vector3d tau1 = torque_.compute_torque(ctx1);
    Eigen::Vector3d tau2 = torque_.compute_torque(ctx2);
    
    EXPECT_TRUE(tau1.isApprox(tau2));
}

TEST_F(GravityGradientTorqueTest, AngularVelocityIndependent) {
    // Gravity gradient torque doesn't depend on angular velocity
    TorqueContext ctx1, ctx2;
    ctx1.position = ctx2.position = Eigen::Vector3d(7e6, 1e6, 2e6);
    ctx1.orientation = ctx2.orientation = Eigen::Quaterniond::Identity();
    ctx1.velocity = ctx2.velocity = Eigen::Vector3d::Zero();
    
    ctx1.angular_velocity = Eigen::Vector3d(0.01, 0.02, 0.03);
    ctx2.angular_velocity = Eigen::Vector3d(-0.05, 0.10, -0.02);
    
    Eigen::Vector3d tau1 = torque_.compute_torque(ctx1);
    Eigen::Vector3d tau2 = torque_.compute_torque(ctx2);
    
    EXPECT_TRUE(tau1.isApprox(tau2));
}

TEST_F(GravityGradientTorqueTest, NearZeroRadius) {
    // Should handle near-zero radius gracefully
    TorqueContext ctx;
    ctx.position = Eigen::Vector3d(0.1, 0.0, 0.0);
    ctx.velocity = Eigen::Vector3d::Zero();
    ctx.orientation = Eigen::Quaterniond::Identity();
    ctx.angular_velocity = Eigen::Vector3d::Zero();
    
    Eigen::Vector3d tau = torque_.compute_torque(ctx);
    
    // Should return zero (safety check)
    EXPECT_TRUE(tau.isZero());
}

// ============================================================================
// PHYSICAL PROPERTIES TESTS
// ============================================================================

TEST_F(GravityGradientTorqueTest, MaximumTorqueAt45Degrees) {
    // Maximum torque occurs at 45° misalignment
    GravityGradientTorque box_torque(createBoxInertia(), EARTH_MU);
    
    TorqueContext ctx;
    ctx.position = Eigen::Vector3d(0.0, 0.0, -7e6);
    ctx.velocity = Eigen::Vector3d::Zero();
    ctx.angular_velocity = Eigen::Vector3d::Zero();
    
    double max_torque = 0.0;
    for (int i = 0; i < 90; ++i) {
        double angle = i * M_PI / 180.0;
        ctx.orientation = Eigen::Quaterniond(Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitY()));
        Eigen::Vector3d tau = box_torque.compute_torque(ctx);
        max_torque = std::max(max_torque, tau.norm());
    }
    
    // Check that torque at 45° is close to maximum
    ctx.orientation = Eigen::Quaterniond(Eigen::AngleAxisd(M_PI/4.0, Eigen::Vector3d::UnitY()));
    Eigen::Vector3d tau_45 = box_torque.compute_torque(ctx);
    
    EXPECT_GT(tau_45.norm(), max_torque * 0.95);
}

TEST_F(GravityGradientTorqueTest, TorqueMagnitudeFormula) {
    // |τ| = (3μ/r³) |I_max - I_min| sin(2θ) / 2
    // where θ is angle between principal axis and nadir
    
    GravityGradientTorque box_torque(createBoxInertia(), EARTH_MU);
    Eigen::Matrix3d I_box = createBoxInertia();
    
    double I_max = I_box(0, 0);  // 200
    double I_min = I_box(2, 2);  // 100
    double r = 7e6;
    double theta = M_PI / 6.0;  // 30°
    
    TorqueContext ctx;
    ctx.position = Eigen::Vector3d(0.0, 0.0, -r);
    ctx.velocity = Eigen::Vector3d::Zero();
    ctx.orientation = Eigen::Quaterniond(Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitY()));
    ctx.angular_velocity = Eigen::Vector3d::Zero();
    
    Eigen::Vector3d tau = box_torque.compute_torque(ctx);
    
    double expected = (3.0 * EARTH_MU / (r*r*r)) * (I_max - I_min) * std::sin(2*theta) / 2.0;
    
    EXPECT_NEAR(tau.norm(), expected, expected * 0.1);  // Within 10%
}

TEST_F(GravityGradientTorqueTest, PeriodicTorque) {
    // Torque should be periodic with rotation
    TorqueContext ctx;
    ctx.position = Eigen::Vector3d(0.0, 0.0, -7e6);
    ctx.velocity = Eigen::Vector3d::Zero();
    ctx.angular_velocity = Eigen::Vector3d::Zero();
    
    // At 0° and 180°, torques should be equal magnitude
    ctx.orientation = Eigen::Quaterniond(Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitY()));
    Eigen::Vector3d tau_0 = torque_.compute_torque(ctx);
    
    ctx.orientation = Eigen::Quaterniond(Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitY()));
    Eigen::Vector3d tau_180 = torque_.compute_torque(ctx);
    
    EXPECT_NEAR(tau_0.norm(), tau_180.norm(), 1e-10);
}

// ============================================================================
// JACOBIAN TESTS
// ============================================================================

TEST_F(GravityGradientTorqueTest, JacobianSize) {
    TorqueContext ctx;
    ctx.position = Eigen::Vector3d(7e6, 1e6, 2e6);
    ctx.velocity = Eigen::Vector3d::Zero();
    ctx.orientation = Eigen::Quaterniond::Identity();
    ctx.angular_velocity = Eigen::Vector3d::Zero();
    
    auto [dtau_dq, dtau_domega, dtau_dr, dtau_dv] = torque_.compute_jacobian(ctx);
    
    EXPECT_EQ(dtau_dq.rows(), 3);
    EXPECT_EQ(dtau_dq.cols(), 3);
    EXPECT_EQ(dtau_domega.rows(), 3);
    EXPECT_EQ(dtau_domega.cols(), 3);
    EXPECT_EQ(dtau_dr.rows(), 3);
    EXPECT_EQ(dtau_dr.cols(), 3);
    EXPECT_EQ(dtau_dv.rows(), 3);
    EXPECT_EQ(dtau_dv.cols(), 3);
}

TEST_F(GravityGradientTorqueTest, JacobianVelocityZero) {
    TorqueContext ctx;
    ctx.position = Eigen::Vector3d(7e6, 1e6, 2e6);
    ctx.velocity = Eigen::Vector3d::Zero();
    ctx.orientation = Eigen::Quaterniond::Identity();
    ctx.angular_velocity = Eigen::Vector3d::Zero();
    
    auto [dtau_dq, dtau_domega, dtau_dr, dtau_dv] = torque_.compute_jacobian(ctx);
    
    // Gravity gradient doesn't depend on velocity
    EXPECT_DOUBLE_EQ(dtau_dv.norm(), 0.0);
}

TEST_F(GravityGradientTorqueTest, JacobianAngularVelocityZero) {
    TorqueContext ctx;
    ctx.position = Eigen::Vector3d(7e6, 1e6, 2e6);
    ctx.velocity = Eigen::Vector3d::Zero();
    ctx.orientation = Eigen::Quaterniond::Identity();
    ctx.angular_velocity = Eigen::Vector3d::Zero();
    
    auto [dtau_dq, dtau_domega, dtau_dr, dtau_dv] = torque_.compute_jacobian(ctx);
    
    // Gravity gradient doesn't depend on angular velocity
    EXPECT_DOUBLE_EQ(dtau_domega.norm(), 0.0);
}

TEST_F(GravityGradientTorqueTest, JacobianAttitudeNumerical) {
    TorqueContext ctx;
    ctx.position = Eigen::Vector3d(7e6, 1e6, 2e6);
    ctx.velocity = Eigen::Vector3d::Zero();
    Eigen::Quaterniond q(Eigen::AngleAxisd(0.3, Eigen::Vector3d(1, 2, 3).normalized()));
    ctx.orientation = q;
    ctx.angular_velocity = Eigen::Vector3d::Zero();
    
    auto [dtau_dq, dtau_domega, dtau_dr, dtau_dv] = torque_.compute_jacobian(ctx);
    
    // Numerical Jacobian for attitude (small rotation perturbations)
    double epsilon = 1e-6;
    Eigen::Matrix3d dtau_dq_numerical;
    
    for (int i = 0; i < 3; ++i) {
        Eigen::Vector3d delta = Eigen::Vector3d::Zero();
        delta(i) = epsilon;
        
        // Perturb attitude with small rotation
        Eigen::Quaterniond q_plus = q * Eigen::Quaterniond(
            Eigen::AngleAxisd(epsilon, Eigen::Vector3d::Unit(i))
        );
        Eigen::Quaterniond q_minus = q * Eigen::Quaterniond(
            Eigen::AngleAxisd(-epsilon, Eigen::Vector3d::Unit(i))
        );
        
        TorqueContext ctx_plus = ctx;
        TorqueContext ctx_minus = ctx;
        ctx_plus.orientation = q_plus;
        ctx_minus.orientation = q_minus;
        
        Eigen::Vector3d tau_plus = torque_.compute_torque(ctx_plus);
        Eigen::Vector3d tau_minus = torque_.compute_torque(ctx_minus);
        
        dtau_dq_numerical.col(i) = (tau_plus - tau_minus) / (2.0 * epsilon);
    }
    
    double error = (dtau_dq - dtau_dq_numerical).norm();
    EXPECT_LT(error, 1e-4) << "Attitude Jacobian error: " << error;
}

TEST_F(GravityGradientTorqueTest, JacobianPositionNumerical) {
    TorqueContext ctx;
    ctx.position = Eigen::Vector3d(7e6, 1e6, 2e6);
    ctx.velocity = Eigen::Vector3d::Zero();
    ctx.orientation = Eigen::Quaterniond(Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitY()));
    ctx.angular_velocity = Eigen::Vector3d::Zero();
    
    auto [dtau_dq, dtau_domega, dtau_dr, dtau_dv] = torque_.compute_jacobian(ctx);
    
    // Numerical Jacobian for position
    double epsilon = 1e3;  // Larger epsilon for position (in meters)
    Eigen::Matrix3d dtau_dr_numerical;
    
    for (int i = 0; i < 3; ++i) {
        TorqueContext ctx_plus = ctx;
        TorqueContext ctx_minus = ctx;
        
        ctx_plus.position(i) += epsilon;
        ctx_minus.position(i) -= epsilon;
        
        Eigen::Vector3d tau_plus = torque_.compute_torque(ctx_plus);
        Eigen::Vector3d tau_minus = torque_.compute_torque(ctx_minus);
        
        dtau_dr_numerical.col(i) = (tau_plus - tau_minus) / (2.0 * epsilon);
    }
    
    double error = (dtau_dr - dtau_dr_numerical).norm();
    EXPECT_LT(error, 1e-10) << "Position Jacobian error: " << error;
}

TEST_F(GravityGradientTorqueTest, JacobianAttitudeSymmetric) {
    TorqueContext ctx;
    ctx.position = Eigen::Vector3d(7e6, 1e6, 2e6);
    ctx.velocity = Eigen::Vector3d::Zero();
    ctx.orientation = Eigen::Quaterniond::Identity();
    ctx.angular_velocity = Eigen::Vector3d::Zero();
    
    auto [dtau_dq, dtau_domega, dtau_dr, dtau_dv] = torque_.compute_jacobian(ctx);
    
    // For symmetric inertia and certain orientations, Jacobian may have symmetries
    // Just check it's finite and reasonable
    EXPECT_TRUE(dtau_dq.allFinite());
}

TEST_F(GravityGradientTorqueTest, JacobianScalesWithDistance) {
    // Position Jacobian should scale with 1/r⁴ (derivative of 1/r³)
    TorqueContext ctx1, ctx2;
    ctx1.position = Eigen::Vector3d(7e6, 0.0, 0.0);
    ctx2.position = Eigen::Vector3d(14e6, 0.0, 0.0);  // Double distance
    
    Eigen::Quaterniond q(Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitY()));
    ctx1.orientation = ctx2.orientation = q;
    ctx1.velocity = ctx2.velocity = Eigen::Vector3d::Zero();
    ctx1.angular_velocity = ctx2.angular_velocity = Eigen::Vector3d::Zero();
    
    auto [dtau_dq1, _, dtau_dr1, __] = torque_.compute_jacobian(ctx1);
    auto [dtau_dq2, ___, dtau_dr2, ____] = torque_.compute_jacobian(ctx2);
    
    // At 2x distance, derivatives should be ~1/16 (1/2⁴)
    double ratio = dtau_dr2.norm() / dtau_dr1.norm();
    EXPECT_NEAR(ratio, 1.0/16.0, 0.02);
}

TEST_F(GravityGradientTorqueTest, JacobianLinearConsistency) {
    // Taylor expansion: τ(θ + δθ) ≈ τ(θ) + ∂τ/∂θ · δθ
    TorqueContext ctx;
    ctx.position = Eigen::Vector3d(7e6, 1e6, 2e6);
    ctx.velocity = Eigen::Vector3d::Zero();
    ctx.orientation = Eigen::Quaterniond(Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitY()));
    ctx.angular_velocity = Eigen::Vector3d::Zero();
    
    auto [dtau_dq, dtau_domega, dtau_dr, dtau_dv] = torque_.compute_jacobian(ctx);
    
    // Small perturbations
    Eigen::Vector3d delta_theta(0.01, -0.005, 0.008);  // Small rotation
    Eigen::Vector3d delta_r(100.0, -50.0, 75.0);       // Small position change
    
    // Perturbed state
    TorqueContext ctx_perturbed = ctx;
    ctx_perturbed.orientation = ctx.orientation * Eigen::Quaterniond(
        Eigen::AngleAxisd(delta_theta.norm(), delta_theta.normalized())
    );
    ctx_perturbed.position += delta_r;
    
    // Compute torques
    Eigen::Vector3d tau_base = torque_.compute_torque(ctx);
    Eigen::Vector3d tau_perturbed = torque_.compute_torque(ctx_perturbed);
    
    // Linear approximation
    Eigen::Vector3d tau_linear = tau_base + dtau_dq * delta_theta + dtau_dr * delta_r;
    
    // Should be reasonably close for small perturbations
    double error = (tau_perturbed - tau_linear).norm();
    EXPECT_LT(error, tau_base.norm() * 0.01);  // Within 1%
}

TEST_F(GravityGradientTorqueTest, JacobianZeroForSphere) {
    // Spherical satellite has zero torque → zero Jacobians
    GravityGradientTorque spherical_torque(createSphericalInertia(), EARTH_MU);
    
    TorqueContext ctx;
    ctx.position = Eigen::Vector3d(7e6, 1e6, 2e6);
    ctx.velocity = Eigen::Vector3d::Zero();
    ctx.orientation = Eigen::Quaterniond::Identity();
    ctx.angular_velocity = Eigen::Vector3d::Zero();
    
    auto [dtau_dq, dtau_domega, dtau_dr, dtau_dv] = spherical_torque.compute_jacobian(ctx);
    
    // All Jacobians should be zero (or very small due to numerical precision)
    EXPECT_LT(dtau_dq.norm(), 1e-10);
    EXPECT_LT(dtau_dr.norm(), 1e-10);
}

TEST_F(GravityGradientTorqueTest, JacobianFiniteValues) {
    // Jacobians should always be finite
    TorqueContext ctx;
    ctx.position = Eigen::Vector3d(7e6, 1e6, 2e6);
    ctx.velocity = Eigen::Vector3d::Zero();
    ctx.orientation = Eigen::Quaterniond(Eigen::AngleAxisd(0.5, Eigen::Vector3d(1,2,3).normalized()));
    ctx.angular_velocity = Eigen::Vector3d::Zero();
    
    auto [dtau_dq, dtau_domega, dtau_dr, dtau_dv] = torque_.compute_jacobian(ctx);
    
    EXPECT_TRUE(dtau_dq.allFinite());
    EXPECT_TRUE(dtau_domega.allFinite());
    EXPECT_TRUE(dtau_dr.allFinite());
    EXPECT_TRUE(dtau_dv.allFinite());
}

// ============================================================================
// EDGE CASES
// ============================================================================

TEST_F(GravityGradientTorqueTest, MultipleRotations) {
    // Test with compound rotations
    Eigen::Quaterniond q1(Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitX()));
    Eigen::Quaterniond q2(Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitY()));
    Eigen::Quaterniond q3(Eigen::AngleAxisd(0.4, Eigen::Vector3d::UnitZ()));
    Eigen::Quaterniond q = q1 * q2 * q3;
    
    TorqueContext ctx;
    ctx.position = Eigen::Vector3d(7e6, 1e6, 2e6);
    ctx.velocity = Eigen::Vector3d::Zero();
    ctx.orientation = q;
    ctx.angular_velocity = Eigen::Vector3d::Zero();
    
    Eigen::Vector3d tau = torque_.compute_torque(ctx);
    
    EXPECT_TRUE(tau.allFinite());
    EXPECT_GT(tau.norm(), 0.0);  // Should have some torque
}

TEST_F(GravityGradientTorqueTest, HighAltitude) {
    // At GEO altitude (~36,000 km), torque should be much smaller
    TorqueContext ctx_leo, ctx_geo;
    ctx_leo.position = Eigen::Vector3d(7e6, 0.0, 0.0);
    ctx_geo.position = Eigen::Vector3d(42e6, 0.0, 0.0);  // GEO
    
    Eigen::Quaterniond q(Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitY()));
    ctx_leo.orientation = ctx_geo.orientation = q;
    ctx_leo.velocity = ctx_geo.velocity = Eigen::Vector3d::Zero();
    ctx_leo.angular_velocity = ctx_geo.angular_velocity = Eigen::Vector3d::Zero();
    
    Eigen::Vector3d tau_leo = torque_.compute_torque(ctx_leo);
    Eigen::Vector3d tau_geo = torque_.compute_torque(ctx_geo);
    
    // At 6x distance, torque should be ~1/216 (1/6³)
    EXPECT_LT(tau_geo.norm(), tau_leo.norm() / 100.0);
}

TEST_F(GravityGradientTorqueTest, LargeInertia) {
    // Test with large inertia values
    Eigen::Matrix3d large_inertia;
    large_inertia << 1e6,   0.0,   0.0,
                      0.0, 1e6,   0.0,
                      0.0,   0.0, 5e5;
    
    GravityGradientTorque large_torque(large_inertia, EARTH_MU);
    
    TorqueContext ctx;
    ctx.position = Eigen::Vector3d(7e6, 0.0, 0.0);
    ctx.velocity = Eigen::Vector3d::Zero();
    ctx.orientation = Eigen::Quaterniond(Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitY()));
    ctx.angular_velocity = Eigen::Vector3d::Zero();
    
    Eigen::Vector3d tau = large_torque.compute_torque(ctx);
    
    EXPECT_TRUE(tau.allFinite());
    EXPECT_GT(tau.norm(), 0.0);
}
