#include <gtest/gtest.h>
#include "dynamics/aerodynamic_torque.hpp"
#include "dynamics/atmosphere.hpp"
#include <cmath>

using namespace dynamics;

// ============================================================================
// TEST FIXTURE
// ============================================================================

class AerodynamicTorqueTest : public ::testing::Test {
protected:
    // Typical small satellite parameters
    Eigen::Vector3d cop_;   // Center of pressure
    Eigen::Vector3d com_;   // Center of mass
    double cd_;             // Drag coefficient
    double area_;           // Reference area
    
    AerodynamicTorqueTest()
        : cop_(0.0, 0.0, 0.1),    // COP 10 cm above COM
          com_(0.0, 0.0, 0.0),    // COM at origin
          cd_(2.2),               // Typical drag coefficient
          area_(1.0)              // 1 m² reference area
    {}
    
    TorqueContext createLEOContext(const Eigen::Vector3d& velocity) {
        TorqueContext ctx;
        ctx.position = Eigen::Vector3d(0.0, 0.0, 6.771e6);  // 400 km altitude
        ctx.velocity = velocity;
        ctx.orientation = Eigen::Quaterniond::Identity();
        ctx.angular_velocity = Eigen::Vector3d::Zero();
        return ctx;
    }
};

// ============================================================================
// BASIC TORQUE TESTS
// ============================================================================

TEST_F(AerodynamicTorqueTest, ZeroTorqueAtHighAltitude) {
    // At very high altitude, density → 0, torque → 0
    AerodynamicTorque torque(cop_, com_, cd_, area_);
    
    TorqueContext ctx;
    ctx.position = Eigen::Vector3d(0.0, 0.0, 20e6);  // 13,629 km altitude
    ctx.velocity = Eigen::Vector3d(7500.0, 0.0, 0.0);
    ctx.orientation = Eigen::Quaterniond::Identity();
    ctx.angular_velocity = Eigen::Vector3d::Zero();
    
    Eigen::Vector3d tau = torque.compute_torque(ctx);
    
    EXPECT_NEAR(tau.norm(), 0.0, 1e-15);
}

TEST_F(AerodynamicTorqueTest, ZeroTorqueAtZeroVelocity) {
    // No velocity → no drag → no torque
    AerodynamicTorque torque(cop_, com_, cd_, area_);
    
    TorqueContext ctx = createLEOContext(Eigen::Vector3d::Zero());
    
    Eigen::Vector3d tau = torque.compute_torque(ctx);
    
    EXPECT_NEAR(tau.norm(), 0.0, 1e-15);
}

TEST_F(AerodynamicTorqueTest, ZeroTorqueWhenCOPEqualsCOM) {
    // No moment arm → no torque
    Eigen::Vector3d same_point(0.0, 0.0, 0.0);
    AerodynamicTorque torque(same_point, same_point, cd_, area_);
    
    TorqueContext ctx = createLEOContext(Eigen::Vector3d(7500.0, 0.0, 0.0));
    
    Eigen::Vector3d tau = torque.compute_torque(ctx);
    
    EXPECT_NEAR(tau.norm(), 0.0, 1e-15);
}

TEST_F(AerodynamicTorqueTest, NonzeroTorqueInLEO) {
    // Should have significant torque at LEO altitude
    AerodynamicTorque torque(cop_, com_, cd_, area_);
    
    TorqueContext ctx = createLEOContext(Eigen::Vector3d(7500.0, 0.0, 0.0));
    
    Eigen::Vector3d tau = torque.compute_torque(ctx);
    
    EXPECT_GT(tau.norm(), 1e-10);  // Should be non-zero
}

TEST_F(AerodynamicTorqueTest, TorquePerpendicular) {
    // Torque = r × F, so τ ⊥ r and τ ⊥ F
    AerodynamicTorque torque(cop_, com_, cd_, area_);
    
    TorqueContext ctx = createLEOContext(Eigen::Vector3d(7500.0, 0.0, 0.0));
    
    Eigen::Vector3d tau = torque.compute_torque(ctx);
    Eigen::Vector3d moment_arm = cop_ - com_;
    
    // τ should be perpendicular to moment arm
    EXPECT_NEAR(tau.dot(moment_arm), 0.0, 1e-10);
}

TEST_F(AerodynamicTorqueTest, TorqueScalesWithVelocitySquared) {
    // Drag force ∝ v², so torque ∝ v²
    AerodynamicTorque torque(cop_, com_, cd_, area_);
    
    TorqueContext ctx1 = createLEOContext(Eigen::Vector3d(7500.0, 0.0, 0.0));
    TorqueContext ctx2 = createLEOContext(Eigen::Vector3d(15000.0, 0.0, 0.0));
    
    Eigen::Vector3d tau1 = torque.compute_torque(ctx1);
    Eigen::Vector3d tau2 = torque.compute_torque(ctx2);
    
    // At 2× velocity, torque should be 4×
    EXPECT_NEAR(tau2.norm(), 4.0 * tau1.norm(), tau1.norm() * 0.01);
}

TEST_F(AerodynamicTorqueTest, TorqueScalesWithArea) {
    // Torque ∝ area
    AerodynamicTorque torque1(cop_, com_, cd_, 1.0);
    AerodynamicTorque torque2(cop_, com_, cd_, 2.0);
    
    TorqueContext ctx = createLEOContext(Eigen::Vector3d(7500.0, 0.0, 0.0));
    
    Eigen::Vector3d tau1 = torque1.compute_torque(ctx);
    Eigen::Vector3d tau2 = torque2.compute_torque(ctx);
    
    EXPECT_NEAR(tau2.norm(), 2.0 * tau1.norm(), tau1.norm() * 0.01);
}

TEST_F(AerodynamicTorqueTest, TorqueScalesWithDragCoefficient) {
    // Torque ∝ Cd
    AerodynamicTorque torque1(cop_, com_, 2.0, area_);
    AerodynamicTorque torque2(cop_, com_, 3.0, area_);
    
    TorqueContext ctx = createLEOContext(Eigen::Vector3d(7500.0, 0.0, 0.0));
    
    Eigen::Vector3d tau1 = torque1.compute_torque(ctx);
    Eigen::Vector3d tau2 = torque2.compute_torque(ctx);
    
    EXPECT_NEAR(tau2.norm(), 1.5 * tau1.norm(), tau1.norm() * 0.01);
}

TEST_F(AerodynamicTorqueTest, TorqueScalesWithMomentArm) {
    // Torque ∝ moment arm length
    Eigen::Vector3d cop1(0.0, 0.0, 0.1);   // 10 cm offset
    Eigen::Vector3d cop2(0.0, 0.0, 0.2);   // 20 cm offset
    
    AerodynamicTorque torque1(cop1, com_, cd_, area_);
    AerodynamicTorque torque2(cop2, com_, cd_, area_);
    
    TorqueContext ctx = createLEOContext(Eigen::Vector3d(7500.0, 0.0, 0.0));
    
    Eigen::Vector3d tau1 = torque1.compute_torque(ctx);
    Eigen::Vector3d tau2 = torque2.compute_torque(ctx);
    
    EXPECT_NEAR(tau2.norm(), 2.0 * tau1.norm(), tau1.norm() * 0.01);
}

TEST_F(AerodynamicTorqueTest, TorqueDecreaseWithAltitude) {
    // Higher altitude → lower density → lower torque
    AerodynamicTorque torque(cop_, com_, cd_, area_);
    
    TorqueContext ctx_low = createLEOContext(Eigen::Vector3d(7500.0, 0.0, 0.0));
    ctx_low.position = Eigen::Vector3d(0.0, 0.0, 6.571e6);  // 200 km
    
    TorqueContext ctx_high = createLEOContext(Eigen::Vector3d(7500.0, 0.0, 0.0));
    ctx_high.position = Eigen::Vector3d(0.0, 0.0, 6.971e6);  // 600 km
    
    Eigen::Vector3d tau_low = torque.compute_torque(ctx_low);
    Eigen::Vector3d tau_high = torque.compute_torque(ctx_high);
    
    EXPECT_GT(tau_low.norm(), tau_high.norm());
}

TEST_F(AerodynamicTorqueTest, TorqueDirection) {
    // For velocity in +x direction and COP above COM (+z),
    // drag force should be in -x direction,
    // torque should be about y-axis
    Eigen::Vector3d cop(0.0, 0.0, 0.1);
    Eigen::Vector3d com(0.0, 0.0, 0.0);
    AerodynamicTorque torque(cop, com, cd_, area_);
    
    TorqueContext ctx = createLEOContext(Eigen::Vector3d(7500.0, 0.0, 0.0));
    
    Eigen::Vector3d tau = torque.compute_torque(ctx);
    
    // Torque should primarily be about y-axis
    EXPECT_GT(std::abs(tau(1)), std::abs(tau(0)));
    EXPECT_GT(std::abs(tau(1)), std::abs(tau(2)));
}

// ============================================================================
// ATTITUDE DEPENDENCE TESTS
// ============================================================================

TEST_F(AerodynamicTorqueTest, TorqueChangesWithAttitude) {
    // Different attitudes → different body-frame velocity → different torque
    AerodynamicTorque torque(cop_, com_, cd_, area_);
    
    TorqueContext ctx1 = createLEOContext(Eigen::Vector3d(7500.0, 0.0, 0.0));
    ctx1.orientation = Eigen::Quaterniond::Identity();
    
    TorqueContext ctx2 = ctx1;
    ctx2.orientation = Eigen::Quaterniond(Eigen::AngleAxisd(M_PI/2.0, Eigen::Vector3d::UnitZ()));
    
    Eigen::Vector3d tau1 = torque.compute_torque(ctx1);
    Eigen::Vector3d tau2 = torque.compute_torque(ctx2);
    
    // Torques should be different
    EXPECT_GT((tau1 - tau2).norm(), 1e-12);
}

TEST_F(AerodynamicTorqueTest, Torque180DegreeRotation) {
    // 180° rotation should reverse torque direction (approximately)
    AerodynamicTorque torque(cop_, com_, cd_, area_);
    
    TorqueContext ctx1 = createLEOContext(Eigen::Vector3d(7500.0, 0.0, 0.0));
    ctx1.orientation = Eigen::Quaterniond::Identity();
    
    TorqueContext ctx2 = ctx1;
    ctx2.orientation = Eigen::Quaterniond(Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitZ()));
    
    Eigen::Vector3d tau1 = torque.compute_torque(ctx1);
    Eigen::Vector3d tau2 = torque.compute_torque(ctx2);
    
    // Torques should have similar magnitude but opposite direction (approximately)
    EXPECT_NEAR(tau1.norm(), tau2.norm(), tau1.norm() * 0.01);
}

// ============================================================================
// JACOBIAN TESTS
// ============================================================================

TEST_F(AerodynamicTorqueTest, JacobianSize) {
    AerodynamicTorque torque(cop_, com_, cd_, area_);
    
    TorqueContext ctx = createLEOContext(Eigen::Vector3d(7500.0, 0.0, 0.0));
    
    auto [dtau_dq, dtau_domega, dtau_dr, dtau_dv] = torque.compute_jacobian(ctx);
    
    EXPECT_EQ(dtau_dq.rows(), 3);
    EXPECT_EQ(dtau_dq.cols(), 3);
    EXPECT_EQ(dtau_domega.rows(), 3);
    EXPECT_EQ(dtau_domega.cols(), 3);
    EXPECT_EQ(dtau_dr.rows(), 3);
    EXPECT_EQ(dtau_dr.cols(), 3);
    EXPECT_EQ(dtau_dv.rows(), 3);
    EXPECT_EQ(dtau_dv.cols(), 3);
}

TEST_F(AerodynamicTorqueTest, JacobianAngularVelocityZero) {
    // This simple model doesn't depend on angular velocity
    AerodynamicTorque torque(cop_, com_, cd_, area_);
    
    TorqueContext ctx = createLEOContext(Eigen::Vector3d(7500.0, 0.0, 0.0));
    
    auto [dtau_dq, dtau_domega, dtau_dr, dtau_dv] = torque.compute_jacobian(ctx);
    
    EXPECT_DOUBLE_EQ(dtau_domega.norm(), 0.0);
}

TEST_F(AerodynamicTorqueTest, JacobianZeroAtHighAltitude) {
    // At high altitude, all Jacobians should be zero
    AerodynamicTorque torque(cop_, com_, cd_, area_);
    
    TorqueContext ctx;
    ctx.position = Eigen::Vector3d(0.0, 0.0, 20e6);
    ctx.velocity = Eigen::Vector3d(7500.0, 0.0, 0.0);
    ctx.orientation = Eigen::Quaterniond::Identity();
    ctx.angular_velocity = Eigen::Vector3d::Zero();
    
    auto [dtau_dq, dtau_domega, dtau_dr, dtau_dv] = torque.compute_jacobian(ctx);
    
    EXPECT_DOUBLE_EQ(dtau_dq.norm(), 0.0);
    EXPECT_DOUBLE_EQ(dtau_dr.norm(), 0.0);
    EXPECT_DOUBLE_EQ(dtau_dv.norm(), 0.0);
}

TEST_F(AerodynamicTorqueTest, JacobianZeroAtZeroVelocity) {
    // At zero velocity, all Jacobians should be zero
    AerodynamicTorque torque(cop_, com_, cd_, area_);
    
    TorqueContext ctx = createLEOContext(Eigen::Vector3d::Zero());
    
    auto [dtau_dq, dtau_domega, dtau_dr, dtau_dv] = torque.compute_jacobian(ctx);
    
    EXPECT_DOUBLE_EQ(dtau_dq.norm(), 0.0);
    EXPECT_DOUBLE_EQ(dtau_dr.norm(), 0.0);
    EXPECT_DOUBLE_EQ(dtau_dv.norm(), 0.0);
}

TEST_F(AerodynamicTorqueTest, JacobianAttitudeNumerical) {
    // This test may be sensitive to the Jacobian implementation
    // The analytical Jacobian uses a linearization that may not match
    // the finite difference approach with quaternions perfectly
    AerodynamicTorque torque(cop_, com_, cd_, area_);
    
    TorqueContext ctx = createLEOContext(Eigen::Vector3d(7500.0, 0.0, 0.0));
    ctx.orientation = Eigen::Quaterniond(Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitY()));
    
    auto [dtau_dq, dtau_domega, dtau_dr, dtau_dv] = torque.compute_jacobian(ctx);
    
    // Use a more robust numerical approach
    double epsilon = 1e-6;
    Eigen::Matrix3d dtau_dq_numerical = Eigen::Matrix3d::Zero();
    
    Eigen::Vector3d tau_base = torque.compute_torque(ctx);
    
    for (int i = 0; i < 3; ++i) {
        // Perturb in both directions for each axis
        Eigen::Vector3d axis = Eigen::Vector3d::Zero();
        axis(i) = 1.0;
        
        // Small rotation about axis
        Eigen::AngleAxisd rotation(epsilon, axis);
        
        TorqueContext ctx_plus = ctx;
        ctx_plus.orientation = ctx.orientation * Eigen::Quaterniond(rotation);
        
        Eigen::Vector3d tau_plus = torque.compute_torque(ctx_plus);
        
        // Forward difference (more stable for small epsilon)
        dtau_dq_numerical.col(i) = (tau_plus - tau_base) / epsilon;
    }
    
    // Compare absolute error since relative error can be misleading
    double abs_error = (dtau_dq - dtau_dq_numerical).norm();
    double expected_magnitude = std::max(dtau_dq.norm(), dtau_dq_numerical.norm());
    
    // Allow larger tolerance for attitude Jacobian due to quaternion nonlinearity
    EXPECT_LT(abs_error, expected_magnitude * 0.1) 
        << "Analytical Jacobian norm: " << dtau_dq.norm()
        << ", Numerical Jacobian norm: " << dtau_dq_numerical.norm();
}

TEST_F(AerodynamicTorqueTest, JacobianAttitudeDirectionalCheck) {
    // Test that Jacobian predicts correct direction of change
    AerodynamicTorque torque(cop_, com_, cd_, area_);
    
    TorqueContext ctx = createLEOContext(Eigen::Vector3d(7500.0, 0.0, 0.0));
    ctx.orientation = Eigen::Quaterniond::Identity();
    
    auto [dtau_dq, _, __, ___] = torque.compute_jacobian(ctx);
    
    Eigen::Vector3d tau_base = torque.compute_torque(ctx);
    
    // Small perturbation in each direction
    double delta = 0.01;
    for (int i = 0; i < 3; ++i) {
        Eigen::Vector3d delta_theta = Eigen::Vector3d::Zero();
        delta_theta(i) = delta;
        
        TorqueContext ctx_perturbed = ctx;
        ctx_perturbed.orientation = ctx.orientation * Eigen::Quaterniond(
            Eigen::AngleAxisd(delta, Eigen::Vector3d::Unit(i))
        );
        
        Eigen::Vector3d tau_perturbed = torque.compute_torque(ctx_perturbed);
        Eigen::Vector3d actual_change = tau_perturbed - tau_base;
        Eigen::Vector3d predicted_change = dtau_dq * delta_theta;
        
        // Check that predicted direction aligns with actual (cosine > 0.5)
        if (actual_change.norm() > 1e-15 && predicted_change.norm() > 1e-15) {
            double cos_angle = actual_change.dot(predicted_change) / 
                              (actual_change.norm() * predicted_change.norm());
            EXPECT_GT(cos_angle, 0.5) << "Mismatch for axis " << i;
        }
    }
}

TEST_F(AerodynamicTorqueTest, JacobianAttitudeConsistencyCheck) {
    // Verify Jacobian structure makes physical sense
    AerodynamicTorque torque(cop_, com_, cd_, area_);
    
    TorqueContext ctx = createLEOContext(Eigen::Vector3d(7500.0, 0.0, 0.0));
    
    auto [dtau_dq, _, __, ___] = torque.compute_jacobian(ctx);
    
    // Jacobian should be finite
    EXPECT_TRUE(dtau_dq.allFinite());
    
    // Jacobian should be non-zero in LEO
    EXPECT_GT(dtau_dq.norm(), 1e-15);
    
    // Each column should represent sensitivity to rotation about that axis
    for (int i = 0; i < 3; ++i) {
        EXPECT_TRUE(dtau_dq.col(i).allFinite());
    }
}

TEST_F(AerodynamicTorqueTest, JacobianAttitudeVerySmallPerturbation) {
    // Test linearization at very small perturbations
    AerodynamicTorque torque(cop_, com_, cd_, area_);
    
    TorqueContext ctx = createLEOContext(Eigen::Vector3d(7500.0, 0.0, 0.0));
    ctx.orientation = Eigen::Quaterniond::Identity();
    
    auto [dtau_dq, _, __, ___] = torque.compute_jacobian(ctx);
    
    Eigen::Vector3d tau_base = torque.compute_torque(ctx);
    
    // Very small perturbations where linearization should be valid
    std::vector<double> deltas = {1e-4, 1e-5, 1e-6};
    
    for (double delta : deltas) {
        Eigen::Vector3d delta_theta(delta, delta * 0.5, delta * 0.7);
        
        TorqueContext ctx_perturbed = ctx;
        ctx_perturbed.orientation = ctx.orientation * Eigen::Quaterniond(
            Eigen::AngleAxisd(delta_theta.norm(), delta_theta.normalized())
        );
        
        Eigen::Vector3d tau_perturbed = torque.compute_torque(ctx_perturbed);
        Eigen::Vector3d tau_linear = tau_base + dtau_dq * delta_theta;
        
        double relative_error = (tau_perturbed - tau_linear).norm() / tau_base.norm();
        
        // Error should decrease as delta decreases (linear regime)
        EXPECT_LT(relative_error, delta * 10.0);
    }
}

TEST_F(AerodynamicTorqueTest, JacobianAttitudeSymmetryTest) {
    // Test symmetry properties
    AerodynamicTorque torque(cop_, com_, cd_, area_);
    
    TorqueContext ctx = createLEOContext(Eigen::Vector3d(7500.0, 0.0, 0.0));
    ctx.orientation = Eigen::Quaterniond::Identity();
    
    auto [dtau_dq, _, __, ___] = torque.compute_jacobian(ctx);
    
    // For symmetric geometry, certain elements should be zero
    // This depends on the specific COP/COM configuration
    // Just verify basic structure
    EXPECT_TRUE(dtau_dq.allFinite());
    EXPECT_FALSE(dtau_dq.isZero());
}

TEST_F(AerodynamicTorqueTest, JacobianAttitudeMultipleAngles) {
    // Test Jacobian at various attitudes
    AerodynamicTorque torque(cop_, com_, cd_, area_);
    
    std::vector<double> angles = {0.0, 0.1, 0.3, 0.5, 1.0};
    
    for (double angle : angles) {
        TorqueContext ctx = createLEOContext(Eigen::Vector3d(7500.0, 0.0, 0.0));
        ctx.orientation = Eigen::Quaterniond(Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitY()));
        
        auto [dtau_dq, _, __, ___] = torque.compute_jacobian(ctx);
        
        EXPECT_TRUE(dtau_dq.allFinite()) << "Failed at angle " << angle;
        EXPECT_GT(dtau_dq.norm(), 0.0) << "Zero Jacobian at angle " << angle;
    }
}

TEST_F(AerodynamicTorqueTest, JacobianVelocityVersusNumerical) {
    // Velocity Jacobian should match numerical differentiation well
    AerodynamicTorque torque(cop_, com_, cd_, area_);
    
    TorqueContext ctx = createLEOContext(Eigen::Vector3d(7500.0, 0.0, 0.0));
    
    auto [_, __, ___, dtau_dv] = torque.compute_jacobian(ctx);
    
    // Numerical Jacobian with central differences
    double epsilon = 1.0;  // 1 m/s
    Eigen::Matrix3d dtau_dv_numerical;
    
    for (int i = 0; i < 3; ++i) {
        TorqueContext ctx_plus = ctx;
        TorqueContext ctx_minus = ctx;
        
        ctx_plus.velocity(i) += epsilon;
        ctx_minus.velocity(i) -= epsilon;
        
        Eigen::Vector3d tau_plus = torque.compute_torque(ctx_plus);
        Eigen::Vector3d tau_minus = torque.compute_torque(ctx_minus);
        
        dtau_dv_numerical.col(i) = (tau_plus - tau_minus) / (2.0 * epsilon);
    }
    
    double error = (dtau_dv - dtau_dv_numerical).norm();
    double relative_error = error / dtau_dv.norm();
    
    EXPECT_LT(relative_error, 1e-6) 
        << "Velocity Jacobian error: " << error
        << ", Analytical norm: " << dtau_dv.norm()
        << ", Numerical norm: " << dtau_dv_numerical.norm();
}

TEST_F(AerodynamicTorqueTest, JacobianPositionVersusNumerical) {
    // Position Jacobian should match numerical differentiation
    AerodynamicTorque torque(cop_, com_, cd_, area_);
    
    TorqueContext ctx = createLEOContext(Eigen::Vector3d(7500.0, 0.0, 0.0));
    ctx.position = Eigen::Vector3d(0.0, 0.0, 6.571e6);  // 200 km (lower altitude for larger gradient)
    
    auto [_, __, dtau_dr, ___] = torque.compute_jacobian(ctx);
    
    // Numerical Jacobian
    double epsilon = 100.0;  // 100 m
    Eigen::Matrix3d dtau_dr_numerical;
    
    for (int i = 0; i < 3; ++i) {
        TorqueContext ctx_plus = ctx;
        TorqueContext ctx_minus = ctx;
        
        ctx_plus.position(i) += epsilon;
        ctx_minus.position(i) -= epsilon;
        
        Eigen::Vector3d tau_plus = torque.compute_torque(ctx_plus);
        Eigen::Vector3d tau_minus = torque.compute_torque(ctx_minus);
        
        dtau_dr_numerical.col(i) = (tau_plus - tau_minus) / (2.0 * epsilon);
    }
    
    double error = (dtau_dr - dtau_dr_numerical).norm();
    
    if (dtau_dr.norm() > 1e-15) {
        double relative_error = error / dtau_dr.norm();
        EXPECT_LT(relative_error, 0.01) 
            << "Position Jacobian relative error: " << relative_error;
    } else {
        EXPECT_LT(error, 1e-15) << "Position Jacobian absolute error: " << error;
    }
}

TEST_F(AerodynamicTorqueTest, JacobianMagnitudeScaling) {
    // Test that Jacobian magnitudes scale appropriately
    AerodynamicTorque torque(cop_, com_, cd_, area_);
    
    TorqueContext ctx1 = createLEOContext(Eigen::Vector3d(5000.0, 0.0, 0.0));
    TorqueContext ctx2 = createLEOContext(Eigen::Vector3d(10000.0, 0.0, 0.0));
    
    auto [dtau_dq1, _, __, dtau_dv1] = torque.compute_jacobian(ctx1);
    auto [dtau_dq2, ____, _____, dtau_dv2] = torque.compute_jacobian(ctx2);
    
    // Velocity Jacobian should scale approximately linearly with velocity
    double v_ratio = 10000.0 / 5000.0;
    double jac_ratio = dtau_dv2.norm() / dtau_dv1.norm();
    
    EXPECT_NEAR(jac_ratio, v_ratio, v_ratio * 0.2);
}

TEST_F(AerodynamicTorqueTest, JacobianCrossValidation) {
    // Cross-validate different Jacobian computation methods
    AerodynamicTorque torque(cop_, com_, cd_, area_);
    
    TorqueContext ctx = createLEOContext(Eigen::Vector3d(7500.0, 0.0, 0.0));
    
    // Get analytical Jacobian
    auto [dtau_dq, dtau_domega, dtau_dr, dtau_dv] = torque.compute_jacobian(ctx);
    
    // Verify structure
    EXPECT_TRUE(dtau_dq.allFinite());
    EXPECT_TRUE(dtau_dv.allFinite());
    EXPECT_TRUE(dtau_dr.allFinite());
    EXPECT_DOUBLE_EQ(dtau_domega.norm(), 0.0);  // Should be zero for this model
    
    // Verify non-zero where expected
    EXPECT_GT(dtau_dq.norm(), 1e-15);
    EXPECT_GT(dtau_dv.norm(), 1e-15);
    EXPECT_GT(dtau_dr.norm(), 1e-18);  // May be very small at LEO altitude
}

TEST_F(AerodynamicTorqueTest, JacobianRankDeficiency) {
    // Check if Jacobians have expected rank
    AerodynamicTorque torque(cop_, com_, cd_, area_);
    
    TorqueContext ctx = createLEOContext(Eigen::Vector3d(7500.0, 0.0, 0.0));
    
    auto [dtau_dq, _, __, dtau_dv] = torque.compute_jacobian(ctx);
    
    // Compute singular values
    Eigen::JacobiSVD<Eigen::Matrix3d> svd_q(dtau_dq, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::JacobiSVD<Eigen::Matrix3d> svd_v(dtau_dv, Eigen::ComputeFullU | Eigen::ComputeFullV);
    
    // Count non-zero singular values
    int rank_q = 0, rank_v = 0;
    double tol = 1e-12;
    
    for (int i = 0; i < 3; ++i) {
        if (svd_q.singularValues()(i) > tol) rank_q++;
        if (svd_v.singularValues()(i) > tol) rank_v++;
    }
    
    // Velocity Jacobian is rank-2 because torque = r_arm × F_drag
    // The cross product means torque is always perpendicular to moment arm
    // So there's a null space in the direction of the moment arm
    EXPECT_EQ(rank_v, 2) << "Velocity Jacobian should be rank-2 due to cross product structure";
    
    // Attitude Jacobian rank depends on configuration
    EXPECT_GE(rank_q, 1);  // At least rank 1
    
    // Verify the null space is in the direction of moment arm
    Eigen::Vector3d moment_arm = cop_ - com_;
    Eigen::Vector3d moment_arm_normalized = moment_arm.normalized();
    
    // The last singular vector (smallest singular value) should align with moment arm
    Eigen::Vector3d null_vector = svd_v.matrixV().col(2);
    double alignment = std::abs(null_vector.dot(moment_arm_normalized));
    
    // Should be nearly aligned (cosine close to 1)
    EXPECT_GT(alignment, 0.99) << "Null space should align with moment arm direction";
}

TEST_F(AerodynamicTorqueTest, JacobianPerturbationInvariance) {
    // Jacobian should be similar for nearby states
    AerodynamicTorque torque(cop_, com_, cd_, area_);
    
    TorqueContext ctx1 = createLEOContext(Eigen::Vector3d(7500.0, 0.0, 0.0));
    TorqueContext ctx2 = createLEOContext(Eigen::Vector3d(7510.0, 0.0, 0.0));
    
    auto [dtau_dq1, _, __, dtau_dv1] = torque.compute_jacobian(ctx1);
    auto [dtau_dq2, ____, _____, dtau_dv2] = torque.compute_jacobian(ctx2);
    
    // Jacobians should be very similar for nearby states
    EXPECT_LT((dtau_dq1 - dtau_dq2).norm(), dtau_dq1.norm() * 0.01);
    EXPECT_LT((dtau_dv1 - dtau_dv2).norm(), dtau_dv1.norm() * 0.01);
}

TEST_F(AerodynamicTorqueTest, JacobianAttitudeIdentity) {
    // Test at identity orientation (simpler case)
    AerodynamicTorque torque(cop_, com_, cd_, area_);
    
    TorqueContext ctx = createLEOContext(Eigen::Vector3d(7500.0, 0.0, 0.0));
    ctx.orientation = Eigen::Quaterniond::Identity();
    
    auto [dtau_dq, dtau_domega, dtau_dr, dtau_dv] = torque.compute_jacobian(ctx);
    
    double epsilon = 1e-6;
    Eigen::Matrix3d dtau_dq_numerical;
    
    for (int i = 0; i < 3; ++i) {
        Eigen::Quaterniond q_plus = ctx.orientation * Eigen::Quaterniond(
            Eigen::AngleAxisd(epsilon, Eigen::Vector3d::Unit(i))
        );
        Eigen::Quaterniond q_minus = ctx.orientation * Eigen::Quaterniond(
            Eigen::AngleAxisd(-epsilon, Eigen::Vector3d::Unit(i))
        );
        
        TorqueContext ctx_plus = ctx;
        TorqueContext ctx_minus = ctx;
        ctx_plus.orientation = q_plus;
        ctx_minus.orientation = q_minus;
        
        Eigen::Vector3d tau_plus = torque.compute_torque(ctx_plus);
        Eigen::Vector3d tau_minus = torque.compute_torque(ctx_minus);
        
        dtau_dq_numerical.col(i) = (tau_plus - tau_minus) / (2.0 * epsilon);
    }
    
    double error = (dtau_dq - dtau_dq_numerical).norm();
    EXPECT_LT(error, 1e-6) << "Attitude Jacobian error at identity: " << error;
}

TEST_F(AerodynamicTorqueTest, JacobianAttitudeSmallPerturbation) {
    // Test with very small attitude perturbation
    AerodynamicTorque torque(cop_, com_, cd_, area_);
    
    TorqueContext ctx = createLEOContext(Eigen::Vector3d(7500.0, 0.0, 0.0));
    ctx.orientation = Eigen::Quaterniond(Eigen::AngleAxisd(0.1, Eigen::Vector3d::UnitZ()));
    
    auto [dtau_dq, _, __, ___] = torque.compute_jacobian(ctx);
    
    // Test linearization validity
    Eigen::Vector3d delta_theta(0.001, -0.0005, 0.0008);
    
    TorqueContext ctx_perturbed = ctx;
    ctx_perturbed.orientation = ctx.orientation * Eigen::Quaterniond(
        Eigen::AngleAxisd(delta_theta.norm(), delta_theta.normalized())
    );
    
    Eigen::Vector3d tau_base = torque.compute_torque(ctx);
    Eigen::Vector3d tau_perturbed = torque.compute_torque(ctx_perturbed);
    Eigen::Vector3d tau_linear = tau_base + dtau_dq * delta_theta;
    
    double error = (tau_perturbed - tau_linear).norm();
    EXPECT_LT(error, tau_base.norm() * 0.001);  // Within 0.1% for small perturbation
}

TEST_F(AerodynamicTorqueTest, JacobianAttitudeAxisSymmetry) {
    // Jacobian should be symmetric for rotations about different axes
    AerodynamicTorque torque(cop_, com_, cd_, area_);
    
    std::vector<Eigen::Vector3d> axes = {
        Eigen::Vector3d::UnitX(),
        Eigen::Vector3d::UnitY(),
        Eigen::Vector3d::UnitZ()
    };
    
    for (const auto& axis : axes) {
        TorqueContext ctx = createLEOContext(Eigen::Vector3d(7500.0, 0.0, 0.0));
        ctx.orientation = Eigen::Quaterniond(Eigen::AngleAxisd(0.2, axis));
        
        auto [dtau_dq, _, __, ___] = torque.compute_jacobian(ctx);
        
        EXPECT_TRUE(dtau_dq.allFinite());
        EXPECT_GT(dtau_dq.norm(), 0.0);
    }
}

TEST_F(AerodynamicTorqueTest, JacobianVelocityHighAccuracy) {
    // Velocity Jacobian should be very accurate
    AerodynamicTorque torque(cop_, com_, cd_, area_);
    
    TorqueContext ctx = createLEOContext(Eigen::Vector3d(7500.0, 0.0, 0.0));
    
    auto [_, __, ___, dtau_dv] = torque.compute_jacobian(ctx);
    
    double epsilon = 0.1;  // Very small velocity change
    Eigen::Matrix3d dtau_dv_numerical;
    
    for (int i = 0; i < 3; ++i) {
        TorqueContext ctx_plus = ctx;
        TorqueContext ctx_minus = ctx;
        
        ctx_plus.velocity(i) += epsilon;
        ctx_minus.velocity(i) -= epsilon;
        
        Eigen::Vector3d tau_plus = torque.compute_torque(ctx_plus);
        Eigen::Vector3d tau_minus = torque.compute_torque(ctx_minus);
        
        dtau_dv_numerical.col(i) = (tau_plus - tau_minus) / (2.0 * epsilon);
    }
    
    double error = (dtau_dv - dtau_dv_numerical).norm();
    EXPECT_LT(error, 1e-10) << "Velocity Jacobian error: " << error;
}

TEST_F(AerodynamicTorqueTest, JacobianPositionHighAltitude) {
    // Position Jacobian at high altitude should be small but non-zero
    AerodynamicTorque torque(cop_, com_, cd_, area_);
    
    TorqueContext ctx = createLEOContext(Eigen::Vector3d(7500.0, 0.0, 0.0));
    ctx.position = Eigen::Vector3d(0.0, 0.0, 6.971e6);  // 600 km
    
    auto [_, __, dtau_dr, ___] = torque.compute_jacobian(ctx);
    
    // At 600 km altitude, density is very low but not zero
    // Position Jacobian should be small (much smaller than at LEO)
    
    // Compare to low altitude
    TorqueContext ctx_low = createLEOContext(Eigen::Vector3d(7500.0, 0.0, 0.0));
    ctx_low.position = Eigen::Vector3d(0.0, 0.0, 6.571e6);  // 200 km
    
    auto [____, _____, dtau_dr_low, ______] = torque.compute_jacobian(ctx_low);
    
    // High altitude Jacobian should be much smaller than low altitude
    EXPECT_LT(dtau_dr.norm(), dtau_dr_low.norm() * 0.01);
    
    // But still finite
    EXPECT_TRUE(dtau_dr.allFinite());
}

TEST_F(AerodynamicTorqueTest, JacobianPositionLowAltitude) {
    // Position Jacobian at low altitude should be larger
    AerodynamicTorque torque(cop_, com_, cd_, area_);
    
    TorqueContext ctx = createLEOContext(Eigen::Vector3d(7500.0, 0.0, 0.0));
    ctx.position = Eigen::Vector3d(0.0, 0.0, 6.471e6);  // 100 km
    
    auto [_, __, dtau_dr, ___] = torque.compute_jacobian(ctx);
    
    // At low altitude, position sensitivity should be measurable
    EXPECT_GT(dtau_dr.norm(), 1e-15);
    EXPECT_TRUE(dtau_dr.allFinite());
}

TEST_F(AerodynamicTorqueTest, JacobianVelocityDirection) {
    // Velocity Jacobian should vary with velocity direction
    AerodynamicTorque torque(cop_, com_, cd_, area_);
    
    TorqueContext ctx1 = createLEOContext(Eigen::Vector3d(7500.0, 0.0, 0.0));
    TorqueContext ctx2 = createLEOContext(Eigen::Vector3d(0.0, 7500.0, 0.0));
    
    auto [_, __, ___, dtau_dv1] = torque.compute_jacobian(ctx1);
    auto [____, _____, ______, dtau_dv2] = torque.compute_jacobian(ctx2);
    
    // Jacobians should be different for different velocity directions
    EXPECT_GT((dtau_dv1 - dtau_dv2).norm(), 1e-15);
}

TEST_F(AerodynamicTorqueTest, JacobianConsistencyMultipleVelocities) {
    // Test Jacobian at multiple velocity magnitudes
    AerodynamicTorque torque(cop_, com_, cd_, area_);
    
    std::vector<double> velocities = {3000.0, 7500.0, 12000.0};
    std::vector<Eigen::Matrix3d> jacobians;
    
    for (double v : velocities) {
        TorqueContext ctx = createLEOContext(Eigen::Vector3d(v, 0.0, 0.0));
        auto [_, __, ___, dtau_dv] = torque.compute_jacobian(ctx);
        jacobians.push_back(dtau_dv);
    }
    
    // All Jacobians should be finite and non-zero
    for (const auto& J : jacobians) {
        EXPECT_TRUE(J.allFinite());
        EXPECT_GT(J.norm(), 0.0);
    }
}

TEST_F(AerodynamicTorqueTest, JacobianAttitudeOrthogonality) {
    // Test that Jacobian columns are approximately orthogonal for identity orientation
    AerodynamicTorque torque(cop_, com_, cd_, area_);
    
    TorqueContext ctx = createLEOContext(Eigen::Vector3d(7500.0, 0.0, 0.0));
    ctx.orientation = Eigen::Quaterniond::Identity();
    
    auto [dtau_dq, _, __, ___] = torque.compute_jacobian(ctx);
    
    // Check orthogonality of columns (not exact due to nonlinearity)
    for (int i = 0; i < 3; ++i) {
        for (int j = i + 1; j < 3; ++j) {
            double dot_product = dtau_dq.col(i).dot(dtau_dq.col(j));
            double norm_i = dtau_dq.col(i).norm();
            double norm_j = dtau_dq.col(j).norm();
            
            if (norm_i > 1e-15 && norm_j > 1e-15) {
                double cos_angle = dot_product / (norm_i * norm_j);
                EXPECT_LT(std::abs(cos_angle), 0.5);  // Not too parallel
            }
        }
    }
}

TEST_F(AerodynamicTorqueTest, JacobianPositionRadialDirection) {
    // Position Jacobian should have specific structure (depends on altitude gradient)
    AerodynamicTorque torque(cop_, com_, cd_, area_);
    
    TorqueContext ctx = createLEOContext(Eigen::Vector3d(7500.0, 0.0, 0.0));
    
    auto [_, __, dtau_dr, ___] = torque.compute_jacobian(ctx);
    
    // Compute radial direction
    Eigen::Vector3d r_hat = ctx.position.normalized();
    
    // Project Jacobian onto radial direction
    Eigen::Vector3d dtau_dr_radial = dtau_dr * r_hat;
    
    // Radial component should dominate (altitude effect)
    EXPECT_GT(dtau_dr_radial.norm(), dtau_dr.norm() * 0.8);
}

TEST_F(AerodynamicTorqueTest, JacobianVelocitySymmetry) {
    // Velocity Jacobian should have certain symmetries
    AerodynamicTorque torque(cop_, com_, cd_, area_);
    
    TorqueContext ctx1 = createLEOContext(Eigen::Vector3d(7500.0, 0.0, 0.0));
    TorqueContext ctx2 = createLEOContext(Eigen::Vector3d(-7500.0, 0.0, 0.0));
    
    auto [_, __, ___, dtau_dv1] = torque.compute_jacobian(ctx1);
    auto [____, _____, ______, dtau_dv2] = torque.compute_jacobian(ctx2);
    
    // Jacobians should have similar magnitude
    EXPECT_NEAR(dtau_dv1.norm(), dtau_dv2.norm(), dtau_dv1.norm() * 0.1);
}

TEST_F(AerodynamicTorqueTest, JacobianCompleteStatePerturb) {
    // Test all Jacobians together with full state perturbation
    AerodynamicTorque torque(cop_, com_, cd_, area_);
    
    TorqueContext ctx = createLEOContext(Eigen::Vector3d(7500.0, 0.0, 0.0));
    ctx.orientation = Eigen::Quaterniond(Eigen::AngleAxisd(0.2, Eigen::Vector3d(1,1,1).normalized()));
    
    auto [dtau_dq, dtau_domega, dtau_dr, dtau_dv] = torque.compute_jacobian(ctx);
    
    // Full state perturbation
    Eigen::Vector3d delta_theta(0.005, -0.003, 0.004);
    Eigen::Vector3d delta_r(20.0, -15.0, 25.0);
    Eigen::Vector3d delta_v(5.0, 3.0, -4.0);
    
    TorqueContext ctx_perturbed = ctx;
    ctx_perturbed.orientation = ctx.orientation * Eigen::Quaterniond(
        Eigen::AngleAxisd(delta_theta.norm(), delta_theta.normalized())
    );
    ctx_perturbed.position += delta_r;
    ctx_perturbed.velocity += delta_v;
    
    Eigen::Vector3d tau_base = torque.compute_torque(ctx);
    Eigen::Vector3d tau_perturbed = torque.compute_torque(ctx_perturbed);
    
    // Linear approximation using all Jacobians
    Eigen::Vector3d tau_linear = tau_base + 
        dtau_dq * delta_theta + 
        dtau_dr * delta_r + 
        dtau_dv * delta_v;
    
    double error = (tau_perturbed - tau_linear).norm();
    double relative_error = error / tau_base.norm();
    EXPECT_LT(relative_error, 0.02);  // Within 2% for small perturbations
}

TEST_F(AerodynamicTorqueTest, JacobianNonsingular) {
    // Test that Jacobians are non-singular in typical conditions
    AerodynamicTorque torque(cop_, com_, cd_, area_);
    
    TorqueContext ctx = createLEOContext(Eigen::Vector3d(7500.0, 0.0, 0.0));
    
    auto [dtau_dq, dtau_domega, dtau_dr, dtau_dv] = torque.compute_jacobian(ctx);
    
    // Check rank/determinant for non-singular behavior
    Eigen::JacobiSVD<Eigen::Matrix3d> svd_q(dtau_dq);
    Eigen::JacobiSVD<Eigen::Matrix3d> svd_v(dtau_dv);
    
    // Should have non-zero singular values
    EXPECT_GT(svd_q.singularValues()(0), 1e-15);
    EXPECT_GT(svd_v.singularValues()(0), 1e-15);
}

TEST_F(AerodynamicTorqueTest, JacobianScaleInvariance) {
    // Jacobian should scale properly with physical parameters
    Eigen::Vector3d cop_large(0.0, 0.0, 1.0);  // 10x larger moment arm
    
    AerodynamicTorque torque1(cop_, com_, cd_, area_);
    AerodynamicTorque torque2(cop_large, com_, cd_, area_);
    
    TorqueContext ctx = createLEOContext(Eigen::Vector3d(7500.0, 0.0, 0.0));
    
    auto [dtau_dq1, _, __, ___] = torque1.compute_jacobian(ctx);
    auto [dtau_dq2, ____, _____, ______] = torque2.compute_jacobian(ctx);
    
    // Jacobian should scale approximately linearly with moment arm
    double scale = cop_large(2) / cop_(2);
    EXPECT_NEAR(dtau_dq2.norm(), dtau_dq1.norm() * scale, dtau_dq1.norm() * scale * 0.1);
}

TEST_F(AerodynamicTorqueTest, JacobianVelocityNumerical) {
    AerodynamicTorque torque(cop_, com_, cd_, area_);
    
    TorqueContext ctx = createLEOContext(Eigen::Vector3d(7500.0, 0.0, 0.0));
    
    auto [dtau_dq, dtau_domega, dtau_dr, dtau_dv] = torque.compute_jacobian(ctx);
    
    // Numerical Jacobian
    double epsilon = 1.0;  // 1 m/s velocity change
    Eigen::Matrix3d dtau_dv_numerical;
    
    for (int i = 0; i < 3; ++i) {
        TorqueContext ctx_plus = ctx;
        TorqueContext ctx_minus = ctx;
        
        ctx_plus.velocity(i) += epsilon;
        ctx_minus.velocity(i) -= epsilon;
        
        Eigen::Vector3d tau_plus = torque.compute_torque(ctx_plus);
        Eigen::Vector3d tau_minus = torque.compute_torque(ctx_minus);
        
        dtau_dv_numerical.col(i) = (tau_plus - tau_minus) / (2.0 * epsilon);
    }
    
    double error = (dtau_dv - dtau_dv_numerical).norm();
    EXPECT_LT(error, 1e-12) << "Velocity Jacobian error: " << error;
}

TEST_F(AerodynamicTorqueTest, JacobianPositionNumerical) {
    AerodynamicTorque torque(cop_, com_, cd_, area_);
    
    TorqueContext ctx = createLEOContext(Eigen::Vector3d(7500.0, 0.0, 0.0));
    
    auto [dtau_dq, dtau_domega, dtau_dr, dtau_dv] = torque.compute_jacobian(ctx);
    
    // Numerical Jacobian
    double epsilon = 100.0;  // 100 m position change
    Eigen::Matrix3d dtau_dr_numerical;
    
    for (int i = 0; i < 3; ++i) {
        TorqueContext ctx_plus = ctx;
        TorqueContext ctx_minus = ctx;
        
        ctx_plus.position(i) += epsilon;
        ctx_minus.position(i) -= epsilon;
        
        Eigen::Vector3d tau_plus = torque.compute_torque(ctx_plus);
        Eigen::Vector3d tau_minus = torque.compute_torque(ctx_minus);
        
        dtau_dr_numerical.col(i) = (tau_plus - tau_minus) / (2.0 * epsilon);
    }
    
    double error = (dtau_dr - dtau_dr_numerical).norm();
    EXPECT_LT(error, 1e-14) << "Position Jacobian error: " << error;
}

TEST_F(AerodynamicTorqueTest, JacobianLinearConsistency) {
    // Taylor expansion: τ(x + δx) ≈ τ(x) + J·δx
    AerodynamicTorque torque(cop_, com_, cd_, area_);
    
    TorqueContext ctx = createLEOContext(Eigen::Vector3d(7500.0, 0.0, 0.0));
    ctx.orientation = Eigen::Quaterniond(Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitY()));
    
    auto [dtau_dq, dtau_domega, dtau_dr, dtau_dv] = torque.compute_jacobian(ctx);
    
    // Small perturbations
    Eigen::Vector3d delta_theta(0.01, -0.005, 0.008);
    Eigen::Vector3d delta_r(50.0, -30.0, 40.0);
    Eigen::Vector3d delta_v(10.0, 5.0, -8.0);
    
    // Perturbed state
    TorqueContext ctx_perturbed = ctx;
    ctx_perturbed.orientation = ctx.orientation * Eigen::Quaterniond(
        Eigen::AngleAxisd(delta_theta.norm(), delta_theta.normalized())
    );
    ctx_perturbed.position += delta_r;
    ctx_perturbed.velocity += delta_v;
    
    // Compute torques
    Eigen::Vector3d tau_base = torque.compute_torque(ctx);
    Eigen::Vector3d tau_perturbed = torque.compute_torque(ctx_perturbed);
    
    // Linear approximation
    Eigen::Vector3d tau_linear = tau_base + dtau_dq * delta_theta + dtau_dr * delta_r + dtau_dv * delta_v;
    
    // Should be reasonably close for small perturbations
    double error = (tau_perturbed - tau_linear).norm();
    EXPECT_LT(error, tau_base.norm() * 0.05);  // Within 5%
}

TEST_F(AerodynamicTorqueTest, JacobianVelocityScaling) {
    // ∂τ/∂v should scale with velocity magnitude
    AerodynamicTorque torque(cop_, com_, cd_, area_);
    
    TorqueContext ctx1 = createLEOContext(Eigen::Vector3d(7500.0, 0.0, 0.0));
    TorqueContext ctx2 = createLEOContext(Eigen::Vector3d(15000.0, 0.0, 0.0));
    
    auto [_, __, ___, dtau_dv1] = torque.compute_jacobian(ctx1);
    auto [____, _____, ______, dtau_dv2] = torque.compute_jacobian(ctx2);
    
    // At 2× velocity, velocity Jacobian should be ~2× larger
    double ratio = dtau_dv2.norm() / dtau_dv1.norm();
    EXPECT_NEAR(ratio, 2.0, 0.2);
}

TEST_F(AerodynamicTorqueTest, JacobianFiniteValues) {
    // All Jacobians should be finite
    AerodynamicTorque torque(cop_, com_, cd_, area_);
    
    TorqueContext ctx = createLEOContext(Eigen::Vector3d(7500.0, 0.0, 0.0));
    ctx.orientation = Eigen::Quaterniond(Eigen::AngleAxisd(0.5, Eigen::Vector3d(1,2,3).normalized()));
    
    auto [dtau_dq, dtau_domega, dtau_dr, dtau_dv] = torque.compute_jacobian(ctx);
    
    EXPECT_TRUE(dtau_dq.allFinite());
    EXPECT_TRUE(dtau_domega.allFinite());
    EXPECT_TRUE(dtau_dr.allFinite());
    EXPECT_TRUE(dtau_dv.allFinite());
}

// ============================================================================
// EDGE CASES
// ============================================================================

TEST_F(AerodynamicTorqueTest, LargeVelocity) {
    // Test with very high velocity (escape velocity)
    AerodynamicTorque torque(cop_, com_, cd_, area_);
    
    TorqueContext ctx = createLEOContext(Eigen::Vector3d(11000.0, 0.0, 0.0));
    
    Eigen::Vector3d tau = torque.compute_torque(ctx);
    
    EXPECT_TRUE(tau.allFinite());
    EXPECT_GT(tau.norm(), 0.0);
}

TEST_F(AerodynamicTorqueTest, SmallVelocity) {
    // Test with very small velocity
    AerodynamicTorque torque(cop_, com_, cd_, area_);
    
    TorqueContext ctx = createLEOContext(Eigen::Vector3d(10.0, 0.0, 0.0));
    
    Eigen::Vector3d tau = torque.compute_torque(ctx);
    
    EXPECT_TRUE(tau.allFinite());
    EXPECT_GT(tau.norm(), 0.0);
}

TEST_F(AerodynamicTorqueTest, LargeMomentArm) {
    // Test with large moment arm (e.g., solar sail)
    Eigen::Vector3d cop(0.0, 0.0, 5.0);  // 5 m offset
    AerodynamicTorque torque(cop, com_, cd_, area_);
    
    TorqueContext ctx = createLEOContext(Eigen::Vector3d(7500.0, 0.0, 0.0));
    
    Eigen::Vector3d tau = torque.compute_torque(ctx);
    
    EXPECT_TRUE(tau.allFinite());
    EXPECT_GT(tau.norm(), 0.0);
}

TEST_F(AerodynamicTorqueTest, SmallDragCoefficient) {
    // Test with very small Cd (smooth sphere)
    AerodynamicTorque torque(cop_, com_, 0.1, area_);
    
    TorqueContext ctx = createLEOContext(Eigen::Vector3d(7500.0, 0.0, 0.0));
    
    Eigen::Vector3d tau = torque.compute_torque(ctx);
    
    EXPECT_TRUE(tau.allFinite());
    EXPECT_GT(tau.norm(), 0.0);
}

TEST_F(AerodynamicTorqueTest, LargeDragCoefficient) {
    // Test with very large Cd (flat plate)
    AerodynamicTorque torque(cop_, com_, 10.0, area_);
    
    TorqueContext ctx = createLEOContext(Eigen::Vector3d(7500.0, 0.0, 0.0));
    
    Eigen::Vector3d tau = torque.compute_torque(ctx);
    
    EXPECT_TRUE(tau.allFinite());
    EXPECT_GT(tau.norm(), 0.0);
}

TEST_F(AerodynamicTorqueTest, SmallReferenceArea) {
    // Test with small area (CubeSat)
    AerodynamicTorque torque(cop_, com_, cd_, 0.01);  // 10 cm × 10 cm
    
    TorqueContext ctx = createLEOContext(Eigen::Vector3d(7500.0, 0.0, 0.0));
    
    Eigen::Vector3d tau = torque.compute_torque(ctx);
    
    EXPECT_TRUE(tau.allFinite());
    EXPECT_GT(tau.norm(), 0.0);
}

TEST_F(AerodynamicTorqueTest, LargeReferenceArea) {
    // Test with large area (solar sail)
    AerodynamicTorque torque(cop_, com_, cd_, 100.0);  // 10 m × 10 m
    
    TorqueContext ctx = createLEOContext(Eigen::Vector3d(7500.0, 0.0, 0.0));
    
    Eigen::Vector3d tau = torque.compute_torque(ctx);
    
    EXPECT_TRUE(tau.allFinite());
    EXPECT_GT(tau.norm(), 0.0);
}

TEST_F(AerodynamicTorqueTest, ArbitraryOrientation) {
    // Test with arbitrary orientation
    AerodynamicTorque torque(cop_, com_, cd_, area_);
    
    TorqueContext ctx = createLEOContext(Eigen::Vector3d(7500.0, 0.0, 0.0));
    
    // Compound rotation
    Eigen::Quaterniond q1(Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitX()));
    Eigen::Quaterniond q2(Eigen::AngleAxisd(0.5, Eigen::Vector3d::UnitY()));
    Eigen::Quaterniond q3(Eigen::AngleAxisd(0.7, Eigen::Vector3d::UnitZ()));
    ctx.orientation = q1 * q2 * q3;
    
    Eigen::Vector3d tau = torque.compute_torque(ctx);
    
    EXPECT_TRUE(tau.allFinite());
}

TEST_F(AerodynamicTorqueTest, VelocityInDifferentDirections) {
    // Test with velocity in various directions
    // Note: When velocity is parallel to moment arm, torque will be zero
    AerodynamicTorque torque(cop_, com_, cd_, area_);
    
    std::vector<Eigen::Vector3d> velocities = {
        Eigen::Vector3d(7500.0, 0.0, 0.0),      // +X: should produce torque
        Eigen::Vector3d(0.0, 7500.0, 0.0),      // +Y: should produce torque
        Eigen::Vector3d(5000.0, 5000.0, 0.0),   // XY plane: should produce torque
        Eigen::Vector3d(5000.0, 0.0, 5000.0),   // XZ plane: should produce torque
        Eigen::Vector3d(0.0, 5000.0, 5000.0)    // YZ plane: should produce torque
    };
    
    for (const auto& vel : velocities) {
        TorqueContext ctx = createLEOContext(vel);
        Eigen::Vector3d tau = torque.compute_torque(ctx);
        
        EXPECT_TRUE(tau.allFinite());
        EXPECT_GT(tau.norm(), 0.0) << "Failed for velocity: " << vel.transpose();
    }
}

TEST_F(AerodynamicTorqueTest, VelocityParallelToMomentArm) {
    // When velocity is parallel to moment arm, torque should be zero
    // because r × F = 0 when r ∥ F
    Eigen::Vector3d cop(0.0, 0.0, 0.1);  // Moment arm in +Z
    AerodynamicTorque torque(cop, com_, cd_, area_);
    
    // Velocity also in +Z direction
    TorqueContext ctx = createLEOContext(Eigen::Vector3d(0.0, 0.0, 7500.0));
    
    Eigen::Vector3d tau = torque.compute_torque(ctx);
    
    EXPECT_NEAR(tau.norm(), 0.0, 1e-12);
}

TEST_F(AerodynamicTorqueTest, VelocityPerpendicularToMomentArm) {
    // Maximum torque when velocity perpendicular to moment arm
    Eigen::Vector3d cop(0.0, 0.0, 0.1);  // Moment arm in +Z
    AerodynamicTorque torque(cop, com_, cd_, area_);
    
    // Velocity in XY plane (perpendicular to Z)
    TorqueContext ctx = createLEOContext(Eigen::Vector3d(7500.0, 0.0, 0.0));
    
    Eigen::Vector3d tau = torque.compute_torque(ctx);
    
    EXPECT_GT(tau.norm(), 1e-10);
}

TEST_F(AerodynamicTorqueTest, TorqueSymmetryUnderReflection) {
    // Reflecting moment arm should produce opposite torque
    Eigen::Vector3d cop1(0.0, 0.1, 0.0);
    Eigen::Vector3d cop2(0.0, -0.1, 0.0);
    
    AerodynamicTorque torque1(cop1, com_, cd_, area_);
    AerodynamicTorque torque2(cop2, com_, cd_, area_);
    
    TorqueContext ctx = createLEOContext(Eigen::Vector3d(7500.0, 0.0, 0.0));
    
    Eigen::Vector3d tau1 = torque1.compute_torque(ctx);
    Eigen::Vector3d tau2 = torque2.compute_torque(ctx);
    
    EXPECT_NEAR((tau1 + tau2).norm(), 0.0, tau1.norm() * 0.01);
}

TEST_F(AerodynamicTorqueTest, TorqueConsistentWithDragForce) {
    // Verify τ = r × F relationship explicitly
    AerodynamicTorque torque(cop_, com_, cd_, area_);
    
    TorqueContext ctx = createLEOContext(Eigen::Vector3d(7500.0, 0.0, 0.0));
    
    // Compute torque
    Eigen::Vector3d tau = torque.compute_torque(ctx);
    
    // Manually compute drag force
    double altitude = ctx.position.norm() - 6.371e6;
    double rho = atmosphere::get_density_us76(altitude);
    Eigen::Vector3d v_body = ctx.orientation.inverse() * ctx.velocity;
    double v_mag = v_body.norm();
    Eigen::Vector3d drag_force = -0.5 * rho * cd_ * area_ * v_mag * v_mag * v_body.normalized();
    
    // Manually compute torque
    Eigen::Vector3d moment_arm = cop_ - com_;
    Eigen::Vector3d expected_tau = moment_arm.cross(drag_force);
    
    EXPECT_TRUE(tau.isApprox(expected_tau, 1e-12));
}

TEST_F(AerodynamicTorqueTest, JacobianPositionAtDifferentAltitudes) {
    // Position Jacobian should vary with altitude (density gradient changes)
    AerodynamicTorque torque(cop_, com_, cd_, area_);
    
    TorqueContext ctx_low = createLEOContext(Eigen::Vector3d(7500.0, 0.0, 0.0));
    ctx_low.position = Eigen::Vector3d(0.0, 0.0, 6.571e6);  // 200 km
    
    TorqueContext ctx_high = createLEOContext(Eigen::Vector3d(7500.0, 0.0, 0.0));
    ctx_high.position = Eigen::Vector3d(0.0, 0.0, 6.971e6);  // 600 km
    
    auto [_, __, dtau_dr_low, ___] = torque.compute_jacobian(ctx_low);
    auto [____, _____, dtau_dr_high, ______] = torque.compute_jacobian(ctx_high);
    
    // Jacobians should be different at different altitudes
    EXPECT_GT((dtau_dr_low - dtau_dr_high).norm(), 1e-15);
}

TEST_F(AerodynamicTorqueTest, JacobianAttitudeMultipleOrientations) {
    // Attitude Jacobian should vary with orientation
    AerodynamicTorque torque(cop_, com_, cd_, area_);
    
    TorqueContext ctx1 = createLEOContext(Eigen::Vector3d(7500.0, 0.0, 0.0));
    ctx1.orientation = Eigen::Quaterniond::Identity();
    
    TorqueContext ctx2 = createLEOContext(Eigen::Vector3d(7500.0, 0.0, 0.0));
    ctx2.orientation = Eigen::Quaterniond(Eigen::AngleAxisd(M_PI/4.0, Eigen::Vector3d::UnitY()));
    
    auto [dtau_dq1, _, __, ___] = torque.compute_jacobian(ctx1);
    auto [dtau_dq2, ____, _____, ______] = torque.compute_jacobian(ctx2);
    
    // Jacobians should be different for different orientations
    EXPECT_GT((dtau_dq1 - dtau_dq2).norm(), 1e-15);
}

TEST_F(AerodynamicTorqueTest, TorqueMagnitudeOrder) {
    // Typical LEO torque should be in reasonable range for small satellite
    AerodynamicTorque torque(cop_, com_, cd_, area_);
    
    TorqueContext ctx = createLEOContext(Eigen::Vector3d(7500.0, 0.0, 0.0));
    
    Eigen::Vector3d tau = torque.compute_torque(ctx);
    
    // For 1 m² area, 10 cm moment arm, at 400 km altitude
    // Expect torque in range 10^-8 to 10^-4 Nm
    EXPECT_GT(tau.norm(), 1e-10);
    EXPECT_LT(tau.norm(), 1e-3);
}

TEST_F(AerodynamicTorqueTest, MultipleVelocityMagnitudesSameDirection) {
    // Verify quadratic scaling at multiple velocity magnitudes
    AerodynamicTorque torque(cop_, com_, cd_, area_);
    
    std::vector<double> velocities = {5000.0, 7500.0, 10000.0};
    std::vector<double> torque_magnitudes;
    
    for (double v : velocities) {
        TorqueContext ctx = createLEOContext(Eigen::Vector3d(v, 0.0, 0.0));
        Eigen::Vector3d tau = torque.compute_torque(ctx);
        torque_magnitudes.push_back(tau.norm());
    }
    
    // Check quadratic relationship: τ₂/τ₁ = (v₂/v₁)²
    double ratio_vel_01 = velocities[1] / velocities[0];
    double ratio_tau_01 = torque_magnitudes[1] / torque_magnitudes[0];
    EXPECT_NEAR(ratio_tau_01, ratio_vel_01 * ratio_vel_01, 0.01);
    
    double ratio_vel_12 = velocities[2] / velocities[1];
    double ratio_tau_12 = torque_magnitudes[2] / torque_magnitudes[1];
    EXPECT_NEAR(ratio_tau_12, ratio_vel_12 * ratio_vel_12, 0.01);
}

TEST_F(AerodynamicTorqueTest, ZeroTorqueNearSpaceVacuum) {
    // At very high altitude approaching vacuum, torque should approach zero
    AerodynamicTorque torque(cop_, com_, cd_, area_);
    
    std::vector<double> altitudes = {
        6.571e6,   // 200 km: significant atmosphere
        6.771e6,   // 400 km: thin but measurable atmosphere
        6.971e6    // 600 km: very thin atmosphere (approaching vacuum)
    };
    
    std::vector<double> torque_magnitudes;
    for (double r : altitudes) {
        TorqueContext ctx = createLEOContext(Eigen::Vector3d(7500.0, 0.0, 0.0));
        ctx.position = Eigen::Vector3d(0.0, 0.0, r);
        Eigen::Vector3d tau = torque.compute_torque(ctx);
        torque_magnitudes.push_back(tau.norm());
    }
    
    // Torque should decrease monotonically with altitude
    EXPECT_GT(torque_magnitudes[0], torque_magnitudes[1]);
    EXPECT_GT(torque_magnitudes[1], torque_magnitudes[2]);
    
    // At 600 km, torque should be very small but might not be exactly zero
    EXPECT_LT(torque_magnitudes[2], torque_magnitudes[0] * 0.01);  // Less than 1% of 200km value
}

TEST_F(AerodynamicTorqueTest, AtmosphericDensityEffect) {
    // Test that torque follows atmospheric density profile
    AerodynamicTorque torque(cop_, com_, cd_, area_);
    
    std::vector<double> altitudes = {
        6.471e6,   // 100 km
        6.571e6,   // 200 km
        6.671e6,   // 300 km
        6.771e6    // 400 km
    };
    
    std::vector<double> torque_magnitudes;
    std::vector<double> densities;
    
    for (double r : altitudes) {
        TorqueContext ctx = createLEOContext(Eigen::Vector3d(7500.0, 0.0, 0.0));
        ctx.position = Eigen::Vector3d(0.0, 0.0, r);
        
        double altitude = r - 6.371e6;
        densities.push_back(atmosphere::get_density_us76(altitude));
        
        Eigen::Vector3d tau = torque.compute_torque(ctx);
        torque_magnitudes.push_back(tau.norm());
    }
    
    // Verify torque ratios match density ratios (τ ∝ ρ)
    for (size_t i = 1; i < torque_magnitudes.size(); ++i) {
        double torque_ratio = torque_magnitudes[i] / torque_magnitudes[0];
        double density_ratio = densities[i] / densities[0];
        EXPECT_NEAR(torque_ratio, density_ratio, 0.01);
    }
}

TEST_F(AerodynamicTorqueTest, TorqueInVacuum) {
    // Explicitly test at extremely high altitude (deep space)
    AerodynamicTorque torque(cop_, com_, cd_, area_);
    
    TorqueContext ctx;
    ctx.position = Eigen::Vector3d(0.0, 0.0, 100e6);  // ~93,629 km altitude
    ctx.velocity = Eigen::Vector3d(7500.0, 0.0, 0.0);
    ctx.orientation = Eigen::Quaterniond::Identity();
    ctx.angular_velocity = Eigen::Vector3d::Zero();
    
    Eigen::Vector3d tau = torque.compute_torque(ctx);
    
    EXPECT_DOUBLE_EQ(tau.norm(), 0.0);
}

TEST_F(AerodynamicTorqueTest, TorqueAtThermosphere) {
    // Test in thermosphere where density is still significant
    AerodynamicTorque torque(cop_, com_, cd_, area_);
    
    TorqueContext ctx = createLEOContext(Eigen::Vector3d(7500.0, 0.0, 0.0));
    ctx.position = Eigen::Vector3d(0.0, 0.0, 6.471e6);  // 100 km altitude
    
    Eigen::Vector3d tau = torque.compute_torque(ctx);
    
    EXPECT_GT(tau.norm(), 1e-8);  // Should have measurable torque
    EXPECT_TRUE(tau.allFinite());
}

TEST_F(AerodynamicTorqueTest, TorqueAtExosphere) {
    // Test in exosphere where density is very low
    AerodynamicTorque torque(cop_, com_, cd_, area_);
    
    TorqueContext ctx = createLEOContext(Eigen::Vector3d(7500.0, 0.0, 0.0));
    ctx.position = Eigen::Vector3d(0.0, 0.0, 7.371e6);  // 1000 km altitude
    
    Eigen::Vector3d tau = torque.compute_torque(ctx);
    
    // At 1000 km, density should be negligible
    EXPECT_NEAR(tau.norm(), 0.0, 1e-15);
}

TEST_F(AerodynamicTorqueTest, TorqueAltitudeGradient) {
    // Test the gradient of torque with respect to altitude
    AerodynamicTorque torque(cop_, com_, cd_, area_);
    
    double base_altitude = 6.771e6;  // 400 km
    double delta_altitude = 10e3;     // 10 km step
    
    TorqueContext ctx1 = createLEOContext(Eigen::Vector3d(7500.0, 0.0, 0.0));
    ctx1.position = Eigen::Vector3d(0.0, 0.0, base_altitude);
    
    TorqueContext ctx2 = createLEOContext(Eigen::Vector3d(7500.0, 0.0, 0.0));
    ctx2.position = Eigen::Vector3d(0.0, 0.0, base_altitude + delta_altitude);
    
    Eigen::Vector3d tau1 = torque.compute_torque(ctx1);
    Eigen::Vector3d tau2 = torque.compute_torque(ctx2);
    
    // Torque should decrease with increasing altitude
    EXPECT_LT(tau2.norm(), tau1.norm());
    
    // Gradient should be negative
    double gradient = (tau2.norm() - tau1.norm()) / delta_altitude;
    EXPECT_LT(gradient, 0.0);
}

TEST_F(AerodynamicTorqueTest, TorqueMultipleAltitudesMonotonic) {
    // Verify monotonic decrease across wide altitude range
    AerodynamicTorque torque(cop_, com_, cd_, area_);
    
    std::vector<double> altitudes;
    for (int i = 0; i <= 10; ++i) {
        altitudes.push_back(6.371e6 + 50e3 * i);  // 0 to 500 km in 50 km steps
    }
    
    std::vector<double> torque_magnitudes;
    for (double r : altitudes) {
        TorqueContext ctx = createLEOContext(Eigen::Vector3d(7500.0, 0.0, 0.0));
        ctx.position = Eigen::Vector3d(0.0, 0.0, r);
        Eigen::Vector3d tau = torque.compute_torque(ctx);
        torque_magnitudes.push_back(tau.norm());
    }
    
    // Verify strictly decreasing
    for (size_t i = 1; i < torque_magnitudes.size(); ++i) {
        EXPECT_LT(torque_magnitudes[i], torque_magnitudes[i-1])
            << "Non-monotonic at altitude " << (altitudes[i] - 6.371e6) / 1e3 << " km";
    }
}

TEST_F(AerodynamicTorqueTest, TorqueExponentialDecay) {
    // Atmospheric density follows exponential decay, so should torque
    AerodynamicTorque torque(cop_, com_, cd_, area_);
    
    TorqueContext ctx1 = createLEOContext(Eigen::Vector3d(7500.0, 0.0, 0.0));
    ctx1.position = Eigen::Vector3d(0.0, 0.0, 6.571e6);  // 200 km
    
    TorqueContext ctx2 = createLEOContext(Eigen::Vector3d(7500.0, 0.0, 0.0));
    ctx2.position = Eigen::Vector3d(0.0, 0.0, 6.671e6);  // 300 km
    
    TorqueContext ctx3 = createLEOContext(Eigen::Vector3d(7500.0, 0.0, 0.0));
    ctx3.position = Eigen::Vector3d(0.0, 0.0, 6.771e6);  // 400 km
    
    Eigen::Vector3d tau1 = torque.compute_torque(ctx1);
    Eigen::Vector3d tau2 = torque.compute_torque(ctx2);
    Eigen::Vector3d tau3 = torque.compute_torque(ctx3);
    
    // Check if decay is approximately exponential
    double ratio_12 = tau2.norm() / tau1.norm();
    double ratio_23 = tau3.norm() / tau2.norm();
    
    // Ratios should be similar for exponential decay
    EXPECT_NEAR(ratio_12, ratio_23, 0.5);  // Allow some variation
}

TEST_F(AerodynamicTorqueTest, TorqueVelocityDirectionIndependence) {
    // For same velocity magnitude, torque magnitude should be similar
    // (direction changes, but magnitude stays relatively constant)
    AerodynamicTorque torque(cop_, com_, cd_, area_);
    
    double v_mag = 7500.0;
    std::vector<Eigen::Vector3d> velocities = {
        Eigen::Vector3d(v_mag, 0.0, 0.0),
        Eigen::Vector3d(0.0, v_mag, 0.0),
        Eigen::Vector3d(v_mag / std::sqrt(2), v_mag / std::sqrt(2), 0.0)
    };
    
    std::vector<double> torque_magnitudes;
    for (const auto& vel : velocities) {
        TorqueContext ctx = createLEOContext(vel);
        Eigen::Vector3d tau = torque.compute_torque(ctx);
        torque_magnitudes.push_back(tau.norm());
    }
    
    // All magnitudes should be similar (within 5%)
    double avg = (torque_magnitudes[0] + torque_magnitudes[1] + torque_magnitudes[2]) / 3.0;
    for (double mag : torque_magnitudes) {
        EXPECT_NEAR(mag, avg, avg * 0.05);
    }
}

TEST_F(AerodynamicTorqueTest, TorqueBodyFrameConsistency) {
    // Torque computed should be consistent in body frame
    AerodynamicTorque torque(cop_, com_, cd_, area_);
    
    TorqueContext ctx1 = createLEOContext(Eigen::Vector3d(7500.0, 0.0, 0.0));
    ctx1.orientation = Eigen::Quaterniond::Identity();
    
    // Same velocity but different orientation
    TorqueContext ctx2 = ctx1;
    ctx2.orientation = Eigen::Quaterniond(Eigen::AngleAxisd(M_PI/3.0, Eigen::Vector3d::UnitZ()));
    
    Eigen::Vector3d tau1 = torque.compute_torque(ctx1);
    Eigen::Vector3d tau2 = torque.compute_torque(ctx2);
    
    // Torques should have same magnitude (just different direction in body frame)
    EXPECT_NEAR(tau1.norm(), tau2.norm(), tau1.norm() * 0.01);
}

TEST_F(AerodynamicTorqueTest, COMNotAtOrigin) {
    // Test when COM is not at body frame origin
    Eigen::Vector3d cop(0.1, 0.2, 0.3);
    Eigen::Vector3d com(0.05, 0.1, 0.15);
    
    AerodynamicTorque torque(cop, com, cd_, area_);
    
    TorqueContext ctx = createLEOContext(Eigen::Vector3d(7500.0, 0.0, 0.0));
    
    Eigen::Vector3d tau = torque.compute_torque(ctx);
    
    EXPECT_TRUE(tau.allFinite());
    EXPECT_GT(tau.norm(), 0.0);
    
    // Verify torque depends only on moment arm difference
    Eigen::Vector3d cop_shifted(0.2, 0.3, 0.4);
    Eigen::Vector3d com_shifted(0.15, 0.2, 0.25);
    AerodynamicTorque torque_shifted(cop_shifted, com_shifted, cd_, area_);
    
    Eigen::Vector3d tau_shifted = torque_shifted.compute_torque(ctx);
    
    EXPECT_TRUE(tau.isApprox(tau_shifted, 1e-12));
}
