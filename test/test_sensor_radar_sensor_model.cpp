#include <gtest/gtest.h>

#include "sensor/radar_sensor_model.hpp"

#include <cmath>

using namespace sensor;

class RadarSensorModelTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Radar at origin
        radar_position_ = Eigen::Vector3d(0, 0, 0);
        range_noise_ = 10.0;   // 10m
        angle_noise_ = 0.001;  // ~0.057 degrees
        
        radar_ = std::make_shared<RadarSensorModel>(
            radar_position_,
            range_noise_,
            angle_noise_
        );
    }

    Eigen::Vector3d radar_position_;
    double range_noise_;
    double angle_noise_;
    std::shared_ptr<RadarSensorModel> radar_;
};

TEST_F(RadarSensorModelTest, MeasuresDimension) {
    EXPECT_EQ(radar_->get_dimension(), 3);
}

TEST_F(RadarSensorModelTest, MeasuresRangeCorrectly) {
    // Target at 1000m on x-axis
    SensorContext ctx;
    ctx.state = Eigen::VectorXd(6);
    ctx.state << 1000, 0, 0, 0, 0, 0;
    ctx.time = 0.0;
    
    Eigen::VectorXd z = radar_->compute_measurement(ctx);
    
    EXPECT_NEAR(z(0), 1000.0, 1e-6);  // Range
    EXPECT_NEAR(z(1), 0.0, 1e-6);     // Azimuth
    EXPECT_NEAR(z(2), 0.0, 1e-6);     // Elevation
}

TEST_F(RadarSensorModelTest, MeasuresAzimuthCorrectly) {
    // Target on y-axis (90 degrees azimuth)
    SensorContext ctx;
    ctx.state = Eigen::VectorXd(6);
    ctx.state << 0, 1000, 0, 0, 0, 0;
    ctx.time = 0.0;
    
    Eigen::VectorXd z = radar_->compute_measurement(ctx);
    
    EXPECT_NEAR(z(0), 1000.0, 1e-6);           // Range
    EXPECT_NEAR(z(1), M_PI / 2.0, 1e-6);       // Azimuth = 90 degrees
    EXPECT_NEAR(z(2), 0.0, 1e-6);              // Elevation
}

TEST_F(RadarSensorModelTest, MeasuresElevationCorrectly) {
    // Target directly above (45 degree elevation)
    double dist = 1000.0 / std::sqrt(2.0);  // sqrt(x^2 + z^2) = 1000
    SensorContext ctx;
    ctx.state = Eigen::VectorXd(6);
    ctx.state << dist, 0, dist, 0, 0, 0;
    ctx.time = 0.0;
    
    Eigen::VectorXd z = radar_->compute_measurement(ctx);
    
    EXPECT_NEAR(z(0), 1000.0, 1e-6);           // Range
    EXPECT_NEAR(z(1), 0.0, 1e-6);              // Azimuth
    EXPECT_NEAR(z(2), M_PI / 4.0, 1e-6);       // Elevation = 45 degrees
}

TEST_F(RadarSensorModelTest, MeasuresNegativeAzimuth) {
    // Target in 4th quadrant (negative y)
    SensorContext ctx;
    ctx.state = Eigen::VectorXd(6);
    ctx.state << 1000, -1000, 0, 0, 0, 0;
    ctx.time = 0.0;
    
    Eigen::VectorXd z = radar_->compute_measurement(ctx);
    
    double expected_range = std::sqrt(2.0) * 1000.0;
    EXPECT_NEAR(z(0), expected_range, 1e-6);
    EXPECT_NEAR(z(1), -M_PI / 4.0, 1e-6);  // -45 degrees
    EXPECT_NEAR(z(2), 0.0, 1e-6);
}

TEST_F(RadarSensorModelTest, MeasuresNegativeElevation) {
    // Target below radar
    double dist = 1000.0 / std::sqrt(2.0);
    SensorContext ctx;
    ctx.state = Eigen::VectorXd(6);
    ctx.state << dist, 0, -dist, 0, 0, 0;  // Negative z
    ctx.time = 0.0;
    
    Eigen::VectorXd z = radar_->compute_measurement(ctx);
    
    EXPECT_NEAR(z(0), 1000.0, 1e-6);
    EXPECT_NEAR(z(1), 0.0, 1e-6);
    EXPECT_NEAR(z(2), -M_PI / 4.0, 1e-6);  // -45 degrees elevation
}

TEST_F(RadarSensorModelTest, HandlesOffsetRadar) {
    // Radar not at origin
    Eigen::Vector3d radar_pos(100, 200, 50);
    auto radar = std::make_shared<RadarSensorModel>(
        radar_pos, range_noise_, angle_noise_
    );
    
    // Target position
    Eigen::Vector3d target_pos(1100, 200, 50);  // 1000m away on x-axis from radar
    
    SensorContext ctx;
    ctx.state = Eigen::VectorXd(6);
    ctx.state.head<3>() = target_pos;
    ctx.time = 0.0;
    
    Eigen::VectorXd z = radar->compute_measurement(ctx);
    
    EXPECT_NEAR(z(0), 1000.0, 1e-6);
    EXPECT_NEAR(z(1), 0.0, 1e-6);
    EXPECT_NEAR(z(2), 0.0, 1e-6);
}

TEST_F(RadarSensorModelTest, NoiseCovarianceHasCorrectStructure) {
    Eigen::MatrixXd R = radar_->get_noise_covariance();
    
    EXPECT_EQ(R.rows(), 3);
    EXPECT_EQ(R.cols(), 3);
    
    // Check diagonal elements
    EXPECT_NEAR(R(0, 0), range_noise_ * range_noise_, 1e-9);
    EXPECT_NEAR(R(1, 1), angle_noise_ * angle_noise_, 1e-9);
    EXPECT_NEAR(R(2, 2), angle_noise_ * angle_noise_, 1e-9);
    
    // Check off-diagonal elements are zero
    EXPECT_NEAR(R(0, 1), 0.0, 1e-9);
    EXPECT_NEAR(R(0, 2), 0.0, 1e-9);
    EXPECT_NEAR(R(1, 0), 0.0, 1e-9);
    EXPECT_NEAR(R(1, 2), 0.0, 1e-9);
    EXPECT_NEAR(R(2, 0), 0.0, 1e-9);
    EXPECT_NEAR(R(2, 1), 0.0, 1e-9);
}

TEST_F(RadarSensorModelTest, NoiseCovarianceIsSymmetric) {
    Eigen::MatrixXd R = radar_->get_noise_covariance();
    Eigen::MatrixXd R_transpose = R.transpose();
    
    EXPECT_TRUE(R.isApprox(R_transpose));
}

TEST_F(RadarSensorModelTest, NoiseCovarianceIsPositiveDefinite) {
    Eigen::MatrixXd R = radar_->get_noise_covariance();
    
    // Check eigenvalues are all positive
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(R);
    Eigen::VectorXd eigenvalues = solver.eigenvalues();
    
    for (int i = 0; i < eigenvalues.size(); ++i) {
        EXPECT_GT(eigenvalues(i), 0.0);
    }
}

TEST_F(RadarSensorModelTest, HandlesLargeStateVectors) {
    // State with more than 6 elements (e.g., with attitude)
    SensorContext ctx;
    ctx.state = Eigen::VectorXd(13);
    ctx.state << 1000, 0, 0,      // Position
                 0, 0, 0,          // Velocity
                 1, 0, 0, 0,       // Quaternion
                 0, 0, 0;          // Angular velocity
    ctx.time = 0.0;
    
    Eigen::VectorXd z = radar_->compute_measurement(ctx);
    
    EXPECT_EQ(z.size(), 3);
    EXPECT_NEAR(z(0), 1000.0, 1e-6);
}

TEST_F(RadarSensorModelTest, MeasurementIsConsistent) {
    // Same input should give same output
    SensorContext ctx;
    ctx.state = Eigen::VectorXd(6);
    ctx.state << 1234, 5678, 910, 0, 0, 0;
    ctx.time = 0.0;
    
    Eigen::VectorXd z1 = radar_->compute_measurement(ctx);
    Eigen::VectorXd z2 = radar_->compute_measurement(ctx);
    
    EXPECT_TRUE(z1.isApprox(z2));
}

TEST_F(RadarSensorModelTest, RangeIsAlwaysPositive) {
    // Test various positions
    std::vector<Eigen::Vector3d> positions = {
        {1000, 0, 0},
        {-1000, 0, 0},
        {0, 1000, 0},
        {0, -1000, 0},
        {0, 0, 1000},
        {0, 0, -1000},
        {500, 500, 500},
        {-500, -500, -500}
    };
    
    for (const auto& pos : positions) {
        SensorContext ctx;
        ctx.state = Eigen::VectorXd(6);
        ctx.state.head<3>() = pos;
        ctx.time = 0.0;
        
        Eigen::VectorXd z = radar_->compute_measurement(ctx);
        EXPECT_GT(z(0), 0.0) << "Range should be positive for position: " 
                              << pos.transpose();
    }
}

TEST_F(RadarSensorModelTest, AzimuthIsInCorrectRange) {
    // Test positions in all quadrants
    std::vector<Eigen::Vector3d> positions = {
        {1000, 1000, 0},    // Q1: 0 to π/2
        {-1000, 1000, 0},   // Q2: π/2 to π
        {-1000, -1000, 0},  // Q3: -π to -π/2
        {1000, -1000, 0}    // Q4: -π/2 to 0
    };
    
    for (const auto& pos : positions) {
        SensorContext ctx;
        ctx.state = Eigen::VectorXd(6);
        ctx.state.head<3>() = pos;
        ctx.time = 0.0;
        
        Eigen::VectorXd z = radar_->compute_measurement(ctx);
        EXPECT_GE(z(1), -M_PI) << "Azimuth should be >= -π";
        EXPECT_LE(z(1), M_PI) << "Azimuth should be <= π";
    }
}

TEST_F(RadarSensorModelTest, ElevationIsInCorrectRange) {
    // Test positions at different elevations
    std::vector<Eigen::Vector3d> positions = {
        {1000, 0, 1000},   // 45° elevation
        {1000, 0, 0},      // 0° elevation
        {1000, 0, -1000},  // -45° elevation
        {0, 0, 1000}       // 90° elevation
    };
    
    for (const auto& pos : positions) {
        SensorContext ctx;
        ctx.state = Eigen::VectorXd(6);
        ctx.state.head<3>() = pos;
        ctx.time = 0.0;
        
        Eigen::VectorXd z = radar_->compute_measurement(ctx);
        EXPECT_GE(z(2), -M_PI / 2.0) << "Elevation should be >= -π/2";
        EXPECT_LE(z(2), M_PI / 2.0) << "Elevation should be <= π/2";
    }
}

TEST_F(RadarSensorModelTest, SphericalToCartesianRoundTrip) {
    // Original position
    Eigen::Vector3d pos_original(1000, 500, 200);
    
    SensorContext ctx;
    ctx.state = Eigen::VectorXd(6);
    ctx.state.head<3>() = pos_original;
    ctx.time = 0.0;
    
    // Compute measurement (spherical)
    Eigen::VectorXd z = radar_->compute_measurement(ctx);
    double range = z(0);
    double azimuth = z(1);
    double elevation = z(2);
    
    // Convert back to Cartesian
    Eigen::Vector3d pos_reconstructed;
    pos_reconstructed.x() = range * std::cos(elevation) * std::cos(azimuth);
    pos_reconstructed.y() = range * std::cos(elevation) * std::sin(azimuth);
    pos_reconstructed.z() = range * std::sin(elevation);
    
    EXPECT_TRUE(pos_original.isApprox(pos_reconstructed, 1e-6))
        << "Original: " << pos_original.transpose()
        << "\nReconstructed: " << pos_reconstructed.transpose();
}

TEST_F(RadarSensorModelTest, DifferentNoiseValues) {
    // Test with different noise parameters
    double large_range_noise = 100.0;
    double large_angle_noise = 0.01;
    
    auto radar_noisy = std::make_shared<RadarSensorModel>(
        radar_position_,
        large_range_noise,
        large_angle_noise
    );
    
    Eigen::MatrixXd R = radar_noisy->get_noise_covariance();
    
    EXPECT_NEAR(R(0, 0), large_range_noise * large_range_noise, 1e-9);
    EXPECT_NEAR(R(1, 1), large_angle_noise * large_angle_noise, 1e-9);
    EXPECT_NEAR(R(2, 2), large_angle_noise * large_angle_noise, 1e-9);
}

// Edge case tests
TEST_F(RadarSensorModelTest, HandlesVerySmallRange) {
    // Target very close to radar (1 meter)
    SensorContext ctx;
    ctx.state = Eigen::VectorXd(6);
    ctx.state << 1, 0, 0, 0, 0, 0;
    ctx.time = 0.0;
    
    Eigen::VectorXd z = radar_->compute_measurement(ctx);
    
    EXPECT_NEAR(z(0), 1.0, 1e-6);
    EXPECT_NEAR(z(1), 0.0, 1e-6);
    EXPECT_NEAR(z(2), 0.0, 1e-6);
}

TEST_F(RadarSensorModelTest, HandlesVeryLargeRange) {
    // Target very far from radar (1000 km)
    SensorContext ctx;
    ctx.state = Eigen::VectorXd(6);
    ctx.state << 1e6, 0, 0, 0, 0, 0;
    ctx.time = 0.0;
    
    Eigen::VectorXd z = radar_->compute_measurement(ctx);
    
    EXPECT_NEAR(z(0), 1e6, 1.0);  // Within 1m tolerance
    EXPECT_NEAR(z(1), 0.0, 1e-6);
    EXPECT_NEAR(z(2), 0.0, 1e-6);
}

TEST_F(RadarSensorModelTest, IgnoresVelocityComponents) {
    // Measurements should not depend on velocity
    SensorContext ctx1;
    ctx1.state = Eigen::VectorXd(6);
    ctx1.state << 1000, 0, 0, 0, 0, 0;
    
    SensorContext ctx2;
    ctx2.state = Eigen::VectorXd(6);
    ctx2.state << 1000, 0, 0, 100, 200, 300;  // Different velocity
    
    Eigen::VectorXd z1 = radar_->compute_measurement(ctx1);
    Eigen::VectorXd z2 = radar_->compute_measurement(ctx2);
    
    EXPECT_TRUE(z1.isApprox(z2));
}

// ============================================================================
// JACOBIAN TESTS
// ============================================================================

TEST_F(RadarSensorModelTest, JacobianHasCorrectDimensions) {
    SensorContext ctx;
    ctx.state = Eigen::VectorXd(6);
    ctx.state << 1000, 500, 200, 0, 0, 0;
    ctx.time = 0.0;
    
    Eigen::MatrixXd H = radar_->compute_jacobian(ctx);
    
    EXPECT_EQ(H.rows(), 3);  // 3 measurements: range, azimuth, elevation
    EXPECT_EQ(H.cols(), 6);  // 6 state elements: position + velocity
}

TEST_F(RadarSensorModelTest, JacobianVelocityColumnsAreZero) {
    // Measurement doesn't depend on velocity, so ∂h/∂v = 0
    SensorContext ctx;
    ctx.state = Eigen::VectorXd(6);
    ctx.state << 1000, 500, 200, 10, 20, 30;
    ctx.time = 0.0;
    
    Eigen::MatrixXd H = radar_->compute_jacobian(ctx);
    
    // Columns 3-5 (velocity components) should be zero
    EXPECT_TRUE((H.block<3, 3>(0, 3).isApprox(Eigen::Matrix3d::Zero(), 1e-12)));
}

TEST_F(RadarSensorModelTest, JacobianNumericalVsAnalytical) {
    // Compare analytical Jacobian to numerical differentiation
    SensorContext ctx;
    ctx.state = Eigen::VectorXd(6);
    ctx.state << 1000, 500, 200, 0, 0, 0;
    ctx.time = 0.0;
    
    Eigen::MatrixXd H_analytical = radar_->compute_jacobian(ctx);
    
    // Compute numerical Jacobian using finite differences
    double epsilon = 1e-6;
    Eigen::MatrixXd H_numerical = Eigen::MatrixXd::Zero(3, 6);
    
    for (int col = 0; col < 3; ++col) {  // Only test position columns
        SensorContext ctx_plus = ctx;
        SensorContext ctx_minus = ctx;
        
        ctx_plus.state(col) += epsilon;
        ctx_minus.state(col) -= epsilon;
        
        Eigen::VectorXd z_plus = radar_->compute_measurement(ctx_plus);
        Eigen::VectorXd z_minus = radar_->compute_measurement(ctx_minus);
        
        H_numerical.col(col) = (z_plus - z_minus) / (2.0 * epsilon);
    }
    
    // Compare only position columns (0-2)
    for (int col = 0; col < 3; ++col) {
        double error = (H_analytical.col(col) - H_numerical.col(col)).norm();
        EXPECT_LT(error, 1e-5) << "Column " << col << " error: " << error;
    }
}

TEST_F(RadarSensorModelTest, JacobianRangeDerivatives) {
    // Test range derivatives ∂ρ/∂x = dx/ρ, ∂ρ/∂y = dy/ρ, ∂ρ/∂z = dz/ρ
    SensorContext ctx;
    ctx.state = Eigen::VectorXd(6);
    ctx.state << 1000, 0, 0, 0, 0, 0;
    ctx.time = 0.0;
    
    Eigen::MatrixXd H = radar_->compute_jacobian(ctx);
    
    // For target at (1000, 0, 0), range = 1000
    // ∂ρ/∂x = 1000/1000 = 1
    // ∂ρ/∂y = 0/1000 = 0
    // ∂ρ/∂z = 0/1000 = 0
    EXPECT_NEAR(H(0, 0), 1.0, 1e-6);  // ∂ρ/∂x
    EXPECT_NEAR(H(0, 1), 0.0, 1e-6);  // ∂ρ/∂y
    EXPECT_NEAR(H(0, 2), 0.0, 1e-6);  // ∂ρ/∂z
}

TEST_F(RadarSensorModelTest, JacobianAzimuthDerivatives) {
    // Test azimuth derivatives at known points
    SensorContext ctx;
    ctx.state = Eigen::VectorXd(6);
    ctx.state << 1000, 0, 0, 0, 0, 0;
    ctx.time = 0.0;
    
    Eigen::MatrixXd H = radar_->compute_jacobian(ctx);
    
    // For target at (1000, 0, 0):
    // ∂α/∂x = -dy/(dx² + dy²) = 0/(1000² + 0²) = 0
    // ∂α/∂y = dx/(dx² + dy²) = 1000/(1000² + 0²) = 1/1000
    // ∂α/∂z = 0
    EXPECT_NEAR(H(1, 0), 0.0, 1e-9);       // ∂α/∂x
    EXPECT_NEAR(H(1, 1), 1.0/1000.0, 1e-9); // ∂α/∂y
    EXPECT_NEAR(H(1, 2), 0.0, 1e-9);       // ∂α/∂z
}

TEST_F(RadarSensorModelTest, JacobianElevationDerivatives) {
    // Test elevation derivatives at known points
    SensorContext ctx;
    ctx.state = Eigen::VectorXd(6);
    ctx.state << 1000, 0, 0, 0, 0, 0;
    ctx.time = 0.0;
    
    Eigen::MatrixXd H = radar_->compute_jacobian(ctx);
    
    // For target at (1000, 0, 0) on xy-plane:
    // ∂ε/∂x = -dx*dz / (ρ² * sqrt(dx² + dy²)) = 0 (since dz=0)
    // ∂ε/∂y = -dy*dz / (ρ² * sqrt(dx² + dy²)) = 0 (since dz=0)
    // ∂ε/∂z = sqrt(dx² + dy²) / ρ² = 1000/1000² = 1/1000
    EXPECT_NEAR(H(2, 0), 0.0, 1e-9);       // ∂ε/∂x
    EXPECT_NEAR(H(2, 1), 0.0, 1e-9);       // ∂ε/∂y
    EXPECT_NEAR(H(2, 2), 1.0/1000.0, 1e-9); // ∂ε/∂z
}

TEST_F(RadarSensorModelTest, JacobianAt45DegreeAzimuth) {
    // Test at 45° azimuth (equal x and y)
    double dist = 1000.0 / std::sqrt(2.0);
    SensorContext ctx;
    ctx.state = Eigen::VectorXd(6);
    ctx.state << dist, dist, 0, 0, 0, 0;
    ctx.time = 0.0;
    
    Eigen::MatrixXd H = radar_->compute_jacobian(ctx);
    
    // Range derivatives should be symmetric
    EXPECT_NEAR(H(0, 0), H(0, 1), 1e-6);  // ∂ρ/∂x ≈ ∂ρ/∂y
    EXPECT_NEAR(H(0, 0), 1.0/std::sqrt(2.0), 1e-6);
}

TEST_F(RadarSensorModelTest, JacobianAt45DegreeElevation) {
    // Test at 45° elevation
    double dist = 1000.0 / std::sqrt(2.0);
    SensorContext ctx;
    ctx.state = Eigen::VectorXd(6);
    ctx.state << dist, 0, dist, 0, 0, 0;
    ctx.time = 0.0;
    
    Eigen::MatrixXd H = radar_->compute_jacobian(ctx);
    
    // At 45° elevation:
    // ∂ρ/∂x ≈ ∂ρ/∂z ≈ 1/√2
    EXPECT_NEAR(H(0, 0), 1.0/std::sqrt(2.0), 1e-6);
    EXPECT_NEAR(H(0, 2), 1.0/std::sqrt(2.0), 1e-6);
}

TEST_F(RadarSensorModelTest, JacobianIsFinite) {
    // Test at various positions
    std::vector<Eigen::Vector3d> positions = {
        {1000, 0, 0},
        {0, 1000, 0},
        {0, 0, 1000},
        {500, 500, 0},
        {500, 0, 500},
        {0, 500, 500},
        {333, 333, 333}
    };
    
    for (const auto& pos : positions) {
        SensorContext ctx;
        ctx.state = Eigen::VectorXd(6);
        ctx.state.head<3>() = pos;
        ctx.time = 0.0;
        
        Eigen::MatrixXd H = radar_->compute_jacobian(ctx);
        
        EXPECT_TRUE(H.allFinite()) << "Jacobian should be finite for position: " 
                                   << pos.transpose();
    }
}

TEST_F(RadarSensorModelTest, JacobianAtOriginHandlesSingularity) {
    // Target at radar position (singular point)
    SensorContext ctx;
    ctx.state = Eigen::VectorXd(6);
    ctx.state << 0, 0, 0, 0, 0, 0;
    ctx.time = 0.0;
    
    Eigen::MatrixXd H = radar_->compute_jacobian(ctx);
    
    // Should return zero Jacobian at singularity
    EXPECT_TRUE(H.isApprox(Eigen::MatrixXd::Zero(3, 6), 1e-12));
}

TEST_F(RadarSensorModelTest, JacobianNearOriginHandlesSingularity) {
    // Target very close to radar (near-singular)
    SensorContext ctx;
    ctx.state = Eigen::VectorXd(6);
    ctx.state << 1e-13, 0, 0, 0, 0, 0;
    ctx.time = 0.0;
    
    Eigen::MatrixXd H = radar_->compute_jacobian(ctx);
    
    // Should return zero Jacobian or handle gracefully
    EXPECT_TRUE(H.allFinite() || H.isApprox(Eigen::MatrixXd::Zero(3, 6), 1e-12));
}

TEST_F(RadarSensorModelTest, JacobianWithOffsetRadar) {
    // Test Jacobian when radar is not at origin
    Eigen::Vector3d radar_pos(100, 200, 50);
    auto radar = std::make_shared<RadarSensorModel>(
        radar_pos, range_noise_, angle_noise_
    );
    
    // Target position
    Eigen::Vector3d target_pos(1100, 200, 50);  // 1000m away on x-axis
    
    SensorContext ctx;
    ctx.state = Eigen::VectorXd(6);
    ctx.state.head<3>() = target_pos;
    ctx.time = 0.0;
    
    Eigen::MatrixXd H = radar->compute_jacobian(ctx);
    
    EXPECT_EQ(H.rows(), 3);
    EXPECT_EQ(H.cols(), 6);
    EXPECT_TRUE(H.allFinite());
    
    // Range derivative should be ~1 in x-direction
    EXPECT_NEAR(H(0, 0), 1.0, 1e-6);
}

TEST_F(RadarSensorModelTest, JacobianSymmetryInXY) {
    // For symmetric positions, certain derivatives should match
    // Test at (x, y, 0) where x = y
    double val = 707.0;  // Arbitrary value
    SensorContext ctx;
    ctx.state = Eigen::VectorXd(6);
    ctx.state << val, val, 0, 0, 0, 0;
    ctx.time = 0.0;
    
    Eigen::MatrixXd H = radar_->compute_jacobian(ctx);
    
    // ∂ρ/∂x should equal ∂ρ/∂y due to symmetry
    EXPECT_NEAR(H(0, 0), H(0, 1), 1e-6);
}

TEST_F(RadarSensorModelTest, JacobianLinearizationAccuracy) {
    // Test that Jacobian accurately predicts measurement changes
    SensorContext ctx;
    ctx.state = Eigen::VectorXd(6);
    ctx.state << 1000, 500, 200, 0, 0, 0;
    ctx.time = 0.0;
    
    Eigen::MatrixXd H = radar_->compute_jacobian(ctx);
    Eigen::VectorXd z_base = radar_->compute_measurement(ctx);
    
    // Small perturbation in position
    Eigen::VectorXd delta_state = Eigen::VectorXd::Zero(6);
    delta_state(0) = 10.0;  // 10m in x
    delta_state(1) = 5.0;   // 5m in y
    delta_state(2) = 2.0;   // 2m in z
    
    SensorContext ctx_perturbed;
    ctx_perturbed.state = ctx.state + delta_state;
    ctx_perturbed.time = 0.0;
    
    Eigen::VectorXd z_perturbed = radar_->compute_measurement(ctx_perturbed);
    
    // Linear approximation
    Eigen::VectorXd z_linear = z_base + H * delta_state;
    
    double error = (z_perturbed - z_linear).norm();
    double relative_error = error / z_perturbed.norm();
    
    EXPECT_LT(relative_error, 0.01);  // Within 1% for small perturbations
}

TEST_F(RadarSensorModelTest, JacobianConsistentWithMeasurement) {
    // Verify Jacobian structure matches measurement function
    SensorContext ctx;
    ctx.state = Eigen::VectorXd(6);
    ctx.state << 1234, 5678, 910, 11, 12, 13;
    ctx.time = 0.0;
    
    Eigen::MatrixXd H = radar_->compute_jacobian(ctx);
    Eigen::VectorXd z = radar_->compute_measurement(ctx);
    
    EXPECT_EQ(H.rows(), z.size());  // Jacobian rows = measurement dimension
    EXPECT_EQ(H.cols(), ctx.state.size());  // Jacobian cols = state dimension
}

TEST_F(RadarSensorModelTest, JacobianWithLargeStateVector) {
    // Test with 13-element state (position, velocity, quaternion, angular velocity)
    SensorContext ctx;
    ctx.state = Eigen::VectorXd(13);
    ctx.state << 1000, 500, 200,      // Position
                 10, 20, 30,           // Velocity
                 1, 0, 0, 0,           // Quaternion
                 0.01, 0.02, 0.03;     // Angular velocity
    ctx.time = 0.0;
    
    Eigen::MatrixXd H = radar_->compute_jacobian(ctx);
    
    EXPECT_EQ(H.rows(), 3);
    EXPECT_EQ(H.cols(), 13);
    
    // Only first 3 columns should be non-zero
    EXPECT_FALSE((H.block<3, 3>(0, 0).isApprox(Eigen::Matrix3d::Zero(), 1e-12)));
    EXPECT_TRUE((H.block<3, 10>(0, 3).isApprox(Eigen::MatrixXd::Zero(3, 10), 1e-12)));
}

TEST_F(RadarSensorModelTest, JacobianRangeRowNormalized) {
    // The range Jacobian row [∂ρ/∂x, ∂ρ/∂y, ∂ρ/∂z] should be a unit vector
    // because it points in the direction of the relative position
    SensorContext ctx;
    ctx.state = Eigen::VectorXd(6);
    ctx.state << 600, 800, 0, 0, 0, 0;  // 3-4-5 triangle in xy-plane
    ctx.time = 0.0;
    
    Eigen::MatrixXd H = radar_->compute_jacobian(ctx);
    
    // Extract range gradient
    Eigen::Vector3d range_gradient = H.block<1, 3>(0, 0).transpose();
    
    // Should be unit vector
    EXPECT_NEAR(range_gradient.norm(), 1.0, 1e-6);
}

TEST_F(RadarSensorModelTest, JacobianAzimuthIndependentOfZ) {
    // Azimuth should not depend on z-coordinate
    std::vector<double> z_values = {0, 100, 500, -100, -500};
    
    SensorContext ctx;
    ctx.state = Eigen::VectorXd(6);
    ctx.state << 1000, 500, 0, 0, 0, 0;
    ctx.time = 0.0;
    
    Eigen::MatrixXd H_base = radar_->compute_jacobian(ctx);
    double az_dz_base = H_base(1, 2);  // ∂α/∂z
    
    for (double z_val : z_values) {
        ctx.state(2) = z_val;
        Eigen::MatrixXd H = radar_->compute_jacobian(ctx);
        
        // ∂α/∂z should always be 0
        EXPECT_NEAR(H(1, 2), 0.0, 1e-12) << "z = " << z_val;
    }
}
