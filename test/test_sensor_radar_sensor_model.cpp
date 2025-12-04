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
