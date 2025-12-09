#include <gtest/gtest.h>
#include "common/types.hpp"

TEST(MeasurementTest, DefaultConstructorCreatesInvalidMeasurement) {
    common::Measurement meas;
    EXPECT_FALSE(meas.is_valid());
}

TEST(MeasurementTest, ValidConstructor) {
    Eigen::Vector3d z(1000.0, 0.5, 0.1);
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity() * 100.0;
    
    common::Measurement meas(z, R, 10.0);
    
    EXPECT_TRUE(meas.is_valid());
    EXPECT_EQ(meas.dimension(), 3);
    EXPECT_DOUBLE_EQ(meas.time, 10.0);
}

TEST(MeasurementTest, ThrowsOnEmptyVector) {
    Eigen::VectorXd empty_z;
    Eigen::MatrixXd R;
    
    EXPECT_THROW(
        common::Measurement meas(empty_z, R, 10.0),
        std::invalid_argument
    );
}

TEST(MeasurementTest, ThrowsOnDimensionMismatch) {
    Eigen::Vector3d z(1.0, 2.0, 3.0);
    Eigen::Matrix2d R = Eigen::Matrix2d::Identity();  // Wrong size!
    
    EXPECT_THROW(
        common::Measurement meas(z, R, 10.0),
        std::invalid_argument
    );
}

TEST(MeasurementTest, ThrowsOnNegativeTime) {
    Eigen::Vector3d z(1.0, 2.0, 3.0);
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    
    EXPECT_THROW(
        common::Measurement meas(z, R, -1.0),
        std::invalid_argument
    );
}

TEST(MeasurementTest, ThrowsOnNonSymmetricCovariance) {
    Eigen::Vector3d z(1.0, 2.0, 3.0);
    Eigen::Matrix3d R;
    R << 1, 2, 3,
         2, 1, 4,  // Not symmetric!
         5, 4, 1;
    
    EXPECT_THROW(
        common::Measurement meas(z, R, 10.0),
        std::invalid_argument
    );
}

TEST(MeasurementTest, ThrowsOnNonPositiveDefiniteCovariance) {
    Eigen::Vector3d z(1.0, 2.0, 3.0);
    Eigen::Matrix3d R = -Eigen::Matrix3d::Identity();  // Negative eigenvalues!
    
    EXPECT_THROW(
        common::Measurement meas(z, R, 10.0),
        std::invalid_argument
    );
}

TEST(MeasurementTest, DetectsNaNValues) {
    Eigen::Vector3d z(1.0, std::nan(""), 3.0);
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    
    // Constructor might not catch NaN (depends on checks)
    // But is_valid() should catch it
    common::Measurement meas;
    meas.z = z;
    meas.R = R;
    meas.time = 10.0;
    
    EXPECT_FALSE(meas.is_valid());
}

TEST(MeasurementTest, MetadataInitialization) {
    Eigen::Vector3d z(1.0, 2.0, 3.0);
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    
    common::Measurement meas(z, R, 10.0);
    
    EXPECT_EQ(meas.sensor_id, "");
    EXPECT_EQ(meas.measurement_id, -1);
    EXPECT_FALSE(meas.is_associated);
    EXPECT_EQ(meas.track_id, -1);
}

TEST(MeasurementTest, MetadataCanBeSet) {
    Eigen::Vector3d z(1.0, 2.0, 3.0);
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    
    common::Measurement meas(z, R, 10.0);
    meas.sensor_id = "radar_1";
    meas.measurement_id = 42;
    meas.is_associated = true;
    meas.track_id = 5;
    
    EXPECT_EQ(meas.sensor_id, "radar_1");
    EXPECT_EQ(meas.measurement_id, 42);
    EXPECT_TRUE(meas.is_associated);
    EXPECT_EQ(meas.track_id, 5);
}
