#include <gtest/gtest.h>

#include "sensor/space_based_azel_sensor_model.hpp"

#include <cmath>
#include <limits>

namespace sensor {
namespace {

class SpaceBasedAzElSensorModelTest : public ::testing::Test {
protected:
    void SetUp() override {
        fixed_sensor_pos_ = Eigen::Vector3d(7000e3, 0.0, 0.0);
        az_noise_ = 100e-6;
        el_noise_ = 120e-6;

        model_fixed_ = std::make_shared<SpaceBasedAzElSensorModel>(
            az_noise_,
            el_noise_
        );
    }

    static auto make_ctx(
        const Eigen::VectorXd& state,
        double time,
        const Eigen::Vector3d& sensor_position,
        const Eigen::Quaterniond& sensor_orientation = Eigen::Quaterniond::Identity()
    ) -> SensorContext {
        SensorContext ctx;
        ctx.state = state;
        ctx.time = time;
        ctx.sensor_position = sensor_position;
        ctx.sensor_orientation = sensor_orientation;
        return ctx;
    }

    Eigen::Vector3d fixed_sensor_pos_;
    double az_noise_ = 0.0;
    double el_noise_ = 0.0;
    std::shared_ptr<SpaceBasedAzElSensorModel> model_fixed_;
};

TEST_F(SpaceBasedAzElSensorModelTest, MeasurementDimensionIsTwo) {
    EXPECT_EQ(model_fixed_->get_dimension(), 2);
}

TEST_F(SpaceBasedAzElSensorModelTest, MeasuresAzimuthAndElevationInSensorFrame) {
    Eigen::VectorXd state(6);
    state << 7001e3, 1000.0, 1000.0, 0.0, 0.0, 0.0;

    const Eigen::VectorXd z =
        model_fixed_->compute_measurement(make_ctx(state, 0.0, fixed_sensor_pos_));

    EXPECT_EQ(z.size(), 2);

    const Eigen::Vector3d los = state.head<3>() - fixed_sensor_pos_;
    const double expected_az = std::atan2(los.y(), los.x());
    const double expected_el = std::atan2(
        los.z(),
        std::sqrt(los.x() * los.x() + los.y() * los.y())
    );

    EXPECT_NEAR(z(0), expected_az, 1e-12);
    EXPECT_NEAR(z(1), expected_el, 1e-12);
}

TEST_F(SpaceBasedAzElSensorModelTest, OrientationChangesMeasurement) {
    Eigen::VectorXd state(6);
    state << 7001e3, 2000.0, 300.0, 0.0, 0.0, 0.0;

    const Eigen::Quaterniond nominal = Eigen::Quaterniond::Identity();
    const Eigen::Quaterniond rotated(
        Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitZ())
    );

    const Eigen::VectorXd z_nominal =
        model_fixed_->compute_measurement(make_ctx(state, 0.0, fixed_sensor_pos_, nominal));
    const Eigen::VectorXd z_rotated =
        model_fixed_->compute_measurement(make_ctx(state, 0.0, fixed_sensor_pos_, rotated));

    EXPECT_FALSE(z_nominal.isApprox(z_rotated, 1e-12));
}

TEST_F(SpaceBasedAzElSensorModelTest, NoiseCovarianceMatchesConstructorInputs) {
    const Eigen::MatrixXd R = model_fixed_->get_noise_covariance();

    ASSERT_EQ(R.rows(), 2);
    ASSERT_EQ(R.cols(), 2);
    EXPECT_NEAR(R(0, 0), az_noise_ * az_noise_, 1e-18);
    EXPECT_NEAR(R(1, 1), el_noise_ * el_noise_, 1e-18);
    EXPECT_NEAR(R(0, 1), 0.0, 1e-18);
    EXPECT_NEAR(R(1, 0), 0.0, 1e-18);
}

TEST_F(SpaceBasedAzElSensorModelTest, JacobianShapeAndVelocityColumns) {
    Eigen::VectorXd state(6);
    state << 7100e3, 120e3, 80e3, 10.0, -20.0, 5.0;

    const Eigen::MatrixXd H =
        model_fixed_->compute_jacobian(make_ctx(state, 0.0, fixed_sensor_pos_));

    ASSERT_EQ(H.rows(), 2);
    ASSERT_EQ(H.cols(), 6);
    EXPECT_TRUE(H.block(0, 3, 2, 3).isZero(1e-15));
}

TEST_F(SpaceBasedAzElSensorModelTest, JacobianMatchesNumericalDerivative) {
    Eigen::VectorXd state(6);
    state << 7120e3, 180e3, 90e3, 0.0, 0.0, 0.0;
    const Eigen::Quaterniond orientation(
        Eigen::AngleAxisd(-0.15, Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitZ())
    );
    const SensorContext ctx = make_ctx(state, 0.0, fixed_sensor_pos_, orientation);

    const Eigen::MatrixXd H_analytic = model_fixed_->compute_jacobian(ctx);

    Eigen::MatrixXd H_numeric = Eigen::MatrixXd::Zero(2, 6);
    constexpr double h = 1e-3;

    for (int i = 0; i < 3; ++i) {
        Eigen::VectorXd xp = state;
        Eigen::VectorXd xm = state;
        xp(i) += h;
        xm(i) -= h;

        const Eigen::VectorXd zp =
            model_fixed_->compute_measurement(make_ctx(xp, 0.0, fixed_sensor_pos_, orientation));
        const Eigen::VectorXd zm =
            model_fixed_->compute_measurement(make_ctx(xm, 0.0, fixed_sensor_pos_, orientation));
        H_numeric.col(i) = (zp - zm) / (2.0 * h);
    }

    EXPECT_TRUE(
        H_analytic.block(0, 0, 2, 3).isApprox(H_numeric.block(0, 0, 2, 3), 5e-7)
    );
}

TEST_F(SpaceBasedAzElSensorModelTest, ThrowsIfStateDimensionTooSmall) {
    Eigen::VectorXd bad_state(2);
    bad_state << 1.0, 2.0;

    EXPECT_THROW(
        model_fixed_->compute_measurement(make_ctx(bad_state, 0.0, fixed_sensor_pos_)),
        std::invalid_argument
    );
    EXPECT_THROW(
        model_fixed_->compute_jacobian(make_ctx(bad_state, 0.0, fixed_sensor_pos_)),
        std::invalid_argument
    );
}

TEST(SpaceBasedAzElSensorModelCtorTest, RejectsInvalidNoise) {
    EXPECT_THROW(
        SpaceBasedAzElSensorModel(0.0, 1e-4),
        std::invalid_argument
    );
    EXPECT_THROW(
        SpaceBasedAzElSensorModel(1e-4, -1e-4),
        std::invalid_argument
    );
}

TEST_F(SpaceBasedAzElSensorModelTest, ThrowsIfSensorPositionIsNotFinite) {
    Eigen::VectorXd state(6);
    state << 7100e3, 0.0, 0.0, 0.0, 0.0, 0.0;

    SensorContext ctx;
    ctx.state = state;
    ctx.time = 0.0;
    ctx.sensor_position = Eigen::Vector3d::Constant(
        std::numeric_limits<double>::quiet_NaN()
    );

    EXPECT_THROW(model_fixed_->compute_measurement(ctx), std::invalid_argument);
    EXPECT_THROW(model_fixed_->compute_jacobian(ctx), std::invalid_argument);
}

TEST_F(SpaceBasedAzElSensorModelTest, ThrowsIfSensorOrientationIsNotFinite) {
    Eigen::VectorXd state(6);
    state << 7100e3, 10e3, 5e3, 0.0, 0.0, 0.0;

    SensorContext ctx = make_ctx(state, 0.0, fixed_sensor_pos_);
    ctx.sensor_orientation.coeffs()(0) = std::numeric_limits<double>::quiet_NaN();

    EXPECT_THROW(model_fixed_->compute_measurement(ctx), std::invalid_argument);
    EXPECT_THROW(model_fixed_->compute_jacobian(ctx), std::invalid_argument);
}

} // namespace
} // namespace sensor
