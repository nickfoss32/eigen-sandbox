#include <gtest/gtest.h>

#include "sensor/space_based_optical_sensor_model.hpp"

#include <cmath>
#include <limits>

namespace sensor {
namespace {

class SpaceBasedOpticalSensorModelTest : public ::testing::Test {
protected:
    void SetUp() override {
        fixed_sensor_pos_ = Eigen::Vector3d(7000e3, 0.0, 0.0);
        ra_noise_ = 100e-6;
        dec_noise_ = 120e-6;

        model_fixed_ = std::make_shared<SpaceBasedOpticalSensorModel>(
            ra_noise_,
            dec_noise_
        );
    }

    static auto make_ctx(
        const Eigen::VectorXd& state,
        double time,
        const Eigen::Vector3d& sensor_position
    ) -> SensorContext {
        SensorContext ctx;
        ctx.state = state;
        ctx.time = time;
        ctx.sensor_position = sensor_position;
        ctx.sensor_orientation = Eigen::Quaterniond::Identity();
        return ctx;
    }

    Eigen::Vector3d fixed_sensor_pos_;
    double ra_noise_ = 0.0;
    double dec_noise_ = 0.0;
    std::shared_ptr<SpaceBasedOpticalSensorModel> model_fixed_;
};

TEST_F(SpaceBasedOpticalSensorModelTest, MeasurementDimensionIsTwo) {
    EXPECT_EQ(model_fixed_->get_dimension(), 2);
}

TEST_F(SpaceBasedOpticalSensorModelTest, MeasuresRightAscensionAndDeclination) {
    Eigen::VectorXd state(6);
    state << 7001e3, 1000.0, 1000.0, 0.0, 0.0, 0.0;

    const Eigen::VectorXd z = model_fixed_->compute_measurement(make_ctx(state, 0.0, fixed_sensor_pos_));

    EXPECT_EQ(z.size(), 2);

    const Eigen::Vector3d los = state.head<3>() - fixed_sensor_pos_;
    const double expected_ra = std::atan2(los.y(), los.x());
    const double expected_dec = std::atan2(los.z(), std::sqrt(los.x() * los.x() + los.y() * los.y()));

    EXPECT_NEAR(z(0), expected_ra, 1e-12);
    EXPECT_NEAR(z(1), expected_dec, 1e-12);
}

TEST_F(SpaceBasedOpticalSensorModelTest, MovingSensorPositionChangesMeasurement) {
    Eigen::VectorXd state(6);
    state << 7001e3, 2000.0, 300.0, 0.0, 0.0, 0.0;

    const Eigen::Vector3d sensor_t0(7000e3, 0.0, 0.0);
    const Eigen::Vector3d sensor_t10(7000e3, 7500.0 * 10.0, 0.0);

    const Eigen::VectorXd z_t0 = model_fixed_->compute_measurement(make_ctx(state, 0.0, sensor_t0));
    const Eigen::VectorXd z_t10 = model_fixed_->compute_measurement(make_ctx(state, 10.0, sensor_t10));

    EXPECT_FALSE(z_t0.isApprox(z_t10, 1e-12));
}

TEST_F(SpaceBasedOpticalSensorModelTest, NoiseCovarianceMatchesConstructorInputs) {
    const Eigen::MatrixXd R = model_fixed_->get_noise_covariance();

    ASSERT_EQ(R.rows(), 2);
    ASSERT_EQ(R.cols(), 2);
    EXPECT_NEAR(R(0, 0), ra_noise_ * ra_noise_, 1e-18);
    EXPECT_NEAR(R(1, 1), dec_noise_ * dec_noise_, 1e-18);
    EXPECT_NEAR(R(0, 1), 0.0, 1e-18);
    EXPECT_NEAR(R(1, 0), 0.0, 1e-18);
}

TEST_F(SpaceBasedOpticalSensorModelTest, JacobianShapeAndVelocityColumns) {
    Eigen::VectorXd state(6);
    state << 7100e3, 120e3, 80e3, 10.0, -20.0, 5.0;

    const Eigen::MatrixXd H = model_fixed_->compute_jacobian(make_ctx(state, 0.0, fixed_sensor_pos_));

    ASSERT_EQ(H.rows(), 2);
    ASSERT_EQ(H.cols(), 6);
    EXPECT_TRUE(H.block(0, 3, 2, 3).isZero(1e-15));
}

TEST_F(SpaceBasedOpticalSensorModelTest, JacobianMatchesNumericalDerivative) {
    Eigen::VectorXd state(6);
    state << 7120e3, 180e3, 90e3, 0.0, 0.0, 0.0;
    const SensorContext ctx = make_ctx(state, 0.0, fixed_sensor_pos_);

    const Eigen::MatrixXd H_analytic = model_fixed_->compute_jacobian(ctx);

    Eigen::MatrixXd H_numeric = Eigen::MatrixXd::Zero(2, 6);
    constexpr double h = 1e-3;

    for (int i = 0; i < 3; ++i) {
        Eigen::VectorXd xp = state;
        Eigen::VectorXd xm = state;
        xp(i) += h;
        xm(i) -= h;

        const Eigen::VectorXd zp = model_fixed_->compute_measurement(make_ctx(xp, 0.0, fixed_sensor_pos_));
        const Eigen::VectorXd zm = model_fixed_->compute_measurement(make_ctx(xm, 0.0, fixed_sensor_pos_));
        H_numeric.col(i) = (zp - zm) / (2.0 * h);
    }

    EXPECT_TRUE(H_analytic.block(0, 0, 2, 3).isApprox(H_numeric.block(0, 0, 2, 3), 5e-7));
}

TEST_F(SpaceBasedOpticalSensorModelTest, ThrowsIfStateDimensionTooSmall) {
    Eigen::VectorXd bad_state(2);
    bad_state << 1.0, 2.0;

    EXPECT_THROW(model_fixed_->compute_measurement(make_ctx(bad_state, 0.0, fixed_sensor_pos_)), std::invalid_argument);
    EXPECT_THROW(model_fixed_->compute_jacobian(make_ctx(bad_state, 0.0, fixed_sensor_pos_)), std::invalid_argument);
}

TEST(SpaceBasedOpticalSensorModelCtorTest, RejectsInvalidNoise) {
    EXPECT_THROW(
        SpaceBasedOpticalSensorModel(0.0, 1e-4),
        std::invalid_argument
    );
    EXPECT_THROW(
        SpaceBasedOpticalSensorModel(1e-4, -1e-4),
        std::invalid_argument
    );
}

TEST_F(SpaceBasedOpticalSensorModelTest, ThrowsIfSensorPositionIsNotFinite) {
    Eigen::VectorXd state(6);
    state << 7100e3, 0.0, 0.0, 0.0, 0.0, 0.0;

    SensorContext ctx;
    ctx.state = state;
    ctx.time = 0.0;
    ctx.sensor_position = Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());

    EXPECT_THROW(model_fixed_->compute_measurement(ctx), std::invalid_argument);
    EXPECT_THROW(model_fixed_->compute_jacobian(ctx), std::invalid_argument);
}

} // namespace
} // namespace sensor
