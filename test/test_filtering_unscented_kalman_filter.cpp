#include <gtest/gtest.h>

#include "common/types.hpp"
#include "dynamics/forces/gravity.hpp"
#include "dynamics/point_mass_dynamics.hpp"
#include "filtering/unscented_kalman_filter.hpp"
#include "integrator/rk4.hpp"
#include "propagator/numerical_propagator.hpp"
#include "sensor/radar_sensor_model.hpp"

#include <Eigen/Dense>

#include <cmath>
#include <memory>
#include <random>
#include <vector>

class UnscentedKalmanFilterTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initial state: Low Earth Orbit at ~400 km altitude.
        const double earth_radius = 6.371e6;
        const double altitude = 400e3;
        const double orbital_radius = earth_radius + altitude;

        x0_ = Eigen::VectorXd(6);
        x0_ << orbital_radius, 0.0, 0.0,
               0.0, 7670.0, 0.0;

        P0_ = Eigen::MatrixXd::Identity(6, 6);
        P0_.block<3, 3>(0, 0) *= 100.0 * 100.0;
        P0_.block<3, 3>(3, 3) *= 10.0 * 10.0;

        auto gravity = std::make_shared<dynamics::PointMassGravity>();
        dynamics_ = std::make_shared<dynamics::PointMassDynamics>(
            std::vector<std::shared_ptr<dynamics::IForce>>{gravity}
        );

        auto integrator = std::make_shared<integrator::RK4Integrator>();
        propagator_ = std::make_shared<propagator::NumericalPropagator>(
            dynamics_, integrator, 0.1
        );

        sensor_model_ = std::make_shared<sensor::RadarSensorModel>(
            Eigen::Vector3d::Zero(),
            10.0,
            0.001
        );

        Q_func_ = [](double dt) {
            Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(6, 6);
            Q.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * (1.0 * dt);
            Q.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * (0.1 * dt);
            return Q;
        };

        ut_params_.alpha = 0.3;
        ut_params_.beta = 2.0;
        ut_params_.kappa = 0.0;
    }

    Eigen::VectorXd x0_;
    Eigen::MatrixXd P0_;
    std::shared_ptr<dynamics::PointMassDynamics> dynamics_;
    std::shared_ptr<propagator::NumericalPropagator> propagator_;
    std::shared_ptr<sensor::RadarSensorModel> sensor_model_;
    filtering::UnscentedKalmanFilter::ProcessNoiseFunction Q_func_;
    filtering::UnscentedTransformParameters ut_params_;
};

TEST_F(UnscentedKalmanFilterTest, ConstructorInitializesStateAndCovariance) {
    filtering::UnscentedKalmanFilter ukf(
        x0_, P0_, propagator_, sensor_model_, Q_func_, 0.0, ut_params_
    );

    EXPECT_TRUE(ukf.get_state().isApprox(x0_));
    EXPECT_TRUE(ukf.get_covariance().isApprox(P0_));
    EXPECT_DOUBLE_EQ(ukf.get_time(), 0.0);
}

TEST_F(UnscentedKalmanFilterTest, PredictUpdatesStateAndTime) {
    filtering::UnscentedKalmanFilter ukf(
        x0_, P0_, propagator_, sensor_model_, Q_func_, 0.0, ut_params_
    );

    const Eigen::VectorXd state_before = ukf.get_state();
    ukf.predict(1.0);

    const Eigen::VectorXd state_after = ukf.get_state();
    EXPECT_FALSE(state_after.isApprox(state_before, 1e-6));
    EXPECT_DOUBLE_EQ(ukf.get_time(), 1.0);
}

TEST_F(UnscentedKalmanFilterTest, UpdateReducesCovarianceTrace) {
    filtering::UnscentedKalmanFilter ukf(
        x0_, P0_, propagator_, sensor_model_, Q_func_, 0.0, ut_params_
    );

    ukf.predict(1.0);
    const double trace_before = ukf.get_covariance().trace();

    sensor::SensorContext ctx;
    ctx.state = ukf.get_state();
    ctx.time = ukf.get_time();
    const Eigen::VectorXd z = sensor_model_->compute_measurement(ctx);

    common::Measurement measurement(z, sensor_model_->get_noise_covariance(), ukf.get_time());
    ukf.update(measurement);

    const double trace_after = ukf.get_covariance().trace();
    EXPECT_LT(trace_after, trace_before);
}

TEST_F(UnscentedKalmanFilterTest, InnovationLikelihoodIsFinitePositive) {
    filtering::UnscentedKalmanFilter ukf(
        x0_, P0_, propagator_, sensor_model_, Q_func_, 0.0, ut_params_
    );

    sensor::SensorContext ctx;
    ctx.state = ukf.get_state();
    ctx.time = ukf.get_time();
    const Eigen::VectorXd z = sensor_model_->compute_measurement(ctx);

    common::Measurement measurement(z, sensor_model_->get_noise_covariance(), ukf.get_time());
    const double likelihood = ukf.get_innovation_likelihood(measurement);

    EXPECT_TRUE(std::isfinite(likelihood));
    EXPECT_GT(likelihood, 0.0);
}

TEST_F(UnscentedKalmanFilterTest, TracksOrbitWithNoisyRadarMeasurements) {
    Eigen::VectorXd x_true = x0_;

    Eigen::VectorXd x_filter(6);
    x_filter << x0_(0) + 1200.0, -300.0, 150.0,
                12.0, x0_(4) - 20.0, -8.0;

    Eigen::MatrixXd P_init = Eigen::MatrixXd::Identity(6, 6);
    P_init.block<3, 3>(0, 0) *= 1200.0 * 1200.0;
    P_init.block<3, 3>(3, 3) *= 50.0 * 50.0;

    filtering::UnscentedKalmanFilter ukf(
        x_filter,
        P_init,
        propagator_,
        sensor_model_,
        Q_func_,
        0.0,
        ut_params_
    );

    std::mt19937 rng(42);
    std::normal_distribution<double> range_noise(0.0, 10.0);
    std::normal_distribution<double> angle_noise(0.0, 0.001);

    const double dt = 1.0;
    const int num_steps = 80;

    for (int k = 0; k < num_steps; ++k) {
        const double t = k * dt;

        const auto truth_traj = propagator_->propagate(t, x_true, t + dt);
        x_true = truth_traj.back().second;

        ukf.predict(dt);

        sensor::SensorContext ctx;
        ctx.state = x_true;
        ctx.time = t + dt;
        Eigen::VectorXd z = sensor_model_->compute_measurement(ctx);
        z(0) += range_noise(rng);
        z(1) += angle_noise(rng);
        z(2) += angle_noise(rng);

        common::Measurement measurement(z, sensor_model_->get_noise_covariance(), t + dt);
        ukf.update(measurement);

        ASSERT_TRUE(ukf.get_state().allFinite());
        ASSERT_TRUE(ukf.get_covariance().allFinite());
    }

    const Eigen::VectorXd x_est = ukf.get_state();
    const double pos_err = (x_est.head<3>() - x_true.head<3>()).norm();
    const double vel_err = (x_est.tail<3>() - x_true.tail<3>()).norm();

    EXPECT_LT(pos_err, 5000.0);
    EXPECT_LT(vel_err, 80.0);
}
