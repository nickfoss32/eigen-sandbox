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

namespace {

auto make_simplex_params() -> filtering::UnscentedTransformParameters {
    filtering::UnscentedTransformParameters params;
    params.sigma_point_scheme =
        filtering::UnscentedSigmaPointScheme::JulierSphericalSimplex;
    params.simplex_weight_0 = 0.5;
    return params;
}

class NonlinearProcessPropagator : public propagator::IPropagator {
public:
    auto propagate(double t0, const Eigen::VectorXd& initial_state, double tf) const
        -> std::vector<std::pair<double, Eigen::VectorXd>> override {
        const double dt = tf - t0;
        Eigen::VectorXd propagated(2);
        propagated(0) =
            initial_state(0) * initial_state(0) + 0.35 * initial_state(1) + 0.1 * dt;
        propagated(1) = initial_state(1) + 0.4 * initial_state(0) * initial_state(1);
        return {{t0, initial_state}, {tf, propagated}};
    }

    auto compute_transition_jacobian(double, const Eigen::VectorXd& state, double) const
        -> Eigen::MatrixXd override {
        Eigen::MatrixXd F = Eigen::MatrixXd::Identity(2, 2);
        F(0, 0) = 2.0 * state(0);
        F(0, 1) = 0.35;
        F(1, 0) = 0.4 * state(1);
        F(1, 1) = 1.0 + 0.4 * state(0);
        return F;
    }
};

class NonlinearMeasurementModel : public sensor::ISensorModel {
public:
    auto compute_measurement(const sensor::SensorContext& ctx) const -> Eigen::VectorXd override {
        Eigen::VectorXd measurement(1);
        measurement(0) =
            ctx.state(0) * ctx.state(0) + 0.25 * ctx.state(0) * ctx.state(1);
        return measurement;
    }

    auto get_noise_covariance() const -> Eigen::MatrixXd override {
        Eigen::MatrixXd R(1, 1);
        R(0, 0) = 0.05;
        return R;
    }

    auto compute_jacobian(const sensor::SensorContext& ctx) const -> Eigen::MatrixXd override {
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(1, ctx.state.size());
        H(0, 0) = 2.0 * ctx.state(0) + 0.25 * ctx.state(1);
        H(0, 1) = 0.25 * ctx.state(0);
        return H;
    }

    int get_dimension() const override { return 1; }
};

auto propagate_sigma_state(const Eigen::VectorXd& state) -> Eigen::VectorXd {
    Eigen::VectorXd propagated(2);
    propagated(0) = state(0) * state(0) + 0.35 * state(1) + 0.1;
    propagated(1) = state(1) + 0.4 * state(0) * state(1);
    return propagated;
}

auto measure_sigma_state(const Eigen::VectorXd& state) -> Eigen::VectorXd {
    Eigen::VectorXd measurement(1);
    measurement(0) = state(0) * state(0) + 0.25 * state(0) * state(1);
    return measurement;
}

} // namespace

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

        ut_params_ = make_simplex_params();
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

TEST(UnscentedKalmanFilterRegressionTest, UpdateReusesPredictedSigmaPoints) {
    Eigen::VectorXd x0(2);
    x0 << 1.1, -0.7;

    Eigen::MatrixXd P0(2, 2);
    P0 << 0.45, 0.08,
          0.08, 0.30;

    const auto ut_params = make_simplex_params();
    auto propagator = std::make_shared<NonlinearProcessPropagator>();
    auto sensor_model = std::make_shared<NonlinearMeasurementModel>();
    auto Q_func = [](double) {
        return Eigen::MatrixXd::Zero(2, 2);
    };

    filtering::UnscentedKalmanFilter ukf(
        x0,
        P0,
        propagator,
        sensor_model,
        Q_func,
        0.0,
        ut_params
    );

    ukf.predict(1.0);

    Eigen::VectorXd z(1);
    z << 1.35;
    common::Measurement measurement(z, sensor_model->get_noise_covariance(), 1.0);
    ukf.update(measurement);

    const auto predicted = filtering::UnscentedTransform::transform_distribution(
        x0,
        P0,
        propagate_sigma_state,
        Eigen::MatrixXd::Zero(2, 2),
        ut_params
    );

    std::vector<Eigen::VectorXd> sigma_measurements;
    sigma_measurements.reserve(predicted.transformed_sigma_points.size());
    for (const Eigen::VectorXd& sigma_state : predicted.transformed_sigma_points) {
        sigma_measurements.push_back(measure_sigma_state(sigma_state));
    }

    const auto measurement_moments = filtering::UnscentedTransform::recover_gaussian(
        sigma_measurements,
        predicted.weights,
        sensor_model->get_noise_covariance()
    );
    const Eigen::MatrixXd cross_covariance =
        filtering::UnscentedTransform::compute_cross_covariance(
            predicted.transformed_sigma_points,
            predicted.moments.mean,
            sigma_measurements,
            measurement_moments.mean,
            predicted.weights
        );

    const Eigen::MatrixXd kalman_gain =
        cross_covariance * measurement_moments.covariance.inverse();
    const Eigen::VectorXd expected_state =
        predicted.moments.mean + kalman_gain * (z - measurement_moments.mean);
    const Eigen::MatrixXd expected_covariance =
        0.5 * (
            predicted.moments.covariance -
            kalman_gain * measurement_moments.covariance * kalman_gain.transpose() +
            (predicted.moments.covariance -
             kalman_gain * measurement_moments.covariance * kalman_gain.transpose()
            ).transpose()
        );

    EXPECT_TRUE(ukf.get_state().isApprox(expected_state, 1e-10));
    EXPECT_TRUE(ukf.get_covariance().isApprox(expected_covariance, 1e-10));
}
