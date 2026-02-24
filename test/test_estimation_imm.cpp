#include <gtest/gtest.h>

#include <common/types.hpp>
#include <dynamics/forces/constant_acceleration_force.hpp>
#include <dynamics/forces/coordinated_turn_force.hpp>
#include <dynamics/point_mass_dynamics.hpp>
#include <estimation/imm.hpp>
#include <filtering/extended_kalman_filter.hpp>
#include <integrator/rk4.hpp>
#include <propagator/numerical_propagator.hpp>
#include <sensor/radar_sensor_model.hpp>

#include <array>
#include <memory>
#include <random>
#include <vector>

namespace {

auto make_process_noise(double q_pos, double q_vel)
    -> filtering::ExtendedKalmanFilter::ProcessNoiseFunction {
    return [q_pos, q_vel](double dt) {
        Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(6, 6);
        Q.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * (q_pos * dt);
        Q.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * (q_vel * dt);
        return Q;
    };
}

class IMMTest : public ::testing::Test {
protected:
    static constexpr double kDtSeconds = 1.0;
    static constexpr double kIntegratorStepSeconds = 0.1;

    void SetUp() override {
        truth_state_ = Eigen::VectorXd(6);
        truth_state_ << 4000.0, 0.0, 500.0,
                        120.0, 35.0, 0.0;

        initial_estimate_ = Eigen::VectorXd(6);
        initial_estimate_ << 4200.0, -150.0, 460.0,
                             100.0, 45.0, 2.0;

        initial_covariance_ = Eigen::MatrixXd::Identity(6, 6);
        initial_covariance_.block<3, 3>(0, 0) *= 300.0 * 300.0;
        initial_covariance_.block<3, 3>(3, 3) *= 20.0 * 20.0;

        sensor_model_ = std::make_shared<sensor::RadarSensorModel>(
            Eigen::Vector3d::Zero(),
            12.0,
            0.0012
        );

        auto integrator = std::make_shared<integrator::RK4Integrator>();
        auto truth_dynamics = std::make_shared<dynamics::PointMassDynamics>(
            std::vector<std::shared_ptr<dynamics::IForce>>{
                std::make_shared<dynamics::CoordinatedTurnForce>(0.028)
            }
        );
        truth_propagator_ = std::make_shared<propagator::NumericalPropagator>(
            truth_dynamics, integrator, kIntegratorStepSeconds
        );

        current_time_ = 0.0;
    }

    auto make_imm() const -> estimation::IMM {
        auto integrator = std::make_shared<integrator::RK4Integrator>();

        auto model_cv_dynamics = std::make_shared<dynamics::PointMassDynamics>(
            std::vector<std::shared_ptr<dynamics::IForce>>{}
        );
        auto model_ca_dynamics = std::make_shared<dynamics::PointMassDynamics>(
            std::vector<std::shared_ptr<dynamics::IForce>>{
                std::make_shared<dynamics::ConstantAccelerationForce>(
                    Eigen::Vector3d(-1.2, 0.5, 0.0)
                )
            }
        );
        auto model_ct_dynamics = std::make_shared<dynamics::PointMassDynamics>(
            std::vector<std::shared_ptr<dynamics::IForce>>{
                std::make_shared<dynamics::CoordinatedTurnForce>(0.024)
            }
        );

        auto model_cv_propagator = std::make_shared<propagator::NumericalPropagator>(
            model_cv_dynamics, integrator, kIntegratorStepSeconds
        );
        auto model_ca_propagator = std::make_shared<propagator::NumericalPropagator>(
            model_ca_dynamics, integrator, kIntegratorStepSeconds
        );
        auto model_ct_propagator = std::make_shared<propagator::NumericalPropagator>(
            model_ct_dynamics, integrator, kIntegratorStepSeconds
        );

        std::vector<std::unique_ptr<filtering::IKalmanFilter>> filters;
        filters.push_back(std::make_unique<filtering::ExtendedKalmanFilter>(
            initial_estimate_,
            initial_covariance_,
            model_cv_propagator,
            sensor_model_,
            make_process_noise(180.0, 4.0),
            0.0
        ));
        filters.push_back(std::make_unique<filtering::ExtendedKalmanFilter>(
            initial_estimate_,
            initial_covariance_,
            model_ca_propagator,
            sensor_model_,
            make_process_noise(70.0, 1.8),
            0.0
        ));
        filters.push_back(std::make_unique<filtering::ExtendedKalmanFilter>(
            initial_estimate_,
            initial_covariance_,
            model_ct_propagator,
            sensor_model_,
            make_process_noise(40.0, 0.8),
            0.0
        ));

        Eigen::VectorXd initial_mode_probs(3);
        initial_mode_probs << (1.0 / 3.0), (1.0 / 3.0), (1.0 / 3.0);

        Eigen::MatrixXd transition(3, 3);
        transition << 0.95, 0.025, 0.025,
                      0.025, 0.95, 0.025,
                      0.025, 0.025, 0.95;

        return estimation::IMM(std::move(filters), initial_mode_probs, transition);
    }

    auto make_measurement(std::mt19937& rng) -> common::Measurement {
        auto truth_traj = truth_propagator_->propagate(current_time_, truth_state_, current_time_ + kDtSeconds);
        truth_state_ = truth_traj.back().second;
        current_time_ += kDtSeconds;

        sensor::SensorContext ctx;
        ctx.state = truth_state_;
        ctx.time = current_time_;

        Eigen::VectorXd z = sensor_model_->compute_measurement(ctx);

        std::normal_distribution<double> range_noise(0.0, 12.0);
        std::normal_distribution<double> angle_noise(0.0, 0.0012);
        z(0) += range_noise(rng);
        z(1) += angle_noise(rng);
        z(2) += angle_noise(rng);

        return common::Measurement(z, sensor_model_->get_noise_covariance(), current_time_);
    }

    Eigen::VectorXd truth_state_;
    Eigen::VectorXd initial_estimate_;
    Eigen::MatrixXd initial_covariance_;
    double current_time_ = 0.0;
    std::shared_ptr<sensor::RadarSensorModel> sensor_model_;
    std::shared_ptr<propagator::NumericalPropagator> truth_propagator_;
};

TEST_F(IMMTest, ModeProbabilitiesStayNormalizedAndTimeIsConsistent) {
    auto imm = make_imm();
    std::mt19937 rng(42);

    for (int k = 0; k < 40; ++k) {
        const auto measurement = make_measurement(rng);
        imm.predict(kDtSeconds);
        imm.update(measurement);

        const Eigen::VectorXd mu = imm.get_model_probabilities();
        ASSERT_EQ(mu.size(), 3);
        EXPECT_TRUE(mu.allFinite());
        EXPECT_NEAR(mu.sum(), 1.0, 1e-9);
        for (int i = 0; i < mu.size(); ++i) {
            EXPECT_GE(mu(i), 0.0);
            EXPECT_LE(mu(i), 1.0);
        }

        EXPECT_NEAR(imm.get_time(), current_time_, 1e-12);
    }
}

TEST_F(IMMTest, CoordinatedTurnModelBecomesMostLikely) {
    auto imm = make_imm();
    std::mt19937 rng(42);

    for (int k = 0; k < 60; ++k) {
        const auto measurement = make_measurement(rng);
        imm.predict(kDtSeconds);
        imm.update(measurement);
    }

    const Eigen::VectorXd mu = imm.get_model_probabilities();
    EXPECT_EQ(imm.get_most_likely_model(), 2);
    EXPECT_GT(mu(2), mu(0));
    EXPECT_GT(mu(2), mu(1));
    EXPECT_GT(mu(2), 0.70);
}

TEST_F(IMMTest, CombinedEstimateTracksTruthReasonably) {
    auto imm = make_imm();
    std::mt19937 rng(42);

    for (int k = 0; k < 90; ++k) {
        const auto measurement = make_measurement(rng);
        imm.predict(kDtSeconds);
        imm.update(measurement);
    }

    const Eigen::VectorXd estimate = imm.get_state();
    const double pos_err = (estimate.head<3>() - truth_state_.head<3>()).norm();
    const double vel_err = (estimate.tail<3>() - truth_state_.tail<3>()).norm();

    EXPECT_LT(pos_err, 50.0);
    EXPECT_LT(vel_err, 8.0);
}

} // namespace
