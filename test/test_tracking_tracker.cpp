#include <gtest/gtest.h>

#include "common/types.hpp"
#include "dynamics/point_mass_dynamics.hpp"
#include "estimation/imm.hpp"
#include "filtering/extended_kalman_filter.hpp"
#include "filtering/kalman_filter_base.hpp"
#include "integrator/rk4.hpp"
#include "propagator/numerical_propagator.hpp"
#include "sensor/radar_sensor_model.hpp"
#include "tracking/nearest_neighbor_associator.hpp"
#include "tracking/tracker.hpp"

#include <Eigen/Dense>

#include <cmath>
#include <memory>
#include <optional>
#include <vector>

namespace {

constexpr double kRangeNoise = 10.0;
constexpr double kAngleNoise = 1e-3;

auto measurement_to_cartesian(
    const common::Measurement& measurement
) -> Eigen::Vector3d {
    const double range = measurement.z(0);
    const double azimuth = measurement.z(1);
    const double elevation = measurement.z(2);

    return measurement.sensor_position + Eigen::Vector3d(
        range * std::cos(elevation) * std::cos(azimuth),
        range * std::cos(elevation) * std::sin(azimuth),
        range * std::sin(elevation)
    );
}

class LinearTestKalmanFilter : public filtering::IKalmanFilter {
public:
    explicit LinearTestKalmanFilter(
        const Eigen::VectorXd& initial_state,
        double initial_time = 0.0
    )
        : x_(initial_state)
        , P_(Eigen::MatrixXd::Identity(initial_state.size(), initial_state.size()) * 100.0)
        , time_(initial_time) {}

    void predict(double dt) override {
        x_.head<3>() += x_.segment<3>(3) * dt;
        time_ += dt;
    }

    void update(const common::Measurement& measurement) override {
        x_.head<3>() = measurement_to_cartesian(measurement);
        time_ = measurement.time;
    }

    Eigen::VectorXd get_state() const override { return x_; }
    Eigen::MatrixXd get_covariance() const override { return P_; }
    void set_state(const Eigen::VectorXd& state) override { x_ = state; }
    void set_covariance(const Eigen::MatrixXd& covariance) override { P_ = covariance; }
    double get_time() const override { return time_; }
    double get_innovation_likelihood(const common::Measurement& measurement) const override {
        const double position_error =
            (x_.head<3>() - measurement_to_cartesian(measurement)).norm();
        return std::exp(-0.5 * position_error / 100.0);
    }

private:
    Eigen::VectorXd x_;
    Eigen::MatrixXd P_;
    double time_ = 0.0;
};

auto make_radar_measurement(
    const sensor::RadarSensorModel& sensor_model,
    const Eigen::Vector3d& sensor_position,
    const Eigen::Vector3d& target_position,
    double time_seconds
) -> common::Measurement {
    sensor::SensorContext ctx;
    ctx.state = Eigen::VectorXd::Zero(6);
    ctx.state.head<3>() = target_position;
    ctx.time = time_seconds;
    ctx.sensor_position = sensor_position;

    common::Measurement measurement(
        sensor_model.compute_measurement(ctx),
        sensor_model.get_noise_covariance(),
        time_seconds
    );
    measurement.sensor_position = sensor_position;
    measurement.sensor_orientation = Eigen::Quaterniond::Identity();
    return measurement;
}

auto make_test_track(
    int id,
    const std::shared_ptr<sensor::ISensorModel>& sensor_model,
    const Eigen::Vector3d& position
) -> tracking::Track {
    Eigen::VectorXd state(6);
    state << position, Eigen::Vector3d::Zero();
    return tracking::Track(
        id,
        std::make_unique<estimation::IMM>(
            std::make_unique<LinearTestKalmanFilter>(state)
        ),
        sensor_model
    );
}

TEST(NearestNeighborAssociatorTest, AssignsMeasurementsBySmallestMahalanobisDistance) {
    const Eigen::Vector3d radar_position = Eigen::Vector3d::Zero();
    auto sensor_model = std::make_shared<sensor::RadarSensorModel>(
        radar_position,
        kRangeNoise,
        kAngleNoise
    );

    std::vector<tracking::Track> tracks;
    tracks.push_back(make_test_track(0, sensor_model, Eigen::Vector3d(1000.0, 0.0, 0.0)));
    tracks.push_back(make_test_track(1, sensor_model, Eigen::Vector3d(0.0, 1200.0, 0.0)));

    common::MeasurementBatch measurements;
    measurements.push_back(make_radar_measurement(
        *sensor_model,
        radar_position,
        Eigen::Vector3d(1005.0, 4.0, 0.0),
        0.0
    ));
    measurements.push_back(make_radar_measurement(
        *sensor_model,
        radar_position,
        Eigen::Vector3d(3.0, 1190.0, 0.0),
        0.0
    ));

    tracking::NearestNeighborAssociator associator(25.0);
    const auto result = associator.associate(tracks, measurements);

    ASSERT_EQ(result.matches.size(), 2U);
    EXPECT_EQ(result.matches[0].track_index, 0);
    EXPECT_EQ(result.matches[0].measurement_index, 0);
    EXPECT_EQ(result.matches[1].track_index, 1);
    EXPECT_EQ(result.matches[1].measurement_index, 1);
    EXPECT_TRUE(result.unmatched_track_indices.empty());
    EXPECT_TRUE(result.unmatched_measurement_indices.empty());
}

TEST(TrackerTest, CreatesConfirmsCoastsAndDeletesTrack) {
    const Eigen::Vector3d radar_position = Eigen::Vector3d::Zero();
    auto sensor_model = std::make_shared<sensor::RadarSensorModel>(
        radar_position,
        kRangeNoise,
        kAngleNoise
    );

    tracking::Tracker tracker(
        tracking::TrackerConfig{25.0, 2, 1, 0},
        [sensor_model](const common::Measurement& measurement)
            -> std::optional<tracking::TrackSeed> {
            Eigen::VectorXd state(6);
            state << measurement_to_cartesian(measurement), Eigen::Vector3d::Zero();
            return tracking::TrackSeed{
                std::make_unique<estimation::IMM>(
                    std::make_unique<LinearTestKalmanFilter>(state, measurement.time)
                ),
                sensor_model
            };
        }
    );

    const common::Measurement measurement = make_radar_measurement(
        *sensor_model,
        radar_position,
        Eigen::Vector3d(1000.0, 0.0, 0.0),
        0.0
    );

    const auto first_step = tracker.step(0.0, {measurement});
    ASSERT_EQ(first_step.created_track_ids.size(), 1U);
    ASSERT_EQ(tracker.get_tracks().size(), 1U);
    EXPECT_EQ(tracker.get_tracks().front().get_quality(), common::TrackQuality::TENTATIVE);

    common::Measurement second_measurement = measurement;
    second_measurement.time = 1.0;
    const auto second_step = tracker.step(1.0, {second_measurement});
    ASSERT_EQ(second_step.updated_track_ids.size(), 1U);
    EXPECT_EQ(tracker.get_tracks().front().get_quality(), common::TrackQuality::CONFIRMED);

    const auto third_step = tracker.step(2.0, {});
    ASSERT_EQ(third_step.missed_track_ids.size(), 1U);
    ASSERT_EQ(tracker.get_tracks().size(), 1U);
    EXPECT_EQ(tracker.get_tracks().front().get_quality(), common::TrackQuality::COASTING);

    const auto fourth_step = tracker.step(3.0, {});
    ASSERT_EQ(fourth_step.deleted_track_ids.size(), 1U);
    EXPECT_TRUE(tracker.get_tracks().empty());
}

TEST(TrackerTest, SingleModelImmTracksSingleRadarTarget) {
    const Eigen::Vector3d radar_position = Eigen::Vector3d::Zero();
    auto sensor_model = std::make_shared<sensor::RadarSensorModel>(
        radar_position,
        5.0,
        5e-4
    );

    auto integrator = std::make_shared<integrator::RK4Integrator>();
    auto dynamics_model = std::make_shared<dynamics::PointMassDynamics>(
        std::vector<std::shared_ptr<dynamics::IForce>>{}
    );
    auto propagator = std::make_shared<propagator::NumericalPropagator>(
        dynamics_model,
        integrator,
        0.1
    );

    tracking::Tracker tracker(
        tracking::TrackerConfig{36.0, 2, 1, 0},
        [sensor_model, propagator](const common::Measurement& measurement)
            -> std::optional<tracking::TrackSeed> {
            Eigen::VectorXd initial_state(6);
            initial_state << measurement_to_cartesian(measurement), Eigen::Vector3d::Zero();

            Eigen::MatrixXd initial_covariance = Eigen::MatrixXd::Zero(6, 6);
            initial_covariance.block<3, 3>(0, 0) =
                Eigen::Matrix3d::Identity() * (100.0 * 100.0);
            initial_covariance.block<3, 3>(3, 3) =
                Eigen::Matrix3d::Identity() * (200.0 * 200.0);

            auto ekf = std::make_unique<filtering::ExtendedKalmanFilter>(
                initial_state,
                initial_covariance,
                propagator,
                sensor_model,
                [](double dt) {
                    const double sigma2 = 25.0;
                    const double dt2 = dt * dt;
                    const double dt3 = dt2 * dt;
                    const double dt4 = dt2 * dt2;
                    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(6, 6);
                    const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
                    Q.block<3, 3>(0, 0) = 0.25 * dt4 * sigma2 * I;
                    Q.block<3, 3>(0, 3) = 0.5 * dt3 * sigma2 * I;
                    Q.block<3, 3>(3, 0) = 0.5 * dt3 * sigma2 * I;
                    Q.block<3, 3>(3, 3) = dt2 * sigma2 * I;
                    return Q;
                },
                measurement.time
            );

            return tracking::TrackSeed{
                std::make_unique<estimation::IMM>(
                    std::move(ekf)
                ),
                sensor_model
            };
        }
    );

    const Eigen::Vector3d initial_position(1500.0, 400.0, 250.0);
    const Eigen::Vector3d velocity(35.0, -12.0, 6.0);

    for (int step = 0; step < 6; ++step) {
        const double time_seconds = static_cast<double>(step);
        const Eigen::Vector3d truth_position = initial_position + time_seconds * velocity;
        const common::Measurement measurement = make_radar_measurement(
            *sensor_model,
            radar_position,
            truth_position,
            time_seconds
        );
        tracker.step(time_seconds, {measurement});
    }

    ASSERT_EQ(tracker.get_tracks().size(), 1U);
    const auto& track = tracker.get_tracks().front();
    EXPECT_EQ(track.get_quality(), common::TrackQuality::CONFIRMED);
    EXPECT_EQ(track.get_imm().get_type(), "IMM(1 model)");

    const Eigen::Vector3d final_truth_position = initial_position + 5.0 * velocity;
    const double final_position_error =
        (track.get_state().head<3>() - final_truth_position).norm();
    EXPECT_LT(final_position_error, 60.0);
}

} // namespace
