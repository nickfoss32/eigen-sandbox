#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <numbers>
#include <optional>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <boost/program_options.hpp>
#include <nlohmann/json.hpp>

#include <common/types.hpp>
#include <dynamics/forces/atmospheric_drag.hpp>
#include <dynamics/forces/constant_acceleration_force.hpp>
#include <dynamics/forces/gravity.hpp>
#include <dynamics/forces/time_window_force.hpp>
#include <dynamics/point_mass_dynamics.hpp>
#include <estimation/imm.hpp>
#include <filtering/extended_kalman_filter.hpp>
#include <integrator/rk4.hpp>
#include <propagator/numerical_propagator.hpp>
#include <sensor/space_based_azel_sensor_model.hpp>
#include <tracking/nearest_neighbor_associator.hpp>
#include <tracking/track.hpp>

namespace po = boost::program_options;

namespace {

constexpr double kEarthRadiusMeters = 6378137.0;
constexpr double kEarthMu = 3.986004418e14;
constexpr double kDegToRad = std::numbers::pi / 180.0;

struct SensorKinematics {
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
    Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();
};

struct SensorSpec {
    std::string id;
    double altitude_m = 0.0;
    double lead_angle_deg = 0.0;
    double heading_deg = 0.0;
    double az_noise_sigma_rad = 0.0;
    double el_noise_sigma_rad = 0.0;
};

struct SensorPlatform {
    SensorSpec spec;
    std::function<SensorKinematics(double)> state_eci;
};

struct ModelResource {
    std::string name;
    std::string motion_model;
    std::string description;
    double process_noise_sigma_accel = 0.0;
    std::shared_ptr<propagator::IPropagator> propagator;
    filtering::ExtendedKalmanFilter::ProcessNoiseFunction process_noise;
};

struct TargetSpec {
    int id = -1;
    std::string name;
    double birth_time_seconds = 0.0;
    common::TargetType target_type = common::TargetType::MISSILE;
    Eigen::VectorXd initial_state;
    std::shared_ptr<propagator::IPropagator> truth_propagator;
};

struct TargetRuntime {
    TargetSpec spec;
    bool active = false;
    double last_time_seconds = 0.0;
    Eigen::VectorXd state;
};

struct MeasurementRecord {
    common::Measurement measurement;
    int truth_id = -1;
    bool is_clutter = false;
    Eigen::Vector3d los_world = Eigen::Vector3d::Zero();
};

struct EpochData {
    double time_seconds = 0.0;
    std::vector<std::pair<int, Eigen::VectorXd>> truth_states;
    std::vector<SensorKinematics> sensor_states;
    std::vector<common::MeasurementBatch> sensor_batches;
    std::vector<MeasurementRecord> measurements;
};

struct TriangulatedPosition {
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    double rms_cross_track_residual_m = 0.0;
    int sensors_used = 0;
};

struct InitiationCandidate {
    double time_seconds = 0.0;
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    double residual_m = 0.0;
    std::vector<common::Measurement> measurements;
};

struct PendingInitiation {
    double time_seconds = 0.0;
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    double residual_m = 0.0;
    int sensor_count = 0;
};

struct CreatedTrackRecord {
    int track_id = -1;
    std::vector<int> measurement_ids;
    double residual_m = 0.0;
    int sensors_used = 0;
    Eigen::Vector3d bootstrap_position = Eigen::Vector3d::Zero();
    Eigen::Vector3d bootstrap_velocity = Eigen::Vector3d::Zero();
};

struct DemoConfig {
    int steps = 120;
    double dt_seconds = 1.0;
    unsigned int seed = 42;
    int print_every = 10;
    int sensor_count = 3;
    double detection_probability = 0.94;
    int clutter_per_sensor = 1;
    double association_gate_squared = 18.0;
    int confirmation_updates = 2;
    int max_coast_steps = 3;
    int max_tentative_missed_steps = 1;
    double fov_half_angle_deg = 55.0;
    double clutter_az_limit_deg = 35.0;
    double clutter_el_min_deg = -10.0;
    double clutter_el_max_deg = 35.0;
    double initiation_residual_gate_m = 3500.0;
    double initiation_link_distance_m = 16000.0;
    int target_count = 2;
};

auto vector_to_json(const Eigen::VectorXd& vector) -> nlohmann::json {
    nlohmann::json values = nlohmann::json::array();
    for (int i = 0; i < vector.size(); ++i) {
        values.push_back(vector(i));
    }
    return values;
}

auto vector3_to_json(const Eigen::Vector3d& vector) -> nlohmann::json {
    return {vector.x(), vector.y(), vector.z()};
}

auto track_quality_to_string(common::TrackQuality quality) -> std::string {
    switch (quality) {
    case common::TrackQuality::TENTATIVE:
        return "TENTATIVE";
    case common::TrackQuality::CONFIRMED:
        return "CONFIRMED";
    case common::TrackQuality::COASTING:
        return "COASTING";
    case common::TrackQuality::TERMINATED:
        return "TERMINATED";
    }

    return "UNKNOWN";
}

auto target_type_to_string(common::TargetType target_type) -> std::string {
    switch (target_type) {
    case common::TargetType::UNKNOWN:
        return "UNKNOWN";
    case common::TargetType::AIRCRAFT:
        return "AIRCRAFT";
    case common::TargetType::MISSILE:
        return "MISSILE";
    case common::TargetType::SATELLITE:
        return "SATELLITE";
    case common::TargetType::DEBRIS:
        return "DEBRIS";
    }

    return "UNKNOWN";
}

auto make_process_noise(double sigma_accel) -> filtering::ExtendedKalmanFilter::ProcessNoiseFunction {
    return [sigma_accel](double dt) {
        const double sigma2 = sigma_accel * sigma_accel;
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
    };
}

auto make_propagator(
    const std::shared_ptr<integrator::RK4Integrator>& integrator,
    double integrator_step_seconds,
    std::vector<std::shared_ptr<dynamics::IForce>> forces
) -> std::shared_ptr<propagator::IPropagator> {
    auto dynamics_model = std::make_shared<dynamics::PointMassDynamics>(std::move(forces));
    return std::make_shared<propagator::NumericalPropagator>(
        dynamics_model,
        integrator,
        integrator_step_seconds
    );
}

auto make_space_sensor_kinematics(
    const Eigen::Vector3d& initial_radial_direction,
    const Eigen::Vector3d& initial_velocity_direction,
    double orbital_radius_m,
    double gravitational_parameter,
    double time_seconds
) -> SensorKinematics {
    const Eigen::Vector3d r_hat0 = initial_radial_direction.normalized();
    Eigen::Vector3d v_hat0 =
        initial_velocity_direction - initial_velocity_direction.dot(r_hat0) * r_hat0;
    v_hat0.normalize();

    const double mean_motion =
        std::sqrt(gravitational_parameter / std::pow(orbital_radius_m, 3));
    const double phase = mean_motion * time_seconds;

    const Eigen::Vector3d radial_direction =
        std::cos(phase) * r_hat0 + std::sin(phase) * v_hat0;
    const Eigen::Vector3d tangential_direction =
        -std::sin(phase) * r_hat0 + std::cos(phase) * v_hat0;

    SensorKinematics sensor_state;
    sensor_state.position = orbital_radius_m * radial_direction;
    sensor_state.velocity = orbital_radius_m * mean_motion * tangential_direction;

    const Eigen::Vector3d z_sensor = -sensor_state.position.normalized();
    Eigen::Vector3d x_sensor = sensor_state.velocity.normalized();
    Eigen::Vector3d y_sensor = z_sensor.cross(x_sensor).normalized();
    x_sensor = y_sensor.cross(z_sensor).normalized();

    Eigen::Matrix3d sensor_to_eci;
    sensor_to_eci.col(0) = x_sensor;
    sensor_to_eci.col(1) = y_sensor;
    sensor_to_eci.col(2) = z_sensor;
    sensor_state.orientation = Eigen::Quaterniond(sensor_to_eci);

    return sensor_state;
}

auto make_sensor_platform(
    const SensorSpec& spec,
    const Eigen::Vector3d& up,
    const Eigen::Matrix3d& enu_to_eci,
    const Eigen::Vector3d& downrange_unit_enu
) -> SensorPlatform {
    const double sensor_radius = kEarthRadiusMeters + spec.altitude_m;
    const double sensor_lead_angle = spec.lead_angle_deg * kDegToRad;
    const Eigen::Vector3d downrange_unit_eci = enu_to_eci * downrange_unit_enu;
    const Eigen::Vector3d sensor_initial_radial =
        (std::cos(sensor_lead_angle) * up + std::sin(sensor_lead_angle) * downrange_unit_eci)
            .normalized();

    const Eigen::Vector3d sensor_east =
        Eigen::Vector3d::UnitZ().cross(sensor_initial_radial).normalized();
    const Eigen::Vector3d sensor_north = sensor_initial_radial.cross(sensor_east).normalized();
    const double sensor_heading = spec.heading_deg * kDegToRad;
    const Eigen::Vector3d sensor_initial_velocity_direction =
        (std::sin(sensor_heading) * sensor_east + std::cos(sensor_heading) * sensor_north)
            .normalized();

    SensorPlatform platform;
    platform.spec = spec;
    platform.state_eci =
        [sensor_initial_radial, sensor_initial_velocity_direction, sensor_radius](double t) {
            return make_space_sensor_kinematics(
                sensor_initial_radial,
                sensor_initial_velocity_direction,
                sensor_radius,
                kEarthMu,
                t
            );
        };
    return platform;
}

auto azel_measurement_to_world_los(const common::Measurement& measurement)
    -> std::optional<Eigen::Vector3d> {
    if (measurement.z.size() < 2 || !measurement.sensor_position.allFinite()) {
        return std::nullopt;
    }
    if (!measurement.sensor_orientation.coeffs().allFinite() ||
        measurement.sensor_orientation.norm() <= 0.0) {
        return std::nullopt;
    }

    const double azimuth = measurement.z(0);
    const double elevation = measurement.z(1);
    const Eigen::Vector3d los_sensor(
        std::cos(elevation) * std::cos(azimuth),
        std::cos(elevation) * std::sin(azimuth),
        std::sin(elevation)
    );

    const Eigen::Quaterniond q_sensor_to_world = measurement.sensor_orientation.normalized();
    return (q_sensor_to_world * los_sensor).normalized();
}

auto triangulate_position_from_measurements(const std::vector<common::Measurement>& measurements)
    -> std::optional<TriangulatedPosition> {
    Eigen::Matrix3d normal_matrix = Eigen::Matrix3d::Zero();
    Eigen::Vector3d rhs = Eigen::Vector3d::Zero();
    int sensors_used = 0;
    double residual_sq_sum = 0.0;

    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> lines;
    for (const auto& measurement : measurements) {
        const auto los_world = azel_measurement_to_world_los(measurement);
        if (!los_world.has_value()) {
            continue;
        }

        const Eigen::Vector3d unit_los = los_world.value();
        const Eigen::Matrix3d projector =
            Eigen::Matrix3d::Identity() - unit_los * unit_los.transpose();
        normal_matrix += projector;
        rhs += projector * measurement.sensor_position;
        lines.emplace_back(measurement.sensor_position, unit_los);
        ++sensors_used;
    }

    if (sensors_used < 2) {
        return std::nullopt;
    }

    Eigen::ColPivHouseholderQR<Eigen::Matrix3d> qr(normal_matrix);
    if (qr.rank() < 3) {
        return std::nullopt;
    }

    const Eigen::Vector3d position = qr.solve(rhs);
    if (!position.allFinite()) {
        return std::nullopt;
    }

    for (const auto& [sensor_position, unit_los] : lines) {
        const Eigen::Vector3d miss_vector =
            (Eigen::Matrix3d::Identity() - unit_los * unit_los.transpose()) *
            (position - sensor_position);
        residual_sq_sum += miss_vector.squaredNorm();
    }

    TriangulatedPosition solution;
    solution.position = position;
    solution.rms_cross_track_residual_m =
        std::sqrt(residual_sq_sum / static_cast<double>(sensors_used));
    solution.sensors_used = sensors_used;
    return solution;
}

auto is_line_of_sight_clear(
    const Eigen::Vector3d& sensor_position,
    const Eigen::Vector3d& target_position
) -> bool {
    const Eigen::Vector3d segment = target_position - sensor_position;
    const double segment_norm_sq = segment.squaredNorm();
    if (!(segment_norm_sq > 0.0)) {
        return false;
    }

    const double t = std::clamp(-sensor_position.dot(segment) / segment_norm_sq, 0.0, 1.0);
    const Eigen::Vector3d closest = sensor_position + t * segment;
    return closest.norm() > kEarthRadiusMeters;
}

auto is_in_sensor_fov(
    const SensorKinematics& sensor_state,
    const Eigen::Vector3d& target_position,
    double fov_half_angle_rad
) -> bool {
    const Eigen::Vector3d los_world = target_position - sensor_state.position;
    if (!(los_world.squaredNorm() > 0.0)) {
        return false;
    }

    const Eigen::Vector3d los_sensor = sensor_state.orientation.conjugate() * los_world;
    if (!(los_sensor.x() > 0.0)) {
        return false;
    }

    const double off_boresight =
        std::atan2(std::sqrt(los_sensor.y() * los_sensor.y() + los_sensor.z() * los_sensor.z()),
                   los_sensor.x());
    return off_boresight <= fov_half_angle_rad;
}

auto build_model_resources(
    const std::shared_ptr<integrator::RK4Integrator>& integrator,
    double integrator_step_seconds
) -> std::vector<ModelResource> {
    return {
        {
            "Gravity",
            "Gravity",
            "Point-mass gravity only.",
            4.0,
            make_propagator(
                integrator,
                integrator_step_seconds,
                {std::make_shared<dynamics::PointMassGravity>()}
            ),
            make_process_noise(4.0)
        },
        {
            "GravityDrag",
            "GravityPlusDrag",
            "Point-mass gravity plus atmospheric drag.",
            6.0,
            make_propagator(
                integrator,
                integrator_step_seconds,
                {
                    std::make_shared<dynamics::PointMassGravity>(),
                    std::make_shared<dynamics::AtmosphericDrag>(900.0, 0.35, 1.1, kEarthRadiusMeters)
                }
            ),
            make_process_noise(6.0)
        },
        {
            "J2Drag",
            "J2PlusDrag",
            "J2 gravity plus atmospheric drag.",
            7.5,
            make_propagator(
                integrator,
                integrator_step_seconds,
                {
                    std::make_shared<dynamics::J2Gravity>(),
                    std::make_shared<dynamics::AtmosphericDrag>(900.0, 0.35, 1.1, kEarthRadiusMeters)
                }
            ),
            make_process_noise(7.5)
        }
    };
}

auto build_track_imm(
    const Eigen::VectorXd& initial_state,
    const Eigen::MatrixXd& initial_covariance,
    double initial_time_seconds,
    const std::shared_ptr<sensor::ISensorModel>& sensor_model,
    const std::vector<ModelResource>& model_resources
) -> std::unique_ptr<estimation::IMM> {
    std::vector<std::unique_ptr<filtering::IKalmanFilter>> filters;
    filters.reserve(model_resources.size());

    for (const auto& model : model_resources) {
        filters.push_back(std::make_unique<filtering::ExtendedKalmanFilter>(
            initial_state,
            initial_covariance,
            model.propagator,
            sensor_model,
            model.process_noise,
            initial_time_seconds
        ));
    }

    const int num_models = static_cast<int>(model_resources.size());
    Eigen::VectorXd initial_model_probabilities =
        Eigen::VectorXd::Constant(num_models, 1.0 / static_cast<double>(num_models));

    Eigen::MatrixXd transition_matrix =
        Eigen::MatrixXd::Constant(num_models, num_models, 0.08 / static_cast<double>(num_models - 1));
    for (int i = 0; i < num_models; ++i) {
        transition_matrix.row(i).setConstant(0.08 / static_cast<double>(num_models - 1));
        transition_matrix(i, i) = 0.92;
    }

    return std::make_unique<estimation::IMM>(
        std::move(filters),
        initial_model_probabilities,
        transition_matrix
    );
}

auto make_target_spec(
    int id,
    const std::string& name,
    double birth_time_seconds,
    double latitude_deg,
    double longitude_deg,
    double heading_deg,
    double flight_path_deg,
    double altitude_m,
    double speed_mps,
    double downrange_offset_m,
    double crossrange_offset_m,
    double mass_kg,
    double drag_coefficient,
    double reference_area_m2,
    double boost_duration_seconds,
    double divert_start_offset_seconds,
    double divert_duration_seconds,
    double boost_downrange_accel_mps2,
    double boost_up_accel_mps2,
    double divert_crossrange_accel_mps2,
    double divert_up_accel_mps2,
    const std::shared_ptr<integrator::RK4Integrator>& integrator,
    double integrator_step_seconds
) -> TargetSpec {
    const double latitude = latitude_deg * kDegToRad;
    const double longitude = longitude_deg * kDegToRad;
    const double heading = heading_deg * kDegToRad;
    const double flight_path = flight_path_deg * kDegToRad;

    const Eigen::Vector3d up(
        std::cos(latitude) * std::cos(longitude),
        std::cos(latitude) * std::sin(longitude),
        std::sin(latitude)
    );
    const Eigen::Vector3d east(-std::sin(longitude), std::cos(longitude), 0.0);
    const Eigen::Vector3d north = up.cross(east);

    Eigen::Matrix3d enu_to_eci;
    enu_to_eci.col(0) = east;
    enu_to_eci.col(1) = north;
    enu_to_eci.col(2) = up;

    const Eigen::Vector3d downrange_unit_enu(
        std::sin(heading),
        std::cos(heading),
        0.0
    );
    const Eigen::Vector3d crossrange_unit_enu(
        std::cos(heading),
        -std::sin(heading),
        0.0
    );

    const Eigen::Vector3d initial_offset_enu =
        downrange_offset_m * downrange_unit_enu +
        crossrange_offset_m * crossrange_unit_enu +
        Eigen::Vector3d(0.0, 0.0, altitude_m);
    const Eigen::Vector3d initial_position =
        kEarthRadiusMeters * up + enu_to_eci * initial_offset_enu;

    const Eigen::Vector3d initial_velocity_enu =
        speed_mps * std::cos(flight_path) * downrange_unit_enu +
        Eigen::Vector3d(0.0, 0.0, speed_mps * std::sin(flight_path));
    const Eigen::Vector3d initial_velocity = enu_to_eci * initial_velocity_enu;

    Eigen::VectorXd initial_state(6);
    initial_state << initial_position, initial_velocity;

    const Eigen::Vector3d boost_accel_eci =
        enu_to_eci * (boost_downrange_accel_mps2 * downrange_unit_enu +
                      Eigen::Vector3d(0.0, 0.0, boost_up_accel_mps2));
    const Eigen::Vector3d divert_accel_eci =
        enu_to_eci * (divert_crossrange_accel_mps2 * crossrange_unit_enu +
                      Eigen::Vector3d(0.0, 0.0, divert_up_accel_mps2));

    const double boost_start_seconds = birth_time_seconds;
    const double boost_end_seconds = birth_time_seconds + boost_duration_seconds;
    const double divert_start_seconds = birth_time_seconds + divert_start_offset_seconds;
    const double divert_end_seconds = divert_start_seconds + divert_duration_seconds;

    TargetSpec spec;
    spec.id = id;
    spec.name = name;
    spec.birth_time_seconds = birth_time_seconds;
    spec.target_type = common::TargetType::MISSILE;
    spec.initial_state = initial_state;
    spec.truth_propagator = make_propagator(
        integrator,
        integrator_step_seconds,
        {
            std::make_shared<dynamics::J2Gravity>(),
            std::make_shared<dynamics::AtmosphericDrag>(
                mass_kg,
                drag_coefficient,
                reference_area_m2,
                kEarthRadiusMeters
            ),
            std::make_shared<dynamics::TimeWindowForce>(
                std::make_shared<dynamics::ConstantAccelerationForce>(boost_accel_eci),
                boost_start_seconds,
                boost_end_seconds
            ),
            std::make_shared<dynamics::TimeWindowForce>(
                std::make_shared<dynamics::ConstantAccelerationForce>(divert_accel_eci),
                divert_start_seconds,
                divert_end_seconds
            )
        }
    );
    return spec;
}

auto build_targets(
    int target_count,
    const std::shared_ptr<integrator::RK4Integrator>& integrator,
    double integrator_step_seconds
) -> std::vector<TargetSpec> {
    std::vector<TargetSpec> targets;
    targets.push_back(make_target_spec(
        0,
        "Ballistic-A",
        0.0,
        28.57,
        -80.65,
        63.0,
        22.0,
        120000.0,
        6100.0,
        45000.0,
        -6000.0,
        900.0,
        0.35,
        1.1,
        26.0,
        82.0,
        12.0,
        12.0,
        5.0,
        3.5,
        -0.8,
        integrator,
        integrator_step_seconds
    ));

    if (target_count >= 2) {
        targets.push_back(make_target_spec(
            1,
            "Ballistic-B",
            28.0,
            30.10,
            -79.10,
            51.0,
            19.0,
            126000.0,
            5750.0,
            35000.0,
            8000.0,
            860.0,
            0.38,
            1.0,
            24.0,
            70.0,
            14.0,
            10.5,
            4.0,
            -4.0,
            -0.5,
            integrator,
            integrator_step_seconds
        ));
    }

    return targets;
}

auto build_sensor_platforms(int sensor_count) -> std::vector<SensorPlatform> {
    constexpr double kLaunchLatitudeDeg = 28.573255;
    constexpr double kLaunchLongitudeDeg = -80.646895;
    constexpr double kLaunchHeadingDeg = 64.0;
    constexpr double kAzNoiseRad = 70e-6;
    constexpr double kElNoiseRad = 70e-6;

    const double launch_lat = kLaunchLatitudeDeg * kDegToRad;
    const double launch_lon = kLaunchLongitudeDeg * kDegToRad;
    const double launch_heading = kLaunchHeadingDeg * kDegToRad;

    const Eigen::Vector3d up(
        std::cos(launch_lat) * std::cos(launch_lon),
        std::cos(launch_lat) * std::sin(launch_lon),
        std::sin(launch_lat)
    );
    const Eigen::Vector3d east(-std::sin(launch_lon), std::cos(launch_lon), 0.0);
    const Eigen::Vector3d north = up.cross(east);
    Eigen::Matrix3d enu_to_eci;
    enu_to_eci.col(0) = east;
    enu_to_eci.col(1) = north;
    enu_to_eci.col(2) = up;

    const Eigen::Vector3d downrange_unit_enu(
        std::sin(launch_heading),
        std::cos(launch_heading),
        0.0
    );

    const std::vector<SensorSpec> sensor_catalog = {
        {"S1", 900000.0, 7.0, 96.0, kAzNoiseRad, kElNoiseRad},
        {"S2", 980000.0, 19.0, 128.0, kAzNoiseRad, kElNoiseRad},
        {"S3", 860000.0, -11.0, 62.0, kAzNoiseRad, kElNoiseRad},
        {"S4", 1040000.0, 28.0, 156.0, kAzNoiseRad, kElNoiseRad},
        {"S5", 930000.0, -24.0, 28.0, kAzNoiseRad, kElNoiseRad}
    };

    std::vector<SensorPlatform> sensor_platforms;
    sensor_platforms.reserve(static_cast<std::size_t>(sensor_count));
    for (int i = 0; i < sensor_count; ++i) {
        sensor_platforms.push_back(make_sensor_platform(
            sensor_catalog.at(static_cast<std::size_t>(i)),
            up,
            enu_to_eci,
            downrange_unit_enu
        ));
    }
    return sensor_platforms;
}

auto simulate_epochs(
    const DemoConfig& config,
    const std::vector<TargetSpec>& targets,
    const std::vector<SensorPlatform>& sensor_platforms,
    const std::shared_ptr<sensor::SpaceBasedAzElSensorModel>& sensor_model
) -> std::vector<EpochData> {
    std::vector<EpochData> epochs;
    epochs.reserve(static_cast<std::size_t>(config.steps));

    std::vector<TargetRuntime> runtimes;
    runtimes.reserve(targets.size());
    for (const auto& target : targets) {
        TargetRuntime runtime;
        runtime.spec = target;
        runtime.state = target.initial_state;
        runtimes.push_back(std::move(runtime));
    }

    std::mt19937 rng(config.seed);
    std::uniform_real_distribution<double> unit_uniform(0.0, 1.0);
    std::uniform_real_distribution<double> clutter_azimuth(
        -config.clutter_az_limit_deg * kDegToRad,
        config.clutter_az_limit_deg * kDegToRad
    );
    std::uniform_real_distribution<double> clutter_elevation(
        config.clutter_el_min_deg * kDegToRad,
        config.clutter_el_max_deg * kDegToRad
    );

    int next_measurement_id = 0;
    for (int step = 0; step < config.steps; ++step) {
        const double time_seconds = (step + 1) * config.dt_seconds;
        EpochData epoch;
        epoch.time_seconds = time_seconds;
        epoch.sensor_states.reserve(sensor_platforms.size());
        epoch.sensor_batches.resize(sensor_platforms.size());

        for (const auto& sensor_platform : sensor_platforms) {
            epoch.sensor_states.push_back(sensor_platform.state_eci(time_seconds));
        }

        for (auto& runtime : runtimes) {
            if (!runtime.active && time_seconds >= runtime.spec.birth_time_seconds) {
                runtime.active = true;
                runtime.state = runtime.spec.initial_state;
                runtime.last_time_seconds = runtime.spec.birth_time_seconds;
            }

            if (!runtime.active) {
                continue;
            }

            if (time_seconds > runtime.last_time_seconds) {
                const auto truth_trajectory = runtime.spec.truth_propagator->propagate(
                    runtime.last_time_seconds,
                    runtime.state,
                    time_seconds
                );
                runtime.state = truth_trajectory.back().second;
                runtime.last_time_seconds = time_seconds;
            }

            epoch.truth_states.push_back({runtime.spec.id, runtime.state});

            for (int sensor_index = 0;
                 sensor_index < static_cast<int>(sensor_platforms.size());
                 ++sensor_index) {
                const auto& sensor_platform =
                    sensor_platforms.at(static_cast<std::size_t>(sensor_index));
                const SensorKinematics& sensor_state =
                    epoch.sensor_states.at(static_cast<std::size_t>(sensor_index));

                if (!is_line_of_sight_clear(sensor_state.position, runtime.state.head<3>()) ||
                    unit_uniform(rng) > config.detection_probability) {
                    continue;
                }

                sensor::SensorContext ctx;
                ctx.state = runtime.state;
                ctx.time = time_seconds;
                ctx.sensor_position = sensor_state.position;
                ctx.sensor_orientation = sensor_state.orientation;

                Eigen::VectorXd measurement_vector = sensor_model->compute_measurement(ctx);
                std::normal_distribution<double> az_noise(
                    0.0,
                    sensor_platform.spec.az_noise_sigma_rad
                );
                std::normal_distribution<double> el_noise(
                    0.0,
                    sensor_platform.spec.el_noise_sigma_rad
                );
                measurement_vector(0) += az_noise(rng);
                measurement_vector(1) += el_noise(rng);

                common::Measurement measurement(
                    measurement_vector,
                    sensor_model->get_noise_covariance(),
                    time_seconds
                );
                measurement.sensor_position = sensor_state.position;
                measurement.sensor_orientation = sensor_state.orientation;
                measurement.sensor_id = sensor_platform.spec.id;
                measurement.measurement_id = next_measurement_id++;

                epoch.sensor_batches[static_cast<std::size_t>(sensor_index)].push_back(measurement);
                epoch.measurements.push_back({
                    measurement,
                    runtime.spec.id,
                    false,
                    (runtime.state.head<3>() - sensor_state.position).normalized()
                });
            }
        }

        for (int sensor_index = 0; sensor_index < static_cast<int>(sensor_platforms.size()); ++sensor_index) {
            const auto& sensor_platform = sensor_platforms.at(static_cast<std::size_t>(sensor_index));
            const SensorKinematics& sensor_state =
                epoch.sensor_states.at(static_cast<std::size_t>(sensor_index));

            for (int clutter_idx = 0; clutter_idx < config.clutter_per_sensor; ++clutter_idx) {
                Eigen::Vector2d clutter_measurement;
                clutter_measurement(0) = clutter_azimuth(rng);
                clutter_measurement(1) = clutter_elevation(rng);

                common::Measurement measurement(
                    clutter_measurement,
                    sensor_model->get_noise_covariance(),
                    time_seconds
                );
                measurement.sensor_position = sensor_state.position;
                measurement.sensor_orientation = sensor_state.orientation;
                measurement.sensor_id = sensor_platform.spec.id;
                measurement.measurement_id = next_measurement_id++;

                const double azimuth = clutter_measurement(0);
                const double elevation = clutter_measurement(1);
                const Eigen::Vector3d los_sensor(
                    std::cos(elevation) * std::cos(azimuth),
                    std::cos(elevation) * std::sin(azimuth),
                    std::sin(elevation)
                );
                const Eigen::Vector3d los_world =
                    (sensor_state.orientation * los_sensor).normalized();

                epoch.sensor_batches[static_cast<std::size_t>(sensor_index)].push_back(measurement);
                epoch.measurements.push_back({measurement, -1, true, los_world});
            }
        }

        epochs.push_back(std::move(epoch));
    }

    return epochs;
}

void update_track_quality(
    tracking::Track& track,
    const DemoConfig& config
) {
    if (track.get_total_updates() >= config.confirmation_updates &&
        track.get_consecutive_misses() == 0) {
        track.set_quality(common::TrackQuality::CONFIRMED);
        return;
    }

    if (track.get_total_updates() < config.confirmation_updates) {
        if (track.get_consecutive_misses() > config.max_tentative_missed_steps) {
            track.set_quality(common::TrackQuality::TERMINATED);
        } else {
            track.set_quality(common::TrackQuality::TENTATIVE);
        }
        return;
    }

    if (track.get_consecutive_misses() == 0) {
        track.set_quality(common::TrackQuality::CONFIRMED);
    } else if (track.get_consecutive_misses() <= config.max_coast_steps) {
        track.set_quality(common::TrackQuality::COASTING);
    } else {
        track.set_quality(common::TrackQuality::TERMINATED);
    }
}

auto generate_initiation_candidates(
    double time_seconds,
    const std::vector<common::Measurement>& unmatched_measurements,
    double residual_gate_m
) -> std::vector<InitiationCandidate> {
    std::map<std::string, std::vector<common::Measurement>> by_sensor;
    for (const auto& measurement : unmatched_measurements) {
        by_sensor[measurement.sensor_id].push_back(measurement);
    }

    std::vector<std::string> sensor_ids;
    sensor_ids.reserve(by_sensor.size());
    for (const auto& [sensor_id, _] : by_sensor) {
        sensor_ids.push_back(sensor_id);
    }

    const int combo_size = std::min(3, static_cast<int>(sensor_ids.size()));
    if (combo_size < 2) {
        return {};
    }

    std::vector<std::vector<std::string>> sensor_subsets;
    std::vector<std::string> current_subset;
    std::function<void(int)> build_subsets = [&](int start_index) {
        if (static_cast<int>(current_subset.size()) == combo_size) {
            sensor_subsets.push_back(current_subset);
            return;
        }
        for (int i = start_index; i < static_cast<int>(sensor_ids.size()); ++i) {
            current_subset.push_back(sensor_ids[static_cast<std::size_t>(i)]);
            build_subsets(i + 1);
            current_subset.pop_back();
        }
    };
    build_subsets(0);

    std::vector<InitiationCandidate> candidates;
    for (const auto& sensor_subset : sensor_subsets) {
        std::vector<common::Measurement> current_combo;
        std::function<void(int)> enumerate_measurements = [&](int depth) {
            if (depth == static_cast<int>(sensor_subset.size())) {
                const auto triangulated = triangulate_position_from_measurements(current_combo);
                if (!triangulated.has_value()) {
                    return;
                }
                const double altitude_m = triangulated->position.norm() - kEarthRadiusMeters;
                if (triangulated->rms_cross_track_residual_m > residual_gate_m ||
                    altitude_m < 70000.0) {
                    return;
                }

                candidates.push_back({
                    time_seconds,
                    triangulated->position,
                    triangulated->rms_cross_track_residual_m,
                    current_combo
                });
                return;
            }

            const auto& measurements = by_sensor.at(sensor_subset[static_cast<std::size_t>(depth)]);
            for (const auto& measurement : measurements) {
                current_combo.push_back(measurement);
                enumerate_measurements(depth + 1);
                current_combo.pop_back();
            }
        };
        enumerate_measurements(0);
    }

    std::sort(
        candidates.begin(),
        candidates.end(),
        [](const InitiationCandidate& lhs, const InitiationCandidate& rhs) {
            if (lhs.measurements.size() != rhs.measurements.size()) {
                return lhs.measurements.size() > rhs.measurements.size();
            }
            return lhs.residual_m < rhs.residual_m;
        }
    );

    std::vector<InitiationCandidate> accepted;
    std::set<int> used_measurement_ids;
    for (const auto& candidate : candidates) {
        bool reuses_measurement = false;
        for (const auto& measurement : candidate.measurements) {
            if (used_measurement_ids.contains(measurement.measurement_id)) {
                reuses_measurement = true;
                break;
            }
        }
        if (reuses_measurement) {
            continue;
        }

        bool too_close_to_existing = false;
        for (const auto& existing : accepted) {
            if ((existing.position - candidate.position).norm() < 5000.0) {
                too_close_to_existing = true;
                break;
            }
        }
        if (too_close_to_existing) {
            continue;
        }

        accepted.push_back(candidate);
        for (const auto& measurement : candidate.measurements) {
            used_measurement_ids.insert(measurement.measurement_id);
        }
    }

    return accepted;
}

auto find_best_truth_match(
    const Eigen::Vector3d& position,
    const std::vector<std::pair<int, Eigen::VectorXd>>& truth_states
) -> std::pair<int, double> {
    int best_truth_id = -1;
    double best_error_m = std::numeric_limits<double>::infinity();
    for (const auto& [truth_id, truth_state] : truth_states) {
        const double error_m = (position - truth_state.head<3>()).norm();
        if (error_m < best_error_m) {
            best_error_m = error_m;
            best_truth_id = truth_id;
        }
    }

    if (!std::isfinite(best_error_m)) {
        return {-1, 0.0};
    }
    return {best_truth_id, best_error_m};
}

auto build_track_point_json(
    const tracking::Track& track,
    const std::vector<std::pair<int, Eigen::VectorXd>>& truth_states,
    const std::vector<ModelResource>& model_resources
) -> nlohmann::json {
    const Eigen::VectorXd state = track.get_state();
    const auto [nearest_truth_id, position_error_m] =
        find_best_truth_match(state.head<3>(), truth_states);

    nlohmann::json point = {
        {"time_seconds", track.get_time()},
        {"state", vector_to_json(state)},
        {"position", vector3_to_json(state.head<3>())},
        {"velocity", vector3_to_json(state.segment<3>(3))},
        {"quality", track_quality_to_string(track.get_quality())},
        {"total_updates", track.get_total_updates()},
        {"consecutive_hits", track.get_consecutive_hits()},
        {"consecutive_misses", track.get_consecutive_misses()},
        {"age_steps", track.get_age_steps()},
        {"nearest_truth_id", nearest_truth_id},
        {"position_error_m", position_error_m},
        {"most_likely_model_index", track.get_imm().get_most_likely_model()}
    };

    const Eigen::VectorXd model_probabilities = track.get_imm().get_model_probabilities();
    point["model_probabilities"] = vector_to_json(model_probabilities);

    if (track.get_imm().get_most_likely_model() >= 0 &&
        track.get_imm().get_most_likely_model() < static_cast<int>(model_resources.size())) {
        point["most_likely_model_name"] =
            model_resources[static_cast<std::size_t>(track.get_imm().get_most_likely_model())].name;
    } else {
        point["most_likely_model_name"] = "";
    }

    return point;
}

} // namespace

int main(int argc, char* argv[]) {
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "Produce help message")
        ("output,o", po::value<std::string>()->default_value("tracker_demo.json"),
            "Output JSON file with tracker demo data")
        ("steps", po::value<int>()->default_value(120),
            "Number of sensor epochs to simulate")
        ("dt", po::value<double>()->default_value(1.0),
            "Sensor update period in seconds")
        ("seed", po::value<unsigned int>()->default_value(42),
            "Random seed for measurements and clutter")
        ("print-every", po::value<int>()->default_value(10),
            "Print a console summary every N epochs")
        ("sensor-count", po::value<int>()->default_value(3),
            "Number of space-based az/el sensors to use (2 to 5)")
        ("target-count", po::value<int>()->default_value(2),
            "Number of truth targets to simulate (1 or 2)")
        ("detection-probability", po::value<double>()->default_value(0.94),
            "Per-sensor probability of detection for visible truth targets")
        ("clutter-per-sensor", po::value<int>()->default_value(1),
            "Number of angle-only clutter measurements per sensor per epoch");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    if (vm.count("help")) {
        std::cout << desc << '\n';
        return 0;
    }

    po::notify(vm);

    DemoConfig config;
    config.steps = vm["steps"].as<int>();
    config.dt_seconds = vm["dt"].as<double>();
    config.seed = vm["seed"].as<unsigned int>();
    config.print_every = vm["print-every"].as<int>();
    config.sensor_count = vm["sensor-count"].as<int>();
    config.target_count = vm["target-count"].as<int>();
    config.detection_probability = vm["detection-probability"].as<double>();
    config.clutter_per_sensor = vm["clutter-per-sensor"].as<int>();

    if (config.steps <= 0 || config.dt_seconds <= 0.0 || config.print_every <= 0) {
        std::cerr << "steps, dt, and print-every must be positive\n";
        return 1;
    }
    if (config.sensor_count < 2 || config.sensor_count > 5) {
        std::cerr << "sensor-count must be between 2 and 5 for angle-only triangulation\n";
        return 1;
    }
    if (config.target_count < 1 || config.target_count > 2) {
        std::cerr << "target-count must be 1 or 2\n";
        return 1;
    }
    if (config.detection_probability <= 0.0 || config.detection_probability > 1.0) {
        std::cerr << "detection-probability must be in (0, 1]\n";
        return 1;
    }
    if (config.clutter_per_sensor < 0) {
        std::cerr << "clutter-per-sensor must be non-negative\n";
        return 1;
    }

    constexpr double kIntegratorStepSeconds = 0.1;
    const std::string output_file = vm["output"].as<std::string>();

    auto integrator = std::make_shared<integrator::RK4Integrator>();
    auto sensor_model = std::make_shared<sensor::SpaceBasedAzElSensorModel>(70e-6, 70e-6);
    const std::vector<SensorPlatform> sensor_platforms = build_sensor_platforms(config.sensor_count);
    const std::vector<TargetSpec> targets =
        build_targets(config.target_count, integrator, kIntegratorStepSeconds);
    const std::vector<ModelResource> model_resources =
        build_model_resources(integrator, kIntegratorStepSeconds);

    const std::vector<EpochData> epochs =
        simulate_epochs(config, targets, sensor_platforms, sensor_model);

    tracking::NearestNeighborAssociator associator(config.association_gate_squared);
    std::vector<tracking::Track> tracks;
    std::vector<PendingInitiation> pending_initiations;
    int next_track_id = 0;

    nlohmann::json truth_json = nlohmann::json::array();
    std::map<int, nlohmann::json> truth_history_by_id;
    for (const auto& target : targets) {
        truth_history_by_id[target.id] = nlohmann::json{
            {"truth_id", target.id},
            {"name", target.name},
            {"target_type", target_type_to_string(target.target_type)},
            {"birth_time_seconds", target.birth_time_seconds},
            {"points", nlohmann::json::array()}
        };
    }

    nlohmann::json sensor_json = nlohmann::json::array();
    for (const auto& sensor_platform : sensor_platforms) {
        sensor_json.push_back({
            {"sensor_id", sensor_platform.spec.id},
            {"altitude_m", sensor_platform.spec.altitude_m},
            {"lead_angle_deg", sensor_platform.spec.lead_angle_deg},
            {"heading_deg", sensor_platform.spec.heading_deg},
            {"points", nlohmann::json::array()}
        });
    }

    nlohmann::json measurement_json = nlohmann::json::array();
    std::map<int, nlohmann::json> track_history_by_id;
    nlohmann::json steps_json = nlohmann::json::array();

    int total_created_tracks = 0;
    int total_deleted_tracks = 0;
    int total_confirmed_snapshots = 0;
    double confirmed_position_error_sum_m = 0.0;
    int max_active_tracks = 0;

    std::cout << "Tracker demo with space-based az/el sensors only\n";
    std::cout << "Targets: " << targets.size()
              << ", sensors: " << sensor_platforms.size()
              << ", track IMM models: ";
    for (std::size_t i = 0; i < model_resources.size(); ++i) {
        std::cout << model_resources[i].name;
        if (i + 1 != model_resources.size()) {
            std::cout << ", ";
        }
    }
    std::cout << '\n';

    for (int epoch_index = 0; epoch_index < static_cast<int>(epochs.size()); ++epoch_index) {
        const EpochData& epoch = epochs[static_cast<std::size_t>(epoch_index)];

        for (const auto& [truth_id, truth_state] : epoch.truth_states) {
            truth_history_by_id[truth_id]["points"].push_back({
                {"time_seconds", epoch.time_seconds},
                {"state", vector_to_json(truth_state)},
                {"position", vector3_to_json(truth_state.head<3>())},
                {"velocity", vector3_to_json(truth_state.segment<3>(3))}
            });
        }

        for (int sensor_index = 0; sensor_index < static_cast<int>(sensor_platforms.size()); ++sensor_index) {
            const SensorKinematics& sensor_state =
                epoch.sensor_states.at(static_cast<std::size_t>(sensor_index));
            sensor_json[static_cast<std::size_t>(sensor_index)]["points"].push_back({
                {"time_seconds", epoch.time_seconds},
                {"position", vector3_to_json(sensor_state.position)},
                {"velocity", vector3_to_json(sensor_state.velocity)}
            });
        }

        for (const auto& measurement_record : epoch.measurements) {
            measurement_json.push_back({
                {"measurement_id", measurement_record.measurement.measurement_id},
                {"time_seconds", measurement_record.measurement.time},
                {"sensor_id", measurement_record.measurement.sensor_id},
                {"z", vector_to_json(measurement_record.measurement.z)},
                {"sensor_position", vector3_to_json(measurement_record.measurement.sensor_position)},
                {"los_world", vector3_to_json(measurement_record.los_world)},
                {"truth_id", measurement_record.truth_id >= 0
                                 ? nlohmann::json(measurement_record.truth_id)
                                 : nlohmann::json(nullptr)},
                {"is_clutter", measurement_record.is_clutter}
            });
        }

        for (auto& track : tracks) {
            track.predict_to(epoch.time_seconds);
        }

        std::vector<bool> updated_this_epoch(tracks.size(), false);
        nlohmann::json sensor_events_json = nlohmann::json::array();
        std::vector<common::Measurement> unmatched_measurements_for_initiation;

        for (int sensor_index = 0; sensor_index < static_cast<int>(epoch.sensor_batches.size()); ++sensor_index) {
            const common::MeasurementBatch& sensor_batch =
                epoch.sensor_batches.at(static_cast<std::size_t>(sensor_index));
            std::vector<int> track_ids_before;
            track_ids_before.reserve(tracks.size());
            for (const auto& track : tracks) {
                track_ids_before.push_back(track.get_id());
            }

            const auto association_result = associator.associate(tracks, sensor_batch);
            nlohmann::json sensor_event = {
                {"sensor_id", sensor_platforms.at(static_cast<std::size_t>(sensor_index)).spec.id},
                {"measurement_ids", nlohmann::json::array()},
                {"matches", nlohmann::json::array()},
                {"unmatched_measurement_ids", nlohmann::json::array()}
            };

            for (const auto& measurement : sensor_batch) {
                sensor_event["measurement_ids"].push_back(measurement.measurement_id);
            }

            for (const auto& match : association_result.matches) {
                const common::Measurement& measurement =
                    sensor_batch.at(static_cast<std::size_t>(match.measurement_index));
                tracks[static_cast<std::size_t>(match.track_index)].assimilate_measurement(measurement);
                updated_this_epoch[static_cast<std::size_t>(match.track_index)] = true;
                sensor_event["matches"].push_back({
                    {"track_id", track_ids_before.at(static_cast<std::size_t>(match.track_index))},
                    {"measurement_id", measurement.measurement_id},
                    {"squared_mahalanobis", match.squared_mahalanobis}
                });
            }

            for (const int unmatched_index : association_result.unmatched_measurement_indices) {
                const common::Measurement& measurement =
                    sensor_batch.at(static_cast<std::size_t>(unmatched_index));
                unmatched_measurements_for_initiation.push_back(measurement);
                sensor_event["unmatched_measurement_ids"].push_back(measurement.measurement_id);
            }

            sensor_events_json.push_back(sensor_event);
        }

        nlohmann::json missed_track_ids = nlohmann::json::array();
        for (int track_index = 0; track_index < static_cast<int>(tracks.size()); ++track_index) {
            tracks[static_cast<std::size_t>(track_index)].finalize_epoch(
                updated_this_epoch[static_cast<std::size_t>(track_index)]
            );
            update_track_quality(tracks[static_cast<std::size_t>(track_index)], config);
            if (!updated_this_epoch[static_cast<std::size_t>(track_index)]) {
                missed_track_ids.push_back(tracks[static_cast<std::size_t>(track_index)].get_id());
            }
        }

        const std::vector<InitiationCandidate> candidates = generate_initiation_candidates(
            epoch.time_seconds,
            unmatched_measurements_for_initiation,
            config.initiation_residual_gate_m
        );

        std::vector<PendingInitiation> next_pending_initiations;
        nlohmann::json created_tracks_json = nlohmann::json::array();
        for (const auto& candidate : candidates) {
            const auto best_pending_it = std::min_element(
                pending_initiations.begin(),
                pending_initiations.end(),
                [&](const PendingInitiation& lhs, const PendingInitiation& rhs) {
                    return (candidate.position - lhs.position).norm() <
                           (candidate.position - rhs.position).norm();
                }
            );

            const bool has_link =
                best_pending_it != pending_initiations.end() &&
                (candidate.position - best_pending_it->position).norm() <=
                    config.initiation_link_distance_m &&
                epoch.time_seconds > best_pending_it->time_seconds;

            if (!has_link) {
                next_pending_initiations.push_back({
                    candidate.time_seconds,
                    candidate.position,
                    candidate.residual_m,
                    static_cast<int>(candidate.measurements.size())
                });
                continue;
            }

            const double dt_seconds = candidate.time_seconds - best_pending_it->time_seconds;
            const Eigen::Vector3d velocity =
                (candidate.position - best_pending_it->position) / dt_seconds;

            const double position_sigma_m = std::clamp(
                3.0 * std::max(candidate.residual_m, best_pending_it->residual_m),
                250.0,
                2500.0
            );
            const double velocity_sigma_mps = std::clamp(
                std::sqrt(2.0) * position_sigma_m / dt_seconds,
                80.0,
                700.0
            );

            Eigen::VectorXd initial_state(6);
            initial_state << candidate.position, velocity;
            Eigen::MatrixXd initial_covariance = Eigen::MatrixXd::Zero(6, 6);
            initial_covariance.block<3, 3>(0, 0) =
                Eigen::Matrix3d::Identity() * (position_sigma_m * position_sigma_m);
            initial_covariance.block<3, 3>(3, 3) =
                Eigen::Matrix3d::Identity() * (velocity_sigma_mps * velocity_sigma_mps);

            auto imm = build_track_imm(
                initial_state,
                initial_covariance,
                epoch.time_seconds,
                sensor_model,
                model_resources
            );

            tracking::Track track(
                next_track_id,
                std::move(imm),
                sensor_model,
                common::TrackQuality::CONFIRMED,
                common::TargetType::MISSILE,
                2
            );
            update_track_quality(track, config);

            std::vector<int> measurement_ids;
            measurement_ids.reserve(candidate.measurements.size());
            for (const auto& measurement : candidate.measurements) {
                measurement_ids.push_back(measurement.measurement_id);
            }

            created_tracks_json.push_back({
                {"track_id", next_track_id},
                {"measurement_ids", measurement_ids},
                {"sensors_used", static_cast<int>(candidate.measurements.size())},
                {"triangulation_residual_m", candidate.residual_m},
                {"bootstrap_position", vector3_to_json(candidate.position)},
                {"bootstrap_velocity", vector3_to_json(velocity)}
            });

            track_history_by_id[next_track_id] = {
                {"track_id", next_track_id},
                {"target_type", target_type_to_string(common::TargetType::MISSILE)},
                {"points", nlohmann::json::array()}
            };

            tracks.push_back(std::move(track));
            ++next_track_id;
            ++total_created_tracks;
        }

        pending_initiations = std::move(next_pending_initiations);

        nlohmann::json deleted_track_ids = nlohmann::json::array();
        std::vector<tracking::Track> surviving_tracks;
        surviving_tracks.reserve(tracks.size());
        for (auto& track : tracks) {
            if (track.get_quality() == common::TrackQuality::TERMINATED) {
                deleted_track_ids.push_back(track.get_id());
                ++total_deleted_tracks;
                continue;
            }
            surviving_tracks.push_back(std::move(track));
        }
        tracks = std::move(surviving_tracks);

        nlohmann::json active_track_ids = nlohmann::json::array();
        for (const auto& track : tracks) {
            active_track_ids.push_back(track.get_id());
            if (!track_history_by_id.contains(track.get_id())) {
                track_history_by_id[track.get_id()] = {
                    {"track_id", track.get_id()},
                    {"target_type", target_type_to_string(track.get_target_type())},
                    {"points", nlohmann::json::array()}
                };
            }

            const nlohmann::json point = build_track_point_json(
                track,
                epoch.truth_states,
                model_resources
            );
            track_history_by_id[track.get_id()]["points"].push_back(point);

            if (track.get_quality() == common::TrackQuality::CONFIRMED) {
                confirmed_position_error_sum_m += point["position_error_m"].get<double>();
                ++total_confirmed_snapshots;
            }
        }

        max_active_tracks = std::max(max_active_tracks, static_cast<int>(tracks.size()));

        steps_json.push_back({
            {"time_seconds", epoch.time_seconds},
            {"truth_ids", [&epoch]() {
                nlohmann::json ids = nlohmann::json::array();
                for (const auto& [truth_id, _] : epoch.truth_states) {
                    ids.push_back(truth_id);
                }
                return ids;
            }()},
            {"sensor_events", sensor_events_json},
            {"missed_track_ids", missed_track_ids},
            {"created_tracks", created_tracks_json},
            {"deleted_track_ids", deleted_track_ids},
            {"active_track_ids", active_track_ids},
            {"pending_initiations", [&pending_initiations]() {
                nlohmann::json pending_json = nlohmann::json::array();
                for (const auto& pending : pending_initiations) {
                    pending_json.push_back({
                        {"time_seconds", pending.time_seconds},
                        {"position", vector3_to_json(pending.position)},
                        {"residual_m", pending.residual_m},
                        {"sensor_count", pending.sensor_count}
                    });
                }
                return pending_json;
            }()}
        });

        if ((epoch_index + 1) % config.print_every == 0 || epoch_index + 1 == config.steps) {
            int confirmed_tracks = 0;
            int coasting_tracks = 0;
            for (const auto& track : tracks) {
                if (track.get_quality() == common::TrackQuality::CONFIRMED) {
                    ++confirmed_tracks;
                } else if (track.get_quality() == common::TrackQuality::COASTING) {
                    ++coasting_tracks;
                }
            }

            std::cout << std::fixed << std::setprecision(1)
                      << "t=" << epoch.time_seconds << " s"
                      << "  active=" << tracks.size()
                      << "  confirmed=" << confirmed_tracks
                      << "  coasting=" << coasting_tracks
                      << "  created=" << created_tracks_json.size()
                      << "  deleted=" << deleted_track_ids.size()
                      << "  pending=" << pending_initiations.size()
                      << '\n';
        }
    }

    for (auto& [_, truth] : truth_history_by_id) {
        truth_json.push_back(truth);
    }

    nlohmann::json track_histories_json = nlohmann::json::array();
    for (auto& [track_id, history] : track_history_by_id) {
        std::map<int, int> truth_counts;
        double error_sum_m = 0.0;
        int error_samples = 0;
        for (const auto& point : history["points"]) {
            const int nearest_truth_id = point["nearest_truth_id"].get<int>();
            if (nearest_truth_id >= 0) {
                ++truth_counts[nearest_truth_id];
                error_sum_m += point["position_error_m"].get<double>();
                ++error_samples;
            }
        }

        int dominant_truth_id = -1;
        int dominant_count = 0;
        for (const auto& [truth_id, count] : truth_counts) {
            if (count > dominant_count) {
                dominant_count = count;
                dominant_truth_id = truth_id;
            }
        }

        history["dominant_truth_id"] =
            dominant_truth_id >= 0 ? nlohmann::json(dominant_truth_id) : nlohmann::json(nullptr);
        history["mean_position_error_m"] =
            (error_samples > 0) ? error_sum_m / static_cast<double>(error_samples) : 0.0;
        track_histories_json.push_back(history);
    }

    nlohmann::json model_summary_json = nlohmann::json::array();
    for (int i = 0; i < static_cast<int>(model_resources.size()); ++i) {
        const auto& model = model_resources[static_cast<std::size_t>(i)];
        model_summary_json.push_back({
            {"index", i},
            {"name", model.name},
            {"motion_model", model.motion_model},
            {"description", model.description},
            {"process_noise_sigma_accel", model.process_noise_sigma_accel}
        });
    }

    nlohmann::json data_json;
    data_json["truth_targets"] = truth_json;
    data_json["sensors"] = sensor_json;
    data_json["measurements"] = measurement_json;
    data_json["track_histories"] = track_histories_json;
    data_json["steps"] = steps_json;
    data_json["summary"] = {
        {"scenario",
         {
             {"steps", config.steps},
             {"dt_seconds", config.dt_seconds},
             {"seed", config.seed},
             {"target_count", static_cast<int>(targets.size())},
             {"sensor_count", static_cast<int>(sensor_platforms.size())},
             {"detection_probability", config.detection_probability},
             {"clutter_per_sensor", config.clutter_per_sensor},
             {"association_gate_squared", config.association_gate_squared},
             {"sensor_type", "SpaceBasedAzEl[azimuth,elevation]"}
         }},
        {"tracking",
         {
             {"confirmation_updates", config.confirmation_updates},
             {"max_coast_steps", config.max_coast_steps},
             {"max_tentative_missed_steps", config.max_tentative_missed_steps},
             {"tracks_created", total_created_tracks},
             {"tracks_deleted", total_deleted_tracks},
             {"tracks_final", static_cast<int>(tracks.size())},
             {"max_active_tracks", max_active_tracks},
             {"mean_confirmed_position_error_m",
              (total_confirmed_snapshots > 0)
                  ? confirmed_position_error_sum_m / static_cast<double>(total_confirmed_snapshots)
                  : 0.0},
             {"pending_initiations_final", static_cast<int>(pending_initiations.size())}
         }},
        {"initialization",
         {
             {"mode", "TwoEpochTriangulation"},
             {"detail",
              "Track initiation uses two consecutive multi-sensor az/el epochs to triangulate position and estimate velocity."},
             {"residual_gate_m", config.initiation_residual_gate_m},
             {"link_distance_m", config.initiation_link_distance_m}
         }},
        {"imm_models", model_summary_json}
    };

    std::ofstream out_file(output_file);
    if (!out_file.is_open()) {
        std::cerr << "Error opening output file: " << output_file << '\n';
        return 1;
    }
    out_file << data_json.dump(2);
    out_file.close();

    std::cout << "Tracker demo data written to " << output_file << '\n';
    return 0;
}
