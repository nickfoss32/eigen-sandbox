#include <Eigen/Dense>

#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numbers>
#include <random>
#include <string>
#include <vector>

#include <boost/program_options.hpp>
#include <nlohmann/json.hpp>

#include <common/types.hpp>
#include <dynamics/forces/atmospheric_drag.hpp>
#include <dynamics/forces/gravity.hpp>
#include <dynamics/point_mass_dynamics.hpp>
#include <estimation/imm.hpp>
#include <filtering/extended_kalman_filter.hpp>
#include <integrator/rk4.hpp>
#include <propagator/numerical_propagator.hpp>
#include <sensor/space_based_optical_sensor_model.hpp>

namespace po = boost::program_options;

int main(int argc, char* argv[]) {
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "Produce help message")
        ("output,o", po::value<std::string>()->default_value("imm_demo.json"),
            "Output JSON file with IMM trajectory data");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    if (vm.count("help")) {
        std::cout << desc << '\n';
        return 0;
    }

    po::notify(vm);
    const std::string output_file = vm["output"].as<std::string>();

    constexpr double kEarthRadiusMeters = 6378137.0;
    constexpr double kEarthMu = 3.986004418e14;
    constexpr double kOrbitAltitudeMeters = 300000.0;
    constexpr double kDtSeconds = 10.0;
    constexpr int kNumSteps = 360;  // 1 hour at 10 second updates
    constexpr double kIntegratorStepSeconds = 2.0;
    constexpr double kDegToRad = std::numbers::pi / 180.0;
    constexpr double kLaunchLatitudeDeg = 28.573255;    // Cape Canaveral
    constexpr double kLaunchLongitudeDeg = -80.646895;  // Cape Canaveral
    constexpr double kLaunchHeadingDeg = 45.0;          // Northeast, ISS-like corridor
    constexpr double kFlightPathDeg = 4.0;              // Shallow climb near orbital insertion

    const double initial_radius = kEarthRadiusMeters + kOrbitAltitudeMeters;
    const double circular_speed = std::sqrt(kEarthMu / initial_radius);
    const double launch_lat = kLaunchLatitudeDeg * kDegToRad;
    const double launch_lon = kLaunchLongitudeDeg * kDegToRad;
    const double launch_heading = kLaunchHeadingDeg * kDegToRad;
    const double flight_path = kFlightPathDeg * kDegToRad;

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

    // Truth: Cape-based, launch-like ECI state with northeast heading and slight climb.
    const Eigen::Vector3d initial_position = initial_radius * up;
    const Eigen::Vector3d initial_velocity_enu(
        circular_speed * std::cos(flight_path) * std::sin(launch_heading),  // east
        circular_speed * std::cos(flight_path) * std::cos(launch_heading),  // north
        circular_speed * std::sin(flight_path)                               // up
    );
    const Eigen::Vector3d initial_velocity = enu_to_eci * initial_velocity_enu;

    Eigen::VectorXd truth_state(6);
    truth_state << initial_position, initial_velocity;

    // Initial estimate with realistic local-frame state errors.
    const Eigen::Vector3d initial_pos_error_enu(1500.0, -1200.0, 900.0);
    const Eigen::Vector3d initial_vel_error_enu(6.0, 4.0, -3.0);
    Eigen::VectorXd initial_estimate(6);
    initial_estimate << initial_position + enu_to_eci * initial_pos_error_enu,
                        initial_velocity + enu_to_eci * initial_vel_error_enu;

    Eigen::MatrixXd initial_covariance = Eigen::MatrixXd::Identity(6, 6);
    initial_covariance.block<3, 3>(0, 0) *= 2000.0 * 2000.0;
    initial_covariance.block<3, 3>(3, 3) *= 10.0 * 10.0;

    // Space-based optical sensor in its own ECI orbit, returning [RA, Dec].
    const double sensor_radius = kEarthRadiusMeters + 700000.0;
    const double sensor_mean_motion = std::sqrt(kEarthMu / (sensor_radius * sensor_radius * sensor_radius));
    const double sensor_raan = 25.0 * kDegToRad;
    const double sensor_inc = 55.0 * kDegToRad;
    const double sensor_phase = 100.0 * kDegToRad;

    const Eigen::Matrix3d sensor_orbit_to_eci =
        (Eigen::AngleAxisd(sensor_raan, Eigen::Vector3d::UnitZ()) *
         Eigen::AngleAxisd(sensor_inc, Eigen::Vector3d::UnitX())).toRotationMatrix();

    auto sensor_position_eci = [sensor_radius, sensor_mean_motion, sensor_phase, sensor_orbit_to_eci](double t) {
        const double u = sensor_mean_motion * t + sensor_phase;
        const Eigen::Vector3d r_orbit(sensor_radius * std::cos(u), sensor_radius * std::sin(u), 0.0);
        return sensor_orbit_to_eci * r_orbit;
    };

    constexpr double kRaNoiseRad = 80e-6;
    constexpr double kDecNoiseRad = 80e-6;
    auto sensor_model = std::make_shared<sensor::SpaceBasedOpticalSensorModel>(
        kRaNoiseRad,
        kDecNoiseRad
    );

    auto integrator = std::make_shared<integrator::RK4Integrator>();

    // Truth model: J2 + atmospheric drag.
    auto truth_dynamics = std::make_shared<dynamics::PointMassDynamics>(
        std::vector<std::shared_ptr<dynamics::IForce>>{
            std::make_shared<dynamics::J2Gravity>(),
            std::make_shared<dynamics::AtmosphericDrag>(
                120.0,   // kg
                2.2,     // Cd
                4.0,     // m^2
                kEarthRadiusMeters
            )
        }
    );

    // IMM model 0: Two-body gravity only.
    auto model_two_body_dynamics = std::make_shared<dynamics::PointMassDynamics>(
        std::vector<std::shared_ptr<dynamics::IForce>>{
            std::make_shared<dynamics::PointMassGravity>()
        }
    );

    // IMM model 1: J2 gravity.
    auto model_j2_dynamics = std::make_shared<dynamics::PointMassDynamics>(
        std::vector<std::shared_ptr<dynamics::IForce>>{
            std::make_shared<dynamics::J2Gravity>()
        }
    );

    // IMM model 2: J2 + drag.
    auto model_j2_drag_dynamics = std::make_shared<dynamics::PointMassDynamics>(
        std::vector<std::shared_ptr<dynamics::IForce>>{
            std::make_shared<dynamics::J2Gravity>(),
            std::make_shared<dynamics::AtmosphericDrag>(
                120.0,
                2.2,
                4.0,
                kEarthRadiusMeters
            )
        }
    );

    auto truth_propagator = std::make_shared<propagator::NumericalPropagator>(
        truth_dynamics, integrator, kIntegratorStepSeconds
    );
    auto model_two_body_propagator = std::make_shared<propagator::NumericalPropagator>(
        model_two_body_dynamics, integrator, kIntegratorStepSeconds
    );
    auto model_j2_propagator = std::make_shared<propagator::NumericalPropagator>(
        model_j2_dynamics, integrator, kIntegratorStepSeconds
    );
    auto model_j2_drag_propagator = std::make_shared<propagator::NumericalPropagator>(
        model_j2_drag_dynamics, integrator, kIntegratorStepSeconds
    );

    filtering::ExtendedKalmanFilter::ProcessNoiseFunction q_two_body =
        [](double dt) {
            Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(6, 6);
            Q.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * (4000.0 * dt);
            Q.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * (2.0 * dt);
            return Q;
        };

    filtering::ExtendedKalmanFilter::ProcessNoiseFunction q_j2 =
        [](double dt) {
            Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(6, 6);
            Q.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * (2000.0 * dt);
            Q.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * (1.0 * dt);
            return Q;
        };

    filtering::ExtendedKalmanFilter::ProcessNoiseFunction q_j2_drag =
        [](double dt) {
            Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(6, 6);
            Q.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * (1000.0 * dt);
            Q.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * (0.5 * dt);
            return Q;
        };

    std::vector<std::unique_ptr<filtering::IKalmanFilter>> filters;
    filters.push_back(std::make_unique<filtering::ExtendedKalmanFilter>(
        initial_estimate,
        initial_covariance,
        model_two_body_propagator,
        sensor_model,
        q_two_body,
        0.0
    ));
    filters.push_back(std::make_unique<filtering::ExtendedKalmanFilter>(
        initial_estimate,
        initial_covariance,
        model_j2_propagator,
        sensor_model,
        q_j2,
        0.0
    ));
    filters.push_back(std::make_unique<filtering::ExtendedKalmanFilter>(
        initial_estimate,
        initial_covariance,
        model_j2_drag_propagator,
        sensor_model,
        q_j2_drag,
        0.0
    ));

    Eigen::VectorXd initial_mode_probs(3);
    initial_mode_probs << 0.80, 0.15, 0.05;

    Eigen::MatrixXd transition(3, 3);
    transition << 0.985, 0.0075, 0.0075,
                  0.0075, 0.985, 0.0075,
                  0.0075, 0.0075, 0.985;

    estimation::IMM imm(std::move(filters), initial_mode_probs, transition);

    std::mt19937 rng(42);
    std::normal_distribution<double> ra_noise(0.0, kRaNoiseRad);
    std::normal_distribution<double> dec_noise(0.0, kDecNoiseRad);

    const std::array<const char*, 3> model_names = {
        "TwoBodyGravity",
        "J2Gravity",
        "J2PlusDrag"
    };

    nlohmann::json truth_points = nlohmann::json::array();
    nlohmann::json imm_points = nlohmann::json::array();
    std::array<nlohmann::json, 3> model_points = {
        nlohmann::json::array(),
        nlohmann::json::array(),
        nlohmann::json::array()
    };

    auto make_state_point = [](double t, const Eigen::VectorXd& state) {
        nlohmann::json point;
        point["time"] = t;
        point["state"] = {
            state(0), state(1), state(2),
            state(3), state(4), state(5)
        };
        return point;
    };

    auto append_imm_point =
        [&](double t, const Eigen::VectorXd& state, const Eigen::VectorXd& model_probs) {
            nlohmann::json point = make_state_point(t, state);
            point["model_probabilities"] = {model_probs(0), model_probs(1), model_probs(2)};
            point["most_likely_model"] = imm.get_most_likely_model();
            imm_points.push_back(std::move(point));
        };

    truth_points.push_back(make_state_point(0.0, truth_state));
    const Eigen::VectorXd initial_mu = imm.get_model_probabilities();
    append_imm_point(0.0, imm.get_state(), initial_mu);
    for (int i = 0; i < 3; ++i) {
        nlohmann::json point = make_state_point(0.0, imm.get_model_state(i));
        point["mode_probability"] = initial_mu(i);
        model_points[static_cast<std::size_t>(i)].push_back(std::move(point));
    }

    std::cout << "IMM demo with orbital force models (3 model filters)\n";
    std::cout << "Measurements: space-based optical [RA, Dec] in ECI\n";
    std::cout << "  model 0: Two-body gravity\n";
    std::cout << "  model 1: J2 gravity\n";
    std::cout << "  model 2: J2 + atmospheric drag\n";
    std::cout << "  truth  : J2 + atmospheric drag\n\n";
    std::cout << "Launch geometry: Cape-like (" << kLaunchLatitudeDeg << " deg lat, "
              << kLaunchLongitudeDeg << " deg lon), heading " << kLaunchHeadingDeg
              << " deg, flight path " << kFlightPathDeg << " deg\n\n";

    std::cout << std::fixed << std::setprecision(3);
    std::cout << std::setw(6) << "t[s]"
              << std::setw(12) << "mu0"
              << std::setw(12) << "mu1"
              << std::setw(12) << "mu2"
              << std::setw(14) << "alt[km]"
              << std::setw(16) << "pos_err[m]"
              << std::setw(16) << "vel_err[m/s]"
              << '\n';

    double time = 0.0;
    for (int k = 0; k < kNumSteps; ++k) {
        const double t_next = time + kDtSeconds;

        auto truth_traj = truth_propagator->propagate(time, truth_state, t_next);
        truth_state = truth_traj.back().second;
        time = t_next;

        sensor::SensorContext ctx;
        ctx.state = truth_state;
        ctx.time = time;
        ctx.sensor_position = sensor_position_eci(time);
        ctx.sensor_orientation = Eigen::Quaterniond::Identity();
        Eigen::VectorXd z = sensor_model->compute_measurement(ctx);
        z(0) += ra_noise(rng);
        z(1) += dec_noise(rng);

        common::Measurement measurement(z, sensor_model->get_noise_covariance(), time);
        measurement.sensor_position = ctx.sensor_position;
        measurement.sensor_orientation = ctx.sensor_orientation;

        imm.predict(kDtSeconds);
        imm.update(measurement);

        const Eigen::VectorXd x_hat = imm.get_state();
        const Eigen::VectorXd mu = imm.get_model_probabilities();
        truth_points.push_back(make_state_point(time, truth_state));
        append_imm_point(time, x_hat, mu);
        for (int i = 0; i < 3; ++i) {
            nlohmann::json point = make_state_point(time, imm.get_model_state(i));
            point["mode_probability"] = mu(i);
            model_points[static_cast<std::size_t>(i)].push_back(std::move(point));
        }

        if ((k + 1) % 30 == 0 || k == 0) {
            const double altitude_km = (truth_state.head<3>().norm() - kEarthRadiusMeters) / 1000.0;
            const double pos_err = (x_hat.head<3>() - truth_state.head<3>()).norm();
            const double vel_err = (x_hat.tail<3>() - truth_state.tail<3>()).norm();

            std::cout << std::setw(6) << time
                      << std::setw(12) << mu(0)
                      << std::setw(12) << mu(1)
                      << std::setw(12) << mu(2)
                      << std::setw(14) << altitude_km
                      << std::setw(16) << pos_err
                      << std::setw(16) << vel_err
                      << '\n';
        }
    }

    const int most_likely = imm.get_most_likely_model();
    std::cout << "\nMost likely model at end: " << most_likely
              << " (" << model_names.at(static_cast<std::size_t>(most_likely)) << ")"
              << '\n';

    const Eigen::VectorXd final_mu = imm.get_model_probabilities();

    nlohmann::json trajectories = nlohmann::json::array();
    trajectories.push_back({
        {"name", "Truth"},
        {"type", "truth"},
        {"points", truth_points}
    });
    trajectories.push_back({
        {"name", "IMMCombined"},
        {"type", "imm_combined"},
        {"points", imm_points}
    });
    for (int i = 0; i < 3; ++i) {
        trajectories.push_back({
            {"name", model_names.at(static_cast<std::size_t>(i))},
            {"type", "imm_model"},
            {"model_index", i},
            {"points", model_points.at(static_cast<std::size_t>(i))}
        });
    }

    nlohmann::json models_summary = nlohmann::json::array();
    for (int i = 0; i < 3; ++i) {
        models_summary.push_back({
            {"index", i},
            {"name", model_names.at(static_cast<std::size_t>(i))}
        });
    }

    nlohmann::json data_json;
    data_json["points"] = imm_points;
    data_json["trajectories"] = trajectories;
    data_json["summary"]["simulation"]["coordinate_frame"] = "ECI";
    data_json["summary"]["simulation"]["start_time"] = 0.0;
    data_json["summary"]["simulation"]["timestep"] = kDtSeconds;
    data_json["summary"]["simulation"]["integrator_timestep"] = kIntegratorStepSeconds;
    data_json["summary"]["simulation"]["steps"] = kNumSteps;
    data_json["summary"]["simulation"]["duration"] = kNumSteps * kDtSeconds;
    data_json["summary"]["simulation"]["earth_radius"] = kEarthRadiusMeters;
    data_json["summary"]["simulation"]["truth_model"] = "J2PlusDrag";
    data_json["summary"]["simulation"]["measurement_type"] = "SpaceBasedOptical[RA,Dec]";
    data_json["summary"]["imm"]["models"] = models_summary;
    data_json["summary"]["imm"]["initial_model_probabilities"] = {
        initial_mode_probs(0), initial_mode_probs(1), initial_mode_probs(2)
    };
    data_json["summary"]["imm"]["transition_matrix"] = {
        {transition(0, 0), transition(0, 1), transition(0, 2)},
        {transition(1, 0), transition(1, 1), transition(1, 2)},
        {transition(2, 0), transition(2, 1), transition(2, 2)}
    };
    data_json["summary"]["imm"]["final_model_probabilities"] = {
        final_mu(0), final_mu(1), final_mu(2)
    };
    data_json["summary"]["imm"]["most_likely_model"] = most_likely;
    data_json["summary"]["imm"]["most_likely_model_name"] =
        model_names.at(static_cast<std::size_t>(most_likely));

    std::ofstream out_file(output_file);
    if (!out_file.is_open()) {
        std::cerr << "Error opening output file: " << output_file << '\n';
        return 1;
    }
    out_file << data_json.dump(4) << '\n';
    std::cout << "Trajectory JSON written to " << output_file << '\n';

    return 0;
}
