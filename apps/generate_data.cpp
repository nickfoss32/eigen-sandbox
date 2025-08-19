#include <transforms/lla_to_ecef.hpp>
#include <propagator/propagator.hpp>
#include <integrator/rk4.hpp>
#include <dynamics/gravity.hpp>
#include <dynamics/ballistic3d.hpp>
#include <noise/gaussian_noise.hpp>

#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include <boost/program_options.hpp>

#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <memory>

namespace po = boost::program_options;

int main(int argc, char* argv[]) {
    // Command-line options
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "Produce help message")
        ("latitude", po::value<double>()->default_value(28.4), "Simulation initial latitude (degrees)")
        ("longitude", po::value<double>()->default_value(-80.6), "Simulation initial longitude (degrees)")
        ("altitude", po::value<double>()->default_value(0.0), "Simulation initial altitude (m)")
        ("elevation", po::value<double>()->default_value(75.0), "Simulation launch elevation angle (degrees)")
        ("azimuth", po::value<double>()->default_value(30.0), "Simulation launch azimuth angle (degrees)")
        ("velocity", po::value<double>()->default_value(3000.0), "Simulation initial velocity (m/s)")
        ("sigma-pos", po::value<double>()->default_value(5000.0), "Simulation position noise standard deviation (m)")
        ("sigma-vel", po::value<double>()->default_value(100.0), "Simulation position noise standard deviation (m/s)")
        ("timestep", po::value<double>()->default_value(0.1), "Timestep size to use for propagator (seconds)")
        ("propagation-time", po::value<double>()->default_value(1000.0), "Amount of time to propagate (seconds).")
        ("output,o", po::value<std::string>()->default_value("track_points.json"), "Output JSON file with simulated trajectory points");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
    }
    po::notify(vm);

    // Simulation parameters
    constexpr double earth_radius = 6371000; // (m)
    double tf = vm["propagation-time"].as<double>(); // Propagate for 20 seconds (adjust as needed)
    double dt = vm["timestep"].as<double>(); // Time step (s)
    double ref_lat = vm["latitude"].as<double>(); // Latitude (degrees)
    double ref_lon = vm["longitude"].as<double>(); // Longitude (degrees)
    double ref_alt = vm["altitude"].as<double>(); // Altitude (meters)
    double v0 = vm["velocity"].as<double>(); // Initial speed (m/s)
    double theta = vm["elevation"].as<double>() * (M_PI/180); // Elevation angle (convert to radians)
    double phi = vm["azimuth"].as<double>() * (M_PI/180); // Azimuth angle (convert to radians)
    double sigma_pos = vm["sigma-pos"].as<double>(); // Standard deviation for position noise (meters)
    double sigma_vel = vm["sigma-vel"].as<double>();  // Standard deviation for velocity noise (m/s)

    // Convert launch point to ECEF
    auto ecef_pos = lla_to_ecef(ref_lat, ref_lon, ref_alt);

    // Compute initial velocity in local ENU frame
    Eigen::Vector3d v_enu;
    v_enu << v0 * std::cos(theta) * std::sin(phi), // v_E (East)
             v0 * std::cos(theta) * std::cos(phi), // v_N (North)
             v0 * std::sin(theta);                 // v_U (Up)

    // Convert ENU velocity to ECEF
    double lat_rad = ref_lat * M_PI / 180.0;
    double lon_rad = ref_lon * M_PI / 180.0;
    Eigen::Matrix3d R_enu_to_ecef;
    R_enu_to_ecef << -std::sin(lon_rad), -std::sin(lat_rad) * std::cos(lon_rad), std::cos(lat_rad) * std::cos(lon_rad),
                     std::cos(lon_rad), -std::sin(lat_rad) * std::sin(lon_rad), std::cos(lat_rad) * std::sin(lon_rad),
                     0.0, std::cos(lat_rad), std::sin(lat_rad);
    Eigen::Vector3d ecef_vel = R_enu_to_ecef * v_enu;

    // Initial state: [x, y, z, vx, vy, vz]
    Eigen::VectorXd initial_state(6);
    initial_state << ecef_pos, ecef_vel;

    // std::cout << "initial state: " << std::fixed << std::setprecision(4) << std::endl << initial_state.transpose() << std::endl;

    // create a propagator to model this trajectory
    auto earth_gravity = std::make_shared<J2Gravity>();
    auto dynamics = std::make_shared<Ballistic3D>(earth_gravity);
    auto integrator = std::make_shared<RK4Integrator>();
    Propagator propagator(dynamics, integrator, dt);

    // Propagate the state
    auto trajectory = propagator.propagate(0.0, initial_state, tf);

    // JSON array to store trajectory data
    nlohmann::json traj_json = nlohmann::json::array();
    double prevTrackAltitude = 0.0;
    int trackFallingCount = 0;
    GaussianNoise noise_generator(sigma_pos, sigma_vel); // Initialize noise generator
    for (auto& entry : trajectory) {
        double t = entry.first;
        auto& state = entry.second;

        // std::cout << "time: " << std::fixed << std::setprecision(4) << t << ", state (pre-noise): " << std::endl << state.transpose() << std::endl;

        // Stop a little after track stops climbing to simulate sensors losing sight of the track due to being post-burnout
        trackFallingCount += ((state.norm() < prevTrackAltitude) ? 1 : 0);
        if (trackFallingCount > 5) {
            // std::cout << "Track falling after 5 consecutive points. Stopping simulation." << std::endl;
            break;
        }
        else if (state.head(3).norm() <= earth_radius)
        {
            // std::cout << "Impact reached. Stopping simulation." << std::endl;
            break;
        }
        prevTrackAltitude = state.norm();

        // Add Gaussian noise to the state before writing to file
        state += noise_generator.generate_noise();

        // std::cout << "time: " << std::fixed << std::setprecision(4) << t << ", state (post-noise): " << std::endl << state.transpose() << std::endl;

        // Create JSON object for current state
        nlohmann::json point;
        point["time"] = t;
        point["state"] = {state(0), state(1), state(2), state(3), state(4), state(5)};
        traj_json.push_back(point);
    }

    nlohmann::json data_json = {};
    data_json["points"] = traj_json;
    data_json["summary"]["simulation"]["launch"]["latitude"] = ref_lat;
    data_json["summary"]["simulation"]["launch"]["longitude"] = ref_lon;
    data_json["summary"]["simulation"]["launch"]["altitude"] = ref_alt;
    data_json["summary"]["simulation"]["launch"]["azimuth"] = vm["azimuth"].as<double>();
    data_json["summary"]["simulation"]["launch"]["elevation"] = vm["elevation"].as<double>();
    data_json["summary"]["simulation"]["launch"]["velocity"] = v0;
    data_json["summary"]["simulation"]["noise"]["sigma_pos"] = sigma_pos;
    data_json["summary"]["simulation"]["noise"]["sigma_vel"] = sigma_vel;
    data_json["summary"]["simulation"]["timestep"] = dt;

    // Write JSON to file
    auto output_file = vm["output"].as<std::string>();
    std::ofstream outFile(output_file);
    if (!outFile.is_open()) {
        std::cerr << "Error opening output file!" << std::endl;
        return 1;
    }
    outFile << data_json.dump(4); // Pretty-print with 4-space indentation
    outFile.close();
    std::cout << "Simulated trajectory data written to " << output_file << std::endl;

    return 0;
}
