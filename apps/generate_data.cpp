#include <transforms/lla_to_ecef.hpp>
#include <propagator/propagator.hpp>
#include <integrator/rk4.hpp>
#include <dynamics/gravity.hpp>
#include <dynamics/ballistic3d.hpp>
#include <noise/gaussian_noise.hpp>

#include <Eigen/Dense>
#include <nlohmann/json.hpp>

#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>

int main() {
    // Simulation parameters
    double earth_radius = 6371000; // (m)
    double v0 = 3000.0; // Initial speed (m/s)
    double theta = 75* (M_PI/180); // Elevation angle (radians)
    double phi = 30 * (M_PI/180); // Azimuth angle (radians)
    double dt = 0.1; // Time step (s)
    double tf = 1000.0; // Propagate for 20 seconds (adjust as needed)
    auto earth_gravity = std::make_shared<J2Gravity>();

    // Noise parameters
    double sigma_pos = 5000.0; // Standard deviation for position noise (meters)
    double sigma_vel = 100.0;  // Standard deviation for velocity noise (m/s)
    GaussianNoise noise_generator(sigma_pos, sigma_vel); // Initialize noise generator

    // Reference launch point (Cape Canaveral)
    double ref_lat = 28.3922; // Latitude (degrees)
    double ref_lon = -80.6077; // Longitude (degrees)
    double ref_alt = 0.0; // Altitude (meters)

    // Convert reference point to ECEF
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

    std::cout << "initial state: " << std::fixed << std::setprecision(4) << std::endl << initial_state.transpose() << std::endl;

    // create a propagator to model this trajectory
    auto dynamics = std::make_shared<Ballistic3D>(earth_gravity);
    auto integrator = std::make_shared<RK4Integrator>();
    Propagator propagator(dynamics, integrator, dt);

    // Propagate the state
    auto trajectory = propagator.propagate(0.0, initial_state, tf);

    // JSON array to store trajectory data
    nlohmann::json traj_json = nlohmann::json::array();
    double prevTrackAltitude = 0.0;
    int trackFallingCount = 0;
    for (auto& entry : trajectory) {
        double t = entry.first;
        auto& state = entry.second;

        std::cout << "time: " << std::fixed << std::setprecision(4) << t << ", state (pre-noise): " << std::endl << state.transpose() << std::endl;

        // Stop a little after track stops climbing to simulate sensors losing sight of the track due to being post-burnout
        trackFallingCount += ((state.norm() < prevTrackAltitude) ? 1 : 0);
        if (trackFallingCount > 5) {
            std::cout << "Track falling after 5 consecutive points. Stopping simulation." << std::endl;
            break;
        }
        else if (state.head(3).norm() <= earth_radius)
        {
            std::cout << "Impact reached. Stopping simulation." << std::endl;
            break;
        }
        prevTrackAltitude = state.norm();

        // Add Gaussian noise to the state before writing to file
        state += noise_generator.generate_noise();

        std::cout << "time: " << std::fixed << std::setprecision(4) << t << ", state (post-noise): " << std::endl << state.transpose() << std::endl;

        // Create JSON object for current state
        nlohmann::json point;
        point["time"] = t;
        point["state"] = {state(0), state(1), state(2), state(3), state(4), state(5)};
        traj_json.push_back(point);
    }

    // Write JSON to file
    std::ofstream outFile("track_data.json");
    if (!outFile.is_open()) {
        std::cerr << "Error opening output file!" << std::endl;
        return 1;
    }
    outFile << traj_json.dump(4); // Pretty-print with 4-space indentation
    outFile.close();
    std::cout << "3D trajectory data written to track_data.json" << std::endl;

    return 0;
}
