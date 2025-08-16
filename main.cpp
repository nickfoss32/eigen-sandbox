#include <propagator/propagator.hpp>
#include <integrator/rk4.hpp>
#include <dynamics/gravity.hpp>
#include <dynamics/ballistic3d.hpp>
#include "lla_to_ecef.hpp"
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

    for (const auto& entry : trajectory) {
        double t = entry.first;
        const auto& state = entry.second;

        std::cout << "time: " << std::fixed << std::setprecision(4) << t << ", state: " << std::endl << state.transpose() << std::endl;

        // Stop if at impact
        if (t != 0.0 && state.norm() <= earth_radius) {
            std::cout << "impact reached!" << std::endl;
            break;
        }

        // Create JSON object for current state
        nlohmann::json point;
        point["time"] = t;
        point["state"] = {state(0), state(1), state(2), state(3), state(4), state(5)};
        traj_json.push_back(point);
    }

    // Write JSON to file
    std::ofstream outFile("trajectory_3d.json");
    if (!outFile.is_open()) {
        std::cerr << "Error opening output file!" << std::endl;
        return 1;
    }
    outFile << traj_json.dump(4); // Pretty-print with 4-space indentation
    outFile.close();
    std::cout << "3D trajectory data written to trajectory_3d.json" << std::endl;

    return 0;
}
