#include <transforms/lla_to_ecef.hpp>
#include <propagator/propagator.hpp>
#include <integrator/rk4.hpp>
#include <dynamics/gravity.hpp>
#include <dynamics/ballistic3d.hpp>

#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include <boost/program_options.hpp>

#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>

namespace po = boost::program_options;

int main(int argc, char* argv[]) {
    // Simulation parameters
    double earth_radius = 6371000; // (m)
    double dt = 0.1; // Time step (s)
    double tf = 1000.0; // Propagate for 1000 seconds (adjust as needed)
    auto earth_gravity = std::make_shared<J2Gravity>();

    // Command-line options
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "Produce help message")
        ("input,i", po::value<std::string>(), "Input JSON file with pre-recorded trajectory points");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
    }

    // Check for input file
    std::string input_file;
    if (vm.count("input")) {
        input_file = vm["input"].as<std::string>();
    } else {
        std::cerr << "Error: Input JSON file must be specified using --input or -i" << std::endl;
        std::cout << desc << std::endl;
        return 1;
    }

    // Read input JSON file
    std::ifstream inFile(input_file);
    if (!inFile.is_open()) {
        std::cerr << "Error: Could not open input file " << input_file << std::endl;
        return 1;
    }

    nlohmann::json input_json;
    try {
        inFile >> input_json;
    } catch (const nlohmann::json::exception& e) {
        std::cerr << "Error: Failed to parse JSON file " << input_file << ": " << e.what() << std::endl;
        inFile.close();
        return 1;
    }
    inFile.close();

    // Validate JSON structure and extract points
    if (!input_json.is_array() || input_json.empty()) {
        std::cerr << "Error: Input JSON must be a non-empty array of points" << std::endl;
        return 1;
    }

    std::vector<std::pair<double, Eigen::VectorXd>> input_points;
    for (const auto& point : input_json) {
        if (!point.contains("time") || !point.contains("state") || !point["state"].is_array() || point["state"].size() != 6) {
            std::cerr << "Error: Invalid point structure in JSON. Each point must have 'time' and 'state' with 6 values" << std::endl;
            return 1;
        }

        double t = point["time"].get<double>();
        Eigen::VectorXd state(6);
        state << point["state"][0].get<double>(),
                 point["state"][1].get<double>(),
                 point["state"][2].get<double>(),
                 point["state"][3].get<double>(),
                 point["state"][4].get<double>(),
                 point["state"][5].get<double>();
        input_points.emplace_back(t, state);
    }

    // Find the point with the latest time
    auto latest_point = std::max_element(input_points.begin(), input_points.end(),
        [](const auto& a, const auto& b) { return a.first < b.first; });
    if (latest_point == input_points.end()) {
        std::cerr << "Error: No valid points found in input JSON" << std::endl;
        return 1;
    }

    double initial_time = latest_point->first;
    Eigen::VectorXd initial_state = latest_point->second;
    std::cout << "Using latest state at time " << std::fixed << std::setprecision(4) << initial_time
              << ": " << initial_state.transpose() << std::endl;

    // create a propagator to model this trajectory
    auto dynamics = std::make_shared<Ballistic3D>(earth_gravity);
    auto integrator = std::make_shared<RK4Integrator>();
    Propagator propagator(dynamics, integrator, dt);

    // Propagate the state
    auto trajectory = propagator.propagate_to_impact(initial_time, initial_state);

    // JSON array to store trajectory data (input + propagated points)
    nlohmann::json traj_json = nlohmann::json::array();

    // Add input points to JSON
    for (const auto& point : input_points) {
        nlohmann::json json_point;
        json_point["time"] = point.first;
        json_point["state"] = {point.second(0), point.second(1), point.second(2),
                              point.second(3), point.second(4), point.second(5)};
        traj_json.push_back(json_point);
    }

    // Add propagated points to JSON
    for (const auto& entry : trajectory) {
        double t = entry.first;
        const auto& state = entry.second;

        std::cout << "time: " << std::fixed << std::setprecision(4) << t << ", state: " << std::endl << state.transpose() << std::endl;

        // Create JSON object for current state
        nlohmann::json point;
        point["time"] = t;
        point["state"] = {state(0), state(1), state(2), state(3), state(4), state(5)};
        traj_json.push_back(point);
    }

    // Sort JSON array by time
    std::sort(traj_json.begin(), traj_json.end(),
        [](const nlohmann::json& a, const nlohmann::json& b) {
            return a["time"].get<double>() < b["time"].get<double>();
        });

    // Write JSON to file
    std::ofstream outFile("predicted_track.json");
    if (!outFile.is_open()) {
        std::cerr << "Error opening output file!" << std::endl;
        return 1;
    }
    outFile << traj_json.dump(4); // Pretty-print with 4-space indentation
    outFile.close();
    std::cout << "3D trajectory data written to predicted_track.json" << std::endl;

    return 0;
}
