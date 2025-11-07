
#include <propagator/propagator.hpp>
#include <integrator/rk4.hpp>
#include <dynamics/gravity.hpp>
#include <dynamics/ballistic3d.hpp>
#include <fitting/plane_fit.hpp>
#include <fitting/factory.hpp>

#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include <boost/program_options.hpp>

#include <algorithm>
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
        ("input,i", po::value<std::string>()->required(), "Input JSON file with pre-recorded trajectory points")
        ("output,o", po::value<std::string>()->default_value("predicted_trajectory.json"), "Output JSON file with predicted trajectory points")
        ("plane-fit-mode,m", po::value<fitting::PlaneFitMode>()->default_value(fitting::PlaneFitMode::TLS), "Type of plane fit to use for predicting trajectory (OLS or TLS)")
        ("timestep,t", po::value<double>()->default_value(0.1), "Timestep size to use for propagator (seconds)");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
    }
    po::notify(vm);

    // Simulation parameters
    double dt = vm["timestep"].as<double>(); // Time step (s)

    // Read input JSON file
    auto input_file = vm["input"].as<std::string>();
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
    if (!input_json["points"].is_array() || input_json["points"].empty()) {
        std::cerr << "Error: Input JSON must be a non-empty array of points" << std::endl;
        return 1;
    }

    auto coordTxfms = std::make_shared<transforms::CoordTransforms>(IERS_EOP_FILE); // IERS_EOP_FILE defined by CMake
    std::vector<std::pair<double, Eigen::VectorXd>> input_points;
    for (const auto& point : input_json["points"]) {
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

        // convert point to ECEF if necessary
        if (input_json["summary"]["simulation"]["coordinate_frame"] == "ECI") {
            state = coordTxfms->eci_to_ecef(state, t);
        }

        input_points.emplace_back(t, state);
    }

    // calculate best fit plane amongst points (fit through origin)
    std::optional<std::pair<Eigen::Vector3d, Eigen::Vector3d>> plane = std::nullopt;
    fitting::PlaneFitMode mode = vm["plane-fit-mode"].as<fitting::PlaneFitMode>();
    std::vector<Eigen::Vector3d> positions;
    std::unique_ptr<fitting::PlaneFitter> planeFitter = nullptr;
    positions.reserve(input_points.size()); // Pre-allocate for efficiency
    std::transform(
        input_points.begin(), input_points.end(), std::back_inserter(positions),
        [](const auto& pair) {
            return pair.second.head(3);
        }
    );
    try {
        planeFitter = fitting::PlaneFitterFactory::create(mode);
        plane = planeFitter->computeFit(positions);
    }
    catch(const std::exception& e) {
        std::cout << "Unable to calculate plane fit! Exception thrown: " << e.what() << std::endl;
    }

    // Find the point with the latest time
    auto latest_point = std::max_element(input_points.begin(), input_points.end(),
        [](const auto& a, const auto& b) { return a.first < b.first; });
    if (latest_point == input_points.end()) {
        std::cerr << "Error: No valid points found in input JSON" << std::endl;
        return 1;
    }
    
    // project last point onto best fit plane
    double initial_time = latest_point->first;
    auto initial_state = latest_point->second;
    if (plane.has_value()) {
        const auto& [normal, point] = plane.value(); // Structured binding for clarity
        initial_state = planeFitter->projectState(initial_state, normal, point);
    }

    // std::cout << "initial state: " << std::fixed << std::setprecision(4) << std::endl << initial_state.transpose() << std::endl;

    // create a propagator to model this trajectory
    auto earth_gravity = std::make_shared<dynamics::J2Gravity>();
    dynamics::CoordinateFrame coordinateFrame = dynamics::CoordinateFrame::ECEF;
    // CoordinateFrame coordinateFrame = ( (input_json["summary"]["simulation"]["coordinate_frame"] == "ECI") ? CoordinateFrame::ECI : CoordinateFrame::ECEF);
    auto dynamics = std::make_shared<dynamics::Ballistic3D>(coordinateFrame, earth_gravity);
    auto integrator = std::make_shared<integrator::RK4Integrator>();
    propagator::Propagator propagator(dynamics, integrator, dt, coordinateFrame, coordTxfms);

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

        // std::cout << "time: " << std::fixed << std::setprecision(4) << t << ", state: " << std::endl << state.transpose() << std::endl;

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

    nlohmann::json data_json = {};
    data_json["points"] = traj_json;
    data_json["summary"] = input_json["summary"];
    if (plane.has_value()) {
        const auto& [normal, point] = plane.value(); // Structured binding for clarity
        data_json["summary"]["fit"]["type"] = (mode ==fitting::PlaneFitMode::OLS ? "OLS" : "TLS");
        data_json["summary"]["fit"]["normal"] = {normal[0], normal[1], normal[2]};
        data_json["summary"]["fit"]["point"] = {point[0], point[1], point[2]};
    }
    else {
        data_json["summary"]["fit"]["normal"] = {};
        data_json["summary"]["fit"]["point"] = {};
    }

    // Write JSON to file
    auto output_file = vm["output"].as<std::string>();
    std::ofstream outFile(output_file);
    if (!outFile.is_open()) {
        std::cerr << "Error opening output file!" << std::endl;
        return 1;
    }
    outFile << data_json.dump(4); // Pretty-print with 4-space indentation
    outFile.close();
    std::cout << "Predicted 3D trajectory data written to " << output_file << std::endl;

    return 0;
}
