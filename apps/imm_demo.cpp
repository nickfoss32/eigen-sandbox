#include <Eigen/Dense>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numbers>
#include <optional>
#include <random>
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
#include <dynamics/smooth_acceleration_point_mass_dynamics.hpp>
#include <estimation/imm.hpp>
#include <estimation/rts_smoother.hpp>
#include <estimation/smoother.hpp>
#include <filtering/extended_kalman_filter.hpp>
#include <filtering/unscented_kalman_filter.hpp>
#include <integrator/rk4.hpp>
#include <propagator/numerical_propagator.hpp>
#include <sensor/space_based_azel_sensor_model.hpp>

namespace po = boost::program_options;

namespace {

constexpr double kEarthRadiusMeters = 6378137.0;
constexpr double kEarthMu = 3.986004418e14;
constexpr double kDegToRad = std::numbers::pi / 180.0;

struct ErrorStats {
    void add_sample(const Eigen::VectorXd& estimate, const Eigen::VectorXd& truth) {
        const double pos_error = (estimate.head<3>() - truth.head<3>()).norm();
        const double vel_error =
            (estimate.segment<3>(3) - truth.segment<3>(3)).norm();

        pos_sq_sum += pos_error * pos_error;
        vel_sq_sum += vel_error * vel_error;
        max_pos_error = std::max(max_pos_error, pos_error);
        max_vel_error = std::max(max_vel_error, vel_error);
        final_pos_error = pos_error;
        final_vel_error = vel_error;
        ++samples;
    }

    auto pos_rmse() const -> double {
        return (samples > 0) ? std::sqrt(pos_sq_sum / static_cast<double>(samples)) : 0.0;
    }

    auto vel_rmse() const -> double {
        return (samples > 0) ? std::sqrt(vel_sq_sum / static_cast<double>(samples)) : 0.0;
    }

    auto to_json() const -> nlohmann::json {
        return {
            {"samples", samples},
            {"position_rmse_m", pos_rmse()},
            {"velocity_rmse_mps", vel_rmse()},
            {"max_position_error_m", max_pos_error},
            {"max_velocity_error_mps", max_vel_error},
            {"final_position_error_m", final_pos_error},
            {"final_velocity_error_mps", final_vel_error}
        };
    }

    double pos_sq_sum = 0.0;
    double vel_sq_sum = 0.0;
    double max_pos_error = 0.0;
    double max_vel_error = 0.0;
    double final_pos_error = 0.0;
    double final_vel_error = 0.0;
    int samples = 0;
};

enum class RunMode {
    Comparison,
    Imm,
    Both
};

enum class FilterFamily {
    EKF,
    UKF,
    Both
};

struct ModelInfo {
    std::string name;
    std::string filter_type;
    std::string motion_model;
    std::string description;
    double process_noise_sigma_accel = 0.0;
};

struct ModelBuild;

using ModelFactory = std::function<ModelBuild(const std::shared_ptr<sensor::ISensorModel>&, FilterFamily)>;

struct ModelSpec {
    std::string name;
    std::string motion_model;
    std::string description;
    double process_noise_sigma_accel = 0.0;
    bool include_in_imm = true;
    std::shared_ptr<propagator::IPropagator> smoother_propagator;
    ModelFactory factory;
};

struct ModelBuild {
    ModelInfo info;
    std::unique_ptr<filtering::IKalmanFilter> filter;
};

struct SensorKinematics {
    Eigen::Vector3d position;
    Eigen::Vector3d velocity;
    Eigen::Quaterniond orientation;
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

struct RecordedSample {
    double time_seconds = 0.0;
    Eigen::VectorXd truth_state;
    std::vector<common::Measurement> measurements;
};

struct InitializationSeed {
    std::string mode = "NominalPrior";
    std::string detail = "Fixed-offset nominal prior.";
    double initial_time_seconds = 0.0;
    int consumed_sample_count = 0;
    int epochs_used = 0;
    int sensors_used_per_epoch = 0;
    double position_sigma_m = 0.0;
    double velocity_sigma_mps = 0.0;
    double first_epoch_triangulation_residual_m = 0.0;
    double second_epoch_triangulation_residual_m = 0.0;
    Eigen::VectorXd initial_estimate;
    Eigen::MatrixXd initial_covariance;
};

struct ScenarioTrace {
    Eigen::VectorXd initial_truth_state;
    Eigen::VectorXd initial_estimate;
    Eigen::MatrixXd initial_covariance;
    double initial_time_seconds = 0.0;
    int first_update_sample_index = 0;
    InitializationSeed initialization;
    nlohmann::json truth_points = nlohmann::json::array();
    nlohmann::json measurement_points = nlohmann::json::array();
    std::vector<RecordedSample> samples;
};

struct PhaseWindow {
    std::string name;
    double start_time_seconds = 0.0;
    double end_time_seconds = 0.0;
};

struct DemoRunResult {
    std::string run_name;
    std::string mode;
    std::string filter_family;
    std::vector<ModelInfo> model_infos;
    std::vector<nlohmann::json> model_points;
    std::vector<ErrorStats> model_stats;
    std::vector<int> ranking;
    nlohmann::json combined_points = nlohmann::json::array();
    ErrorStats combined_stats;
    bool has_combined = false;
    nlohmann::json smoothed_combined_points = nlohmann::json::array();
    ErrorStats smoothed_combined_stats;
    bool has_smoothed_combined = false;
    Eigen::VectorXd initial_model_probabilities;
    Eigen::MatrixXd transition_matrix;
    Eigen::VectorXd final_model_probabilities;
    int most_likely_model = -1;
    int best_model_index = -1;
    std::map<std::string, double> final_motion_model_probabilities;
};

struct ScalarStats {
    void add_sample(double value) {
        if (!std::isfinite(value)) {
            return;
        }

        sum += value;
        max_value = std::max(max_value, value);
        final_value = value;
        ++samples;
    }

    auto mean() const -> double {
        return (samples > 0) ? sum / static_cast<double>(samples) : 0.0;
    }

    auto to_json(const std::string& base_name) const -> nlohmann::json {
        return {
            {"samples", samples},
            {"mean_" + base_name, mean()},
            {"max_" + base_name, max_value},
            {"final_" + base_name, final_value}
        };
    }

    double sum = 0.0;
    double max_value = 0.0;
    double final_value = 0.0;
    int samples = 0;
};

struct ErrorMagnitudeStats {
    void add_sample(double position_error, double velocity_error) {
        if (!std::isfinite(position_error) || !std::isfinite(velocity_error)) {
            return;
        }

        pos_sq_sum += position_error * position_error;
        vel_sq_sum += velocity_error * velocity_error;
        max_pos_error = std::max(max_pos_error, position_error);
        max_vel_error = std::max(max_vel_error, velocity_error);
        final_pos_error = position_error;
        final_vel_error = velocity_error;
        ++samples;
    }

    auto pos_rmse() const -> double {
        return (samples > 0) ? std::sqrt(pos_sq_sum / static_cast<double>(samples)) : 0.0;
    }

    auto vel_rmse() const -> double {
        return (samples > 0) ? std::sqrt(vel_sq_sum / static_cast<double>(samples)) : 0.0;
    }

    auto to_json() const -> nlohmann::json {
        return {
            {"samples", samples},
            {"position_rmse_m", pos_rmse()},
            {"velocity_rmse_mps", vel_rmse()},
            {"max_position_error_m", max_pos_error},
            {"max_velocity_error_mps", max_vel_error},
            {"final_position_error_m", final_pos_error},
            {"final_velocity_error_mps", final_vel_error}
        };
    }

    double pos_sq_sum = 0.0;
    double vel_sq_sum = 0.0;
    double max_pos_error = 0.0;
    double max_vel_error = 0.0;
    double final_pos_error = 0.0;
    double final_vel_error = 0.0;
    int samples = 0;
};

auto make_white_acceleration_process_noise(double sigma_accel)
    -> std::function<Eigen::MatrixXd(double)> {
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

auto make_smooth_acceleration_state_process_noise(double sigma_accel)
    -> std::function<Eigen::MatrixXd(double)> {
    return [sigma_accel](double dt) {
        const double sigma2 = sigma_accel * sigma_accel;
        Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(9, 9);
        Q.block<3, 3>(6, 6) = dt * sigma2 * Eigen::Matrix3d::Identity();
        return Q;
    };
}

auto make_propagator(
    std::shared_ptr<integrator::RK4Integrator> integrator,
    double timestep_seconds,
    std::vector<std::shared_ptr<dynamics::IForce>> forces
) -> std::shared_ptr<propagator::NumericalPropagator> {
    auto dynamics_model = std::make_shared<dynamics::PointMassDynamics>(std::move(forces));
    return std::make_shared<propagator::NumericalPropagator>(
        dynamics_model,
        std::move(integrator),
        timestep_seconds
    );
}

auto make_propagator_from_dynamics(
    std::shared_ptr<integrator::RK4Integrator> integrator,
    double timestep_seconds,
    const std::shared_ptr<dynamics::IDynamics>& dynamics_model
) -> std::shared_ptr<propagator::NumericalPropagator> {
    return std::make_shared<propagator::NumericalPropagator>(
        dynamics_model,
        std::move(integrator),
        timestep_seconds
    );
}

auto make_ekf_model_with_process_noise(
    const Eigen::VectorXd& initial_state,
    const Eigen::MatrixXd& initial_covariance,
    const std::shared_ptr<sensor::ISensorModel>& sensor_model,
    const std::shared_ptr<propagator::IPropagator>& propagator,
    std::string name,
    std::string motion_model,
    std::string description,
    double sigma_accel,
    double initial_time_seconds,
    filtering::ExtendedKalmanFilter::ProcessNoiseFunction process_noise_function
) -> ModelBuild {
    std::unique_ptr<filtering::IKalmanFilter> filter =
        std::make_unique<filtering::ExtendedKalmanFilter>(
            initial_state,
            initial_covariance,
            propagator,
            sensor_model,
            std::move(process_noise_function),
            initial_time_seconds
        );

    return {
        ModelInfo{
            std::move(name),
            "EKF",
            std::move(motion_model),
            std::move(description),
            sigma_accel
        },
        std::move(filter)
    };
}

auto make_ukf_model_with_process_noise(
    const Eigen::VectorXd& initial_state,
    const Eigen::MatrixXd& initial_covariance,
    const std::shared_ptr<sensor::ISensorModel>& sensor_model,
    const std::shared_ptr<propagator::IPropagator>& propagator,
    std::string name,
    std::string motion_model,
    std::string description,
    double sigma_accel,
    double initial_time_seconds,
    filtering::UnscentedKalmanFilter::ProcessNoiseFunction process_noise_function
) -> ModelBuild {
    std::unique_ptr<filtering::IKalmanFilter> filter =
        std::make_unique<filtering::UnscentedKalmanFilter>(
            initial_state,
            initial_covariance,
            propagator,
            sensor_model,
            std::move(process_noise_function),
            initial_time_seconds
        );

    return {
        ModelInfo{
            std::move(name),
            "UKF",
            std::move(motion_model),
            std::move(description),
            sigma_accel
        },
        std::move(filter)
    };
}

auto make_ekf_model(
    const Eigen::VectorXd& initial_state,
    const Eigen::MatrixXd& initial_covariance,
    const std::shared_ptr<sensor::ISensorModel>& sensor_model,
    const std::shared_ptr<propagator::IPropagator>& propagator,
    std::string name,
    std::string motion_model,
    std::string description,
    double sigma_accel,
    double initial_time_seconds
) -> ModelBuild {
    return make_ekf_model_with_process_noise(
        initial_state,
        initial_covariance,
        sensor_model,
        propagator,
        std::move(name),
        std::move(motion_model),
        std::move(description),
        sigma_accel,
        initial_time_seconds,
        make_white_acceleration_process_noise(sigma_accel)
    );
}

auto make_ukf_model(
    const Eigen::VectorXd& initial_state,
    const Eigen::MatrixXd& initial_covariance,
    const std::shared_ptr<sensor::ISensorModel>& sensor_model,
    const std::shared_ptr<propagator::IPropagator>& propagator,
    std::string name,
    std::string motion_model,
    std::string description,
    double sigma_accel,
    double initial_time_seconds
) -> ModelBuild {
    return make_ukf_model_with_process_noise(
        initial_state,
        initial_covariance,
        sensor_model,
        propagator,
        std::move(name),
        std::move(motion_model),
        std::move(description),
        sigma_accel,
        initial_time_seconds,
        make_white_acceleration_process_noise(sigma_accel)
    );
}

auto normalize_cli_value(std::string value) -> std::string {
    std::transform(
        value.begin(),
        value.end(),
        value.begin(),
        [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); }
    );
    return value;
}

auto parse_run_mode(const std::string& value) -> RunMode {
    const std::string normalized = normalize_cli_value(value);
    if (normalized == "comparison") {
        return RunMode::Comparison;
    }
    if (normalized == "imm") {
        return RunMode::Imm;
    }
    if (normalized == "both" || normalized == "all") {
        return RunMode::Both;
    }

    throw std::invalid_argument("mode must be one of: comparison, imm, both");
}

auto parse_filter_family(const std::string& value) -> FilterFamily {
    const std::string normalized = normalize_cli_value(value);
    if (normalized == "ekf") {
        return FilterFamily::EKF;
    }
    if (normalized == "ukf") {
        return FilterFamily::UKF;
    }
    if (normalized == "both" || normalized == "all") {
        return FilterFamily::Both;
    }

    throw std::invalid_argument("filter-family must be one of: ekf, ukf, both");
}

auto run_mode_to_string(RunMode mode) -> std::string {
    switch (mode) {
        case RunMode::Comparison:
            return "comparison";
        case RunMode::Imm:
            return "imm";
        case RunMode::Both:
            return "both";
    }

    return "unknown";
}

auto run_mode_to_display_string(RunMode mode) -> std::string {
    switch (mode) {
        case RunMode::Comparison:
            return "Comparison";
        case RunMode::Imm:
            return "IMM";
        case RunMode::Both:
            return "Both";
    }

    return "Unknown";
}

auto filter_family_to_string(FilterFamily family) -> std::string {
    switch (family) {
        case FilterFamily::EKF:
            return "EKF";
        case FilterFamily::UKF:
            return "UKF";
        case FilterFamily::Both:
            return "both";
    }

    return "unknown";
}

auto expand_run_modes(RunMode selection) -> std::vector<RunMode> {
    if (selection == RunMode::Both) {
        return {RunMode::Comparison, RunMode::Imm};
    }

    return {selection};
}

auto expand_filter_families(FilterFamily selection) -> std::vector<FilterFamily> {
    if (selection == FilterFamily::Both) {
        return {FilterFamily::EKF, FilterFamily::UKF};
    }

    return {selection};
}

auto vector_to_json_array(const Eigen::VectorXd& values) -> nlohmann::json {
    nlohmann::json result = nlohmann::json::array();
    for (int i = 0; i < values.size(); ++i) {
        result.push_back(values(i));
    }
    return result;
}

auto matrix_to_json_array(const Eigen::MatrixXd& matrix) -> nlohmann::json {
    nlohmann::json result = nlohmann::json::array();
    for (int row = 0; row < matrix.rows(); ++row) {
        nlohmann::json row_json = nlohmann::json::array();
        for (int col = 0; col < matrix.cols(); ++col) {
            row_json.push_back(matrix(row, col));
        }
        result.push_back(std::move(row_json));
    }
    return result;
}

auto make_state_point(double time_seconds, const Eigen::VectorXd& state) -> nlohmann::json {
    return {
        {"time", time_seconds},
        {"state", vector_to_json_array(state)}
    };
}

auto make_measurement_point(
    double time_seconds,
    const std::string& sensor_id,
    const Eigen::VectorXd& measurement
) -> nlohmann::json {
    return {
        {"time", time_seconds},
        {"sensor_id", sensor_id},
        {"measurement", vector_to_json_array(measurement)}
    };
}

void append_error_fields(
    nlohmann::json& point,
    const Eigen::VectorXd& estimate,
    const Eigen::VectorXd& truth
) {
    point["position_error_m"] = (estimate.head<3>() - truth.head<3>()).norm();
    point["velocity_error_mps"] =
        (estimate.segment<3>(3) - truth.segment<3>(3)).norm();
}

auto compute_mahalanobis_squared(
    const Eigen::VectorXd& error,
    const Eigen::MatrixXd& covariance
) -> double {
    if (error.size() == 0 ||
        covariance.rows() != error.size() ||
        covariance.cols() != error.size()) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    Eigen::MatrixXd symmetric_covariance = 0.5 * (covariance + covariance.transpose());
    symmetric_covariance.diagonal().array() += 1e-9;

    Eigen::LDLT<Eigen::MatrixXd> ldlt(symmetric_covariance);
    if (ldlt.info() != Eigen::Success) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const Eigen::VectorXd solved = ldlt.solve(error);
    if (ldlt.info() != Eigen::Success || !solved.allFinite()) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const double value = error.dot(solved);
    return (value >= 0.0 && std::isfinite(value))
               ? value
               : std::numeric_limits<double>::quiet_NaN();
}

void append_consistency_fields(
    nlohmann::json& point,
    const Eigen::VectorXd& estimate,
    const Eigen::MatrixXd& covariance,
    const Eigen::VectorXd& truth
) {
    const int compare_dim = std::min(6, std::min(static_cast<int>(estimate.size()), static_cast<int>(truth.size())));
    const Eigen::VectorXd state_error = estimate.head(compare_dim) - truth.head(compare_dim);
    const double state_nees = compute_mahalanobis_squared(
        state_error,
        covariance.block(0, 0, compare_dim, compare_dim)
    );
    const double position_nees = compute_mahalanobis_squared(
        state_error.head<3>(),
        covariance.block(0, 0, 3, 3)
    );

    if (std::isfinite(state_nees)) {
        point["state_nees"] = state_nees;
        point["state_nees_per_dim"] = state_nees / static_cast<double>(compare_dim);
    }
    if (std::isfinite(position_nees)) {
        point["position_nees"] = position_nees;
        point["position_nees_per_dim"] = position_nees / 3.0;
    }
}

auto phase_index_for_time(
    double time_seconds,
    const std::vector<PhaseWindow>& phase_windows
) -> int {
    for (int i = 0; i < static_cast<int>(phase_windows.size()); ++i) {
        const bool is_last_phase = (i + 1 == static_cast<int>(phase_windows.size()));
        const auto& phase = phase_windows[static_cast<std::size_t>(i)];
        if (time_seconds >= phase.start_time_seconds &&
            (time_seconds < phase.end_time_seconds ||
             (is_last_phase && time_seconds <= phase.end_time_seconds))) {
            return i;
        }
    }

    return -1;
}

auto summarize_trajectory_points(
    const nlohmann::json& points,
    const std::vector<PhaseWindow>& phase_windows
) -> nlohmann::json {
    ScalarStats state_nees_stats;
    ScalarStats position_nees_stats;
    std::vector<ErrorMagnitudeStats> phase_error_stats(phase_windows.size());
    std::vector<ScalarStats> phase_state_nees_stats(phase_windows.size());
    std::vector<ScalarStats> phase_position_nees_stats(phase_windows.size());

    for (const auto& point : points) {
        if (!point.contains("time") ||
            !point.contains("position_error_m") ||
            !point.contains("velocity_error_mps")) {
            continue;
        }

        const double time_seconds = point.at("time").get<double>();
        const double position_error = point.at("position_error_m").get<double>();
        const double velocity_error = point.at("velocity_error_mps").get<double>();

        if (point.contains("state_nees")) {
            state_nees_stats.add_sample(point.at("state_nees").get<double>());
        }
        if (point.contains("position_nees")) {
            position_nees_stats.add_sample(point.at("position_nees").get<double>());
        }

        const int phase_index = phase_index_for_time(time_seconds, phase_windows);
        if (phase_index >= 0) {
            phase_error_stats[static_cast<std::size_t>(phase_index)].add_sample(
                position_error,
                velocity_error
            );
            if (point.contains("state_nees")) {
                phase_state_nees_stats[static_cast<std::size_t>(phase_index)].add_sample(
                    point.at("state_nees").get<double>()
                );
            }
            if (point.contains("position_nees")) {
                phase_position_nees_stats[static_cast<std::size_t>(phase_index)].add_sample(
                    point.at("position_nees").get<double>()
                );
            }
        }
    }

    nlohmann::json summary;
    summary["state_nees"] = state_nees_stats.to_json("state_nees");
    summary["position_nees"] = position_nees_stats.to_json("position_nees");

    nlohmann::json phase_metrics = nlohmann::json::array();
    for (int i = 0; i < static_cast<int>(phase_windows.size()); ++i) {
        const auto& phase = phase_windows[static_cast<std::size_t>(i)];
        nlohmann::json phase_json = phase_error_stats[static_cast<std::size_t>(i)].to_json();
        phase_json["name"] = phase.name;
        phase_json["start_time"] = phase.start_time_seconds;
        phase_json["end_time"] = phase.end_time_seconds;
        phase_json["state_nees"] =
            phase_state_nees_stats[static_cast<std::size_t>(i)].to_json("state_nees");
        phase_json["position_nees"] =
            phase_position_nees_stats[static_cast<std::size_t>(i)].to_json("position_nees");
        phase_metrics.push_back(std::move(phase_json));
    }

    summary["phase_metrics"] = phase_metrics;
    return summary;
}

auto clamp_phase_window(
    std::string name,
    double start_time_seconds,
    double end_time_seconds,
    double duration_seconds
) -> PhaseWindow {
    const double clamped_start = std::min(start_time_seconds, duration_seconds);
    const double clamped_end = std::min(end_time_seconds, duration_seconds);
    if (clamped_end < clamped_start) {
        return {std::move(name), duration_seconds, duration_seconds};
    }

    return {std::move(name), clamped_start, clamped_end};
}

auto altitude_from_state(const Eigen::VectorXd& state) -> double {
    return state.head<3>().norm() - kEarthRadiusMeters;
}

void append_model_probabilities(
    nlohmann::json& json_point,
    const Eigen::VectorXd& model_probabilities
) {
    json_point["model_probabilities"] = vector_to_json_array(model_probabilities);
}

auto top_model_indices(
    const Eigen::VectorXd& model_probabilities,
    int max_count
) -> std::vector<int> {
    std::vector<int> indices(static_cast<std::size_t>(model_probabilities.size()));
    for (int i = 0; i < model_probabilities.size(); ++i) {
        indices[static_cast<std::size_t>(i)] = i;
    }

    const int clamped_count = std::min(max_count, static_cast<int>(indices.size()));
    std::partial_sort(
        indices.begin(),
        indices.begin() + clamped_count,
        indices.end(),
        [&model_probabilities](int lhs, int rhs) {
            return model_probabilities(lhs) > model_probabilities(rhs);
        }
    );
    indices.resize(static_cast<std::size_t>(clamped_count));
    return indices;
}

auto top_error_indices(const std::vector<double>& errors, int max_count) -> std::vector<int> {
    std::vector<int> indices(errors.size());
    for (int i = 0; i < static_cast<int>(errors.size()); ++i) {
        indices[static_cast<std::size_t>(i)] = i;
    }

    const int clamped_count = std::min(max_count, static_cast<int>(indices.size()));
    std::partial_sort(
        indices.begin(),
        indices.begin() + clamped_count,
        indices.end(),
        [&errors](int lhs, int rhs) {
            return errors[static_cast<std::size_t>(lhs)] <
                   errors[static_cast<std::size_t>(rhs)];
        }
    );
    indices.resize(static_cast<std::size_t>(clamped_count));
    return indices;
}

auto rank_model_stats(const std::vector<ErrorStats>& model_stats) -> std::vector<int> {
    std::vector<int> ranking(model_stats.size());
    for (int i = 0; i < static_cast<int>(model_stats.size()); ++i) {
        ranking[static_cast<std::size_t>(i)] = i;
    }

    std::sort(
        ranking.begin(),
        ranking.end(),
        [&model_stats](int lhs, int rhs) {
            return model_stats[static_cast<std::size_t>(lhs)].pos_rmse() <
                   model_stats[static_cast<std::size_t>(rhs)].pos_rmse();
        }
    );
    return ranking;
}

auto compute_mixed_transition_jacobian(
    double start_time_seconds,
    const Eigen::VectorXd& filtered_state,
    double dt_seconds,
    const Eigen::VectorXd& model_probabilities,
    const std::vector<ModelSpec>& imm_model_specs
) -> Eigen::MatrixXd {
    const int state_dim = static_cast<int>(filtered_state.size());
    Eigen::MatrixXd mixed_jacobian = Eigen::MatrixXd::Zero(state_dim, state_dim);
    double weight_sum = 0.0;

    for (int i = 0;
         i < static_cast<int>(imm_model_specs.size()) && i < model_probabilities.size();
         ++i) {
        const auto& spec = imm_model_specs[static_cast<std::size_t>(i)];
        if (!spec.smoother_propagator) {
            continue;
        }

        const double weight = std::max(0.0, model_probabilities(i));
        if (!(weight > 0.0)) {
            continue;
        }

        const Eigen::MatrixXd jacobian = spec.smoother_propagator->compute_transition_jacobian(
            start_time_seconds,
            filtered_state,
            dt_seconds
        );
        if (jacobian.rows() != state_dim || jacobian.cols() != state_dim) {
            continue;
        }

        mixed_jacobian += weight * jacobian;
        weight_sum += weight;
    }

    if (!(weight_sum > 0.0)) {
        return Eigen::MatrixXd::Identity(state_dim, state_dim);
    }

    return mixed_jacobian / weight_sum;
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

struct TriangulatedPosition {
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    double rms_cross_track_residual_m = 0.0;
    int sensors_used = 0;
};

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
    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> sensor_lines;
    sensor_lines.reserve(measurements.size());

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
        sensor_lines.emplace_back(measurement.sensor_position, unit_los);
    }

    if (sensor_lines.size() < 2) {
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

    double residual_sq_sum = 0.0;
    for (const auto& [sensor_position, unit_los] : sensor_lines) {
        const Eigen::Vector3d miss_vector =
            (Eigen::Matrix3d::Identity() - unit_los * unit_los.transpose()) *
            (position - sensor_position);
        residual_sq_sum += miss_vector.squaredNorm();
    }

    TriangulatedPosition solution;
    solution.position = position;
    solution.rms_cross_track_residual_m = std::sqrt(
        residual_sq_sum / static_cast<double>(sensor_lines.size())
    );
    solution.sensors_used = static_cast<int>(sensor_lines.size());
    return solution;
}

auto make_initialization_seed(
    const ScenarioTrace& trace,
    const Eigen::VectorXd& fallback_initial_estimate,
    const Eigen::MatrixXd& fallback_initial_covariance
) -> InitializationSeed {
    InitializationSeed seed;
    seed.initial_estimate = fallback_initial_estimate;
    seed.initial_covariance = fallback_initial_covariance;

    if (trace.samples.size() < 2) {
        seed.detail =
            "Fixed-offset nominal prior (bootstrap unavailable: fewer than 2 measurement epochs).";
        return seed;
    }

    const auto first_position = triangulate_position_from_measurements(
        trace.samples[0].measurements
    );
    const auto second_position = triangulate_position_from_measurements(
        trace.samples[1].measurements
    );
    if (!first_position.has_value() || !second_position.has_value()) {
        seed.detail =
            "Fixed-offset nominal prior (bootstrap unavailable: insufficient line-of-sight geometry).";
        return seed;
    }

    const double dt_seconds = trace.samples[1].time_seconds - trace.samples[0].time_seconds;
    if (!(dt_seconds > 0.0)) {
        seed.detail =
            "Fixed-offset nominal prior (bootstrap unavailable: non-positive measurement spacing).";
        return seed;
    }

    const Eigen::Vector3d velocity_estimate =
        (second_position->position - first_position->position) / dt_seconds;
    if (!velocity_estimate.allFinite()) {
        seed.detail =
            "Fixed-offset nominal prior (bootstrap unavailable: non-finite velocity estimate).";
        return seed;
    }

    const double position_sigma_m = std::clamp(
        3.0 * std::max(
            first_position->rms_cross_track_residual_m,
            second_position->rms_cross_track_residual_m
        ),
        150.0,
        1500.0
    );
    const double velocity_sigma_mps = std::clamp(
        std::sqrt(2.0) * position_sigma_m / dt_seconds,
        60.0,
        300.0
    );

    Eigen::VectorXd bootstrap_state(6);
    bootstrap_state << second_position->position, velocity_estimate;

    Eigen::MatrixXd bootstrap_covariance = Eigen::MatrixXd::Zero(6, 6);
    bootstrap_covariance.block<3, 3>(0, 0) =
        Eigen::Matrix3d::Identity() * (position_sigma_m * position_sigma_m);
    bootstrap_covariance.block<3, 3>(3, 3) =
        Eigen::Matrix3d::Identity() * (velocity_sigma_mps * velocity_sigma_mps);

    seed.mode = "TwoEpochTriangulation";
    seed.detail =
        "Two-epoch LOS triangulation bootstrap from the first two space-based az/el batches.";
    seed.initial_time_seconds = trace.samples[1].time_seconds;
    seed.consumed_sample_count = 2;
    seed.epochs_used = 2;
    seed.sensors_used_per_epoch =
        std::min(first_position->sensors_used, second_position->sensors_used);
    seed.position_sigma_m = position_sigma_m;
    seed.velocity_sigma_mps = velocity_sigma_mps;
    seed.first_epoch_triangulation_residual_m = first_position->rms_cross_track_residual_m;
    seed.second_epoch_triangulation_residual_m = second_position->rms_cross_track_residual_m;
    seed.initial_estimate = bootstrap_state;
    seed.initial_covariance = bootstrap_covariance;
    return seed;
}

auto build_model_specs(
    const std::shared_ptr<integrator::RK4Integrator>& integrator,
    double integrator_step_seconds,
    const Eigen::VectorXd& initial_estimate_6d,
    const Eigen::MatrixXd& initial_covariance_6d,
    double initial_time_seconds,
    const Eigen::Vector3d& ca_model_accel_eci,
    const Eigen::Vector3d& smooth_boost_equilibrium_accel_eci,
    double boost_start_seconds,
    double boost_end_seconds,
    double target_mass_kg,
    double target_drag_coefficient,
    double target_reference_area_square_meters
) -> std::vector<ModelSpec> {
    const std::shared_ptr<propagator::IPropagator> cv_propagator =
        make_propagator(integrator, integrator_step_seconds, {});
    const std::shared_ptr<propagator::IPropagator> ca_propagator = make_propagator(
        integrator,
        integrator_step_seconds,
        {
            std::make_shared<dynamics::ConstantAccelerationForce>(ca_model_accel_eci)
        }
    );
    const std::shared_ptr<propagator::IPropagator> gravity_propagator = make_propagator(
        integrator,
        integrator_step_seconds,
        {
            std::make_shared<dynamics::PointMassGravity>()
        }
    );
    const std::shared_ptr<propagator::IPropagator> gravity_drag_propagator = make_propagator(
        integrator,
        integrator_step_seconds,
        {
            std::make_shared<dynamics::PointMassGravity>(),
            std::make_shared<dynamics::AtmosphericDrag>(
                target_mass_kg,
                target_drag_coefficient,
                target_reference_area_square_meters,
                kEarthRadiusMeters
            )
        }
    );
    const std::shared_ptr<propagator::IPropagator> j2_drag_propagator = make_propagator(
        integrator,
        integrator_step_seconds,
        {
            std::make_shared<dynamics::J2Gravity>(),
            std::make_shared<dynamics::AtmosphericDrag>(
                target_mass_kg,
                target_drag_coefficient,
                target_reference_area_square_meters,
                kEarthRadiusMeters
            )
        }
    );
    const std::shared_ptr<propagator::IPropagator> smooth_boost_propagator =
        make_propagator_from_dynamics(
            integrator,
            integrator_step_seconds,
            std::make_shared<dynamics::SmoothAccelerationPointMassDynamics>(
                std::vector<std::shared_ptr<dynamics::IForce>>{
                    std::make_shared<dynamics::PointMassGravity>(),
                    std::make_shared<dynamics::AtmosphericDrag>(
                        target_mass_kg,
                        target_drag_coefficient,
                        target_reference_area_square_meters,
                        kEarthRadiusMeters
                    )
                },
                9.0,
                [boost_start_seconds, boost_end_seconds, smooth_boost_equilibrium_accel_eci](
                    double time_seconds
                ) {
                    return (time_seconds >= boost_start_seconds &&
                            time_seconds <= boost_end_seconds)
                               ? smooth_boost_equilibrium_accel_eci
                               : Eigen::Vector3d::Zero();
                }
            )
        );

    Eigen::VectorXd smooth_boost_initial_estimate(9);
    smooth_boost_initial_estimate << initial_estimate_6d, smooth_boost_equilibrium_accel_eci;

    Eigen::MatrixXd smooth_boost_initial_covariance = Eigen::MatrixXd::Zero(9, 9);
    smooth_boost_initial_covariance.block(0, 0, 6, 6) = initial_covariance_6d;
    smooth_boost_initial_covariance.block<3, 3>(6, 6) =
        Eigen::Matrix3d::Identity() * (6.0 * 6.0);

    auto make_6d_spec = [
        &initial_estimate_6d,
        &initial_covariance_6d,
        initial_time_seconds
    ](
        std::string name,
        std::string motion_model,
        std::string description,
        double process_noise_sigma_accel,
        const std::shared_ptr<propagator::IPropagator>& propagator
    ) -> ModelSpec {
        ModelSpec spec;
        spec.name = name;
        spec.motion_model = motion_model;
        spec.description = description;
        spec.process_noise_sigma_accel = process_noise_sigma_accel;
        spec.include_in_imm = true;
        spec.smoother_propagator = propagator;
        spec.factory =
            [
                initial_estimate_6d,
                initial_covariance_6d,
                propagator,
                name,
                motion_model,
                description,
                process_noise_sigma_accel,
                initial_time_seconds
            ](const std::shared_ptr<sensor::ISensorModel>& sensor_model, FilterFamily family) {
                if (family == FilterFamily::EKF) {
                    return make_ekf_model(
                        initial_estimate_6d,
                        initial_covariance_6d,
                        sensor_model,
                        propagator,
                        name,
                        motion_model,
                        description,
                        process_noise_sigma_accel,
                        initial_time_seconds
                    );
                }
                if (family == FilterFamily::UKF) {
                    return make_ukf_model(
                        initial_estimate_6d,
                        initial_covariance_6d,
                        sensor_model,
                        propagator,
                        name,
                        motion_model,
                        description,
                        process_noise_sigma_accel,
                        initial_time_seconds
                    );
                }

                throw std::invalid_argument("Model factory requires EKF or UKF family");
            };
        return spec;
    };

    return {
        make_6d_spec(
            "CV",
            "ConstantVelocity",
            "Kinematic constant-velocity hypothesis with no explicit forces.",
            35.0,
            cv_propagator
        ),
        make_6d_spec(
            "CA",
            "ConstantAcceleration",
            "Kinematic constant-acceleration hypothesis aligned with the nominal ballistic corridor.",
            18.0,
            ca_propagator
        ),
        make_6d_spec(
            "Gravity",
            "Gravity",
            "Point-mass gravity only, without drag.",
            9.0,
            gravity_propagator
        ),
        make_6d_spec(
            "GravityDrag",
            "GravityPlusDrag",
            "Point-mass gravity with atmospheric drag in the inertial frame.",
            6.0,
            gravity_drag_propagator
        ),
        make_6d_spec(
            "J2Drag",
            "J2PlusDrag",
            "J2 gravity with atmospheric drag for a more detailed ballistic model.",
            4.5,
            j2_drag_propagator
        ),
        ModelSpec{
            "BoostSmooth",
            "BoostSmoothAcceleration",
            "Gravity and drag with a first-order smooth acceleration state tied to a generic boost equilibrium vector instead of a thrust profile.",
            4.0,
            false,
            smooth_boost_propagator,
            [
                smooth_boost_initial_estimate,
                smooth_boost_initial_covariance,
                smooth_boost_propagator,
                initial_time_seconds
            ](const std::shared_ptr<sensor::ISensorModel>& sensor_model, FilterFamily family) {
                if (family == FilterFamily::EKF) {
                    return make_ekf_model_with_process_noise(
                        smooth_boost_initial_estimate,
                        smooth_boost_initial_covariance,
                        sensor_model,
                        smooth_boost_propagator,
                        "BoostSmooth",
                        "BoostSmoothAcceleration",
                        "Gravity and drag with a first-order smooth acceleration state tied to a generic boost equilibrium vector instead of a thrust profile.",
                        4.0,
                        initial_time_seconds,
                        make_smooth_acceleration_state_process_noise(4.0)
                    );
                }
                if (family == FilterFamily::UKF) {
                    return make_ukf_model_with_process_noise(
                        smooth_boost_initial_estimate,
                        smooth_boost_initial_covariance,
                        sensor_model,
                        smooth_boost_propagator,
                        "BoostSmooth",
                        "BoostSmoothAcceleration",
                        "Gravity and drag with a first-order smooth acceleration state tied to a generic boost equilibrium vector instead of a thrust profile.",
                        4.0,
                        initial_time_seconds,
                        make_smooth_acceleration_state_process_noise(4.0)
                    );
                }

                throw std::invalid_argument("Model factory requires EKF or UKF family");
            }
        }
    };
}

auto build_model_bank(
    FilterFamily family,
    const std::shared_ptr<sensor::ISensorModel>& sensor_model,
    const std::vector<ModelSpec>& model_specs,
    bool imm_compatible_only
) -> std::vector<ModelBuild> {
    std::vector<ModelBuild> model_builds;
    model_builds.reserve(model_specs.size());

    for (const auto& spec : model_specs) {
        if (imm_compatible_only && !spec.include_in_imm) {
            continue;
        }
        model_builds.push_back(spec.factory(sensor_model, family));
    }

    return model_builds;
}

auto record_scenario_trace(
    const Eigen::VectorXd& initial_truth_state,
    const Eigen::VectorXd& initial_estimate,
    const Eigen::MatrixXd& initial_covariance,
    const std::shared_ptr<propagator::IPropagator>& truth_propagator,
    const std::shared_ptr<sensor::ISensorModel>& sensor_model,
    const std::vector<SensorPlatform>& sensor_platforms,
    int num_steps,
    double dt_seconds,
    unsigned int seed
) -> ScenarioTrace {
    ScenarioTrace trace;
    trace.initial_truth_state = initial_truth_state;
    trace.initial_estimate = initial_estimate;
    trace.initial_covariance = initial_covariance;

    std::mt19937 rng(seed);

    double time_seconds = 0.0;
    Eigen::VectorXd truth_state = initial_truth_state;
    trace.truth_points.push_back(make_state_point(0.0, truth_state));

    trace.samples.reserve(static_cast<std::size_t>(num_steps));
    for (int step = 0; step < num_steps; ++step) {
        const double next_time_seconds = time_seconds + dt_seconds;
        const auto truth_trajectory = truth_propagator->propagate(
            time_seconds,
            truth_state,
            next_time_seconds
        );
        truth_state = truth_trajectory.back().second;
        time_seconds = next_time_seconds;

        std::vector<common::Measurement> measurements;
        measurements.reserve(sensor_platforms.size());
        for (int sensor_index = 0; sensor_index < static_cast<int>(sensor_platforms.size()); ++sensor_index) {
            const auto& sensor_platform = sensor_platforms[static_cast<std::size_t>(sensor_index)];
            std::normal_distribution<double> az_noise(0.0, sensor_platform.spec.az_noise_sigma_rad);
            std::normal_distribution<double> el_noise(0.0, sensor_platform.spec.el_noise_sigma_rad);

            sensor::SensorContext sensor_ctx;
            sensor_ctx.state = truth_state;
            sensor_ctx.time = time_seconds;
            const SensorKinematics sensor_state = sensor_platform.state_eci(time_seconds);
            sensor_ctx.sensor_position = sensor_state.position;
            sensor_ctx.sensor_orientation = sensor_state.orientation;

            Eigen::VectorXd measurement_vector = sensor_model->compute_measurement(sensor_ctx);
            measurement_vector(0) += az_noise(rng);
            measurement_vector(1) += el_noise(rng);

            common::Measurement measurement(
                measurement_vector,
                sensor_model->get_noise_covariance(),
                time_seconds
            );
            measurement.sensor_position = sensor_ctx.sensor_position;
            measurement.sensor_orientation = sensor_ctx.sensor_orientation;
            measurement.sensor_id = sensor_platform.spec.id;
            measurement.measurement_id =
                step * static_cast<int>(sensor_platforms.size()) + sensor_index;

            measurements.push_back(measurement);
            trace.measurement_points.push_back(make_measurement_point(
                time_seconds,
                sensor_platform.spec.id,
                measurement_vector
            ));
        }

        trace.samples.push_back({time_seconds, truth_state, std::move(measurements)});
        trace.truth_points.push_back(make_state_point(time_seconds, truth_state));
    }

    return trace;
}

void print_model_descriptions(const std::vector<ModelInfo>& model_infos) {
    std::cout << "Models:\n";
    for (int i = 0; i < static_cast<int>(model_infos.size()); ++i) {
        const auto& info = model_infos[static_cast<std::size_t>(i)];
        std::cout << "  [" << i << "] " << info.name << " -> " << info.description << '\n';
    }
}

auto run_standalone_comparison(
    const ScenarioTrace& trace,
    FilterFamily family,
    const std::shared_ptr<sensor::ISensorModel>& sensor_model,
    const std::vector<ModelSpec>& model_specs,
    const std::vector<PhaseWindow>& phase_windows,
    int print_every
) -> DemoRunResult {
    DemoRunResult result;
    result.run_name = "Comparison-" + filter_family_to_string(family);
    result.mode = run_mode_to_string(RunMode::Comparison);
    result.filter_family = filter_family_to_string(family);

    auto model_builds = build_model_bank(
        family,
        sensor_model,
        model_specs,
        false
    );

    std::vector<std::unique_ptr<filtering::IKalmanFilter>> filters;
    filters.reserve(model_builds.size());
    result.model_infos.reserve(model_builds.size());
    for (auto& build : model_builds) {
        result.model_infos.push_back(build.info);
        filters.push_back(std::move(build.filter));
    }

    const int num_models = static_cast<int>(filters.size());
    result.model_points.resize(static_cast<std::size_t>(num_models), nlohmann::json::array());
    result.model_stats.resize(static_cast<std::size_t>(num_models));

    for (int i = 0; i < num_models; ++i) {
        const Eigen::VectorXd state = filters[static_cast<std::size_t>(i)]->get_state();
        nlohmann::json point = make_state_point(trace.initial_time_seconds, state);
        const int phase_index = phase_index_for_time(trace.initial_time_seconds, phase_windows);
        point["phase"] = (phase_index >= 0)
                             ? phase_windows[static_cast<std::size_t>(phase_index)].name
                             : std::string("Unspecified");
        append_error_fields(point, state, trace.initial_truth_state);
        append_consistency_fields(
            point,
            state,
            filters[static_cast<std::size_t>(i)]->get_covariance(),
            trace.initial_truth_state
        );
        result.model_points[static_cast<std::size_t>(i)].push_back(std::move(point));
        result.model_stats[static_cast<std::size_t>(i)].add_sample(state, trace.initial_truth_state);
    }

    std::cout << "\n" << run_mode_to_display_string(RunMode::Comparison)
              << " mode [" << result.filter_family << "]: standalone motion-model evaluation\n";
    print_model_descriptions(result.model_infos);
    std::cout << '\n';
    std::cout << std::fixed << std::setprecision(3);
    std::cout << std::setw(6) << "t[s]"
              << std::setw(14) << "tgt_alt[km]"
              << std::setw(18) << "top model"
              << std::setw(16) << "pos err[m]"
              << std::setw(16) << "vel err[m/s]"
              << std::setw(18) << "runner-up"
              << std::setw(16) << "pos err[m]"
              << '\n';

    for (int step = trace.first_update_sample_index;
         step < static_cast<int>(trace.samples.size());
         ++step) {
        const auto& sample = trace.samples[static_cast<std::size_t>(step)];
        std::vector<double> current_pos_errors(static_cast<std::size_t>(num_models), 0.0);
        std::vector<double> current_vel_errors(static_cast<std::size_t>(num_models), 0.0);

        for (int i = 0; i < num_models; ++i) {
            auto& filter = filters[static_cast<std::size_t>(i)];
            const double dt = sample.time_seconds - filter->get_time();
            filter->predict(dt);
            for (const auto& measurement : sample.measurements) {
                filter->update(measurement);
            }

            const Eigen::VectorXd state = filter->get_state();
            nlohmann::json point = make_state_point(sample.time_seconds, state);
            const int phase_index = phase_index_for_time(sample.time_seconds, phase_windows);
            point["phase"] = (phase_index >= 0)
                                 ? phase_windows[static_cast<std::size_t>(phase_index)].name
                                 : std::string("Unspecified");
            append_error_fields(point, state, sample.truth_state);
            append_consistency_fields(
                point,
                state,
                filter->get_covariance(),
                sample.truth_state
            );
            result.model_points[static_cast<std::size_t>(i)].push_back(std::move(point));
            result.model_stats[static_cast<std::size_t>(i)].add_sample(state, sample.truth_state);

            current_pos_errors[static_cast<std::size_t>(i)] =
                (state.head<3>() - sample.truth_state.head<3>()).norm();
            current_vel_errors[static_cast<std::size_t>(i)] =
                (state.segment<3>(3) - sample.truth_state.segment<3>(3)).norm();
        }

        if ((step + 1) % print_every == 0 ||
            step == trace.first_update_sample_index ||
            step + 1 == trace.samples.size()) {
            const auto best_indices = top_error_indices(current_pos_errors, 2);
            const int best_model = best_indices.empty() ? -1 : best_indices.front();
            const int second_model = (best_indices.size() > 1) ? best_indices[1] : -1;

            std::cout << std::setw(6) << sample.time_seconds
                      << std::setw(14) << altitude_from_state(sample.truth_state) / 1000.0
                      << std::setw(18)
                      << ((best_model >= 0)
                              ? result.model_infos[static_cast<std::size_t>(best_model)].name
                              : std::string("N/A"))
                      << std::setw(16)
                      << ((best_model >= 0)
                              ? current_pos_errors[static_cast<std::size_t>(best_model)]
                              : 0.0)
                      << std::setw(16)
                      << ((best_model >= 0)
                              ? current_vel_errors[static_cast<std::size_t>(best_model)]
                              : 0.0)
                      << std::setw(18)
                      << ((second_model >= 0)
                              ? result.model_infos[static_cast<std::size_t>(second_model)].name
                              : std::string("N/A"))
                      << std::setw(16)
                      << ((second_model >= 0)
                              ? current_pos_errors[static_cast<std::size_t>(second_model)]
                              : 0.0)
                      << '\n';
        }
    }

    result.ranking = rank_model_stats(result.model_stats);
    result.best_model_index = result.ranking.empty() ? -1 : result.ranking.front();

    std::cout << "\nFinal standalone RMSE ranking [" << result.filter_family << "]:\n";
    for (int rank = 0; rank < num_models; ++rank) {
        const int model_index = result.ranking[static_cast<std::size_t>(rank)];
        const auto& info = result.model_infos[static_cast<std::size_t>(model_index)];
        const auto& stats = result.model_stats[static_cast<std::size_t>(model_index)];
        std::cout << "  " << (rank + 1) << ". " << info.name
                  << " -> pos RMSE " << stats.pos_rmse()
                  << " m, vel RMSE " << stats.vel_rmse() << " m/s\n";
    }

    return result;
}

auto run_imm_family(
    const ScenarioTrace& trace,
    FilterFamily family,
    const std::shared_ptr<sensor::ISensorModel>& sensor_model,
    const std::vector<ModelSpec>& model_specs,
    const std::vector<PhaseWindow>& phase_windows,
    int print_every
) -> DemoRunResult {
    DemoRunResult result;
    result.run_name = "IMM-" + filter_family_to_string(family);
    result.mode = run_mode_to_string(RunMode::Imm);
    result.filter_family = filter_family_to_string(family);
    result.has_combined = true;

    auto model_builds = build_model_bank(
        family,
        sensor_model,
        model_specs,
        true
    );
    std::vector<ModelSpec> imm_model_specs;
    imm_model_specs.reserve(model_specs.size());
    for (const auto& spec : model_specs) {
        if (spec.include_in_imm) {
            imm_model_specs.push_back(spec);
        }
    }

    std::vector<std::unique_ptr<filtering::IKalmanFilter>> filters;
    filters.reserve(model_builds.size());
    result.model_infos.reserve(model_builds.size());
    for (auto& build : model_builds) {
        result.model_infos.push_back(build.info);
        filters.push_back(std::move(build.filter));
    }

    const int num_models = static_cast<int>(filters.size());
    std::vector<estimation::SmootherEstimate> filtered_records;
    filtered_records.reserve(trace.samples.size() + 1);
    std::vector<Eigen::VectorXd> filtered_truth_states;
    filtered_truth_states.reserve(trace.samples.size() + 1);
    std::vector<estimation::PredictedEstimate> predicted_transitions;
    predicted_transitions.reserve(trace.samples.size());
    result.initial_model_probabilities =
        Eigen::VectorXd::Constant(num_models, 1.0 / static_cast<double>(num_models));
    result.transition_matrix = Eigen::MatrixXd::Constant(
        num_models,
        num_models,
        0.06 / static_cast<double>(num_models - 1)
    );
    result.transition_matrix.diagonal().setConstant(0.94);

    estimation::IMM imm(
        std::move(filters),
        result.initial_model_probabilities,
        result.transition_matrix
    );

    result.model_points.resize(static_cast<std::size_t>(num_models), nlohmann::json::array());
    result.model_stats.resize(static_cast<std::size_t>(num_models));

    auto append_combined_point = [&](double time_seconds, const Eigen::VectorXd& truth_state) {
        nlohmann::json point = make_state_point(time_seconds, imm.get_state());
        const Eigen::VectorXd model_probabilities = imm.get_model_probabilities();
        const int phase_index = phase_index_for_time(time_seconds, phase_windows);
        point["phase"] = (phase_index >= 0)
                             ? phase_windows[static_cast<std::size_t>(phase_index)].name
                             : std::string("Unspecified");
        append_model_probabilities(point, model_probabilities);
        point["most_likely_model"] = imm.get_most_likely_model();
        append_error_fields(point, imm.get_state(), truth_state);
        append_consistency_fields(point, imm.get_state(), imm.get_covariance(), truth_state);
        result.combined_points.push_back(std::move(point));
    };

    append_combined_point(trace.initial_time_seconds, trace.initial_truth_state);
    result.combined_stats.add_sample(imm.get_state(), trace.initial_truth_state);
    filtered_records.push_back({
        trace.initial_time_seconds,
        imm.get_state(),
        imm.get_covariance()
    });
    filtered_truth_states.push_back(trace.initial_truth_state);

    const Eigen::VectorXd initial_mu = imm.get_model_probabilities();
    for (int i = 0; i < num_models; ++i) {
        const Eigen::VectorXd state = imm.get_model_state(i);
        nlohmann::json point = make_state_point(trace.initial_time_seconds, state);
        const int phase_index = phase_index_for_time(trace.initial_time_seconds, phase_windows);
        point["phase"] = (phase_index >= 0)
                             ? phase_windows[static_cast<std::size_t>(phase_index)].name
                             : std::string("Unspecified");
        point["mode_probability"] = initial_mu(i);
        append_error_fields(point, state, trace.initial_truth_state);
        append_consistency_fields(
            point,
            state,
            imm.get_model_covariance(i),
            trace.initial_truth_state
        );
        result.model_points[static_cast<std::size_t>(i)].push_back(std::move(point));
        result.model_stats[static_cast<std::size_t>(i)].add_sample(state, trace.initial_truth_state);
    }

    std::cout << "\n" << run_mode_to_display_string(RunMode::Imm)
              << " mode [" << result.filter_family
              << "]: motion-model IMM over the same measurement stream\n";
    print_model_descriptions(result.model_infos);
    std::cout << '\n';
    std::cout << std::fixed << std::setprecision(3);
    std::cout << std::setw(6) << "t[s]"
              << std::setw(14) << "tgt_alt[km]"
              << std::setw(16) << "IMM pos[m]"
              << std::setw(16) << "IMM vel[m/s]"
              << std::setw(18) << "top model"
              << std::setw(12) << "mu"
              << std::setw(18) << "runner-up"
              << std::setw(12) << "mu"
              << '\n';

    for (int step = trace.first_update_sample_index;
         step < static_cast<int>(trace.samples.size());
         ++step) {
        const auto& sample = trace.samples[static_cast<std::size_t>(step)];
        const Eigen::VectorXd filtered_state_before_predict = imm.get_state();
        const Eigen::VectorXd model_probabilities_before_predict =
            imm.get_model_probabilities();
        const double predict_start_time = imm.get_time();
        const double dt = sample.time_seconds - predict_start_time;
        const Eigen::MatrixXd mixed_transition_jacobian =
            compute_mixed_transition_jacobian(
                predict_start_time,
                filtered_state_before_predict,
                dt,
                model_probabilities_before_predict,
                imm_model_specs
            );
        imm.predict(dt);
        predicted_transitions.push_back({
            predict_start_time,
            sample.time_seconds,
            imm.get_state(),
            imm.get_covariance(),
            mixed_transition_jacobian
        });
        for (const auto& measurement : sample.measurements) {
            imm.update(measurement);
        }

        append_combined_point(sample.time_seconds, sample.truth_state);
        result.combined_stats.add_sample(imm.get_state(), sample.truth_state);
        filtered_records.push_back({
            sample.time_seconds,
            imm.get_state(),
            imm.get_covariance()
        });
        filtered_truth_states.push_back(sample.truth_state);

        const Eigen::VectorXd mu = imm.get_model_probabilities();
        for (int i = 0; i < num_models; ++i) {
            const Eigen::VectorXd state = imm.get_model_state(i);
            nlohmann::json point = make_state_point(sample.time_seconds, state);
            const int phase_index = phase_index_for_time(sample.time_seconds, phase_windows);
            point["phase"] = (phase_index >= 0)
                                 ? phase_windows[static_cast<std::size_t>(phase_index)].name
                                 : std::string("Unspecified");
            point["mode_probability"] = mu(i);
            append_error_fields(point, state, sample.truth_state);
            append_consistency_fields(
                point,
                state,
                imm.get_model_covariance(i),
                sample.truth_state
            );
            result.model_points[static_cast<std::size_t>(i)].push_back(std::move(point));
            result.model_stats[static_cast<std::size_t>(i)].add_sample(state, sample.truth_state);
        }

        if ((step + 1) % print_every == 0 ||
            step == trace.first_update_sample_index ||
            step + 1 == trace.samples.size()) {
            const auto best_indices = top_model_indices(mu, 2);
            const int best_model = best_indices.empty() ? -1 : best_indices.front();
            const int second_model = (best_indices.size() > 1) ? best_indices[1] : -1;

            std::cout << std::setw(6) << sample.time_seconds
                      << std::setw(14) << altitude_from_state(sample.truth_state) / 1000.0
                      << std::setw(16) << result.combined_stats.final_pos_error
                      << std::setw(16) << result.combined_stats.final_vel_error
                      << std::setw(18)
                      << ((best_model >= 0)
                              ? result.model_infos[static_cast<std::size_t>(best_model)].name
                              : std::string("N/A"))
                      << std::setw(12)
                      << ((best_model >= 0) ? mu(best_model) : 0.0)
                      << std::setw(18)
                      << ((second_model >= 0)
                              ? result.model_infos[static_cast<std::size_t>(second_model)].name
                              : std::string("N/A"))
                      << std::setw(12)
                      << ((second_model >= 0) ? mu(second_model) : 0.0)
                      << '\n';
        }
    }

    const estimation::RTSSmoother smoother;
    const std::vector<estimation::SmootherEstimate> smoothed_records = smoother.smooth(
        filtered_records,
        predicted_transitions
    );
    if (!smoothed_records.empty()) {
        result.has_smoothed_combined = true;
        for (std::size_t i = 0; i < smoothed_records.size(); ++i) {
            const auto& record = smoothed_records[i];
            const Eigen::VectorXd& truth_state = filtered_truth_states[i];

            nlohmann::json point = make_state_point(record.time_seconds, record.state);
            const int phase_index = phase_index_for_time(record.time_seconds, phase_windows);
            point["phase"] = (phase_index >= 0)
                                 ? phase_windows[static_cast<std::size_t>(phase_index)].name
                                 : std::string("Unspecified");
            point["smoother"] = "RTS";
            append_error_fields(point, record.state, truth_state);
            append_consistency_fields(
                point,
                record.state,
                record.covariance,
                truth_state
            );
            result.smoothed_combined_points.push_back(std::move(point));
            result.smoothed_combined_stats.add_sample(
                record.state,
                truth_state
            );
        }
    }

    result.final_model_probabilities = imm.get_model_probabilities();
    result.most_likely_model = imm.get_most_likely_model();
    result.ranking = rank_model_stats(result.model_stats);

    for (int i = 0; i < num_models; ++i) {
        const auto& info = result.model_infos[static_cast<std::size_t>(i)];
        result.final_motion_model_probabilities[info.motion_model] +=
            result.final_model_probabilities(i);
    }

    std::cout << "\nFinal IMM result [" << result.filter_family << "]: "
              << result.model_infos.at(static_cast<std::size_t>(result.most_likely_model)).name
              << " (mu = " << result.final_model_probabilities(result.most_likely_model) << ")\n";
    std::cout << "Final probability by motion model:\n";
    for (const auto& [motion_model, probability] : result.final_motion_model_probabilities) {
        std::cout << "  " << motion_model << " -> " << probability << '\n';
    }
    std::cout << "Position RMSE ranking:\n";
    for (int rank = 0; rank < num_models; ++rank) {
        const int model_index = result.ranking[static_cast<std::size_t>(rank)];
        const auto& info = result.model_infos[static_cast<std::size_t>(model_index)];
        const auto& stats = result.model_stats[static_cast<std::size_t>(model_index)];
        std::cout << "  " << (rank + 1) << ". " << info.name
                  << " -> pos RMSE " << stats.pos_rmse()
                  << " m, vel RMSE " << stats.vel_rmse()
                  << " m/s, final mu " << result.final_model_probabilities(model_index) << '\n';
    }
    std::cout << "  IMMCombined -> pos RMSE " << result.combined_stats.pos_rmse()
              << " m, vel RMSE " << result.combined_stats.vel_rmse() << " m/s\n";
    if (result.has_smoothed_combined) {
        std::cout << "  IMMCombinedRTS -> pos RMSE "
                  << result.smoothed_combined_stats.pos_rmse()
                  << " m, vel RMSE "
                  << result.smoothed_combined_stats.vel_rmse() << " m/s\n";
    }

    return result;
}

auto model_summary_json(const ModelInfo& info, int index) -> nlohmann::json {
    return {
        {"index", index},
        {"name", info.name},
        {"filter_type", info.filter_type},
        {"motion_model", info.motion_model},
        {"description", info.description},
        {"process_noise_sigma_accel_mps2", info.process_noise_sigma_accel}
    };
}

auto run_summary_json(
    const DemoRunResult& run,
    const std::vector<PhaseWindow>& phase_windows
) -> nlohmann::json {
    nlohmann::json run_json;
    run_json["name"] = run.run_name;
    run_json["mode"] = run.mode;
    run_json["filter_family"] = run.filter_family;

    nlohmann::json models_json = nlohmann::json::array();
    nlohmann::json model_performance_json = nlohmann::json::array();
    for (int i = 0; i < static_cast<int>(run.model_infos.size()); ++i) {
        const auto& info = run.model_infos[static_cast<std::size_t>(i)];
        models_json.push_back(model_summary_json(info, i));

        nlohmann::json performance = run.model_stats[static_cast<std::size_t>(i)].to_json();
        performance["index"] = i;
        performance["name"] = info.name;
        performance["filter_type"] = info.filter_type;
        performance["motion_model"] = info.motion_model;
        const nlohmann::json trajectory_summary = summarize_trajectory_points(
            run.model_points[static_cast<std::size_t>(i)],
            phase_windows
        );
        performance["state_nees"] = trajectory_summary.at("state_nees");
        performance["position_nees"] = trajectory_summary.at("position_nees");
        performance["phase_metrics"] = trajectory_summary.at("phase_metrics");
        if (run.has_combined && run.final_model_probabilities.size() == static_cast<int>(run.model_infos.size())) {
            performance["final_model_probability"] = run.final_model_probabilities(i);
        }
        model_performance_json.push_back(std::move(performance));
    }

    nlohmann::json ranking_json = nlohmann::json::array();
    for (std::size_t rank = 0; rank < run.ranking.size(); ++rank) {
        const int model_index = run.ranking[rank];
        ranking_json.push_back({
            {"rank", static_cast<int>(rank + 1)},
            {"model_index", model_index},
            {"name", run.model_infos.at(static_cast<std::size_t>(model_index)).name},
            {"position_rmse_m", run.model_stats.at(static_cast<std::size_t>(model_index)).pos_rmse()},
            {"velocity_rmse_mps", run.model_stats.at(static_cast<std::size_t>(model_index)).vel_rmse()}
        });
    }

    run_json["models"] = models_json;
    run_json["performance"]["models"] = model_performance_json;
    run_json["performance"]["position_rmse_ranking"] = ranking_json;

    if (run.has_combined) {
        nlohmann::json motion_probabilities_json;
        for (const auto& [motion_model, probability] : run.final_motion_model_probabilities) {
            motion_probabilities_json[motion_model] = probability;
        }

        run_json["initial_model_probabilities"] =
            vector_to_json_array(run.initial_model_probabilities);
        run_json["transition_matrix"] = matrix_to_json_array(run.transition_matrix);
        run_json["final_model_probabilities"] =
            vector_to_json_array(run.final_model_probabilities);
        run_json["final_motion_model_probabilities"] = motion_probabilities_json;
        run_json["most_likely_model"] = run.most_likely_model;
        run_json["most_likely_model_name"] =
            run.model_infos.at(static_cast<std::size_t>(run.most_likely_model)).name;
        run_json["performance"]["combined"] = run.combined_stats.to_json();
        const nlohmann::json combined_summary =
            summarize_trajectory_points(run.combined_points, phase_windows);
        run_json["performance"]["combined"]["state_nees"] =
            combined_summary.at("state_nees");
        run_json["performance"]["combined"]["position_nees"] =
            combined_summary.at("position_nees");
        run_json["performance"]["combined"]["phase_metrics"] =
            combined_summary.at("phase_metrics");
        if (run.has_smoothed_combined) {
            run_json["performance"]["smoothed_combined"] =
                run.smoothed_combined_stats.to_json();
            const nlohmann::json smoothed_summary =
                summarize_trajectory_points(run.smoothed_combined_points, phase_windows);
            run_json["performance"]["smoothed_combined"]["state_nees"] =
                smoothed_summary.at("state_nees");
            run_json["performance"]["smoothed_combined"]["position_nees"] =
                smoothed_summary.at("position_nees");
            run_json["performance"]["smoothed_combined"]["phase_metrics"] =
                smoothed_summary.at("phase_metrics");
        }
    } else {
        run_json["best_model_index"] = run.best_model_index;
        run_json["best_model_name"] =
            run.model_infos.at(static_cast<std::size_t>(run.best_model_index)).name;
    }

    return run_json;
}

} // namespace

int main(int argc, char* argv[]) {
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "Produce help message")
        ("output,o", po::value<std::string>()->default_value("imm_demo.json"),
            "Output JSON file with demo trajectory data")
        ("steps", po::value<int>()->default_value(180),
            "Number of sensor update steps to simulate")
        ("dt", po::value<double>()->default_value(1.0),
            "Sensor update period in seconds")
        ("seed", po::value<unsigned int>()->default_value(42),
            "Random seed used for measurement noise")
        ("print-every", po::value<int>()->default_value(15),
            "Print a console summary every N filter updates")
        ("mode", po::value<std::string>()->default_value("both"),
            "Run mode: comparison, imm, or both")
        ("filter-family", po::value<std::string>()->default_value("both"),
            "Filter family: ekf, ukf, or both")
        ("sensor-count", po::value<int>()->default_value(3),
            "Number of space-based az/el sensors to use (1 to 5)");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    if (vm.count("help")) {
        std::cout << desc << '\n';
        return 0;
    }

    po::notify(vm);

    const std::string output_file = vm["output"].as<std::string>();
    const int num_steps = vm["steps"].as<int>();
    const double dt_seconds = vm["dt"].as<double>();
    const unsigned int seed = vm["seed"].as<unsigned int>();
    const int print_every = vm["print-every"].as<int>();
    const int sensor_count = vm["sensor-count"].as<int>();

    RunMode run_mode = RunMode::Both;
    FilterFamily filter_family = FilterFamily::Both;
    try {
        run_mode = parse_run_mode(vm["mode"].as<std::string>());
        filter_family = parse_filter_family(vm["filter-family"].as<std::string>());
    } catch (const std::invalid_argument& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }

    if (num_steps <= 0) {
        std::cerr << "steps must be positive\n";
        return 1;
    }
    if (dt_seconds <= 0.0) {
        std::cerr << "dt must be positive\n";
        return 1;
    }
    if (print_every <= 0) {
        std::cerr << "print-every must be positive\n";
        return 1;
    }
    if (sensor_count < 1 || sensor_count > 5) {
        std::cerr << "sensor-count must be between 1 and 5\n";
        return 1;
    }

    constexpr double kIntegratorStepSeconds = 0.1;
    constexpr double kLaunchLatitudeDeg = 28.573255;
    constexpr double kLaunchLongitudeDeg = -80.646895;
    constexpr double kLaunchHeadingDeg = 64.0;
    constexpr double kFlightPathDeg = 21.0;
    constexpr double kInitialAltitudeMeters = 120000.0;
    constexpr double kInitialSpeedMetersPerSecond = 6100.0;
    constexpr double kTargetMassKg = 900.0;
    constexpr double kTargetDragCoefficient = 0.35;
    constexpr double kTargetReferenceAreaSquareMeters = 1.1;
    constexpr double kBoostStartSeconds = 0.0;
    constexpr double kBoostEndSeconds = 26.0;
    constexpr double kDivertStartSeconds = 145.0;
    constexpr double kDivertEndSeconds = 158.0;
    constexpr double kAzNoiseRad = 70e-6;
    constexpr double kElNoiseRad = 70e-6;

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

    const Eigen::Vector3d downrange_unit_enu(
        std::sin(launch_heading),
        std::cos(launch_heading),
        0.0
    );
    const Eigen::Vector3d crossrange_unit_enu(
        std::cos(launch_heading),
        -std::sin(launch_heading),
        0.0
    );

    const Eigen::Vector3d initial_offset_enu =
        45000.0 * downrange_unit_enu + Eigen::Vector3d(0.0, 0.0, kInitialAltitudeMeters);
    const Eigen::Vector3d initial_position =
        kEarthRadiusMeters * up + enu_to_eci * initial_offset_enu;

    const Eigen::Vector3d initial_velocity_enu =
        kInitialSpeedMetersPerSecond * std::cos(flight_path) * downrange_unit_enu +
        Eigen::Vector3d(0.0, 0.0, kInitialSpeedMetersPerSecond * std::sin(flight_path));
    const Eigen::Vector3d initial_velocity = enu_to_eci * initial_velocity_enu;

    Eigen::VectorXd truth_state(6);
    truth_state << initial_position, initial_velocity;

    const Eigen::Vector3d initial_pos_error_enu(3500.0, -2800.0, 2200.0);
    const Eigen::Vector3d initial_vel_error_enu(-120.0, 75.0, -90.0);
    Eigen::VectorXd fallback_initial_estimate(6);
    fallback_initial_estimate << initial_position + enu_to_eci * initial_pos_error_enu,
                                 initial_velocity + enu_to_eci * initial_vel_error_enu;

    Eigen::MatrixXd fallback_initial_covariance = Eigen::MatrixXd::Zero(6, 6);
    fallback_initial_covariance.block<3, 3>(0, 0) =
        Eigen::Matrix3d::Identity() * (5000.0 * 5000.0);
    fallback_initial_covariance.block<3, 3>(3, 3) =
        Eigen::Matrix3d::Identity() * (180.0 * 180.0);

    const Eigen::Vector3d boost_accel_enu =
        12.0 * downrange_unit_enu + Eigen::Vector3d(0.0, 0.0, 5.0);
    const Eigen::Vector3d divert_accel_enu =
        3.0 * crossrange_unit_enu + Eigen::Vector3d(0.0, 0.0, -0.6);
    const Eigen::Vector3d boost_accel_eci = enu_to_eci * boost_accel_enu;
    const Eigen::Vector3d divert_accel_eci = enu_to_eci * divert_accel_enu;
    const Eigen::Vector3d ca_model_accel_eci =
        enu_to_eci * (5.0 * downrange_unit_enu + Eigen::Vector3d(0.0, 0.0, 2.0));
    const Eigen::Vector3d smooth_boost_equilibrium_accel_eci =
        enu_to_eci * (8.0 * downrange_unit_enu + Eigen::Vector3d(0.0, 0.0, 3.0));

    std::shared_ptr<sensor::ISensorModel> sensor_model =
        std::make_shared<sensor::SpaceBasedAzElSensorModel>(kAzNoiseRad, kElNoiseRad);

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

    auto integrator = std::make_shared<integrator::RK4Integrator>();
    const std::shared_ptr<propagator::IPropagator> truth_propagator = make_propagator(
        integrator,
        kIntegratorStepSeconds,
        {
            std::make_shared<dynamics::J2Gravity>(),
            std::make_shared<dynamics::AtmosphericDrag>(
                kTargetMassKg,
                kTargetDragCoefficient,
                kTargetReferenceAreaSquareMeters,
                kEarthRadiusMeters
            ),
            std::make_shared<dynamics::TimeWindowForce>(
                std::make_shared<dynamics::ConstantAccelerationForce>(boost_accel_eci),
                kBoostStartSeconds,
                kBoostEndSeconds
            ),
            std::make_shared<dynamics::TimeWindowForce>(
                std::make_shared<dynamics::ConstantAccelerationForce>(divert_accel_eci),
                kDivertStartSeconds,
                kDivertEndSeconds
            )
        }
    );

    const double scenario_duration_seconds = num_steps * dt_seconds;
    const std::vector<PhaseWindow> phase_windows = {
        clamp_phase_window("Boost", 0.0, kBoostEndSeconds, scenario_duration_seconds),
        clamp_phase_window(
            "Coast",
            kBoostEndSeconds,
            kDivertStartSeconds,
            scenario_duration_seconds
        ),
        clamp_phase_window(
            "Divert",
            kDivertStartSeconds,
            kDivertEndSeconds,
            scenario_duration_seconds
        ),
        clamp_phase_window(
            "PostDivert",
            kDivertEndSeconds,
            scenario_duration_seconds,
            scenario_duration_seconds
        )
    };

    ScenarioTrace trace = record_scenario_trace(
        truth_state,
        fallback_initial_estimate,
        fallback_initial_covariance,
        truth_propagator,
        sensor_model,
        sensor_platforms,
        num_steps,
        dt_seconds,
        seed
    );

    trace.initialization = make_initialization_seed(
        trace,
        fallback_initial_estimate,
        fallback_initial_covariance
    );
    trace.initial_estimate = trace.initialization.initial_estimate;
    trace.initial_covariance = trace.initialization.initial_covariance;
    trace.initial_time_seconds = trace.initialization.initial_time_seconds;
    trace.first_update_sample_index = trace.initialization.consumed_sample_count;
    if (trace.first_update_sample_index > 0 &&
        trace.first_update_sample_index <= static_cast<int>(trace.samples.size())) {
        trace.initial_truth_state =
            trace.samples[static_cast<std::size_t>(trace.first_update_sample_index - 1)].truth_state;
    }

    const auto model_specs = build_model_specs(
        integrator,
        kIntegratorStepSeconds,
        trace.initial_estimate,
        trace.initial_covariance,
        trace.initial_time_seconds,
        ca_model_accel_eci,
        smooth_boost_equilibrium_accel_eci,
        kBoostStartSeconds,
        kBoostEndSeconds,
        kTargetMassKg,
        kTargetDragCoefficient,
        kTargetReferenceAreaSquareMeters
    );

    std::cout << "High-altitude ballistic tracking demo with separate EKF/UKF evaluation paths\n";
    std::cout << "Frame: ECI, measurements: space-based az/el only from "
              << sensor_count << " sensor"
              << ((sensor_count == 1) ? "" : "s") << '\n';
    std::cout << "Truth: high-altitude ballistic target with J2, drag, and scheduled maneuvers\n";
    std::cout << "Launch site: Cape-like (" << kLaunchLatitudeDeg << " deg lat, "
              << kLaunchLongitudeDeg << " deg lon), heading " << kLaunchHeadingDeg
              << " deg, flight path " << kFlightPathDeg << " deg\n";
    std::cout << "Requested mode: " << run_mode_to_string(run_mode)
              << ", filter family: " << filter_family_to_string(filter_family) << "\n";
    std::cout << "Sensors:\n";
    for (const auto& sensor_platform : sensor_platforms) {
        const auto& spec = sensor_platform.spec;
        std::cout << "  " << spec.id << " -> alt " << (spec.altitude_m / 1000.0)
                  << " km, lead " << spec.lead_angle_deg
                  << " deg, heading " << spec.heading_deg << " deg\n";
    }
    std::cout << "Initialization: " << trace.initialization.mode
              << " at t=" << trace.initial_time_seconds << " s\n";
    std::cout << "  " << trace.initialization.detail << '\n';
    if (trace.initialization.mode == "TwoEpochTriangulation") {
        std::cout << "  epochs used: " << trace.initialization.epochs_used
                  << ", sensors/epoch: " << trace.initialization.sensors_used_per_epoch
                  << ", pos sigma: " << trace.initialization.position_sigma_m
                  << " m, vel sigma: " << trace.initialization.velocity_sigma_mps << " m/s\n";
    }

    std::vector<DemoRunResult> run_results;
    for (const RunMode mode : expand_run_modes(run_mode)) {
        for (const FilterFamily family : expand_filter_families(filter_family)) {
            if (mode == RunMode::Comparison) {
                run_results.push_back(run_standalone_comparison(
                    trace,
                    family,
                    sensor_model,
                    model_specs,
                    phase_windows,
                    print_every
                ));
            } else if (mode == RunMode::Imm) {
                run_results.push_back(run_imm_family(
                    trace,
                    family,
                    sensor_model,
                    model_specs,
                    phase_windows,
                    print_every
                ));
            }
        }
    }

    nlohmann::json trajectories = nlohmann::json::array();
    trajectories.push_back({
        {"name", "Truth"},
        {"type", "truth"},
        {"points", trace.truth_points}
    });

    nlohmann::json run_summaries = nlohmann::json::array();
    for (const auto& run : run_results) {
        if (run.has_combined) {
            trajectories.push_back({
                {"name", run.run_name + "-Combined"},
                {"run_name", run.run_name},
                {"mode", run.mode},
                {"type", "imm_combined"},
                {"filter_family", run.filter_family},
                {"points", run.combined_points}
            });
            if (run.has_smoothed_combined) {
                trajectories.push_back({
                    {"name", run.run_name + "-RTSCombined"},
                    {"run_name", run.run_name},
                    {"mode", run.mode},
                    {"type", "imm_combined_smoothed"},
                    {"filter_family", run.filter_family},
                    {"points", run.smoothed_combined_points}
                });
            }
        }

        for (int i = 0; i < static_cast<int>(run.model_infos.size()); ++i) {
            const auto& info = run.model_infos[static_cast<std::size_t>(i)];
            trajectories.push_back({
                {"name", run.run_name + "-" + info.name},
                {"run_name", run.run_name},
                {"mode", run.mode},
                {"type", run.has_combined ? "imm_model" : "standalone_model"},
                {"model_index", i},
                {"filter_family", run.filter_family},
                {"motion_model", info.motion_model},
                {"points", run.model_points[static_cast<std::size_t>(i)]}
            });
        }

        run_summaries.push_back(run_summary_json(run, phase_windows));
    }

    const DemoRunResult* primary_run = nullptr;
    for (const auto& run : run_results) {
        if (run.has_combined) {
            primary_run = &run;
            break;
        }
    }
    if (primary_run == nullptr && !run_results.empty()) {
        primary_run = &run_results.front();
    }

    std::string primary_trajectory_name = "Truth";
    nlohmann::json primary_points = trace.truth_points;
    if (primary_run != nullptr) {
        if (primary_run->has_combined) {
            primary_trajectory_name = primary_run->run_name + "-Combined";
            primary_points = primary_run->combined_points;
        } else if (primary_run->best_model_index >= 0) {
            primary_trajectory_name =
                primary_run->run_name + "-" +
                primary_run->model_infos.at(
                    static_cast<std::size_t>(primary_run->best_model_index)
                ).name;
            primary_points = primary_run->model_points.at(
                static_cast<std::size_t>(primary_run->best_model_index)
            );
        }
    }

    nlohmann::json data_json;
    data_json["points"] = primary_points;
    data_json["measurements"] = trace.measurement_points;
    data_json["trajectories"] = trajectories;
    data_json["summary"]["primary_trajectory_name"] = primary_trajectory_name;
    data_json["summary"]["requested"]["mode"] = run_mode_to_string(run_mode);
    data_json["summary"]["requested"]["filter_family"] = filter_family_to_string(filter_family);
    data_json["summary"]["simulation"]["coordinate_frame"] = "ECI";
    data_json["summary"]["simulation"]["start_time"] = 0.0;
    data_json["summary"]["simulation"]["filter_start_time"] = trace.initial_time_seconds;
    data_json["summary"]["simulation"]["timestep"] = dt_seconds;
    data_json["summary"]["simulation"]["integrator_timestep"] = kIntegratorStepSeconds;
    data_json["summary"]["simulation"]["steps"] = num_steps;
    data_json["summary"]["simulation"]["duration"] = num_steps * dt_seconds;
    data_json["summary"]["simulation"]["seed"] = seed;
    data_json["summary"]["simulation"]["measurements_per_step"] = sensor_count;
    data_json["summary"]["simulation"]["earth_radius"] = kEarthRadiusMeters;
    data_json["summary"]["simulation"]["launch"]["latitude_deg"] = kLaunchLatitudeDeg;
    data_json["summary"]["simulation"]["launch"]["longitude_deg"] = kLaunchLongitudeDeg;
    data_json["summary"]["simulation"]["launch"]["heading_deg"] = kLaunchHeadingDeg;
    data_json["summary"]["simulation"]["launch"]["flight_path_deg"] = kFlightPathDeg;
    data_json["summary"]["simulation"]["launch"]["initial_speed_mps"] =
        kInitialSpeedMetersPerSecond;
    data_json["summary"]["simulation"]["launch"]["initial_altitude_m"] =
        kInitialAltitudeMeters;
    data_json["summary"]["simulation"]["truth_model"] =
        "HighAltitudeBallistic with J2, drag, and scheduled boost/divert accelerations";
    data_json["summary"]["simulation"]["phases"] = nlohmann::json::array();
    for (const auto& phase : phase_windows) {
        data_json["summary"]["simulation"]["phases"].push_back({
            {"name", phase.name},
            {"start_time", phase.start_time_seconds},
            {"end_time", phase.end_time_seconds}
        });
    }
    data_json["summary"]["simulation"]["truth_schedule"] = {
        {
            {"name", "BoostAcceleration"},
            {"start_time", kBoostStartSeconds},
            {"end_time", kBoostEndSeconds},
            {"acceleration_enu_mps2", {boost_accel_enu(0), boost_accel_enu(1), boost_accel_enu(2)}}
        },
        {
            {"name", "DivertAcceleration"},
            {"start_time", kDivertStartSeconds},
            {"end_time", kDivertEndSeconds},
            {"acceleration_enu_mps2", {divert_accel_enu(0), divert_accel_enu(1), divert_accel_enu(2)}}
        }
    };
    data_json["summary"]["target"]["mass_kg"] = kTargetMassKg;
    data_json["summary"]["target"]["drag_coefficient"] = kTargetDragCoefficient;
    data_json["summary"]["target"]["reference_area_m2"] = kTargetReferenceAreaSquareMeters;
    data_json["summary"]["sensor"]["type"] = "SpaceBasedAzEl[azimuth,elevation]";
    data_json["summary"]["sensor"]["count"] = sensor_count;
    data_json["summary"]["initialization"] = {
        {"mode", trace.initialization.mode},
        {"detail", trace.initialization.detail},
        {"time_seconds", trace.initialization.initial_time_seconds},
        {"consumed_sample_count", trace.initialization.consumed_sample_count},
        {"epochs_used", trace.initialization.epochs_used},
        {"sensors_used_per_epoch", trace.initialization.sensors_used_per_epoch},
        {"position_sigma_m", trace.initialization.position_sigma_m},
        {"velocity_sigma_mps", trace.initialization.velocity_sigma_mps},
        {"first_epoch_triangulation_residual_m",
         trace.initialization.first_epoch_triangulation_residual_m},
        {"second_epoch_triangulation_residual_m",
         trace.initialization.second_epoch_triangulation_residual_m}
    };
    data_json["summary"]["sensor"]["constellation"] = nlohmann::json::array();
    for (const auto& sensor_platform : sensor_platforms) {
        const auto& spec = sensor_platform.spec;
        data_json["summary"]["sensor"]["constellation"].push_back({
            {"id", spec.id},
            {"altitude_m", spec.altitude_m},
            {"lead_angle_deg", spec.lead_angle_deg},
            {"heading_deg", spec.heading_deg},
            {"az_sigma_rad", spec.az_noise_sigma_rad},
            {"el_sigma_rad", spec.el_noise_sigma_rad}
        });
    }
    data_json["summary"]["runs"] = run_summaries;

    std::ofstream out_file(output_file);
    if (!out_file.is_open()) {
        std::cerr << "Error opening output file: " << output_file << '\n';
        return 1;
    }

    out_file << std::setw(2) << data_json << '\n';
    out_file.close();

    std::cout << "\nDemo data written to " << output_file << '\n';
    return 0;
}
