#include "tracking/nearest_neighbor_associator.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace tracking {

NearestNeighborAssociator::NearestNeighborAssociator(double gating_threshold_squared)
    : gating_threshold_squared_(gating_threshold_squared) {
    if (!(gating_threshold_squared_ > 0.0)) {
        throw std::invalid_argument(
            "NearestNeighborAssociator: gating threshold must be positive"
        );
    }
}

auto NearestNeighborAssociator::associate(
    const std::vector<Track>& tracks,
    const common::MeasurementBatch& measurements
) const -> AssociationResult {
    AssociationResult result;

    struct Candidate {
        int track_index = -1;
        int measurement_index = -1;
        double squared_mahalanobis = 0.0;
    };

    std::vector<Candidate> candidates;

    for (int track_index = 0; track_index < static_cast<int>(tracks.size()); ++track_index) {
        for (int measurement_index = 0;
             measurement_index < static_cast<int>(measurements.size());
             ++measurement_index) {
            const double d2 = compute_squared_mahalanobis(
                tracks[static_cast<std::size_t>(track_index)],
                measurements[static_cast<std::size_t>(measurement_index)]
            );
            if (std::isfinite(d2) && d2 <= gating_threshold_squared_) {
                candidates.push_back({track_index, measurement_index, d2});
            }
        }
    }

    std::sort(
        candidates.begin(),
        candidates.end(),
        [](const Candidate& lhs, const Candidate& rhs) {
            return lhs.squared_mahalanobis < rhs.squared_mahalanobis;
        }
    );

    std::vector<bool> track_used(tracks.size(), false);
    std::vector<bool> measurement_used(measurements.size(), false);

    for (const auto& candidate : candidates) {
        if (track_used[static_cast<std::size_t>(candidate.track_index)] ||
            measurement_used[static_cast<std::size_t>(candidate.measurement_index)]) {
            continue;
        }

        track_used[static_cast<std::size_t>(candidate.track_index)] = true;
        measurement_used[static_cast<std::size_t>(candidate.measurement_index)] = true;
        result.matches.push_back({
            candidate.track_index,
            candidate.measurement_index,
            candidate.squared_mahalanobis
        });
    }

    for (int track_index = 0; track_index < static_cast<int>(tracks.size()); ++track_index) {
        if (!track_used[static_cast<std::size_t>(track_index)]) {
            result.unmatched_track_indices.push_back(track_index);
        }
    }
    for (int measurement_index = 0;
         measurement_index < static_cast<int>(measurements.size());
         ++measurement_index) {
        if (!measurement_used[static_cast<std::size_t>(measurement_index)]) {
            result.unmatched_measurement_indices.push_back(measurement_index);
        }
    }

    return result;
}

auto NearestNeighborAssociator::compute_squared_mahalanobis(
    const Track& track,
    const common::Measurement& measurement
) const -> double {
    sensor::SensorContext ctx;
    ctx.state = track.get_state();
    ctx.time = measurement.time;
    ctx.sensor_position = measurement.sensor_position;
    ctx.sensor_orientation = measurement.sensor_orientation;

    const sensor::ISensorModel& sensor_model = track.get_association_sensor_model();
    const Eigen::VectorXd predicted_measurement = sensor_model.compute_measurement(ctx);
    if (predicted_measurement.size() != measurement.z.size()) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const Eigen::MatrixXd H = sensor_model.compute_jacobian(ctx);
    const Eigen::MatrixXd innovation_covariance =
        H * track.get_covariance() * H.transpose() + measurement.R;
    if (innovation_covariance.rows() != measurement.z.size() ||
        innovation_covariance.cols() != measurement.z.size()) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    Eigen::LDLT<Eigen::MatrixXd> ldlt(
        0.5 * (innovation_covariance + innovation_covariance.transpose())
    );
    if (ldlt.info() != Eigen::Success) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const Eigen::VectorXd innovation = measurement.z - predicted_measurement;
    const Eigen::VectorXd solved = ldlt.solve(innovation);
    if (ldlt.info() != Eigen::Success || !solved.allFinite()) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const double value = innovation.dot(solved);
    return (value >= 0.0 && std::isfinite(value))
               ? value
               : std::numeric_limits<double>::quiet_NaN();
}

} // namespace tracking
