#pragma once

#include "common/types.hpp"
#include "tracking/track.hpp"

#include <vector>

namespace tracking {

struct TrackMeasurementPair {
    int track_index = -1;
    int measurement_index = -1;
    double squared_mahalanobis = 0.0;
};

struct AssociationResult {
    std::vector<TrackMeasurementPair> matches;
    std::vector<int> unmatched_track_indices;
    std::vector<int> unmatched_measurement_indices;
};

class NearestNeighborAssociator {
public:
    explicit NearestNeighborAssociator(double gating_threshold_squared = 16.0);

    auto associate(
        const std::vector<Track>& tracks,
        const common::MeasurementBatch& measurements
    ) const -> AssociationResult;

    auto get_gating_threshold_squared() const -> double { return gating_threshold_squared_; }

private:
    auto compute_squared_mahalanobis(
        const Track& track,
        const common::Measurement& measurement
    ) const -> double;

    double gating_threshold_squared_ = 16.0;
};

} // namespace tracking
