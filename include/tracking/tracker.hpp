#pragma once

#include "common/types.hpp"
#include "estimation/imm.hpp"
#include "tracking/nearest_neighbor_associator.hpp"
#include "tracking/track.hpp"

#include <functional>
#include <optional>
#include <vector>

namespace tracking {

struct TrackSeed {
    std::unique_ptr<estimation::IMM> imm;
    std::shared_ptr<sensor::ISensorModel> association_sensor_model;
    common::TargetType target_type = common::TargetType::UNKNOWN;
    common::TrackQuality initial_quality = common::TrackQuality::TENTATIVE;
    int initial_update_count = 1;
};

struct TrackerConfig {
    double association_gate_squared = 16.0;
    int confirmation_updates = 2;
    int max_coast_steps = 2;
    int max_tentative_missed_steps = 0;
};

struct TrackerStepResult {
    std::vector<TrackMeasurementPair> matches;
    std::vector<int> updated_track_ids;
    std::vector<int> missed_track_ids;
    std::vector<int> created_track_ids;
    std::vector<int> deleted_track_ids;
    std::vector<int> unmatched_measurement_indices;
};

class Tracker {
public:
    using TrackInitiator = std::function<std::optional<TrackSeed>(const common::Measurement&)>;

    Tracker(TrackerConfig config, TrackInitiator initiator);

    auto step(double time_seconds, const common::MeasurementBatch& measurements)
        -> TrackerStepResult;
    auto step(const common::MeasurementBatch& measurements) -> TrackerStepResult;

    auto get_tracks() const -> const std::vector<Track>& { return tracks_; }
    auto get_num_tracks() const -> int { return static_cast<int>(tracks_.size()); }

private:
    void validate_measurement_times(
        double time_seconds,
        const common::MeasurementBatch& measurements
    ) const;
    void update_track_quality(Track& track) const;

    TrackerConfig config_;
    TrackInitiator initiator_;
    NearestNeighborAssociator associator_;
    std::vector<Track> tracks_;
    int next_track_id_ = 0;
};

} // namespace tracking
