#include "tracking/tracker.hpp"

#include <algorithm>
#include <stdexcept>

namespace tracking {

namespace {

constexpr double kMeasurementTimeTolerance = 1e-9;

} // namespace

Tracker::Tracker(TrackerConfig config, TrackInitiator initiator)
    : config_(std::move(config))
    , initiator_(std::move(initiator))
    , associator_(config_.association_gate_squared) {
    if (!initiator_) {
        throw std::invalid_argument("Tracker: initiator cannot be empty");
    }
    if (config_.confirmation_updates <= 0) {
        throw std::invalid_argument("Tracker: confirmation_updates must be positive");
    }
    if (config_.max_coast_steps < 0 || config_.max_tentative_missed_steps < 0) {
        throw std::invalid_argument("Tracker: miss limits must be non-negative");
    }
}

auto Tracker::step(double time_seconds, const common::MeasurementBatch& measurements)
    -> TrackerStepResult {
    validate_measurement_times(time_seconds, measurements);

    for (auto& track : tracks_) {
        track.predict_to(time_seconds);
    }

    const AssociationResult association_result = associator_.associate(tracks_, measurements);
    TrackerStepResult step_result;
    step_result.matches = association_result.matches;
    step_result.unmatched_measurement_indices = association_result.unmatched_measurement_indices;

    for (const auto& match : association_result.matches) {
        Track& track = tracks_.at(static_cast<std::size_t>(match.track_index));
        track.update(measurements.at(static_cast<std::size_t>(match.measurement_index)));
        update_track_quality(track);
        step_result.updated_track_ids.push_back(track.get_id());
    }

    for (const int track_index : association_result.unmatched_track_indices) {
        Track& track = tracks_.at(static_cast<std::size_t>(track_index));
        track.mark_missed();
        update_track_quality(track);
        step_result.missed_track_ids.push_back(track.get_id());
    }

    for (const int measurement_index : association_result.unmatched_measurement_indices) {
        const common::Measurement& measurement =
            measurements.at(static_cast<std::size_t>(measurement_index));
        auto seed = initiator_(measurement);
        if (!seed.has_value()) {
            continue;
        }

        Track track(
            next_track_id_,
            std::move(seed->imm),
            seed->association_sensor_model,
            seed->initial_quality,
            seed->target_type,
            seed->initial_update_count
        );
        update_track_quality(track);
        step_result.created_track_ids.push_back(next_track_id_);
        tracks_.push_back(std::move(track));
        ++next_track_id_;
    }

    std::vector<Track> surviving_tracks;
    surviving_tracks.reserve(tracks_.size());
    for (auto& track : tracks_) {
        if (track.get_quality() == common::TrackQuality::TERMINATED) {
            step_result.deleted_track_ids.push_back(track.get_id());
            continue;
        }
        surviving_tracks.push_back(std::move(track));
    }
    tracks_ = std::move(surviving_tracks);

    return step_result;
}

auto Tracker::step(const common::MeasurementBatch& measurements) -> TrackerStepResult {
    if (measurements.empty()) {
        throw std::invalid_argument(
            "Tracker::step: explicit time is required for empty measurement batches"
        );
    }

    return step(measurements.front().time, measurements);
}

void Tracker::validate_measurement_times(
    double time_seconds,
    const common::MeasurementBatch& measurements
) const {
    for (const auto& measurement : measurements) {
        if (std::abs(measurement.time - time_seconds) > kMeasurementTimeTolerance) {
            throw std::invalid_argument(
                "Tracker::step: all measurements must share the batch timestamp"
            );
        }
    }
}

void Tracker::update_track_quality(Track& track) const {
    if (track.get_total_updates() >= config_.confirmation_updates &&
        track.get_consecutive_misses() == 0) {
        track.set_quality(common::TrackQuality::CONFIRMED);
        return;
    }

    if (track.get_total_updates() < config_.confirmation_updates) {
        if (track.get_consecutive_misses() > config_.max_tentative_missed_steps) {
            track.set_quality(common::TrackQuality::TERMINATED);
        } else {
            track.set_quality(common::TrackQuality::TENTATIVE);
        }
        return;
    }

    if (track.get_consecutive_misses() == 0) {
        track.set_quality(common::TrackQuality::CONFIRMED);
    } else if (track.get_consecutive_misses() <= config_.max_coast_steps) {
        track.set_quality(common::TrackQuality::COASTING);
    } else {
        track.set_quality(common::TrackQuality::TERMINATED);
    }
}

} // namespace tracking
