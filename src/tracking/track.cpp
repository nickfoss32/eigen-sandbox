#include "tracking/track.hpp"

#include <stdexcept>

namespace tracking {

Track::Track(
    int id,
    std::unique_ptr<estimation::IMM> imm,
    std::shared_ptr<sensor::ISensorModel> association_sensor_model,
    common::TrackQuality quality,
    common::TargetType target_type,
    int initial_update_count
) : id_(id)
  , quality_(quality)
  , target_type_(target_type)
  , imm_(std::move(imm))
  , association_sensor_model_(std::move(association_sensor_model))
  , total_updates_(initial_update_count)
  , consecutive_hits_(initial_update_count)
  , last_update_time_(0.0)
{
    if (id_ < 0) {
        throw std::invalid_argument("Track: id must be non-negative");
    }
    if (!imm_) {
        throw std::invalid_argument("Track: IMM cannot be null");
    }
    if (!association_sensor_model_) {
        throw std::invalid_argument("Track: association sensor model cannot be null");
    }
    if (initial_update_count < 0) {
        throw std::invalid_argument("Track: initial update count must be non-negative");
    }

    last_update_time_ = imm_->get_time();
}

void Track::predict_to(double time_seconds) {
    const double dt = time_seconds - imm_->get_time();
    if (dt < 0.0) {
        throw std::invalid_argument("Track::predict_to: target time cannot be in the past");
    }
    if (dt > 0.0) {
        imm_->predict(dt);
        ++age_steps_;
    }
}

void Track::assimilate_measurement(const common::Measurement& measurement) {
    imm_->update(measurement);
    last_update_time_ = measurement.time;
}

void Track::finalize_epoch(bool received_measurement) {
    if (received_measurement) {
        ++total_updates_;
        ++consecutive_hits_;
        consecutive_misses_ = 0;
        return;
    }

    consecutive_hits_ = 0;
    ++consecutive_misses_;
}

void Track::update(const common::Measurement& measurement) {
    assimilate_measurement(measurement);
    finalize_epoch(true);
}

void Track::mark_missed() {
    finalize_epoch(false);
}

} // namespace tracking
