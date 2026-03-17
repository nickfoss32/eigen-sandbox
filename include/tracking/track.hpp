#pragma once

#include "common/types.hpp"
#include "estimation/imm.hpp"
#include "sensor/sensor_model.hpp"

#include <memory>

namespace tracking {

class Track {
public:
    Track(
        int id,
        std::unique_ptr<estimation::IMM> imm,
        std::shared_ptr<sensor::ISensorModel> association_sensor_model,
        common::TrackQuality quality = common::TrackQuality::TENTATIVE,
        common::TargetType target_type = common::TargetType::UNKNOWN,
        int initial_update_count = 1
    );

    auto get_id() const -> int { return id_; }
    auto get_quality() const -> common::TrackQuality { return quality_; }
    auto get_target_type() const -> common::TargetType { return target_type_; }
    auto get_state() const -> Eigen::VectorXd { return imm_->get_state(); }
    auto get_covariance() const -> Eigen::MatrixXd { return imm_->get_covariance(); }
    auto get_time() const -> double { return imm_->get_time(); }
    auto get_total_updates() const -> int { return total_updates_; }
    auto get_consecutive_hits() const -> int { return consecutive_hits_; }
    auto get_consecutive_misses() const -> int { return consecutive_misses_; }
    auto get_age_steps() const -> int { return age_steps_; }
    auto get_last_update_time() const -> double { return last_update_time_; }
    auto get_imm() const -> const estimation::IMM& { return *imm_; }
    auto get_association_sensor_model() const -> const sensor::ISensorModel& {
        return *association_sensor_model_;
    }
    auto get_association_sensor_model_ptr() const -> std::shared_ptr<sensor::ISensorModel> {
        return association_sensor_model_;
    }

    void predict_to(double time_seconds);
    void assimilate_measurement(const common::Measurement& measurement);
    void finalize_epoch(bool received_measurement);
    void update(const common::Measurement& measurement);
    void mark_missed();
    void set_quality(common::TrackQuality quality) { quality_ = quality; }

private:
    int id_ = -1;
    common::TrackQuality quality_ = common::TrackQuality::TENTATIVE;
    common::TargetType target_type_ = common::TargetType::UNKNOWN;
    std::unique_ptr<estimation::IMM> imm_;
    std::shared_ptr<sensor::ISensorModel> association_sensor_model_;
    int total_updates_ = 0;
    int consecutive_hits_ = 0;
    int consecutive_misses_ = 0;
    int age_steps_ = 1;
    double last_update_time_ = 0.0;
};

} // namespace tracking
