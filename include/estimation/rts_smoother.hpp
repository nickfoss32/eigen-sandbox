#pragma once

#include "estimation/smoother.hpp"

namespace estimation {

/// @brief Rauch-Tung-Striebel fixed-interval smoother.
class RTSSmoother : public ISmoother {
public:
    auto smooth(
        const std::vector<SmootherEstimate>& filtered_estimates,
        const std::vector<PredictedEstimate>& predicted_estimates
    ) const -> std::vector<SmootherEstimate> override;
};

} // namespace estimation
