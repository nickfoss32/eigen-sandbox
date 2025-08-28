#pragma once

#include <fitting/plane_fit.hpp>
#include <boost/program_options.hpp>
#include <iostream>
#include <memory>
#include <string>

namespace po = boost::program_options;

/// @brief PlaneFitter supported types
enum class PlaneFitMode { TLS, OLS };

std::istream& operator>>(std::istream& is, PlaneFitMode& mode) {
    std::string s;
    is >> s;
    if (s == "TLS" || s == "tls") {
        mode = PlaneFitMode::TLS;
    } else if (s == "OLS" || s == "ols") {
        mode = PlaneFitMode::OLS;
    } else {
        throw po::invalid_option_value(s);
    }
    return is;
}

std::ostream& operator<<(std::ostream& os, const PlaneFitMode& mode) {
    switch (mode) {
        case PlaneFitMode::TLS: os << "TLS"; break;
        case PlaneFitMode::OLS: os << "OLS"; break;
    }
    return os;
}

/// @brief Factory to create the appropriate PlaneFitter based on mode
class PlaneFitterFactory {
public:
    /// @brief Create a PlaneFitter
    /// @param mode Type of PlaneFitter to create
    /// @return A PlaneFitter
    static std::unique_ptr<PlaneFitter> create(PlaneFitMode mode) {
        switch (mode) {
            case PlaneFitMode::TLS:
                return std::make_unique<TotalLeastSquaresPlaneFitter>();
            case PlaneFitMode::OLS:
                return std::make_unique<RegressionPlaneFitter>();
            default:
                throw std::invalid_argument("Unknown plane fit mode");
        }
    }
};
