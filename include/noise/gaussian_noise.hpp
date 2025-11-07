#pragma once

#include <Eigen/Dense>
#include <random>

namespace noise {
/// @brief Uses a Mersenne Twister random number generator to generate gaussian noise.
class GaussianNoise {
public:
    /// @brief Constructor: Initialize with standard deviations for position and velocity noise
    GaussianNoise(double sigma_pos, double sigma_vel);

    /// @brief Generate noise for a 6D state vector [x, y, z, vx, vy, vz]
    auto generate_noise() const -> Eigen::VectorXd;

private:
    /// @brief Mersenne Twister random number generator
    mutable std::mt19937 gen_;
    /// @brief Gaussian noise for position
    mutable std::normal_distribution<double> posNoise_;
    /// @brief Gaussian noise for velocity
    mutable std::normal_distribution<double> velNoise_;
};
} // namespace noise