#pragma once

#include <Eigen/Dense>
#include <random>

class GaussianNoise {
public:
    // Constructor: Initialize with standard deviations for position and velocity noise
    GaussianNoise(double sigma_pos, double sigma_vel) 
        : posNoise_(0.0, sigma_pos), velNoise_(0.0, sigma_vel) {
        std::random_device rd;
        gen_ = std::mt19937(rd());
    }

    // Generate noise for a 6D state vector [x, y, z, vx, vy, vz]
    Eigen::VectorXd generate_noise() const {
        Eigen::VectorXd noise(6);
        noise << posNoise_(gen_), posNoise_(gen_), posNoise_(gen_), // Position noise (x, y, z)
                 velNoise_(gen_), velNoise_(gen_), velNoise_(gen_); // Velocity noise (vx, vy, vz)
        return noise;
    }

private:
    /// @brief Mersenne Twister random number generator
    mutable std::mt19937 gen_;
    /// @brief Gaussian noise for position
    mutable std::normal_distribution<double> posNoise_;
    /// @brief Gaussian noise for velocity
    mutable std::normal_distribution<double> velNoise_;
};
