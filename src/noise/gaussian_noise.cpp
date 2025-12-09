#include "noise/gaussian_noise.hpp"

namespace noise {
GaussianNoise::GaussianNoise(double sigma_pos, double sigma_vel) 
: posNoise_(0.0, sigma_pos), velNoise_(0.0, sigma_vel)
{
    std::random_device rd;
    gen_ = std::mt19937(rd());
}

auto GaussianNoise::generate_noise() const -> Eigen::VectorXd {
    /// @todo Update to support 3,6, and 9 PVA noise generation
    Eigen::VectorXd noise(6);
    noise << posNoise_(gen_), posNoise_(gen_), posNoise_(gen_), // Position noise (x, y, z)
                velNoise_(gen_), velNoise_(gen_), velNoise_(gen_); // Velocity noise (vx, vy, vz)
    return noise;
}
} // namespace noise
