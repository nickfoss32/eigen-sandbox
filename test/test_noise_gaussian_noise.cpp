#include <gtest/gtest.h>

#include "noise/gaussian_noise.hpp"

#include <Eigen/Dense>

#include <cmath>

using namespace noise;

// Test fixture
class GaussianNoiseTest : public ::testing::Test {
protected:
    static constexpr double TOLERANCE = 0.15; // 15% tolerance for statistical tests
};

// Test that noise vector has correct size
TEST_F(GaussianNoiseTest, NoiseVectorSize) {
    GaussianNoise noise(1.0, 0.5);
    Eigen::VectorXd sample = noise.generate_noise();
    EXPECT_EQ(sample.size(), 6);
}

// Test that position noise has approximately correct standard deviation
TEST_F(GaussianNoiseTest, PositionNoiseStandardDeviation) {
    double sigma_pos = 2.0;
    double sigma_vel = 0.5;
    GaussianNoise noise(sigma_pos, sigma_vel);
    
    // Generate many samples
    int num_samples = 10000;
    Eigen::VectorXd sum = Eigen::VectorXd::Zero(6);
    Eigen::VectorXd sum_sq = Eigen::VectorXd::Zero(6);
    
    for (int i = 0; i < num_samples; ++i) {
        Eigen::VectorXd sample = noise.generate_noise();
        sum += sample;
        sum_sq += sample.cwiseProduct(sample);
    }
    
    // Calculate mean and standard deviation for position components
    Eigen::VectorXd mean = sum / num_samples;
    Eigen::VectorXd variance = (sum_sq / num_samples) - mean.cwiseProduct(mean);
    Eigen::VectorXd std_dev = variance.cwiseSqrt();
    
    // Check position components (0, 1, 2)
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(std_dev(i), sigma_pos, sigma_pos * TOLERANCE);
        EXPECT_NEAR(mean(i), 0.0, sigma_pos * TOLERANCE);
    }
}

// Test that velocity noise has approximately correct standard deviation
TEST_F(GaussianNoiseTest, VelocityNoiseStandardDeviation) {
    double sigma_pos = 2.0;
    double sigma_vel = 0.5;
    GaussianNoise noise(sigma_pos, sigma_vel);
    
    // Generate many samples
    int num_samples = 10000;
    Eigen::VectorXd sum = Eigen::VectorXd::Zero(6);
    Eigen::VectorXd sum_sq = Eigen::VectorXd::Zero(6);
    
    for (int i = 0; i < num_samples; ++i) {
        Eigen::VectorXd sample = noise.generate_noise();
        sum += sample;
        sum_sq += sample.cwiseProduct(sample);
    }
    
    // Calculate mean and standard deviation for velocity components
    Eigen::VectorXd mean = sum / num_samples;
    Eigen::VectorXd variance = (sum_sq / num_samples) - mean.cwiseProduct(mean);
    Eigen::VectorXd std_dev = variance.cwiseSqrt();
    
    // Check velocity components (3, 4, 5)
    for (int i = 3; i < 6; ++i) {
        EXPECT_NEAR(std_dev(i), sigma_vel, sigma_vel * TOLERANCE);
        EXPECT_NEAR(mean(i), 0.0, sigma_vel * TOLERANCE);
    }
}

// Test that noise has approximately zero mean
TEST_F(GaussianNoiseTest, ZeroMean) {
    GaussianNoise noise(1.0, 0.5);
    
    int num_samples = 10000;
    Eigen::VectorXd sum = Eigen::VectorXd::Zero(6);
    
    for (int i = 0; i < num_samples; ++i) {
        sum += noise.generate_noise();
    }
    
    Eigen::VectorXd mean = sum / num_samples;
    
    // All components should have mean near zero
    for (int i = 0; i < 6; ++i) {
        EXPECT_NEAR(mean(i), 0.0, 0.1);
    }
}

// Test with zero standard deviation (should produce zeros)
TEST_F(GaussianNoiseTest, ZeroStandardDeviation) {
    GaussianNoise noise(0.0, 0.0);
    
    for (int i = 0; i < 100; ++i) {
        Eigen::VectorXd sample = noise.generate_noise();
        EXPECT_DOUBLE_EQ(sample.norm(), 0.0);
    }
}

// Test that different calls produce different values
TEST_F(GaussianNoiseTest, RandomnessCheck) {
    GaussianNoise noise(1.0, 0.5);
    
    Eigen::VectorXd sample1 = noise.generate_noise();
    Eigen::VectorXd sample2 = noise.generate_noise();
    
    // Samples should be different (probability of identical samples is essentially zero)
    EXPECT_NE(sample1, sample2);
}

// Test with very small standard deviation
TEST_F(GaussianNoiseTest, SmallStandardDeviation) {
    double small_sigma = 0.001;
    GaussianNoise noise(small_sigma, small_sigma);
    
    int num_samples = 1000;
    Eigen::VectorXd sum = Eigen::VectorXd::Zero(6);
    Eigen::VectorXd sum_sq = Eigen::VectorXd::Zero(6);
    
    for (int i = 0; i < num_samples; ++i) {
        Eigen::VectorXd sample = noise.generate_noise();
        sum += sample;
        sum_sq += sample.cwiseProduct(sample);
    }
    
    Eigen::VectorXd mean = sum / num_samples;
    Eigen::VectorXd variance = (sum_sq / num_samples) - mean.cwiseProduct(mean);
    Eigen::VectorXd std_dev = variance.cwiseSqrt();
    
    for (int i = 0; i < 6; ++i) {
        EXPECT_NEAR(std_dev(i), small_sigma, small_sigma * 0.3); // Allow 30% tolerance for small values
    }
}

// Test with large standard deviation
TEST_F(GaussianNoiseTest, LargeStandardDeviation) {
    double large_sigma = 100.0;
    GaussianNoise noise(large_sigma, large_sigma);
    
    int num_samples = 10000;
    Eigen::VectorXd sum = Eigen::VectorXd::Zero(6);
    Eigen::VectorXd sum_sq = Eigen::VectorXd::Zero(6);
    
    for (int i = 0; i < num_samples; ++i) {
        Eigen::VectorXd sample = noise.generate_noise();
        sum += sample;
        sum_sq += sample.cwiseProduct(sample);
    }
    
    Eigen::VectorXd mean = sum / num_samples;
    Eigen::VectorXd variance = (sum_sq / num_samples) - mean.cwiseProduct(mean);
    Eigen::VectorXd std_dev = variance.cwiseSqrt();
    
    for (int i = 0; i < 6; ++i) {
        EXPECT_NEAR(std_dev(i), large_sigma, large_sigma * TOLERANCE);
    }
}

// Test that position and velocity noises are independent
TEST_F(GaussianNoiseTest, PositionVelocityIndependence) {
    double sigma_pos = 2.0;
    double sigma_vel = 0.5;
    GaussianNoise noise(sigma_pos, sigma_vel);
    
    int num_samples = 10000;
    double covariance_xy = 0.0;
    double covariance_xvx = 0.0;
    
    Eigen::VectorXd sum = Eigen::VectorXd::Zero(6);
    
    for (int i = 0; i < num_samples; ++i) {
        Eigen::VectorXd sample = noise.generate_noise();
        sum += sample;
    }
    
    Eigen::VectorXd mean = sum / num_samples;
    
    for (int i = 0; i < num_samples; ++i) {
        Eigen::VectorXd sample = noise.generate_noise();
        // Covariance between x and y (should be ~0)
        covariance_xy += (sample(0) - mean(0)) * (sample(1) - mean(1));
        // Covariance between x and vx (should be ~0)
        covariance_xvx += (sample(0) - mean(0)) * (sample(3) - mean(3));
    }
    
    covariance_xy /= num_samples;
    covariance_xvx /= num_samples;
    
    // Covariances should be near zero (independent)
    EXPECT_NEAR(covariance_xy, 0.0, 0.1);
    EXPECT_NEAR(covariance_xvx, 0.0, 0.1);
}

// Test different sigma values for position and velocity
TEST_F(GaussianNoiseTest, DifferentSigmas) {
    double sigma_pos = 5.0;
    double sigma_vel = 0.1;
    GaussianNoise noise(sigma_pos, sigma_vel);
    
    int num_samples = 10000;
    Eigen::VectorXd sum = Eigen::VectorXd::Zero(6);
    Eigen::VectorXd sum_sq = Eigen::VectorXd::Zero(6);
    
    for (int i = 0; i < num_samples; ++i) {
        Eigen::VectorXd sample = noise.generate_noise();
        sum += sample;
        sum_sq += sample.cwiseProduct(sample);
    }
    
    Eigen::VectorXd mean = sum / num_samples;
    Eigen::VectorXd variance = (sum_sq / num_samples) - mean.cwiseProduct(mean);
    Eigen::VectorXd std_dev = variance.cwiseSqrt();
    
    // Position noise should be much larger than velocity noise
    EXPECT_GT(std_dev(0), std_dev(3));
    EXPECT_NEAR(std_dev(0) / std_dev(3), sigma_pos / sigma_vel, 5.0);
}

// Test that generator is properly initialized (no repeated sequences)
TEST_F(GaussianNoiseTest, GeneratorInitialization) {
    // Create two independent noise generators
    GaussianNoise noise1(1.0, 0.5);
    GaussianNoise noise2(1.0, 0.5);
    
    Eigen::VectorXd sample1 = noise1.generate_noise();
    Eigen::VectorXd sample2 = noise2.generate_noise();
    
    // Different generators should produce different initial values
    EXPECT_NE(sample1, sample2);
}
