# About
This repo contains a sandbox environment for testing a ballistic propagator based on Eigen.

# Dependencies
* Eigen3
* nlohmann_json
* Python 3.9+

# How to use
There are 2 apps and 1 plot script in this repository:
* `generate-data` - Creates simulated noisy (Gaussian) track data and outputs to JSON. Simulation parameters are hard-coded to Cape Canaveral, FL. Feel free to tweak and re-compile.
* `predict-impact` - Reads in output data from `generate-data`, and propagates state forward to impact.
* `scripts/plot.py` - Plots ECEF coordinates from JSON output of either app on a 3D globe.
