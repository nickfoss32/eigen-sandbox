# About
This repo contains a sandbox environment for testing a ballistic propagator based on Eigen.

# Dependencies
* Eigen3
* nlohmann_json
* Python 3.9+
* SOFA (http://www.iausofa.org/index.html)

# How to use
There are 2 apps and 1 plot script in this repository:
* `generate-data` - Creates simulated noisy (Gaussian) track data and outputs to JSON. Simulation parameters are defaulted to Cape Canaveral, FL. Each parameter can be modified via CLI.
* `propagate-track` - Reads in output data from `generate-data`, and propagates state forward in time. A best fit plane is calculated amongst noisy points before propagating. See CLI for details.
* `scripts/plot.py` - Plots ECEF coordinates from JSON output of track trajectory on globe.

## Examples
Generate noisy track data from Cape Canaveral, FL at azimuth 0 degrees, elevation angle 20 degrees:  
`./generate-data -o canaveral_launch_0az_20el.json --azimuth 0 --elevation 20`
  
Propagate noisy track data to impact:  
`./propagate-track -i canaveral_launch_0az_20el.json -o canaveral_launch_0az_20el_predicted_TLS.json -m TLS`. 
  
Plot trajectory on the globe:  
`python plot.py ../build/canaveral_launch_0az_20el_predicted_TLS.json`
