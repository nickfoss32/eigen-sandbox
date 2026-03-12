# About
This repo contains a sandbox for trajectory simulation, propagation, and state estimation in C++ with Eigen.
It includes:
- A noisy track generator and propagator workflow for ballistic-style trajectories.
- An IMM (Interacting Multiple Model) orbital demo that fuses space-based optical measurements (RA/Dec) with selectable EKF or Julier spherical-simplex UKF models.
- A Python globe plotter that can render single or multiple trajectories from JSON outputs.

# Dependencies
* Eigen3
* nlohmann_json
* Boost.ProgramOptions
* Python 3.9+
* SOFA (http://www.iausofa.org/index.html)

# How to use
There are 3 apps and 1 plot script in this repository:
* `generate-data` - Creates simulated noisy (Gaussian) track data and outputs to JSON. Simulation parameters are defaulted to Cape Canaveral, FL. Each parameter can be modified via CLI.
* `propagate-track` - Reads in output data from `generate-data`, and propagates state forward in time. A best fit plane is calculated amongst noisy points before propagating. See CLI for details.
* `imm-demo` - Runs a 3-model IMM (`TwoBodyGravity`, `J2Gravity`, `J2PlusDrag`) against synthetic space-based optical measurements and writes JSON output. Supports `--filter ekf|ukf` and defaults to the Julier spherical-simplex UKF.
* `scripts/plot.py` - Plots trajectory JSON on a globe. Supports both legacy single-trajectory files (`points`) and multi-trajectory files (`trajectories`), including IMM demo output.

`imm-demo` CLI:
- `-h, --help` Show usage.
- `-f, --filter <ekf|ukf>` Select IMM filter type (default: `ukf`).
- `-o, --output <path>` Output JSON path (default: `imm_demo.json`).

`imm-demo` JSON includes:
- `points`: IMM combined estimate (backward compatible with existing tools).
- `trajectories`: Per-trajectory series (`Truth`, `IMMCombined`, and each IMM model estimate).
- `summary.imm`: Model metadata, filter type, transition matrix, initial/final probabilities.

## Examples
Generate noisy track data from Cape Canaveral, FL at azimuth 0 degrees, elevation angle 20 degrees:
`./generate-data -o canaveral_launch_0az_20el.json --azimuth 0 --elevation 20`
  
Propagate noisy track data to impact:  
`./propagate-track -i canaveral_launch_0az_20el.json -o canaveral_launch_0az_20el_predicted_TLS.json -m TLS`. 
  
Plot trajectory on the globe:  
`python scripts/plot.py build/canaveral_launch_0az_20el_predicted_TLS.json`

Run IMM demo and write multi-trajectory JSON:
`./build/imm-demo -o build/imm_demo_cape.json`

Run IMM demo with the Julier spherical-simplex UKF explicitly:
`./build/imm-demo --filter ukf -o build/imm_demo_cape.json`

Plot IMM demo output (all trajectories shown as points):
`python scripts/plot.py build/imm_demo_cape.json`

## Source Organization
```
// Data types (what things are)
common::Measurement
common::StateEstimate
common::Trajectory

// Sensor models (how sensors work)
sensors::ISensorModel
sensors::RadarSensorModel
sensors::SpaceBasedOpticalSensorModel

// Filters (how to estimate state)
filtering::IKalmanFilter
filtering::ExtendedKalmanFilter
filtering::UnscentedKalmanFilter
filtering::UnscentedTransform

// Propagators (how state evolves)
propagator::IPropagator
propagator::NumericalPropagator

// Dynamics (physics)
dynamics::IDynamics
dynamics::PointMassDynamics
```

Folder	Purpose	Contents	Examples
common/	Shared data types	Structs, enums, constants	Measurement, StateEstimate, TrackQuality
sensors/	Sensor models (h(x))	How sensors measure	RadarSensorModel, CameraSensorModel
filtering/	State estimation	Kalman filters	ExtendedKalmanFilter, UKF
propagator/	State propagation (f(x))	How states evolve	NumericalPropagator, AnalyticalPropagator
dynamics/	Physical dynamics	Forces, accelerations	PointMassDynamics, Gravity
tracking/	Multi-target tracking	Track management	Tracker, DataAssociation
