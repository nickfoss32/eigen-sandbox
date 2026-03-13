# About
This repo contains a sandbox for trajectory simulation, propagation, and state estimation in C++ with Eigen.
It includes:
- A noisy track generator and propagator workflow for ballistic-style trajectories.
- An IMM/standalone comparison demo that tracks a high-altitude ballistic point-mass target with space-based az/el sensing, while evaluating `CV`, `CA`, `Gravity`, `Gravity+Drag`, `J2+Drag`, and a smooth-acceleration boost model with EKF and UKF runs kept separate.
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
* `imm-demo` - Replays one synthetic high-altitude ballistic point-mass scenario through two evaluation modes: standalone per-model comparison and motion-model IMM. EKF and UKF are run separately. Standalone comparison evaluates `CV`, `CA`, `Gravity`, `Gravity+Drag`, `J2+Drag`, and `BoostSmooth`, while the IMM keeps the matched 6-state motion-model bank of `CV`, `CA`, `Gravity`, `Gravity+Drag`, and `J2+Drag`. The target can be observed by a configurable 1-5 sensor space-based az/el constellation (default: 3).
* `scripts/plot.py` - Plots trajectory JSON on a globe. Supports both legacy single-trajectory files (`points`) and multi-trajectory files (`trajectories`), including IMM demo output.
* `scripts/plot_filter_eval.py` - Builds a filter-evaluation HTML dashboard from `imm-demo` JSON, including error time series, NEES consistency traces, phase-wise RMSE bars, and IMM mode probability plots.

`imm-demo` CLI:
- `-h, --help` Show usage.
- `-o, --output <path>` Output JSON path (default: `imm_demo.json`).
- `--steps <count>` Number of sensor update steps to simulate.
- `--dt <seconds>` Sensor update period.
- `--seed <value>` Random seed for measurement noise.
- `--print-every <count>` Console print cadence.
- `--mode <comparison|imm|both>` Run standalone comparison, IMM, or both (default: `both`).
- `--filter-family <ekf|ukf|both>` Run EKF, UKF, or both separately (default: `both`).
- `--sensor-count <count>` Number of space-based az/el sensors to use, from 1 to 5 (default: `3`).

`imm-demo` JSON includes:
- `points`: A primary estimate trajectory chosen for backward compatibility. When an IMM run is present, this is the first IMM combined estimate.
- `trajectories`: Per-trajectory series for `Truth`, each standalone model run, and each IMM run.
- `measurements`: Noisy space-based az/el measurements `[azimuth, elevation]`.
- `summary.runs`: Per-run metadata and performance. Standalone runs report model RMSE rankings; IMM runs also report transition matrices, final mode probabilities, and combined-estimate performance.
- `summary.simulation.phases`: Named boost/coast/divert/post-divert windows used for phase-by-phase scoring.
- `summary.sensor.constellation`: Active sensor geometries and noise settings for the selected 1-5 sensor configuration.
- Trajectory points now include `position_error_m`, `velocity_error_mps`, and NEES consistency fields to support tuning plots.

## Examples
Generate noisy track data from Cape Canaveral, FL at azimuth 0 degrees, elevation angle 20 degrees:
`./generate-data -o canaveral_launch_0az_20el.json --azimuth 0 --elevation 20`
  
Propagate noisy track data to impact:  
`./propagate-track -i canaveral_launch_0az_20el.json -o canaveral_launch_0az_20el_predicted_TLS.json -m TLS`. 
  
Plot trajectory on the globe:  
`python scripts/plot.py build/canaveral_launch_0az_20el_predicted_TLS.json`

Run both standalone comparison and IMM modes for both filter families:
`./build/imm-demo -o build/imm_demo_cape.json`

Run only the EKF standalone comparison bank:
`./build/imm-demo --mode comparison --filter-family ekf -o build/imm_demo_cape.json`

Run only the UKF IMM bank for a shorter 90 second scenario:
`./build/imm-demo --mode imm --filter-family ukf --steps 90 --dt 1.0 -o build/imm_demo_cape.json`

Run the default demo with 5 different space-based lines of sight:
`./build/imm-demo --sensor-count 5 -o build/imm_demo_cape.json`

Plot IMM demo output (all trajectories shown as points):
`python scripts/plot.py build/imm_demo_cape.json`

Create an evaluation dashboard for the demo output:
`python scripts/plot_filter_eval.py build/imm_demo_cape.json -o build/imm_demo_cape_eval.html`

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
