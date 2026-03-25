#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Optional


COLORS = [
    "#e63946",
    "#1d3557",
    "#2a9d8f",
    "#f4a261",
    "#6a4c93",
    "#264653",
    "#ff006e",
    "#3a86ff",
]


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise SystemExit(f"Error: file not found: {path}") from exc


def build_trajectory_index(data: dict) -> dict[str, dict]:
    trajectories = data.get("trajectories", [])
    return {
        trajectory["name"]: trajectory
        for trajectory in trajectories
        if trajectory.get("name") and trajectory.get("points")
    }


def select_overview_trajectories(summary_runs: list[dict], top_models: int) -> list[dict]:
    selections = []
    seen = set()

    for run in summary_runs:
        if run.get("mode") == "imm":
            combined_name = f"{run['name']}-Combined"
            selections.append(
                {
                    "trajectory_name": combined_name,
                    "label": f"{run['name']}:Combined",
                }
            )
            seen.add(combined_name)
            if run.get("performance", {}).get("smoothed_combined"):
                smoothed_name = f"{run['name']}-RTSCombined"
                selections.append(
                    {
                        "trajectory_name": smoothed_name,
                        "label": f"{run['name']}:RTS",
                    }
                )
                seen.add(smoothed_name)

        ranking = run.get("performance", {}).get("position_rmse_ranking", [])
        for entry in ranking[:top_models]:
            trajectory_name = f"{run['name']}-{entry['name']}"
            if trajectory_name in seen:
                continue
            selections.append(
                {
                    "trajectory_name": trajectory_name,
                    "label": f"{run['name']}:{entry['name']}",
                }
            )
            seen.add(trajectory_name)

    return selections


def phase_overview_series(summary_runs: list[dict]) -> list[dict]:
    series = []
    for run in summary_runs:
        if run.get("mode") == "comparison":
            best_name = run.get("best_model_name")
            performance_models = run.get("performance", {}).get("models", [])
            matching = next(
                (model for model in performance_models if model.get("name") == best_name),
                None,
            )
            if matching is None:
                continue
            series.append(
                {
                    "label": f"{run['name']}:{best_name}",
                    "phase_metrics": matching.get("phase_metrics", []),
                }
            )
        elif run.get("mode") == "imm":
            combined = run.get("performance", {}).get("combined", {})
            series.append(
                {
                    "label": f"{run['name']}:Combined",
                    "phase_metrics": combined.get("phase_metrics", []),
                }
            )
            smoothed = run.get("performance", {}).get("smoothed_combined", {})
            if smoothed:
                series.append(
                    {
                        "label": f"{run['name']}:RTS",
                        "phase_metrics": smoothed.get("phase_metrics", []),
                    }
                )

    return series


def phase_lines(phases: list[dict], max_time: float) -> list[dict]:
    shapes = []
    for phase in phases:
        start_time = phase.get("start_time", 0.0)
        if not isinstance(start_time, (int, float)) or start_time <= 0.0 or start_time >= max_time:
            continue

        shapes.append(
            {
                "type": "line",
                "xref": "x",
                "yref": "paper",
                "x0": start_time,
                "x1": start_time,
                "y0": 0.0,
                "y1": 1.0,
                "line": {
                    "color": "rgba(80, 80, 80, 0.55)",
                    "dash": "dot",
                    "width": 1,
                },
            }
        )
    return shapes


def empty_layout_annotation(text: str) -> list[dict]:
    return [
        {
            "text": text,
            "xref": "paper",
            "yref": "paper",
            "x": 0.5,
            "y": 0.5,
            "showarrow": False,
            "font": {"size": 12, "color": "gray"},
        }
    ]


def make_line_traces(
    selections: list[dict],
    trajectory_index: dict[str, dict],
    point_key: str,
    hover_label: str,
) -> tuple[list[dict], float]:
    traces = []
    max_time = 0.0

    for idx, selection in enumerate(selections):
        trajectory = trajectory_index.get(selection["trajectory_name"])
        if trajectory is None:
            continue

        points = trajectory.get("points", [])
        times = [point.get("time") for point in points]
        values = [point.get(point_key) for point in points]
        if times:
            max_time = max(max_time, max(times))

        traces.append(
            {
                "type": "scatter",
                "mode": "lines",
                "name": selection["label"],
                "x": times,
                "y": values,
                "line": {"color": COLORS[idx % len(COLORS)], "width": 2},
                "hovertemplate": (
                    f"{selection['label']}<br>t=%{{x:.1f}} s<br>{hover_label}=%{{y:.3f}}"
                    "<extra></extra>"
                ),
            }
        )

    return traces, max_time


def state_metric_from_point(point: dict, metric: str) -> Optional[float]:
    state = point.get("state")
    if not isinstance(state, list) or len(state) < 6:
        return None

    if metric == "altitude_km":
        radius_m = sum(component * component for component in state[:3]) ** 0.5
        return (radius_m - 6378137.0) / 1000.0
    if metric == "speed_mps":
        return sum(component * component for component in state[3:6]) ** 0.5

    return None


def make_truth_state_traces(
    selections: list[dict],
    trajectory_index: dict[str, dict],
    metric: str,
    hover_label: str,
) -> tuple[list[dict], float]:
    traces = []
    max_time = 0.0

    truth_trajectory = trajectory_index.get("Truth")
    if truth_trajectory is not None:
        truth_points = truth_trajectory.get("points", [])
        truth_times = [point.get("time") for point in truth_points]
        truth_values = [state_metric_from_point(point, metric) for point in truth_points]
        if truth_times:
            max_time = max(max_time, max(truth_times))
        traces.append(
            {
                "type": "scatter",
                "mode": "lines",
                "name": "Truth",
                "x": truth_times,
                "y": truth_values,
                "line": {"color": "#222222", "width": 3, "dash": "dash"},
                "hovertemplate": f"Truth<br>t=%{{x:.1f}} s<br>{hover_label}=%{{y:.3f}}<extra></extra>",
            }
        )

    for idx, selection in enumerate(selections):
        trajectory = trajectory_index.get(selection["trajectory_name"])
        if trajectory is None:
            continue

        points = trajectory.get("points", [])
        times = [point.get("time") for point in points]
        values = [state_metric_from_point(point, metric) for point in points]
        if times:
            max_time = max(max_time, max(times))

        traces.append(
            {
                "type": "scatter",
                "mode": "lines",
                "name": selection["label"],
                "x": times,
                "y": values,
                "line": {"color": COLORS[idx % len(COLORS)], "width": 2},
                "hovertemplate": (
                    f"{selection['label']}<br>t=%{{x:.1f}} s<br>{hover_label}=%{{y:.3f}}"
                    "<extra></extra>"
                ),
            }
        )

    return traces, max_time


def make_rmse_bar(summary_runs: list[dict]) -> list[dict]:
    entries = []
    for run in summary_runs:
        for model in run.get("performance", {}).get("models", []):
            entries.append(
                {
                    "label": f"{run['name']}:{model['name']}",
                    "value": model.get("position_rmse_m", 0.0),
                }
            )

    entries.sort(key=lambda entry: entry["value"])
    if not entries:
        return []

    return [
        {
            "type": "bar",
            "x": [entry["label"] for entry in entries],
            "y": [entry["value"] for entry in entries],
            "marker": {"color": [COLORS[idx % len(COLORS)] for idx, _ in enumerate(entries)]},
            "hovertemplate": "%{x}<br>pos RMSE=%{y:.2f} m<extra></extra>",
            "name": "Position RMSE",
        }
    ]


def make_phase_bars(summary_runs: list[dict], phases: list[dict]) -> list[dict]:
    series = phase_overview_series(summary_runs)
    phase_names = [phase.get("name", f"Phase {idx + 1}") for idx, phase in enumerate(phases)]
    traces = []

    for idx, item in enumerate(series):
        phase_lookup = {
            metric.get("name"): metric.get("position_rmse_m", 0.0)
            for metric in item.get("phase_metrics", [])
        }
        traces.append(
            {
                "type": "bar",
                "name": item["label"],
                "x": phase_names,
                "y": [phase_lookup.get(phase_name, 0.0) for phase_name in phase_names],
                "marker": {"color": COLORS[idx % len(COLORS)]},
                "hovertemplate": "%{x}<br>pos RMSE=%{y:.2f} m<extra></extra>",
            }
        )

    return traces


def make_smoother_comparison_bar(summary_runs: list[dict]) -> list[dict]:
    imm_runs = [
        run
        for run in summary_runs
        if run.get("mode") == "imm" and run.get("performance", {}).get("smoothed_combined")
    ]
    if not imm_runs:
        return []

    filtered_values = []
    smoothed_values = []
    labels = []
    for run in imm_runs:
        labels.append(run["name"])
        filtered_values.append(run.get("performance", {}).get("combined", {}).get("position_rmse_m", 0.0))
        smoothed_values.append(
            run.get("performance", {}).get("smoothed_combined", {}).get("position_rmse_m", 0.0)
        )

    return [
        {
            "type": "bar",
            "name": "IMM Combined",
            "x": labels,
            "y": filtered_values,
            "marker": {"color": "#1d3557"},
            "hovertemplate": "%{x}<br>filtered pos RMSE=%{y:.2f} m<extra></extra>",
        },
        {
            "type": "bar",
            "name": "IMM + RTS",
            "x": labels,
            "y": smoothed_values,
            "marker": {"color": "#2a9d8f"},
            "hovertemplate": "%{x}<br>smoothed pos RMSE=%{y:.2f} m<extra></extra>",
        },
    ]


def make_imm_probability_traces(
    summary_runs: list[dict],
    trajectory_index: dict[str, dict],
) -> list[dict]:
    traces = []
    color_index = 0

    for run in summary_runs:
        if run.get("mode") != "imm":
            continue

        combined_name = f"{run['name']}-Combined"
        combined_trajectory = trajectory_index.get(combined_name)
        if combined_trajectory is None:
            continue

        points = combined_trajectory.get("points", [])
        times = [point.get("time") for point in points]

        for model_index, model_info in enumerate(run.get("models", [])):
            probabilities = []
            for point in points:
                model_probabilities = point.get("model_probabilities")
                if isinstance(model_probabilities, list) and model_index < len(model_probabilities):
                    probabilities.append(model_probabilities[model_index])
                else:
                    probabilities.append(None)

            traces.append(
                {
                    "type": "scatter",
                    "mode": "lines",
                    "name": f"{run['name']}:{model_info['name']}",
                    "x": times,
                    "y": probabilities,
                    "line": {"color": COLORS[color_index % len(COLORS)], "width": 2},
                    "hovertemplate": (
                        f"{run['name']}:{model_info['name']}<br>"
                        "t=%{x:.1f} s<br>mu=%{y:.3f}<extra></extra>"
                    ),
                }
            )
            color_index += 1

    return traces


def build_dashboard_payload(data: dict, top_models: int) -> dict:
    summary = data.get("summary", {})
    summary_runs = summary.get("runs", [])
    if not summary_runs:
        raise SystemExit("Error: JSON does not contain summary.runs. Use imm-demo output.")

    trajectory_index = build_trajectory_index(data)
    phases = summary.get("simulation", {}).get("phases", [])
    selections = select_overview_trajectories(summary_runs, max(top_models, 1))
    if not selections:
        raise SystemExit("Error: no trajectories were selected for evaluation plotting.")

    position_traces, max_time = make_line_traces(
        selections,
        trajectory_index,
        "position_error_m",
        "Position error [m]",
    )
    velocity_traces, max_time_velocity = make_line_traces(
        selections,
        trajectory_index,
        "velocity_error_mps",
        "Velocity error [m/s]",
    )
    nees_traces, max_time_nees = make_line_traces(
        selections,
        trajectory_index,
        "state_nees_per_dim",
        "State NEES / dim",
    )
    altitude_traces, max_time_altitude = make_truth_state_traces(
        selections,
        trajectory_index,
        "altitude_km",
        "Altitude [km]",
    )
    speed_traces, max_time_speed = make_truth_state_traces(
        selections,
        trajectory_index,
        "speed_mps",
        "Speed [m/s]",
    )

    max_time = max(max_time, max_time_velocity, max_time_nees, max_time_altitude, max_time_speed)
    shapes = phase_lines(phases, max_time)

    requested = summary.get("requested", {})
    simulation = summary.get("simulation", {})
    sensor = summary.get("sensor", {})

    return {
        "title": f"Filter Evaluation Dashboard: {simulation.get('coordinate_frame', 'ECI')} tracking",
        "subtitle": {
            "requested_mode": requested.get("mode", "N/A"),
            "requested_filter_family": requested.get("filter_family", "N/A"),
            "primary_trajectory_name": summary.get("primary_trajectory_name", "N/A"),
            "duration_seconds": simulation.get("duration", "N/A"),
            "sensor_type": sensor.get("type", "N/A"),
        },
        "position_plot": {
            "div_id": "position-error-plot",
            "traces": position_traces,
            "layout": {
                "title": "Position Error vs Time",
                "xaxis": {"title": "Time [s]"},
                "yaxis": {"title": "Position Error [m]"},
                "shapes": shapes,
                "annotations": empty_layout_annotation("No position-error trajectories found")
                if not position_traces
                else [],
            },
        },
        "altitude_plot": {
            "div_id": "altitude-plot",
            "traces": altitude_traces,
            "layout": {
                "title": "Altitude vs Time (Truth Included)",
                "xaxis": {"title": "Time [s]"},
                "yaxis": {"title": "Altitude [km]"},
                "shapes": shapes,
                "annotations": empty_layout_annotation("No trajectory state data found")
                if not altitude_traces
                else [],
            },
        },
        "speed_plot": {
            "div_id": "speed-plot",
            "traces": speed_traces,
            "layout": {
                "title": "Speed vs Time (Truth Included)",
                "xaxis": {"title": "Time [s]"},
                "yaxis": {"title": "Speed [m/s]"},
                "shapes": shapes,
                "annotations": empty_layout_annotation("No trajectory state data found")
                if not speed_traces
                else [],
            },
        },
        "velocity_plot": {
            "div_id": "velocity-error-plot",
            "traces": velocity_traces,
            "layout": {
                "title": "Velocity Error vs Time",
                "xaxis": {"title": "Time [s]"},
                "yaxis": {"title": "Velocity Error [m/s]"},
                "shapes": shapes,
                "annotations": empty_layout_annotation("No velocity-error trajectories found")
                if not velocity_traces
                else [],
            },
        },
        "nees_plot": {
            "div_id": "state-nees-plot",
            "traces": nees_traces,
            "layout": {
                "title": "State NEES Per Dimension",
                "xaxis": {"title": "Time [s]"},
                "yaxis": {"title": "NEES / dimension"},
                "shapes": shapes,
                "annotations": empty_layout_annotation("No NEES data found")
                if not nees_traces
                else [],
            },
        },
        "rmse_plot": {
            "div_id": "rmse-ranking-plot",
            "traces": make_rmse_bar(summary_runs),
            "layout": {
                "title": "Position RMSE Ranking",
                "xaxis": {"title": "Trajectory", "tickangle": -35},
                "yaxis": {"title": "Position RMSE [m]"},
                "annotations": empty_layout_annotation("No RMSE summary found")
                if not make_rmse_bar(summary_runs)
                else [],
            },
        },
        "phase_plot": {
            "div_id": "phase-rmse-plot",
            "traces": make_phase_bars(summary_runs, phases),
            "layout": {
                "title": "Phase Position RMSE",
                "xaxis": {"title": "Phase"},
                "yaxis": {"title": "Position RMSE [m]"},
                "barmode": "group",
                "annotations": empty_layout_annotation("No phase metrics found")
                if not make_phase_bars(summary_runs, phases)
                else [],
            },
        },
        "smoother_plot": {
            "div_id": "smoother-comparison-plot",
            "traces": make_smoother_comparison_bar(summary_runs),
            "layout": {
                "title": "IMM Combined vs RTS-Smoothed RMSE",
                "xaxis": {"title": "IMM Run"},
                "yaxis": {"title": "Position RMSE [m]"},
                "barmode": "group",
                "annotations": empty_layout_annotation("No RTS-smoothed IMM runs found")
                if not make_smoother_comparison_bar(summary_runs)
                else [],
            },
        },
        "imm_probability_plot": {
            "div_id": "imm-mode-probability-plot",
            "traces": make_imm_probability_traces(summary_runs, trajectory_index),
            "layout": {
                "title": "IMM Mode Probabilities",
                "xaxis": {"title": "Time [s]"},
                "yaxis": {"title": "Mode Probability", "range": [0.0, 1.0]},
                "annotations": empty_layout_annotation("No IMM runs found")
                if not make_imm_probability_traces(summary_runs, trajectory_index)
                else [],
            },
        },
    }


def html_template(payload: dict) -> str:
    payload_json = json.dumps(payload)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{payload['title']}</title>
  <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
  <style>
    body {{
      margin: 0;
      font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
      background: linear-gradient(180deg, #f7f7f2 0%, #eef3f8 100%);
      color: #14213d;
    }}
    .page {{
      max-width: 1440px;
      margin: 0 auto;
      padding: 24px;
    }}
    .hero {{
      background: rgba(255, 255, 255, 0.82);
      border: 1px solid rgba(20, 33, 61, 0.08);
      border-radius: 18px;
      padding: 20px 24px;
      margin-bottom: 20px;
      box-shadow: 0 12px 32px rgba(20, 33, 61, 0.08);
    }}
    .hero h1 {{
      margin: 0 0 10px;
      font-size: 28px;
    }}
    .meta {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-top: 12px;
    }}
    .meta-card {{
      background: rgba(245, 247, 250, 0.95);
      border-radius: 12px;
      padding: 10px 12px;
      border: 1px solid rgba(20, 33, 61, 0.08);
    }}
    .meta-card .label {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #5c677d;
      margin-bottom: 4px;
    }}
    .meta-card .value {{
      font-size: 16px;
      font-weight: 600;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(520px, 1fr));
      gap: 18px;
    }}
    .panel {{
      background: rgba(255, 255, 255, 0.86);
      border-radius: 18px;
      border: 1px solid rgba(20, 33, 61, 0.08);
      box-shadow: 0 12px 32px rgba(20, 33, 61, 0.08);
      padding: 10px;
    }}
    .plot {{
      width: 100%;
      height: 470px;
    }}
    .wide {{
      grid-column: 1 / -1;
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>{payload['title']}</h1>
      <div class="meta">
        <div class="meta-card"><div class="label">Requested Mode</div><div class="value">{payload['subtitle']['requested_mode']}</div></div>
        <div class="meta-card"><div class="label">Filter Family</div><div class="value">{payload['subtitle']['requested_filter_family']}</div></div>
        <div class="meta-card"><div class="label">Primary Trajectory</div><div class="value">{payload['subtitle']['primary_trajectory_name']}</div></div>
        <div class="meta-card"><div class="label">Duration</div><div class="value">{payload['subtitle']['duration_seconds']} s</div></div>
        <div class="meta-card"><div class="label">Sensor</div><div class="value">{payload['subtitle']['sensor_type']}</div></div>
      </div>
    </section>
    <section class="grid">
      <div class="panel"><div id="{payload['altitude_plot']['div_id']}" class="plot"></div></div>
      <div class="panel"><div id="{payload['speed_plot']['div_id']}" class="plot"></div></div>
      <div class="panel"><div id="{payload['position_plot']['div_id']}" class="plot"></div></div>
      <div class="panel"><div id="{payload['velocity_plot']['div_id']}" class="plot"></div></div>
      <div class="panel"><div id="{payload['nees_plot']['div_id']}" class="plot"></div></div>
      <div class="panel"><div id="{payload['rmse_plot']['div_id']}" class="plot"></div></div>
      <div class="panel"><div id="{payload['phase_plot']['div_id']}" class="plot"></div></div>
      <div class="panel"><div id="{payload['smoother_plot']['div_id']}" class="plot"></div></div>
      <div class="panel"><div id="{payload['imm_probability_plot']['div_id']}" class="plot"></div></div>
    </section>
  </div>
  <script>
    const dashboard = {payload_json};
    const commonConfig = {{
      responsive: true,
      displaylogo: false,
    }};

    function renderPlot(section) {{
      Plotly.newPlot(
        section.div_id,
        section.traces,
        Object.assign(
          {{
            paper_bgcolor: "rgba(0,0,0,0)",
            plot_bgcolor: "rgba(0,0,0,0)",
            margin: {{ l: 60, r: 20, t: 70, b: 135 }},
            legend: {{
              orientation: "h",
              yanchor: "top",
              y: -0.28,
              xanchor: "left",
              x: 0.0
            }},
          }},
          section.layout
        ),
        commonConfig
      );
    }}

    renderPlot(dashboard.altitude_plot);
    renderPlot(dashboard.speed_plot);
    renderPlot(dashboard.position_plot);
    renderPlot(dashboard.velocity_plot);
    renderPlot(dashboard.nees_plot);
    renderPlot(dashboard.rmse_plot);
    renderPlot(dashboard.phase_plot);
    renderPlot(dashboard.smoother_plot);
    renderPlot(dashboard.imm_probability_plot);
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an HTML evaluation dashboard for imm-demo JSON output."
    )
    parser.add_argument("trajectory_file", type=Path, help="Path to the imm-demo JSON file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output HTML file path. Defaults to <input>_eval.html.",
    )
    parser.add_argument(
        "--top-models",
        type=int,
        default=3,
        help="Number of top-ranked model trajectories to include per run in the time-series panels.",
    )
    args = parser.parse_args()

    data = load_json(args.trajectory_file)
    payload = build_dashboard_payload(data, args.top_models)

    output_path = args.output or args.trajectory_file.with_name(
        f"{args.trajectory_file.stem}_eval.html"
    )
    output_path.write_text(html_template(payload))
    print(f"Wrote evaluation dashboard to {output_path}")


if __name__ == "__main__":
    main()
