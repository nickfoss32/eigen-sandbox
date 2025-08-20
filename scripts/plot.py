import json
import os
import numpy as np
import plotly.graph_objects as go
import argparse
import geopandas as gpd

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Plot 3D ballistic missile trajectory from a JSON file over a globe with toggleable full/subsection views.")
parser.add_argument("trajectory_file", type=str, help="Path to the JSON file containing trajectory data.")
args = parser.parse_args()

# Load JSON data
try:
    with open(args.trajectory_file, "r") as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Error: File '{args.trajectory_file}' not found.")
    exit(1)

# Extract time and state vectors
times = [point["time"] for point in data["points"]]
x = np.array([point["state"][0] for point in data["points"]])
y = np.array([point["state"][1] for point in data["points"]])
z = np.array([point["state"][2] for point in data["points"]])

# Create 3D trajectory trace
trajectory_trace = go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode="markers",
    name="Trajectory",
    marker=dict(color="red", size=2),
    hovertemplate="Time: %{text:.2f}s<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}",
    text=times,
    visible=True
)

# Get launch point (first point) and convert to spherical coordinates
earth_radius = 6371000  # meters
launch_point = np.array([x[0], y[0], z[0]])
lat_launch = np.degrees(np.arcsin(z[0] / earth_radius))  # Latitude in degrees
lon_launch = np.degrees(np.arctan2(y[0], x[0]))          # Longitude in degrees

# Define zoomed subsection around launch point (±10 degrees)
lat_range = 10
lon_range = 10
lat_min = max(-90, lat_launch - lat_range)
lat_max = min(90, lat_launch + lat_range)
lon_min = lon_launch - lon_range
lon_max = lon_launch + lon_range

# Full globe surface
u_full = np.linspace(0, 2 * np.pi, 50)
v_full = np.linspace(0, np.pi, 50)
u_full, v_full = np.meshgrid(u_full, v_full)
globe_x_full = earth_radius * np.cos(u_full) * np.sin(v_full)
globe_y_full = earth_radius * np.sin(u_full) * np.sin(v_full)
globe_z_full = earth_radius * np.cos(v_full)
globe_trace_full = go.Surface(
    x=globe_x_full,
    y=globe_y_full,
    z=globe_z_full,
    colorscale="Blues",
    opacity=0.6,
    name="Full Globe",
    hovertext="skip",
    showscale=False,
    visible=True  # Start visible (full globe is default)
)

# Zoomed subsection surface
u_zoomed = np.linspace(np.radians(lon_min), np.radians(lon_max), 50)
v_zoomed = np.linspace(np.radians(90 - lat_max), np.radians(90 - lat_min), 50)
u_zoomed, v_zoomed = np.meshgrid(u_zoomed, v_zoomed)
globe_x_zoomed = earth_radius * np.cos(u_zoomed) * np.sin(v_zoomed)
globe_y_zoomed = earth_radius * np.sin(u_zoomed) * np.sin(v_zoomed)
globe_z_zoomed = earth_radius * np.cos(v_zoomed)
globe_trace_zoomed = go.Surface(
    x=globe_x_zoomed,
    y=globe_y_zoomed,
    z=globe_z_zoomed,
    colorscale="Blues",
    opacity=0.6,
    name="Globe Subsection",
    hovertext="skip",
    showscale=False,
    visible=False  # Start hidden (full globe is default)
)

# Load Shapefile for country boundaries
shapefile_path = "resources/ne/ne_110m_land.shp"
try:
    gdf = gpd.read_file(shapefile_path)
except FileNotFoundError:
    print(f"Error: Shapefile '{shapefile_path}' not found. Please download from https://www.naturalearthdata.com/downloads/")
    exit(1)

# Extract country boundaries and convert to 3D coordinates
full_country_traces = []
zoomed_country_traces = []
for _, row in gdf.iterrows():
    geometry = row["geometry"]
    name = row.get("NAME", "Unknown")
    if geometry.geom_type == "Polygon":
        geometries = [geometry]
    elif geometry.geom_type == "MultiPolygon":
        geometries = geometry.geoms
    else:
        continue

    for poly in geometries:
        coords = poly.exterior.coords
        lon_country = np.array([c[0] for c in coords])
        lat_country = np.array([c[1] for c in coords])

        # Full globe boundaries
        lat_rad = np.radians(lat_country)
        lon_rad = np.radians(lon_country)
        x_country = earth_radius * np.cos(lon_rad) * np.cos(lat_rad)
        y_country = earth_radius * np.sin(lon_rad) * np.cos(lat_rad)
        z_country = earth_radius * np.sin(lat_rad)
        full_country_traces.append(
            go.Scatter3d(
                x=x_country,
                y=y_country,
                z=z_country,
                mode="lines",
                line=dict(color="black", width=3),
                name=name,
                hoverinfo="name",
                showlegend=False,
                visible=True  # Visible in full globe view
            )
        )

        # Zoomed subsection boundaries (clipped)
        mask = (lat_country >= lat_min) & (lat_country <= lat_max) & (lon_country >= lon_min) & (lon_country <= lon_max)
        if np.any(mask):
            zoomed_country_traces.append(
                go.Scatter3d(
                    x=x_country[mask],
                    y=y_country[mask],
                    z=z_country[mask],
                    mode="lines",
                    line=dict(color="black", width=3),
                    name=name,
                    hoverinfo="name",
                    showlegend=False,
                    visible=False  # Hidden in full globe view
                )
            )

# Extract best fit plane parameters if available
has_fit = "fit" in data["summary"]
plane_trace = None
if has_fit:
    normal_list = data["summary"]["fit"]["normal"]
    point_list = data["summary"]["fit"]["point"]
    normal = np.array(normal_list)
    point = np.array(point_list)
    normal = normal / np.linalg.norm(normal)  # Unit normal

    # Function to get a perpendicular vector
    def perpendicular_vector(n):
        if n[0] == 0 and n[1] == 0:
            if n[2] == 0:
                raise ValueError("Zero normal vector")
            return np.array([1, 0, 0])
        return np.array([-n[1], n[0], 0])

    # Basis vectors for the plane
    v_temp = perpendicular_vector(normal)
    v1 = v_temp / np.linalg.norm(v_temp)
    v2 = np.cross(normal, v1)

    # Trajectory points
    traj_points = np.column_stack((x, y, z))

    # Project trajectory points onto the plane
    dists = np.dot(traj_points - point, normal)
    projs = traj_points - np.outer(dists, normal)

    # Compute u, v coordinates on the plane
    us = np.dot(projs - point, v1)
    vs = np.dot(projs - point, v2)

    u_min, u_max = np.min(us), np.max(us)
    v_min, v_max = np.min(vs), np.max(vs)
    margin = 0.1 * max(u_max - u_min, v_max - v_min)
    if margin == 0:
        margin = 1e5  # Default margin if trajectory is degenerate

    # Grid for the plane surface
    u_grid = np.linspace(u_min - margin, u_max + margin, 30)
    v_grid = np.linspace(v_min - margin, v_max + margin, 30)
    uu, vv = np.meshgrid(u_grid, v_grid)

    plane_x = point[0] + uu * v1[0] + vv * v2[0]
    plane_y = point[1] + uu * v1[1] + vv * v2[1]
    plane_z = point[2] + uu * v1[2] + vv * v2[2]

    # Best fit plane trace (always visible in both views)
    plane_trace = go.Surface(
        x=plane_x,
        y=plane_y,
        z=plane_z,
        colorscale=[[0, 'rgb(0,255,0)'], [1, 'rgb(0,255,0)']],
        opacity=0.3,
        name="Best Fit Plane",
        showscale=False,
        hoverinfo="skip",
        visible=True
    )

# Define update menus for toggling views
has_plane = plane_trace is not None
full_visible = [True, False, True] + ([True] if has_plane else []) + [True] * len(full_country_traces) + [False] * len(zoomed_country_traces)
zoom_visible = [False, True, True] + ([True] if has_plane else []) + [False] * len(full_country_traces) + [True] * len(zoomed_country_traces)
updatemenus = [
    dict(
        buttons=[
            dict(
                label="Full Globe",
                method="update",
                args=[{"visible": full_visible}]
            ),
            dict(
                label="Zoomed Subsection",
                method="update",
                args=[{"visible": zoom_visible}]
            )
        ],
        direction="up",
        showactive=True,
        x=0.1,
        xanchor="left",
        y=-0.1,
        yanchor="bottom"
    )
]

# Extract and format summary metadata
summary = data.get("summary", {})
simulation = summary.get("simulation", {})
launch = simulation.get("launch", {})
noise = simulation.get("noise", {})

# Format metadata as a string
metadata_text = (
    "<b>Simulation Metadata</b><br>"
    f"Launch Altitude: {launch.get('altitude', 'N/A')} m<br>"
    f"Launch Azimuth: {launch.get('azimuth', 'N/A')}°<br>"
    f"Launch Elevation: {launch.get('elevation', 'N/A')}°<br>"
    f"Launch Latitude: {launch.get('latitude', 'N/A')}°<br>"
    f"Launch Longitude: {launch.get('longitude', 'N/A')}°<br>"
    f"Launch Velocity: {launch.get('velocity', 'N/A')} m/s<br>"
    f"Noise Sigma Pos: {noise.get('sigma_pos', 'N/A')} m<br>"
    f"Noise Sigma Vel: {noise.get('sigma_vel', 'N/A')} m/s<br>"
    f"Timestep: {simulation.get('timestep', 'N/A')} s"
)

# Create figure
data_traces = [globe_trace_full, globe_trace_zoomed, trajectory_trace]
if plane_trace:
    data_traces.append(plane_trace)
data_traces += full_country_traces + zoomed_country_traces
fig = go.Figure(data=data_traces)

# Update the plot layout
filename = os.path.basename(args.trajectory_file)
fig.update_layout(
    title=dict(
        text=f"ECEF Track Trajectory from: {filename}",
        x=0.5,
        xanchor="center",
        y=0.95,
        yanchor="top"
    ),
    scene=dict(
        xaxis_title="X (ECEF M)",
        yaxis_title="Y (ECEF M)",
        zaxis_title="Z (ECEF M)",
        aspectmode="cube",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        zaxis=dict(showgrid=False)
    ),
    updatemenus=updatemenus,
    showlegend=True,
    annotations=[
        dict(
            text=metadata_text,
            x=0.05,  # Position in the left corner
            y=0.95,  # Position near the top
            xref="paper",
            yref="paper",
            showarrow=False,
            align="left",
            bgcolor="rgba(255, 255, 255, 0.8)",  # Semi-transparent white background
            bordercolor="black",
            borderwidth=1,
            font=dict(size=12)
        )
    ]
)

# Show plot
fig.show()
