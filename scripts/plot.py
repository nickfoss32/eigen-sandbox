import json
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

# Define update menus for toggling views
updatemenus = [
    dict(
        buttons=[
            dict(
                label="Full Globe",
                method="update",
                args=[{"visible": [True, False, True] + [True] * len(full_country_traces) + [False] * len(zoomed_country_traces)}]
            ),
            dict(
                label="Zoomed Subsection",
                method="update",
                args=[{"visible": [False, True, True] + [False] * len(full_country_traces) + [True] * len(zoomed_country_traces)}]
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

# Create figure
fig = go.Figure(data=[globe_trace_full, globe_trace_zoomed, trajectory_trace] + full_country_traces + zoomed_country_traces)

# Update layout without custom camera settings
fig.update_layout(
    title=dict(
        text="3D Ballistic Missile Trajectory with Toggleable Globe Views",
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
    showlegend=True
)

# Show plot
fig.show()
