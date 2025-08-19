import json
import numpy as np
import plotly.graph_objects as go
import argparse
import geopandas as gpd

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Plot 3D ballistic missile trajectory from a JSON file over a globe with country boundaries from a Shapefile.")
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
    text=times
)

# Create globe surface (spherical mesh)
u = np.linspace(0, 2 * np.pi, 50)
v = np.linspace(0, np.pi, 50)
u, v = np.meshgrid(u, v)
earth_radius = 6371000  # meters
globe_x = earth_radius * np.cos(u) * np.sin(v)
globe_y = earth_radius * np.sin(u) * np.sin(v)
globe_z = earth_radius * np.cos(v)
globe_trace = go.Surface(
    x=globe_x,
    y=globe_y,
    z=globe_z,
    colorscale="Blues",
    opacity=0.6,
    name="Globe",
    showscale=False
)

# Load Shapefile for country boundaries
shapefile_path = "resources/ne/ne_110m_land.shp"
try:
    gdf = gpd.read_file(shapefile_path)
except FileNotFoundError:
    print(f"Error: Shapefile '{shapefile_path}' not found. Please download from https://www.naturalearthdata.com/downloads/")
    exit(1)

# Extract country boundaries and convert to 3D coordinates
country_traces = []
for _, row in gdf.iterrows():
    geometry = row["geometry"]
    name = row.get("NAME", "Unknown")  # Use country name from Shapefile
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
        # Convert to Cartesian coordinates
        lat_rad = np.radians(lat_country)
        lon_rad = np.radians(lon_country)
        x_country = earth_radius * np.cos(lon_rad) * np.cos(lat_rad)
        y_country = earth_radius * np.sin(lon_rad) * np.cos(lat_rad)
        z_country = earth_radius * np.sin(lat_rad)
        country_traces.append(
            go.Scatter3d(
                x=x_country,
                y=y_country,
                z=z_country,
                mode="lines",
                line=dict(color="black", width=1),
                name=name,
                hoverinfo="name",
                showlegend=False
            )
        )

# Plot best fit plane info

# Create figure
fig = go.Figure(data=[globe_trace, trajectory_trace] + country_traces)

# Update layout
fig.update_layout(
    title="3D Ballistic Missile Trajectory over Globe",
    scene=dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z",
        aspectmode="cube",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        zaxis=dict(showgrid=False)
    ),
    showlegend=True
)

# Show plot
fig.show()
