"""wcp_Hazards_GUI.py

Hazard visualisation layer for CrowdShield (West Coast Park).

Draws a USGS-style banded hazard footprint:
- Red   : highest hazard (core / spill origin)
- Yellow: medium hazard
- Green : lower hazard

Wind-aware behaviour:
- Footprint stretches downwind and compresses upwind using wind direction
  and wind speed from the weather API.

Conventions:
- wind_deg_math is the project's 'math degrees' (0°=+x, 90°=+y).
  This is the same convention already used by wcp_core's Gaussian plume grid.

Integration from wcp_core.park_chart:
    import wcp_Hazards_GUIs as hzvis
    hzvis.add_banded_hazard(fig, HAZARDS, x_domain, y_domain,
                            dx_map_to_m, dy_map_to_m,
                            wind_deg=wind_deg,
                            wind_speed=wind_speed)

"""

import math
import numpy as np
import plotly.graph_objects as go


def _step_colorscale():
    # z in [0..3]:
    #   0 = transparent
    #   1 = green
    #   2 = yellow
    #   3 = red
    return [
        (0.00, "rgba(0,0,0,0)"),
        (0.24, "rgba(0,0,0,0)"),
        (0.25, "rgba(0,180,0,0.28)"),
        (0.49, "rgba(0,180,0,0.28)"),
        (0.50, "rgba(255,215,0,0.35)"),
        (0.74, "rgba(255,215,0,0.35)"),
        (0.75, "rgba(220,0,0,0.45)"),
        (1.00, "rgba(220,0,0,0.45)"),
    ]


def _wind_factors(wind_speed):
    if wind_speed is None or not isinstance(wind_speed, (int, float)):
        return 1.0, 1.0, 1.0

    ws = float(wind_speed)
    if not np.isfinite(ws) or ws < 0:
        return 1.0, 1.0, 1.0

    w = min(ws, 10.0)

    downwind = 1.0 + 0.20 * w
    upwind = 1.0 / (1.0 + 0.10 * w)
    crosswind = 1.0 + 0.04 * w
    return downwind, upwind, crosswind


def _hazard_band_grid_for_one(h, gx, gy, dx_map_to_m, dy_map_to_m, wind_deg, wind_speed):
    hx, hy = float(h["pos"][0]), float(h["pos"][1])

    r0 = max(5.0, float(h.get("r_m", 30.0)))  # red core
    r1 = r0 * 2.2                              # yellow edge
    r2 = r0 * 3.6                              # green edge

    X, Y = np.meshgrid(gx, gy)
    dx_map = X - hx
    dy_map = Y - hy

    dx_m = dx_map_to_m(dx_map)
    dy_m = dy_map_to_m(dy_map)

    if wind_deg is None or not isinstance(wind_deg, (int, float)) or not np.isfinite(wind_deg):
        a = 0.0
    else:
        a = math.radians(float(wind_deg) % 360.0)

    ex, ey = math.cos(a), math.sin(a)

    along = dx_m * ex + dy_m * ey
    cross = -dx_m * ey + dy_m * ex

    downwind, upwind, crosswind = _wind_factors(wind_speed)

    along_scaled = np.where(along >= 0.0, along / downwind, along / upwind)
    cross_scaled = cross / crosswind

    d_eff = np.sqrt(along_scaled * along_scaled + cross_scaled * cross_scaled)

    Z = np.zeros_like(d_eff, dtype=np.uint8)
    Z = np.where(d_eff <= r2, 1, Z)  # green
    Z = np.where(d_eff <= r1, 2, Z)  # yellow
    Z = np.where(d_eff <= r0, 3, Z)  # red
    return Z


def add_banded_hazard(fig,hazards,x_domain,y_domain,dx_map_to_m,dy_map_to_m,wind_deg,wind_speed,grid=120,):
    if not hazards:
        return

    nx = int(max(60, min(220, grid)))
    ny = int(max(50, min(200, int(grid * 0.85))))

    gx = np.linspace(float(x_domain[0]), float(x_domain[1]), nx)
    gy = np.linspace(float(y_domain[0]), float(y_domain[1]), ny)

    Zmax = None
    for h in hazards:
        Z = _hazard_band_grid_for_one(h, gx, gy, dx_map_to_m, dy_map_to_m, wind_deg, wind_speed)
        Zmax = Z if Zmax is None else np.maximum(Zmax, Z)

    if Zmax is None:
        return

    fig.add_trace(
        go.Heatmap(
            x=gx.tolist(),
            y=gy.tolist(),
            z=Zmax,
            zmin=0,
            zmax=3,
            colorscale=_step_colorscale(),
            showscale=False,
            hoverinfo="skip",
            name="Hazard (banded)",
            opacity=1.0,
            zsmooth=False,
        )
    )
