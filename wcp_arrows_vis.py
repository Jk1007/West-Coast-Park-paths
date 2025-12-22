import math
import plotly.graph_objects as go


def _get_bezier_curve(x0, y0, x1, y1, n_points=30, curve_strength=0.18):
    # Quadratic Bezier: P(t) = (1-t)^2 P0 + 2(1-t)t P1 + t^2 P2
    # Control point is offset perpendicular to the straight line for a "curvy" arrow.

    dx = x1 - x0
    dy = y1 - y0
    dist = math.sqrt(dx * dx + dy * dy)

    if dist <= 1e-9:
        return [x0, x1], [y0, y1], x1, y1, 0.0

    mx = (x0 + x1) * 0.5
    my = (y0 + y1) * 0.5

    # Perp unit vector
    px = -dy / dist
    py = dx / dist

    # Offset magnitude
    off = dist * curve_strength

    cx = mx + px * off
    cy = my + py * off

    xs = []
    ys = []
    for i in range(n_points + 1):
        t = float(i) / float(n_points)
        omt = 1.0 - t
        bx = omt * omt * x0 + 2.0 * omt * t * cx + t * t * x1
        by = omt * omt * y0 + 2.0 * omt * t * cy + t * t * y1
        xs.append(bx)
        ys.append(by)

    # angle for arrowhead at end (use last segment direction)
    x_prev = xs[-2]
    y_prev = ys[-2]
    ang = math.degrees(math.atan2(y1 - y_prev, x1 - x_prev))

    return xs, ys, x1, y1, ang


def draw_evac_arrows(fig, arrow_data):
    # arrow_data: list of ((start_x,start_y), (end_x,end_y))

    if not arrow_data:
        return

    for pair in arrow_data:
        (x0, y0), (x1, y1) = pair

        xs, ys, ax, ay, ang = _get_bezier_curve(x0, y0, x1, y1)

        # Curvy orange line
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines",
            line=dict(width=3, color="orange"),
            hoverinfo="skip",
            showlegend=False,
            name="Evac recommendation"
        ))

        # Arrowhead (triangle marker at end)
        fig.add_trace(go.Scatter(
            x=[ax], y=[ay], mode="markers",
            marker=dict(
                size=12,
                color="orange",
                symbol="triangle-up",
                angle=ang - 90.0  # triangle-up points up; rotate to match direction
            ),
            hoverinfo="skip",
            showlegend=False,
            name="Evac arrowhead"
        ))
