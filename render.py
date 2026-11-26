"""Draw every step of a trace as an SVG image.

    python3 render.py traces.json 3 journal/images/0001
    python3 render.py traces.json all journal/images/0003

The first command draws every step of trace 3 as step-0.svg, step-1.svg and so on.
In each image, the orange face is where the next move lands.
The second command draws only the final mesh of every trace, as shape-0.svg and so on.
It uses only the standard library, so it runs under any Python.
"""

import json
import math
import sys

SIZE = 360
LIGHT = (0.35, 0.25, 0.9)


def camera(azimuth, elevation):
    """Return the view direction and the two axes of the picture plane. Blender has z up."""
    a, e = math.radians(azimuth), math.radians(elevation)
    view = (math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e))
    right = (-math.sin(a), math.cos(a), 0.0)
    up = (-math.sin(e) * math.cos(a), -math.sin(e) * math.sin(a), math.cos(e))
    return view, right, up


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def face_normal(points):
    # Newell's method. It also works when a face is not perfectly flat.
    n = [0.0, 0.0, 0.0]
    for p, q in zip(points, points[1:] + points[:1]):
        n[0] += (p[1] - q[1]) * (p[2] + q[2])
        n[1] += (p[2] - q[2]) * (p[0] + q[0])
        n[2] += (p[0] - q[0]) * (p[1] + q[1])
    length = math.sqrt(dot(n, n)) or 1.0
    return [c / length for c in n]


def draw(mesh, highlight, scale, cam, base=(205, 215, 230), line="#1b1e24", paper="#f4f5f7"):
    """Draw one mesh. Pass paper=None to leave the background clear, for drawing on another image."""
    VIEW, RIGHT, UP = cam
    polygons = []
    for index, face in enumerate(mesh["faces"]):
        points = [mesh["verts"][v] for v in face]
        normal = face_normal(points)
        if dot(normal, VIEW) <= 0:
            continue  # The face points away from the camera.
        depth = sum(dot(p, VIEW) for p in points) / len(points)
        shade = 0.75 + 0.25 * dot(normal, LIGHT)
        tint = (255, 160, 50) if index == highlight else base
        color = "rgb(%d,%d,%d)" % tuple(min(255, int(c * shade)) for c in tint)
        xy = " ".join(
            "%.1f,%.1f" % (SIZE / 2 + scale * dot(p, RIGHT), SIZE / 2 - scale * dot(p, UP)) for p in points
        )
        polygons.append((depth, f'<polygon points="{xy}" fill="{color}"/>'))
    # shortcut: faces are drawn far to near by their center, which can misorder deep dents.
    # Upgrade to a real depth buffer when the shapes get complex.
    polygons.sort()
    body = "\n".join(p for _, p in polygons)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SIZE} {SIZE}" '
        f'stroke="{line}" stroke-width="1" stroke-linejoin="round">\n'
        + (f'<rect width="{SIZE}" height="{SIZE}" fill="{paper}" stroke="none"/>\n' if paper else "")
        + f'{body}\n</svg>\n'
    )


def fit(trace):
    """Pick the scale and the camera for one trace."""
    # One scale for the whole trace, so the mesh does not jump in size between steps.
    reach = max(math.sqrt(dot(v, v)) for step in trace["steps"] for v in step["verts"])
    highlights = [m["face"] for m in trace["moves"]] + [-1]

    # Pick the one side view that faces the most moves, so the action is not hidden at the back.
    def facing(cam):
        return sum(
            max(0.0, dot(face_normal([step["verts"][v] for v in step["faces"][h]]), cam[0]))
            for step, h in zip(trace["steps"], highlights) if h >= 0
        )

    cam = max((camera(a, 25) for a in (35, 125, 215, 305)), key=facing)
    return (SIZE / 2 - 12) / reach, cam, highlights


if __name__ == "__main__":
    path, which, out = sys.argv[1], sys.argv[2], sys.argv[3]
    traces = json.load(open(path))
    if which == "all":
        # Draw only the final mesh of every trace, for a grid of sample shapes.
        for i, trace in enumerate(traces):
            scale, cam, _ = fit(trace)
            with open(f"{out}/shape-{i}.svg", "w") as f:
                f.write(draw(trace["steps"][-1], -1, scale, cam))
        print(f"wrote {len(traces)} images to {out}")
    else:
        trace = traces[int(which)]
        scale, cam, highlights = fit(trace)
        for i, step in enumerate(trace["steps"]):
            with open(f"{out}/step-{i}.svg", "w") as f:
                f.write(draw(step, highlights[i], scale, cam))
        print(f"wrote {len(trace['steps'])} images to {out}")
