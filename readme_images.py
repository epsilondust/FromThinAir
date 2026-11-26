"""Make the images in the README from the round zero dataset.

    python readme_images.py

It writes images/build.svg, images/shapes.svg and images/flags.svg.
Every shape in them is a real trace from data/round0.npz.
"""

import os

import numpy as np

import render
import shard

CELL = 180
INK, QUIET, PAPER = "#1b1e24", "#5c6470", "#f4f5f7"
FONT = 'font-family="system-ui, -apple-system, Segoe UI, sans-serif"'


def tile(trace, step, x, y, highlight=True):
    """Draw one step of a trace as a square placed at (x, y)."""
    scale, cam, highlights = render.fit(trace)
    picture = render.draw(trace["steps"][step], highlights[step] if highlight else -1, scale, cam)
    return picture.replace("<svg xmlns", f'<svg x="{x}" y="{y}" width="{CELL}" height="{CELL}" xmlns', 1)


def sheet(width, height, parts):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" {FONT}>\n'
        f'<rect width="{width}" height="{height}" rx="12" fill="{PAPER}"/>\n' + "\n".join(parts) + "\n</svg>\n"
    )


def label(x, y, words, color=QUIET, size=13, weight=400):
    return f'<text x="{x}" y="{y}" text-anchor="middle" font-size="{size}" font-weight="{weight}" fill="{color}">{words}</text>'


def moves_face_camera(trace):
    """Say whether every move lands on a face the camera can see well."""
    _, cam, highlights = render.fit(trace)
    return all(
        render.dot(render.face_normal([step["verts"][v] for v in step["faces"][h]]), cam[0]) > 0.3
        for step, h in zip(trace["steps"], highlights) if h >= 0
    )


if __name__ == "__main__":
    traces = shard.load("data/round0.npz")
    checks = np.load("data/round0-evals.npz")
    surface = np.load("data/round0-surface.npy")
    lengths = np.array([len(t["moves"]) for t in traces])
    os.makedirs("images", exist_ok=True)

    # The build strip: the first clean five-move trace with at least three extrudes whose moves
    # all land where the camera can see them.
    build = next(
        t for i, t in enumerate(traces)
        if checks["clean"][i] and lengths[i] == 5
        and sum(m["op"] == "extrude" for m in t["moves"]) >= 3 and moves_face_camera(t)
    )
    parts = []
    for step in range(6):
        x = 12 + step * (CELL + 8)
        parts.append(tile(build, step, x, 12))
        move = build["moves"][step] if step < 5 else None
        words = f'next: {move["op"]} by {move["amount"]:g}' if move else "the finished shape"
        parts.append(label(x + CELL / 2, CELL + 34, words, INK if move is None else QUIET, 13, 600 if move is None else 400))
    with open("images/build.svg", "w") as f:
        f.write(sheet(12 + 6 * (CELL + 8) + 4, CELL + 52, parts))

    # The grid: 48 clean shapes with four or five moves, one per surface, picked at random.
    rng = np.random.default_rng(0)
    pool = np.flatnonzero(checks["clean"] & (lengths >= 4))
    pool = pool[np.unique(surface[pool], return_index=True)[1]]
    picks = np.sort(rng.choice(pool, size=48, replace=False))
    parts = [tile(traces[i], -1, 12 + (k % 8) * CELL, 12 + (k // 8) * CELL, highlight=False) for k, i in enumerate(picks)]
    with open("images/shapes.svg", "w") as f:
        f.write(sheet(24 + 8 * CELL, 24 + 6 * CELL, parts))

    # The flags: the first six shapes that are clean, that cut themselves, and that have a bent face.
    rows = [
        ("clean", checks["clean"] & (lengths >= 3)),
        ("cuts itself", checks["self_intersects"]),
        ("bent face", (checks["bend"] > 0.01) & ~checks["self_touches"]),
    ]
    parts = []
    for r, (name, mask) in enumerate(rows):
        y = 12 + r * CELL
        parts.append(f'<text x="24" y="{y + CELL / 2 + 5}" font-size="15" font-weight="600" fill="{INK}">{name}</text>')
        for k, i in enumerate(np.flatnonzero(mask)[:6]):
            parts.append(tile(traces[i], -1, 140 + k * CELL, y, highlight=False))
    with open("images/flags.svg", "w") as f:
        f.write(sheet(140 + 6 * CELL + 12, 24 + 3 * CELL, parts))
    print("wrote images/build.svg, images/shapes.svg, images/flags.svg")
