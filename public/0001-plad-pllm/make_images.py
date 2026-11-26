"""Make the images for post 0001 from the round zero dataset.

    python public/0001-plad-pllm/make_images.py

It writes strip, steps, loop, lest and edits into the images folder next to this file, each as SVG.
When rsvg-convert is installed it also writes a PNG of each, for platforms that reject SVG.
Every shape in the images is a real trace from data/round0.npz.
"""

import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

import numpy as np

import render
import shard
from readme_images import INK, QUIET, label, moves_face_camera, sheet

OUT = os.path.join(HERE, "images")
CELL = 240
ARROW = (
    '<defs><marker id="tip" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
    f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{QUIET}"/></marker></defs>'
)


def place(mesh, x, y, scale, cam, highlight=-1, size=CELL):
    """Draw one mesh as a square placed at (x, y)."""
    picture = render.draw(mesh, highlight, scale, cam)
    return picture.replace("<svg xmlns", f'<svg x="{x}" y="{y}" width="{size}" height="{size}" xmlns', 1)


def scale_for(meshes):
    """One scale that fits every mesh given, so sizes can be compared across tiles."""
    reach = max(np.linalg.norm(mesh["verts"], axis=1).max() for mesh in meshes)
    return (render.SIZE / 2 - 12) / reach


def faces_camera(trace, cam):
    """Say whether every move lands on a face this camera can see well."""
    return all(
        render.dot(render.face_normal([step["verts"][v] for v in step["faces"][m["face"]]]), cam[0]) > 0.3
        for step, m in zip(trace["steps"], trace["moves"])
    )


def words(move):
    return f'{move["op"]} face {move["face"]} by {move["amount"]:g}'


def left(x, y, text, color=QUIET, size=16, weight=400):
    return f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{color}">{text}</text>'


def arrow(path):
    return f'<path d="{path}" fill="none" stroke="{QUIET}" stroke-width="2" marker-end="url(#tip)"/>'


def write(name, width, height, parts):
    path = os.path.join(OUT, name + ".svg")
    with open(path, "w") as f:
        f.write(sheet(width, height, parts))
    if shutil.which("rsvg-convert"):
        subprocess.run(["rsvg-convert", "-w", "2000", path, "-o", path[:-4] + ".png"], check=True)


if __name__ == "__main__":
    traces = shard.load(os.path.join(ROOT, "data/round0.npz"))
    checks = np.load(os.path.join(ROOT, "data/round0-evals.npz"))
    surface = np.load(os.path.join(ROOT, "data/round0-surface.npy"))
    clean = checks["clean"]
    lengths = np.array([len(t["moves"]) for t in traces])
    os.makedirs(OUT, exist_ok=True)

    # The strip: one clean shape for each number of moves, 1 to 5, all from one camera at one scale.
    # Each pick is the first trace of that length that is mostly extrudes and shows its moves.
    cam = render.camera(35, 25)
    picks = [
        next(
            t for i, t in enumerate(traces)
            if clean[i] and lengths[i] == n and faces_camera(t, cam)
            and sum(m["op"] == "extrude" for m in t["moves"]) >= (n + 1) // 2
        )
        for n in range(1, 6)
    ]
    scale = scale_for([t["steps"][-1] for t in picks])
    parts = []
    for k, t in enumerate(picks):
        x = 12 + k * (CELL + 8)
        parts.append(place(t["steps"][-1], x, 12, scale, cam))
        parts.append(label(x + CELL / 2, CELL + 40, f"{k + 1} move" + ("s" if k else ""), INK, 18, 600))
    write("strip", 12 + 5 * (CELL + 8) + 4, CELL + 62, parts)

    # The steps: the first clean five-move trace with at least three extrudes whose moves all show.
    # It is drawn as two rows of three, with the next move named under each step.
    build = next(
        t for i, t in enumerate(traces)
        if clean[i] and lengths[i] == 5
        and sum(m["op"] == "extrude" for m in t["moves"]) >= 3 and moves_face_camera(t)
    )
    build_scale, build_cam, highlights = render.fit(build)
    parts = []
    for step in range(6):
        x, y = 12 + (step % 3) * (CELL + 8), 12 + (step // 3) * (CELL + 50)
        parts.append(place(build["steps"][step], x, y, build_scale, build_cam, highlights[step]))
        done = step == 5
        text = "the finished shape" if done else "next: " + words(build["moves"][step])
        parts.append(label(x + CELL / 2, y + CELL + 28, text, INK if done else QUIET, 17, 600 if done else 400))
    write("steps", 12 + 3 * (CELL + 8) + 4, 12 + 2 * (CELL + 50), parts)

    # The target and the guess: two real traces that share every move but the amount of the last one.
    # One plays the target and the other plays the guess. The pairing is staged, not a model's output.
    # The longest such pair is used, because long traces rarely share a start.
    first = {}
    target = guess = None
    for i in np.argsort(-lengths, kind="stable"):
        t = traces[i]
        last = t["moves"][-1]
        if not clean[i] or last["op"] != "extrude" or not moves_face_camera(t):
            continue
        key = repr(t["moves"][:-1]) + str(last["face"])
        if key in first and surface[first[key]] != surface[i]:
            target, guess = traces[first[key]], t
            break
        first.setdefault(key, i)
    scale, cam, _ = render.fit(target)
    scale = min(scale, render.fit(guess)[0])

    # The loop: six steps in a row, with an arrow from the last back to the second.
    # Only the first step is built, and its box has a solid outline. The others are dashed.
    stages = [
        ("1. Seed", ["Random moves on a cube", "make the first examples."], build["steps"][-1], build_scale, build_cam),
        ("2. Imagine", ["A frozen 3D generator", "supplies target shapes."], target["steps"][-1], scale, cam),
        ("3. Attempt", ["The model tries to build", "each target with moves."], guess["steps"][-1], scale, cam),
        ("4. Check", ["Measure how far the built", "shape is from the target."], None, 0, None),
        ("5. Keep", ["Store the moves with the", "shape they really built."], guess["steps"][-1], scale, cam),
        ("6. Retrain", ["Train the model on", "everything kept."], None, 0, None),
    ]
    W, H, GAP = 220, 250, 36
    parts = [ARROW]
    for k, (title, lines, mesh, s, c) in enumerate(stages):
        x = 12 + k * (W + GAP)
        dash = "" if k == 0 else ' stroke-dasharray="7 6"'
        parts.append(f'<rect x="{x}" y="12" width="{W}" height="{H}" rx="10" fill="none" stroke="{QUIET}" stroke-width="2"{dash}/>')
        if mesh:
            parts.append(place(mesh, x + (W - 130) / 2, 20, s, c, size=130))
        elif k == 3:
            # The check compares the target with the built shape, so it shows both.
            parts.append(place(target["steps"][-1], x + 14, 40, scale, cam, size=92))
            parts.append(place(guess["steps"][-1], x + W - 106, 40, scale, cam, size=92))
        parts.append(label(x + W / 2, 180, title, INK, 19, 600))
        for n, line in enumerate(lines):
            parts.append(label(x + W / 2, 208 + n * 22, line, QUIET, 15))
        if k < 5:
            parts.append(arrow(f"M{x + W + 4},{12 + H / 2} L{x + W + GAP - 4},{12 + H / 2}"))
    back, there, low = 12 + 5 * (W + GAP) + W / 2, 12 + (W + GAP) + W / 2, 12 + H + 34
    parts.append(arrow(f"M{back},{12 + H + 4} L{back},{low} L{there},{low} L{there},{12 + H + 6}"))
    parts.append(label((back + there) / 2, low + 26, "the next round, with harder targets", QUIET, 16))
    width = 24 + 6 * W + 5 * GAP
    parts.append(label(width / 2, low + 58, "Solid outline: built. Dashed outline: planned. The shapes are real traces, placed by hand.", QUIET, 15))
    write("loop", width, low + 76, parts)

    # The LEST illustration: the same guess makes two different training pairs.
    # Self-training pairs the guessed moves with the target. LEST pairs them with the shape they built.
    parts = [
        place(target["steps"][-1], 12, 12, scale, cam),
        label(12 + CELL / 2, CELL + 40, "the target", INK, 18, 600),
        place(guess["steps"][-1], 20 + CELL, 12, scale, cam),
        label(20 + CELL + CELL / 2, CELL + 40, "what the guessed moves built", INK, 18, 600),
        left(24, CELL + 70, "guessed moves: " + ", ".join(words(m).replace("face ", "") for m in guess["moves"]), QUIET, 14),
    ]
    x = 36 + 2 * CELL
    pairs = [
        ("self-training keeps", target, "paired with the target", "the label can be wrong"),
        ("LEST keeps", guess, "paired with the shape they built", "the label is always right"),
    ]
    for k, (title, trace, how, result) in enumerate(pairs):
        y = 12 + k * 160
        parts.append(left(x, y + 26, title, INK, 18, 600))
        parts.append(place(trace["steps"][-1], x, y + 36, scale, cam, size=110))
        parts.append(left(x + 124, y + 72, "the guessed moves"))
        parts.append(left(x + 124, y + 96, how))
        parts.append(left(x + 124, y + 126, result, INK, 16, 600))
    width = x + 400
    parts.append(label(width / 2, CELL + 106, "Illustration, not a result. Both shapes are real traces. The guess is staged.", QUIET, 15))
    write("lest", width, CELL + 124, parts)

    # The edits illustration: removing steps from the end of a move list gives more correct pairs.
    # Each shorter list builds exactly the mesh stored for that step of the same trace.
    parts = []
    for k, cut in enumerate((0, 1, 2)):
        x = 12 + k * (CELL + 28)
        kept = build["moves"][: 5 - cut]
        title = ["one list of moves that works", "remove the last move", "remove the last two moves"][k]
        parts.append(place(build["steps"][5 - cut], x, 12, build_scale, build_cam))
        parts.append(label(x + CELL / 2, CELL + 36, title, INK, 17, 600))
        for n, move in enumerate(kept):
            parts.append(left(x + 20, CELL + 68 + n * 24, f"{n + 1}. {words(move)}", QUIET, 15))
        if k:
            parts.append(ARROW if k == 1 else "")
            parts.append(arrow(f"M{x - 24},{12 + CELL / 2} L{x - 4},{12 + CELL / 2}"))
    width = 12 + 3 * CELL + 2 * 28 + 12
    parts.append(label(width / 2, CELL + 206, "Illustration with From Thin Air moves. PLLM edits CAD programs.", QUIET, 15))
    parts.append(label(width / 2, CELL + 228, "Each shorter list still builds exactly the shape shown above it.", QUIET, 15))
    write("edits", width, CELL + 246, parts)
    print("wrote strip, steps, loop, lest and edits to", OUT)
