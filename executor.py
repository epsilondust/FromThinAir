"""Run modeling moves on a mesh and record a trace of every step.

This file runs under Blender's Python, because bmesh only exists there:

    blender -b --factory-startup --python executor.py -- --n 20 --seed 0 --out traces.npz

The output is a shard file. See shard.py for its layout.

A move is a dict: {"op": "extrude" | "inset" | "slide", "face": index, "amount": number}.
A face is named by its index in the mesh that the move is applied to.

A trace is {"moves": [...], "steps": [...]}. steps[0] is the start cube.
steps[i] is the mesh after move i. Each step is {"verts": [[x, y, z]], "faces": [[i, j, ...]]}.
"""

import argparse
import hashlib
import os
import random
import sys
import time

import bmesh

# Blender does not put this folder on the import path by itself.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shard  # noqa: E402

# The trial move set. Each move has a fixed list of amounts, so a move is one of 9 choices per face.
AMOUNTS = {
    "extrude": [0.25, 0.5, 1.0],
    "inset": [0.2, 0.4],
    "slide": [-0.5, -0.25, 0.25, 0.5],
}


def apply_move(bm, move):
    bm.faces.ensure_lookup_table()
    face = bm.faces[move["face"]]
    amount = move["amount"]
    # The normal is copied first, because extrude replaces the face.
    normal = face.normal.copy()
    if move["op"] == "extrude":
        face = bmesh.ops.extrude_discrete_faces(bm, faces=[face])["faces"][0]
        bmesh.ops.translate(bm, verts=face.verts, vec=normal * amount)
    elif move["op"] == "inset":
        # The amount is a fraction of the shortest edge, so the inset always fits inside the face.
        shortest = min(edge.calc_length() for edge in face.edges)
        bmesh.ops.inset_individual(bm, faces=[face], thickness=amount * shortest)
    elif move["op"] == "slide":
        bmesh.ops.translate(bm, verts=face.verts, vec=normal * amount)
    else:
        raise ValueError(f"unknown move: {move['op']}")
    bm.normal_update()


def snapshot(bm):
    bm.verts.index_update()
    return {
        "verts": [list(v.co) for v in bm.verts],
        "faces": [[v.index for v in f.verts] for f in bm.faces],
    }


def run(moves):
    """Apply the moves to a cube of size 2 and return the mesh after every step."""
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=2.0)
    bm.normal_update()
    steps = [snapshot(bm)]
    for move in moves:
        apply_move(bm, move)
        steps.append(snapshot(bm))
    bm.free()
    return steps


def random_trace(rng, max_moves=5):
    """Pick moves one at a time, because the face count changes after every move."""
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=2.0)
    bm.normal_update()
    moves = []
    for _ in range(rng.randint(1, max_moves)):
        op = rng.choice(sorted(AMOUNTS))
        move = {"op": op, "face": rng.randrange(len(bm.faces)), "amount": rng.choice(AMOUNTS[op])}
        apply_move(bm, move)
        moves.append(move)
    bm.free()
    return {"moves": moves, "steps": run(moves)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="traces.npz")
    # Blender keeps its own arguments before "--". Ours come after it.
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])

    clock = time.perf_counter()
    rng = random.Random(args.seed)
    # A trace is kept only when its final mesh is new. Without this, short traces repeat a lot,
    # because a cube has only 54 possible first moves.
    traces, seen, repeats = [], set(), 0
    while len(traces) < args.n:
        trace = random_trace(rng)
        key = hashlib.sha1(repr(trace["steps"][-1]).encode()).digest()
        if key in seen:
            repeats += 1
            continue
        seen.add(key)
        traces.append(trace)
    seconds_generate = time.perf_counter() - clock

    clock = time.perf_counter()
    shard.save(args.out, traces)
    seconds_save = time.perf_counter() - clock

    # The check. Everything read back from the file must rebuild the same mesh when its moves
    # are run again, down to vertex and face order. This covers the executor and the file format.
    clock = time.perf_counter()
    stored = shard.load(args.out)
    seconds_load = time.perf_counter() - clock
    assert len(stored) == len(traces)

    clock = time.perf_counter()
    for trace in stored:
        assert run(trace["moves"]) == trace["steps"], "replay gave a different mesh"
    seconds_replay = time.perf_counter() - clock

    print(f"WROTE {len(traces)} traces to {args.out}, replay check passed, {repeats} repeats skipped")
    # The times go in the journal, so every run prints them.
    print(
        f"TIME generate {seconds_generate:.2f}s, save {seconds_save:.2f}s, "
        f"load {seconds_load:.2f}s, replay {seconds_replay:.2f}s"
    )
