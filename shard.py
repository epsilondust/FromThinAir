"""Read and write shard files. A shard is one .npz file that holds many traces as plain arrays.

    python shard.py traces.npz traces.json 20

That command exports the first 20 traces to JSON, which is what the browser player reads.
It needs NumPy and nothing else.

The arrays in a shard:

    verts             every vertex of every step, end to end (float32, 3 per vertex)
    faces             the vertex indices of every face, end to end
    face_sizes        how many corners each face has
    move_op           one code per move, an index into op_names
    move_face         the face each move acts on
    move_amount       the amount of each move
    op_names          the move name for each code
    step_vert_start   where each step begins in verts
    step_face_start   where each step begins in face_sizes
    trace_step_start  where each trace begins in the list of steps
    trace_move_start  where each trace begins in the list of moves
"""

import json
import sys

import numpy as np


def save(path, traces):
    op_names = sorted({m["op"] for t in traces for m in t["moves"]})
    steps = [s for t in traces for s in t["steps"]]
    moves = [m for t in traces for m in t["moves"]]

    def starts(counts):
        return np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)

    np.savez_compressed(
        path,
        # Blender holds positions as float32, so float32 stores them with no loss.
        verts=np.array([v for s in steps for v in s["verts"]], np.float32).reshape(-1, 3),
        faces=np.array([v for s in steps for f in s["faces"] for v in f], np.int32),
        face_sizes=np.array([len(f) for s in steps for f in s["faces"]], np.uint8),
        move_op=np.array([op_names.index(m["op"]) for m in moves], np.uint8),
        move_face=np.array([m["face"] for m in moves], np.int32),
        # Amounts stay float64. A rounded amount could rebuild a slightly different mesh.
        move_amount=np.array([m["amount"] for m in moves], np.float64),
        op_names=np.array(op_names),
        step_vert_start=starts([len(s["verts"]) for s in steps]),
        step_face_start=starts([len(s["faces"]) for s in steps]),
        trace_step_start=starts([len(t["steps"]) for t in traces]),
        trace_move_start=starts([len(t["moves"]) for t in traces]),
    )


def load(path):
    """Return the traces as a list of {"moves": [...], "steps": [...]}."""
    # shortcut: this unpacks the whole shard into Python lists, which is slow for big shards.
    # Training should slice the arrays directly when it needs speed.
    # Every array is read once here. Reading through the file object would unpack it on each access.
    d = dict(np.load(path))
    corner_start = np.concatenate([[0], np.cumsum(d["face_sizes"], dtype=np.int64)])
    op_names = [str(name) for name in d["op_names"]]
    traces = []
    for t in range(len(d["trace_step_start"]) - 1):
        steps = []
        for s in range(d["trace_step_start"][t], d["trace_step_start"][t + 1]):
            f0, f1 = d["step_face_start"][s], d["step_face_start"][s + 1]
            steps.append({
                "verts": d["verts"][d["step_vert_start"][s]:d["step_vert_start"][s + 1]].tolist(),
                "faces": [d["faces"][corner_start[f]:corner_start[f + 1]].tolist() for f in range(f0, f1)],
            })
        m0, m1 = d["trace_move_start"][t], d["trace_move_start"][t + 1]
        moves = [
            {"op": op_names[d["move_op"][m]], "face": int(d["move_face"][m]), "amount": float(d["move_amount"][m])}
            for m in range(m0, m1)
        ]
        traces.append({"moves": moves, "steps": steps})
    return traces


if __name__ == "__main__":
    traces = load(sys.argv[1])[:int(sys.argv[3])]
    for trace in traces:
        for step in trace["steps"]:
            # Six decimals keep the JSON small. The shard stays the exact copy.
            step["verts"] = [[round(c, 6) for c in v] for v in step["verts"]]
    with open(sys.argv[2], "w") as f:
        json.dump(traces, f)
    print(f"exported {len(traces)} traces to {sys.argv[2]}")
