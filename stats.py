"""Print what a shard holds: how long its traces are, which moves they use, and how many repeat.

    python stats.py data/round0.npz
"""

import collections
import hashlib
import sys

import numpy as np

d = dict(np.load(sys.argv[1]))
traces = len(d["trace_move_start"]) - 1
moves_per_trace = np.diff(d["trace_move_start"])
faces_per_step = np.diff(d["step_face_start"])
verts_per_step = np.diff(d["step_vert_start"])
last_step = d["trace_step_start"][1:] - 1
corner_start = np.concatenate([[0], np.cumsum(d["face_sizes"], dtype=np.int64)])

print(f"traces {traces}, moves {len(d['move_op'])}, steps {len(faces_per_step)}")
print("moves per trace:", dict(sorted(collections.Counter(moves_per_trace.tolist()).items())))
print("move types:", {str(n): int((d["move_op"] == i).sum()) for i, n in enumerate(d["op_names"])})
print("corners per face:", dict(sorted(collections.Counter(d["face_sizes"].tolist()).items())))
print(f"final faces: min {faces_per_step[last_step].min()}, mean {faces_per_step[last_step].mean():.1f}, max {faces_per_step[last_step].max()}")
print(f"final verts: min {verts_per_step[last_step].min()}, mean {verts_per_step[last_step].mean():.1f}, max {verts_per_step[last_step].max()}")

# Two traces repeat when they have the same move list. Two shapes repeat when the final mesh is
# the same, array for array. The same shape built in a different vertex order is not caught here.
move_lists, final_meshes = set(), set()
for t in range(traces):
    m0, m1 = d["trace_move_start"][t], d["trace_move_start"][t + 1]
    move_lists.add((d["move_op"][m0:m1].tobytes(), d["move_face"][m0:m1].tobytes(), d["move_amount"][m0:m1].tobytes()))
    s = last_step[t]
    f0, f1 = d["step_face_start"][s], d["step_face_start"][s + 1]
    final_meshes.add(hashlib.sha1(
        d["verts"][d["step_vert_start"][s]:d["step_vert_start"][s + 1]].tobytes()
        + d["face_sizes"][f0:f1].tobytes()
        + d["faces"][corner_start[f0]:corner_start[f1]].tobytes()
    ).digest())
print(f"unique move lists {len(move_lists)}, unique final meshes {len(final_meshes)}")
