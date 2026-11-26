"""Checks that score a mesh, and a check that scores how close two meshes are.

    python evals.py data/round0.npz

That command runs every check on the final mesh of every trace and prints a summary.
It needs NumPy and nothing else.

A mesh here is (verts, faces). verts is an array of positions, one row per vertex.
faces is a list of arrays, and each array holds the vertex indices of one face.

The checks on one mesh:

    is_closed          every edge is shared by exactly two faces, which face opposite ways
    volume             the space the mesh encloses, which is negative when it is inside out
    bend               how far the worst face is from flat, as a share of its size
    thinness           the worst ratio of shortest edge to longest edge in any face
    self_intersects    whether any two faces cut through each other, or touch when asked

The check on two meshes:

    shape_distance     the average gap between the two surfaces
"""

import sys
import time

import numpy as np

# The limits that turn a measurement into a flag. They are first guesses.
BEND_LIMIT = 0.01
THIN_LIMIT = 0.05


def triangles(verts, faces):
    """Cut every face into a fan of triangles and return their corner positions."""
    index = [(f[0], f[i], f[i + 1]) for f in faces for i in range(1, len(f) - 1)]
    return verts[np.array(index)]


def is_closed(faces):
    # Each edge is stored with its direction. In a closed mesh that faces one way throughout,
    # every directed edge appears once, and so does its reverse.
    edges = set()
    for f in faces:
        for a, b in zip(f, np.roll(f, -1)):
            if (a, b) in edges:
                return False
            edges.add((a, b))
    return all((b, a) in edges for a, b in edges)


def volume(verts, faces):
    t = triangles(verts, faces)
    return float(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6.0)


def bend(verts, faces):
    worst = 0.0
    for f in faces:
        p = verts[f]
        centered = p - p.mean(axis=0)
        # The smallest singular direction is the normal of the best-fit plane.
        normal = np.linalg.svd(centered)[2][-1]
        size = np.linalg.norm(centered, axis=1).max()
        if size > 0:  # A face squeezed to a point has no bend. The thinness check reports it.
            worst = max(worst, float(np.abs(centered @ normal).max() / size))
    return worst


def thinness(verts, faces):
    worst = 1.0
    for f in faces:
        p = verts[f]
        lengths = np.linalg.norm(p - np.roll(p, -1, axis=0), axis=1)
        # A face squeezed to a point scores 0, the same as a face with one edge of no length.
        worst = min(worst, float(lengths.min() / lengths.max()) if lengths.max() > 0 else 0.0)
    return worst


def self_intersects(verts, faces, touching=False, eps=1e-6):
    """Say whether an edge of one face passes through the inside of a face that is not its neighbor.

    With touching=True, contact also counts: a corner resting on another face, two faces lying
    flat against each other, or a face folded over itself in its own plane.
    """
    index, inner_u, inner_v = [], [], []
    for f in faces:
        for i in range(1, len(f) - 1):
            index.append((f[0], f[i], f[i + 1]))
            # A face is cut into triangles along diagonals. A diagonal is inside the face, so a
            # hit exactly on one is a hit inside the face. These two flags mark those sides.
            inner_v.append(i > 1)
            inner_u.append(i + 1 < len(f) - 1)
    index, inner_u, inner_v = np.array(index), np.array(inner_u), np.array(inner_v)
    tris = verts[index]
    a, b = np.triu_indices(len(index), k=1)
    # shortcut: two triangles that share a vertex are skipped, so a fold between neighbors is missed.
    # Upgrade when that shows up.
    apart = ~(index[a][:, :, None] == index[b][:, None, :]).any(axis=(1, 2))
    a, b = a[apart], b[apart]
    for first, second in ((a, b), (b, a)):
        tri = tris[second]
        e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
        if touching:
            u_low = v_low = -eps
            sum_high, t_low, t_high = 1 + eps, -eps, 1 + eps
        else:
            u_low = np.where(inner_u[second], -eps, eps)
            v_low = np.where(inner_v[second], -eps, eps)
            sum_high, t_low, t_high = 1 - eps, eps, 1 - eps
        for k in range(3):
            # Where does edge k of the first triangle meet the plane of the second triangle?
            start = tris[first][:, k]
            ray = tris[first][:, (k + 1) % 3] - start
            h = np.cross(ray, e2)
            det = np.einsum("ij,ij->i", e1, h)
            ok = np.abs(det) > 1e-12
            inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
            s = start - tri[:, 0]
            u = inv * np.einsum("ij,ij->i", s, h)
            q = np.cross(s, e1)
            v = inv * np.einsum("ij,ij->i", ray, q)
            t = inv * np.einsum("ij,ij->i", e2, q)
            if (ok & (u > u_low) & (v > v_low) & (u + v < sum_high) & (t > t_low) & (t < t_high)).any():
                return True
    return False


def sample_points(verts, faces, count, rng):
    """Pick points spread evenly over the surface."""
    t = triangles(verts, faces)
    areas = np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)
    pick = rng.choice(len(t), size=count, p=areas / areas.sum())
    r1, r2 = np.sqrt(rng.random(count)), rng.random(count)
    weights = np.stack([1 - r1, r1 * (1 - r2), r1 * r2], axis=1)
    return np.einsum("ij,ijk->ik", weights, t[pick])


def distance_to_surface(points, tris):
    """The distance from each point to the nearest spot on any of the triangles."""
    p = points[:, None]
    a, b, c = tris[None, :, 0], tris[None, :, 1], tris[None, :, 2]

    def to_segment(start, end):
        along = end - start
        length = np.maximum(np.einsum("...i,...i->...", along, along), 1e-30)
        t = np.clip(np.einsum("...i,...i->...", p - start, along) / length, 0, 1)
        return np.linalg.norm(p - start - t[..., None] * along, axis=-1)

    best = np.minimum(np.minimum(to_segment(a, b), to_segment(b, c)), to_segment(c, a))
    # When the point sits over the inside of the triangle, the flat distance to its plane is closer.
    normal = np.cross(b - a, c - a)
    size = np.linalg.norm(normal, axis=-1)
    unit = normal / np.maximum(size, 1e-30)[..., None]
    height = np.einsum("...i,...i->...", p - a, unit)
    foot = p - height[..., None] * unit
    inside = (
        (np.einsum("...i,...i->...", np.cross(b - a, foot - a), unit) >= 0)
        & (np.einsum("...i,...i->...", np.cross(c - b, foot - b), unit) >= 0)
        & (np.einsum("...i,...i->...", np.cross(a - c, foot - c), unit) >= 0)
        & (size > 1e-12)
    )
    return np.where(inside, np.abs(height), best).min(axis=1)


def shape_distance(mesh_a, mesh_b, count=2000, seed=0):
    """The average gap between two surfaces. It is exactly 0 when the surfaces are the same.

    This is Chamfer distance with one change. Points are sampled on one surface only, and each
    is measured to the other surface itself, not to sampled points on it. Sampling both sides
    leaves a gap of about 0.06 between a cube and itself, which would hide small differences.
    """
    rng = np.random.default_rng(seed)
    a_to_b = distance_to_surface(sample_points(*mesh_a, count, rng), triangles(*mesh_b))
    b_to_a = distance_to_surface(sample_points(*mesh_b, count, rng), triangles(*mesh_a))
    return float((a_to_b.mean() + b_to_a.mean()) / 2)


def shape_key(verts, faces):
    """A key that is the same for two meshes that differ only in vertex order or face order."""
    points = np.round(verts, 5) + 0.0  # Adding 0.0 turns -0.0 into 0.0.
    return tuple(sorted(tuple(sorted(map(tuple, points[f]))) for f in faces))


def step_mesh(d, corner_start, s):
    """Read step s out of the arrays of a shard."""
    f0, f1 = d["step_face_start"][s], d["step_face_start"][s + 1]
    verts = d["verts"][d["step_vert_start"][s]:d["step_vert_start"][s + 1]].astype(np.float64)
    faces = [d["faces"][corner_start[f]:corner_start[f + 1]] for f in range(f0, f1)]
    return verts, faces


def self_check():
    """Run each check on shapes whose answer is known."""
    cube_verts = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    cube_faces = [np.array(f) for f in ([0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3])]
    assert is_closed(cube_faces) and not is_closed(cube_faces[:5])
    assert abs(volume(cube_verts, cube_faces) - 8.0) < 1e-9
    assert volume(cube_verts, [f[::-1] for f in cube_faces]) < 0
    assert bend(cube_verts, cube_faces) < 1e-9 and thinness(cube_verts, cube_faces) == 1.0
    assert not self_intersects(cube_verts, cube_faces)
    # Two cubes that overlap, held as one mesh, must count as cutting through itself.
    both_verts = np.concatenate([cube_verts, cube_verts + 0.5])
    both_faces = cube_faces + [f + 8 for f in cube_faces]
    assert self_intersects(both_verts, both_faces)
    # Two cubes that rest against each other touch, but do not cut.
    resting = np.concatenate([cube_verts, cube_verts + [2, 0.5, 0.5]])
    assert self_intersects(resting, both_faces, touching=True) and not self_intersects(resting, both_faces)
    # Two cubes that sit apart must not.
    assert not self_intersects(np.concatenate([cube_verts, cube_verts + 5]), both_faces)
    # A lifted corner bends three faces.
    bent = cube_verts.copy()
    bent[7] += 0.5
    assert bend(bent, cube_faces) > BEND_LIMIT
    cube = (cube_verts, cube_faces)
    assert shape_distance(cube, cube) < 1e-9
    assert shape_distance(cube, (cube_verts + [3, 0, 0], cube_faces)) > 1.0
    # A cube against the same cube with its top face raised by 0.25 is a small, known gap.
    raised = cube_verts + np.where(cube_verts[:, 2:] > 0, [0, 0, 0.25], 0)
    assert 0.02 < shape_distance(cube, (raised, cube_faces)) < 0.25
    assert shape_key(cube_verts, cube_faces) == shape_key(cube_verts[::-1], [7 - f[::-1] for f in cube_faces[::-1]])


if __name__ == "__main__":
    self_check()
    d = dict(np.load(sys.argv[1]))
    corner_start = np.concatenate([[0], np.cumsum(d["face_sizes"], dtype=np.int64)])
    last_steps = d["trace_step_start"][1:] - 1

    clock = time.perf_counter()
    rows, keys = [], set()
    for s in last_steps:
        verts, faces = step_mesh(d, corner_start, s)
        rows.append((
            is_closed(faces), volume(verts, faces), bend(verts, faces), thinness(verts, faces),
            self_intersects(verts, faces), self_intersects(verts, faces, touching=True),
        ))
        keys.add(shape_key(verts, faces))
    seconds = time.perf_counter() - clock

    closed, vol, bent, thin, cut, touch = map(np.array, zip(*rows))
    clean = closed & (vol > 0) & (bent <= BEND_LIMIT) & (thin >= THIN_LIMIT) & ~touch
    n = len(rows)
    print(f"shapes {n}, self-check passed")
    print(f"closed {closed.sum()}, positive volume {(vol > 0).sum()}")
    print(f"cut through themselves {cut.sum()}, cut or touch themselves {touch.sum()}")
    print(f"bent faces (over {BEND_LIMIT}) {(bent > BEND_LIMIT).sum()}, thin faces (under {THIN_LIMIT}) {(thin < THIN_LIMIT).sum()}")
    print(f"faces with an edge of no length {(thin == 0).sum()}")
    print(f"clean on every check {clean.sum()}")
    print(f"different shapes when order is ignored {len(keys)}")
    print(f"volume: min {vol.min():.2f}, median {np.median(vol):.2f}, max {vol.max():.2f}")
    print(f"bend: median {np.median(bent):.4f}, max {bent.max():.3f}; thinness: median {np.median(thin):.3f}, min {thin.min():.4f}")
    print(f"TIME checks {seconds:.2f}s, {seconds / n * 1000:.2f} ms per shape")
    np.savez(sys.argv[1].replace(".npz", "-evals.npz"), closed=closed, volume=vol, bend=bent, thinness=thin, self_intersects=cut, self_touches=touch, clean=clean)
