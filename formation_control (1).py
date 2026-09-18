"""
Problem 1 -- Multi-Agent Formation Control
===========================================
N = 20 agents on a connected Erdos-Renyi graph move, letter by letter,
through positions spelling out NAME in R^2, using a distributed
(graph-restricted) formation control law. Produces an mp4/gif animation.

Dependencies: numpy, scipy, networkx, matplotlib, scikit-learn
    pip install numpy scipy networkx matplotlib scikit-learn
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.font_manager import FontProperties
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

# ----------------------------------------------------------------------
# 0. Configuration
# ----------------------------------------------------------------------
NAME = "POOJITHA"        # <-- CHANGE THIS to your (commonly used) name
N = 20                    # number of agents
P_EDGE = 0.3               # Erdos-Renyi edge probability
DT = 0.05                  # integration step
K1 = 2.0                   # target-attraction gain
K2 = 1.5                   # graph-consensus (formation) gain
STEPS_PER_LETTER = 120     # simulation steps spent tracking each letter
SEED = 42

rng = np.random.default_rng(SEED)


# ----------------------------------------------------------------------
# 1. Connected Erdos-Renyi communication graph
# ----------------------------------------------------------------------
def generate_connected_er_graph(n, p, seed=None):
    g = nx.erdos_renyi_graph(n, p, seed=seed)
    trial = 0
    while not nx.is_connected(g):
        trial += 1
        g = nx.erdos_renyi_graph(n, p, seed=(seed + trial if seed is not None else None))
    return g


G = generate_connected_er_graph(N, P_EDGE, seed=SEED)
neighbors = {i: list(G.neighbors(i)) for i in G.nodes()}


# ----------------------------------------------------------------------
# 2. Letter point-clouds: N points sampled inside each glyph
# ----------------------------------------------------------------------
def letter_points(letter, n_points, seed=0, dpi=200, fontsize=140):
    """Return n_points 2D points that trace the filled shape of `letter`.

    Rasterizes the glyph (correctly handling holes in letters like O/A/P,
    which a plain point-in-polygon test on the font outline gets wrong)
    and then places n_points evenly over the "ink" pixels via K-means, so
    the resulting cloud is visually recognizable as the letter even with
    only ~20 points.
    """
    fp = FontProperties(family="DejaVu Sans", weight="bold")
    fig = plt.figure(figsize=(2, 2), dpi=dpi)
    fig.patch.set_alpha(0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.5, letter, fontproperties=fp, fontsize=fontsize,
             ha="center", va="center")
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)

    gray = buf[..., :3].mean(axis=2)
    ink = gray < 128                       # dark = glyph ink
    ys, xs = np.nonzero(ink)
    if len(xs) < n_points:
        raise RuntimeError(f"Rendered glyph for '{letter}' too small/thin; "
                            f"increase dpi/fontsize.")
    coords = np.column_stack([xs, -ys]).astype(float)   # flip y (image y is down)

    km = KMeans(n_clusters=n_points, n_init=4, random_state=seed).fit(coords)
    pts = km.cluster_centers_

    # normalize: center at origin, scale so the letter spans ~8 units
    pts = pts - pts.mean(axis=0)
    scale = 8.0 / (pts.max() - pts.min() + 1e-9)
    return pts * scale


letters = list(NAME)
letter_targets = [letter_points(ch, N, seed=SEED + i) for i, ch in enumerate(letters)]


# ----------------------------------------------------------------------
# 3. Sequential Hungarian re-assignment (minimize travel between letters)
# ----------------------------------------------------------------------
def reassign(current_pos, next_targets):
    cost = np.linalg.norm(
        current_pos[:, None, :] - next_targets[None, :, :], axis=2
    )
    row_ind, col_ind = linear_sum_assignment(cost)
    return next_targets[col_ind]


x0 = rng.uniform(-3, 3, size=(N, 2))          # random initial positions x[0]
targets_sequence = [x0]
cur = x0.copy()
for tgt in letter_targets:
    tgt_assigned = reassign(cur, tgt)
    targets_sequence.append(tgt_assigned)
    cur = tgt_assigned


# ----------------------------------------------------------------------
# 4. Distributed formation control law (single-integrator agents)
#    u_i = -K1 (x_i - x_i*) - K2 * sum_{j in N_i} [(x_i-x_j)-(x_i*-x_j*)]
# ----------------------------------------------------------------------
def simulate_segment(x_start, x_star, steps):
    traj = np.zeros((steps, N, 2))
    x = x_start.copy()
    for k in range(steps):
        u = np.zeros_like(x)
        for i in range(N):
            u[i] += -K1 * (x[i] - x_star[i])
            for j in neighbors[i]:
                u[i] += -K2 * ((x[i] - x[j]) - (x_star[i] - x_star[j]))
        x = x + DT * u
        traj[k] = x
    return traj


full_traj = []
x_cur = targets_sequence[0]
for x_star in targets_sequence[1:]:
    seg = simulate_segment(x_cur, x_star, STEPS_PER_LETTER)
    full_traj.append(seg)
    x_cur = seg[-1]
full_traj = np.concatenate(full_traj, axis=0)   # shape (T, N, 2)


# ----------------------------------------------------------------------
# 5. Animate and save
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4))
ax.set_xlim(full_traj[..., 0].min() - 1, full_traj[..., 0].max() + 1)
ax.set_ylim(full_traj[..., 1].min() - 1, full_traj[..., 1].max() + 1)
ax.set_aspect("equal")
ax.set_title(f"N={N} agents forming: {NAME}")

scat = ax.scatter(full_traj[0, :, 0], full_traj[0, :, 1], s=40, c="tab:blue")
lines = [ax.plot([], [], lw=0.5, c="gray", alpha=0.5)[0] for _ in G.edges()]


def update(frame):
    pts = full_traj[frame]
    scat.set_offsets(pts)
    for line, (i, j) in zip(lines, G.edges()):
        line.set_data([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]])
    return [scat] + lines


ani = animation.FuncAnimation(
    fig, update, frames=len(full_traj), interval=20, blit=True
)

# GIF (no external ffmpeg dependency needed)
ani.save("formation_sequence.gif", writer=animation.PillowWriter(fps=30))
print("Saved animation to formation_sequence.gif")

# Optional: also save mp4 if ffmpeg is available
try:
    ani.save("formation_sequence.mp4", writer=animation.FFMpegWriter(fps=30))
    print("Saved animation to formation_sequence.mp4")
except Exception as e:
    print("mp4 export skipped (ffmpeg not found):", e)
