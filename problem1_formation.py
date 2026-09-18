import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

# ============================================================
# PROBLEM 1
# Formation Control for the name POOJITHA
# ============================================================

SEED = 20260918
N = 20
P_EDGE = 0.30
DT = 0.04
KF = 0.80
KG = 1.20
ERROR_TOL = 0.05
MAX_STEPS_PER_LETTER = 80
FPS = 15
VIDEO_FILE = "poojitha_formation.mp4"
GRAPH_FILE = "communication_graph.png"
FINAL_FILE = "final_formation.png"

rng = np.random.default_rng(SEED)


def sample_polyline(points, n):
    """Return n approximately equally spaced points along a polyline."""
    points = np.asarray(points, dtype=float)
    seg = np.diff(points, axis=0)
    lengths = np.linalg.norm(seg, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    total = cumulative[-1]

    if total == 0:
        return np.repeat(points[:1], n, axis=0)

    targets = np.linspace(0.0, total, n)
    result = np.zeros((n, 2))

    for k, d in enumerate(targets):
        idx = np.searchsorted(cumulative, d, side="right") - 1
        idx = min(idx, len(lengths) - 1)
        local = d - cumulative[idx]
        if lengths[idx] == 0:
            result[k] = points[idx]
        else:
            result[k] = points[idx] + (local / lengths[idx]) * seg[idx]

    return result


def letter_P():
    # Start at bottom-left, move upward, then around the upper loop.
    path = [
        [-1.8, -2.4],
        [-1.8,  2.4],
        [ 0.2,  2.4],
        [ 1.0,  2.0],
        [ 1.45, 1.2],
        [ 1.45, 0.4],
        [ 1.0, -0.2],
        [ 0.2, -0.5],
        [-1.8, -0.5],
    ]
    return sample_polyline(path, N)


def letter_O():
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.column_stack((
        1.8 * np.cos(theta),
        2.4 * np.sin(theta)
    ))


def letter_J():
    path = [
        [ 1.6,  2.4],
        [ 1.6, -1.4],
        [ 1.3, -2.0],
        [ 0.7, -2.4],
        [-0.2, -2.4],
        [-0.9, -2.0],
        [-1.3, -1.4],
        [-1.5, -0.8],
    ]
    # Add the top bar by beginning with it, then down the stem/curve.
    top = [[-1.5, 2.4], [1.6, 2.4]]
    stem_curve = [
        [1.6, 2.4], [1.6, -1.4], [1.3, -2.0], [0.7, -2.4],
        [-0.2, -2.4], [-0.9, -2.0], [-1.3, -1.4], [-1.5, -0.8]
    ]
    return sample_polyline(top + stem_curve[1:], N)


def letter_I():
    path = [
        [-1.5, 2.4], [1.5, 2.4],
        [0.0, 2.4], [0.0, -2.4],
        [-1.5, -2.4], [1.5, -2.4],
    ]
    return sample_polyline(path, N)


def letter_T():
    path = [
        [-1.8, 2.4], [1.8, 2.4],
        [0.0, 2.4], [0.0, -2.4],
    ]
    return sample_polyline(path, N)


def letter_H():
    path = [
        [-1.6, -2.4], [-1.6, 2.4],
        [-1.6, 0.0], [1.6, 0.0],
        [1.6, -2.4], [1.6, 2.4],
    ]
    return sample_polyline(path, N)


def letter_A():
    path = [
        [-1.8, -2.4], [0.0, 2.4],
        [1.8, -2.4],
        [1.0, -0.5], [-1.0, -0.5],
    ]
    return sample_polyline(path, N)


letters = {
    "P": letter_P(),
    "O": letter_O(),
    "J": letter_J(),
    "I": letter_I(),
    "T": letter_T(),
    "H": letter_H(),
    "A": letter_A(),
}

# The required sequence contains two O's.
sequence = ["P", "O", "O", "J", "I", "T", "H", "A"]


# ============================================================
# Generate a connected Erdos-Renyi communication graph
# ============================================================
attempts = 0
while True:
    graph_seed = int(rng.integers(0, 2**32 - 1))
    G = nx.erdos_renyi_graph(N, P_EDGE, seed=graph_seed)
    attempts += 1
    if nx.is_connected(G):
        break

A = nx.to_numpy_array(G, dtype=float)

# Save communication graph for the assignment if desired.
plt.figure(figsize=(7, 6))
pos_graph = nx.spring_layout(G, seed=SEED)
nx.draw(
    G,
    pos=pos_graph,
    with_labels=True,
    node_size=550,
    font_size=9,
)
plt.title("Problem 1: Connected Erdos-Renyi Communication Graph")
plt.tight_layout()
plt.savefig(GRAPH_FILE, dpi=200, bbox_inches="tight")
plt.close()


# ============================================================
# Random initial positions
# ============================================================
x = rng.uniform(-5.0, 5.0, size=(N, 2))
initial_positions = x.copy()

# ============================================================
# Simulate sequential formation control
# ============================================================
frames = []
letter_labels = []

for letter in sequence:
    r = letters[letter].copy()

    # Move the target slightly if necessary so all letters remain centered.
    r[:, 0] += 0.0

    for _ in range(MAX_STEPS_PER_LETTER):
        # Distributed formation-control law:
        # u_i = -kf sum_j a_ij[(x_i-x_j)-(r_i-r_j)] - kg(x_i-r_i)
        relative_error = np.zeros_like(x)

        for i in range(N):
            for j in range(N):
                if A[i, j] > 0:
                    relative_error[i] += A[i, j] * (
                        (x[i] - x[j]) - (r[i] - r[j])
                    )

        u = -KF * relative_error - KG * (x - r)

        # Numerical speed limiting for a smooth animation.
        max_speed = 5.0
        speed = np.linalg.norm(u, axis=1, keepdims=True)
        scale = np.minimum(1.0, max_speed / np.maximum(speed, 1e-12))
        u = u * scale

        x = x + DT * u

        error = np.mean(
            np.linalg.norm(x - r, axis=1)
        )

        frames.append(x.copy())
        letter_labels.append(letter)

        if error < ERROR_TOL:
            # Hold the completed formation briefly.
            for _ in range(8):
                frames.append(x.copy())
                letter_labels.append(letter)
            break

# ============================================================
# Plot final formation
# ============================================================
plt.figure(figsize=(8, 6))
plt.scatter(x[:, 0], x[:, 1], s=55)
plt.plot(x[:, 0], x[:, 1], linewidth=1.0, alpha=0.45)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Problem 1: Final Formation - POOJITHA (Letter A)")
plt.axis("equal")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FINAL_FILE, dpi=200, bbox_inches="tight")
plt.close()


# ============================================================
# Create MP4 animation
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(-5.2, 5.2)
ax.set_ylim(-3.3, 3.3)
ax.set_aspect("equal")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(True, alpha=0.25)

scatter = ax.scatter(
    initial_positions[:, 0],
    initial_positions[:, 1],
    s=55,
)

line, = ax.plot(
    initial_positions[:, 0],
    initial_positions[:, 1],
    linewidth=1.0,
    alpha=0.45,
)

ax.set_title("Problem 1: Formation Control - POOJITHA")

# A small text label tells which letter is currently being formed.
letter_text = ax.text(
    0.02,
    0.95,
    "",
    transform=ax.transAxes,
    fontsize=16,
)

writer = FFMpegWriter(
    fps=FPS,
    metadata={
        "title": "POOJITHA Formation Control",
        "artist": "Multi-Agent Systems",
    },
    bitrate=1800,
)

with writer.saving(fig, VIDEO_FILE, dpi=100):
    for positions, label in zip(frames, letter_labels):
        scatter.set_offsets(positions)
        line.set_data(positions[:, 0], positions[:, 1])
        letter_text.set_text(f"Current letter: {label}")
        writer.grab_frame()

plt.close(fig)

print("=" * 60)
print("PROBLEM 1 COMPLETE")
print("=" * 60)
print(f"N = {N}")
print(f"p = {P_EDGE}")
print(f"Connected graph found after {attempts} attempt(s)")
print(f"Sequence = {' -> '.join(sequence)}")
print(f"Frames generated = {len(frames)}")
print(f"Video saved as: {VIDEO_FILE}")
print(f"Graph saved as: {GRAPH_FILE}")
print(f"Final formation saved as: {FINAL_FILE}")
