import os
import random
import torch
import matplotlib.pyplot as plt


seed = 42

box_min = -2.0
box_max = 2.0

neurons = 12
steps = 20000

eta_start = 0.25
eta_decay = 0.99985

neigh_k = 4
neigh_sigma = 1.0

fatigue_lambda = 0.6
fatigue_add = 1.0
fatigue_decay = 0.995

grid_res = 220

out_dir = "wyniki"


def sample_point():
    return box_min + (box_max - box_min) * torch.rand(2)

def normalize(v):
    return v / (torch.norm(v, dim=-1, keepdim=True) + 1e-12)

def learning_rate(t):
    return eta_start * (eta_decay ** t)

def choose_winner(scores):
    max_val = scores.max()
    idx = torch.where(scores == max_val)[0]
    return idx[random.randint(0, len(idx) - 1)].item()

def get_neighbors(w, winner):
    dist = torch.sum((w - w[winner]) ** 2, dim=1)
    return torch.argsort(dist)[:neigh_k]


def train(method, use_norm, use_fatigue):
    torch.manual_seed(seed)
    random.seed(seed)

    w = box_min + (box_max - box_min) * torch.rand(neurons, 2)
    if use_norm:
        w = normalize(w)

    fatigue = torch.zeros(neurons)

    for t in range(steps):
        x = sample_point()
        x_used = normalize(x) if use_norm else x

        if use_norm:
            w = normalize(w)

        scores = w @ x_used

        if use_fatigue:
            scores = scores - fatigue_lambda * fatigue

        winner = choose_winner(scores)

        if use_fatigue:
            fatigue *= fatigue_decay
            fatigue[winner] += fatigue_add

        eta = learning_rate(t)

        if method == "wta":
            idx = [winner]
        else:
            idx = get_neighbors(w, winner)

        for rank, j in enumerate(idx):
            h = torch.exp(torch.tensor(-(rank ** 2) / (2 * neigh_sigma ** 2)))
            w[j] = w[j] + eta * h * (x - w[j])

    return w


def safe_name(method, use_norm, use_fatigue):
    n = "norm-true" if use_norm else "norm-false"
    f = "fatigue-true" if use_fatigue else "fatigue-false"
    return f"{method}_{n}_{f}.png"


def draw_partition(w, use_norm, save_path):
    xs = torch.linspace(box_min, box_max, grid_res)
    ys = torch.linspace(box_min, box_max, grid_res)
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")

    pts = torch.stack([gx.flatten(), gy.flatten()], dim=1)
    pts_used = normalize(pts) if use_norm else pts

    scores = pts_used @ w.t()
    winners = torch.argmax(scores, dim=1).reshape(grid_res, grid_res)

    plt.figure(figsize=(5, 5))
    plt.imshow(
        winners.numpy(),
        origin="lower",
        extent=[box_min, box_max, box_min, box_max],
        aspect="equal"
    )
    plt.scatter(w[:, 0], w[:, 1], marker="x", s=90, c="r", linewidths=2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


if __name__ == "__main__":
    os.makedirs(out_dir, exist_ok=True)

    for method in ["wta", "wtm"]:
        for use_norm in [False, True]:
            for use_fatigue in [False, True]:
                w = train(method, use_norm, use_fatigue)
                filename = safe_name(method, use_norm, use_fatigue)
                path = os.path.join(out_dir, filename)
                draw_partition(w, use_norm, path)

    print(f"Zapisano obrazki do folderu: {out_dir}")
