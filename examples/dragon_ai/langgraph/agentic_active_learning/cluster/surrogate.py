"""
The surrogate: the small trained model the campaign exists to produce.

It learns to predict `quantity` from `dials` -- given the speed and size of the
object, how much extra detail did the simulation have to add? The training data
is whatever real runs have finished so far.

It is a small ensemble of neural networks rather than one. Two reasons: they fit
smooth curves, so the held-out error falls cleanly as more data arrives, and
their disagreement is a usable signal on its own. Where the members disagree, the
campaign has not measured enough, so that is where the next simulations go. The
disagreement ships with the model, so whoever uses it later can also see where it
is guessing.

Same two names as everywhere else: `dials` is the settings of a run, `quantity`
is the single number learned from it. Nothing here knows about Dragon.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


ENSEMBLE_SIZE = 3        # networks per fit; their disagreement drives `propose`
HIDDEN = 64
EPOCHS = 2000
LEARNING_RATE = 0.01
VALIDATE_FRACTION = 0.25
SEED = 0

MODEL_FILE = "surrogate.pt"
CARD_FILE = "model_card.txt"


class Net(nn.Module):
    """Two dials in, one quantity out."""

    def __init__(self, hidden: int = HIDDEN) -> None:
        super().__init__()
        # tanh rather than relu: the quantity being learned is smooth.
        self.stack = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stack(x)


def _tensors(training_pairs: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    dials = torch.tensor([pair["dials"] for pair in training_pairs],
                         dtype=torch.float32)
    quantities = torch.tensor([[pair["quantity"]] for pair in training_pairs],
                              dtype=torch.float32)
    return dials, quantities


def _split(n: int, fraction: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Indices for train and validation, shuffled so later clustered waves do
    not end up as the whole validation set."""
    generator = torch.Generator().manual_seed(seed)
    shuffled = torch.randperm(n, generator=generator)
    n_validate = max(1, int(n * fraction))
    return shuffled[n_validate:], shuffled[:n_validate]


def _prepare(training_pairs: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    """Normalisation constants and the split. Deterministic given data + seed,
    so every member computes the same thing with nothing passed between them."""
    dials, quantities = _tensors(training_pairs)

    dials_mean, dials_std = dials.mean(0), dials.std(0).clamp_min(1e-6)
    quantity_mean = quantities.mean()
    quantity_std = quantities.std().clamp_min(1e-6)
    x_all = (dials - dials_mean) / dials_std
    y_all = (quantities - quantity_mean) / quantity_std

    train_index, validate_index = _split(len(training_pairs),
                                         VALIDATE_FRACTION, seed)
    return {
        "x_train": x_all[train_index], "y_train": y_all[train_index],
        "x_validate": x_all[validate_index],
        "y_validate": y_all[validate_index],
        "dials_mean": dials_mean, "dials_std": dials_std,
        "quantity_mean": quantity_mean, "quantity_std": quantity_std,
        "n_train": len(train_index), "n_validate": len(validate_index),
    }


def train_member(training_pairs: list[dict[str, Any]], seed: int,
                 member: int) -> dict[str, Any]:
    """Fit ONE ensemble member. This is what runs as a Dragon Batch task.

    Members differ only in where they start. Nothing is shared while they run.
    """
    # One thread per member (each is its own process); the 2x64 matmuls are not
    # worth threading.
    torch.set_num_threads(1)
    prepared = _prepare(training_pairs, seed)
    torch.manual_seed(seed + member)
    net = Net()
    optimiser = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    net.train()
    for _ in range(EPOCHS):
        optimiser.zero_grad()
        loss = loss_fn(net(prepared["x_train"]), prepared["y_train"])
        loss.backward()
        optimiser.step()

    net.eval()
    with torch.no_grad():
        predicted = net(prepared["x_validate"])
    # Undo normalised log(1+blocks) back to real block counts, then take the
    # MEDIAN relative error -- robust to the odd run that barely refines.
    pred_blocks = torch.expm1(predicted * prepared["quantity_std"]
                              + prepared["quantity_mean"])
    true_blocks = torch.expm1(prepared["y_validate"] * prepared["quantity_std"]
                              + prepared["quantity_mean"])
    rel = torch.abs(pred_blocks - true_blocks) / true_blocks.clamp_min(1.0)
    return {
        "member": member,
        "weights": {k: v.clone() for k, v in net.state_dict().items()},
        "validate_error": float(rel.median()),
    }


def assemble(training_pairs: list[dict[str, Any]], seed: int,
             members: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine finished members into one saveable, usable model."""
    prepared = _prepare(training_pairs, seed)
    ordered = sorted(members, key=lambda m: m["member"])
    errors = [m["validate_error"] for m in ordered]
    return {
        "member_weights": [m["weights"] for m in ordered],
        "dials_mean": prepared["dials_mean"],
        "dials_std": prepared["dials_std"],
        "quantity_mean": prepared["quantity_mean"],
        "quantity_std": prepared["quantity_std"],
        "validate_error": sum(errors) / len(errors),
        "member_validate_errors": errors,
        "n_training_points": len(training_pairs),
        "n_train": prepared["n_train"], "n_validate": prepared["n_validate"],
    }


def _restore(model: dict[str, Any]) -> list[Net]:
    members = []
    for weights in model["member_weights"]:
        net = Net()
        net.load_state_dict(weights)
        net.eval()
        members.append(net)
    return members


def predict(model: dict[str, Any],
            many_dials: list[list[float]]) -> tuple[list[float], list[float]]:
    """Answer for each point, and how much the ensemble disagreed about it.

    Nets predict log(1+blocks); the mean is turned back into a block count and
    the spread across members (log space, so relative) drives `propose`.
    """
    members = _restore(model)
    x = torch.tensor(many_dials, dtype=torch.float32)
    x = (x - model["dials_mean"]) / model["dials_std"]

    with torch.no_grad():
        # (members, points), so mean and std below reduce over the members.
        member_answers = torch.stack([net(x).squeeze(-1) for net in members])
    predicted_log = (member_answers.mean(0) * model["quantity_std"]
                     + model["quantity_mean"])
    predicted = torch.expm1(predicted_log)
    # No quantity_mean here: the same shift on every member cancels in a spread.
    disagreement = member_answers.std(0) * model["quantity_std"]
    return predicted.tolist(), disagreement.tolist()


def save_surrogate(directory: str | Path, model: dict[str, Any],
                   card: str) -> tuple[Path, Path]:
    """Write the model and its card side by side."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / MODEL_FILE
    card_path = directory / CARD_FILE
    torch.save(model, model_path)
    card_path.write_text(card)
    return model_path, card_path


def load_surrogate(directory: str | Path) -> dict[str, Any]:
    """Load a saved surrogate. Needs nothing from the campaign that made it."""
    path = Path(directory) / MODEL_FILE
    return torch.load(path, map_location="cpu", weights_only=False)
