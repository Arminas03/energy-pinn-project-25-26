import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import math
import h5py


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BaseMLP(nn.Module):
    def __init__(self, input_size):
        super(BaseMLP, self).__init__()

        self._build_layers(input_size)
        self.set_optimizer()
        self.set_loss_fn()

    def _build_layers(self, input_size):
        self.layers = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
        )

    def set_optimizer(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def set_loss_fn(self):
        self.loss_fn = F.mse_loss

    def forward(self, x):
        return self.layers(x)


def get_data():
    return {
        "train": {"X": [], "y": []},
        "val": {"X": [], "y": []},
        "test": {"X": [], "y": []},
    }


def train_one_batch(model: BaseMLP, data, hp, p, i):
    idx = p[i : i + hp["batch_size"]]

    X_batch = data["train"]["X"][idx].to(device)
    y_batch = data["train"]["y"][idx].to(device)

    model.optimizer.zero_grad()
    pred = model(X_batch)
    loss = model.loss_fn(pred, y_batch)
    loss.backward()
    model.optimizer.step()

    return loss.item()


@torch.no_grad()
def eval_on_val_set(model: BaseMLP, data, early_stopping, hp):
    model.eval()

    X_val = data["val"]["X"].to(device)
    y_val = data["val"]["y"].to(device)

    loss = model.loss_fn(model(X_val), y_val).item()

    if loss < early_stopping["best_eval_loss"]:
        early_stopping["best_eval_loss"] = loss
        early_stopping["patience_left"] = hp["patience"]
    else:
        early_stopping["patience_left"] -= 1

    return loss


def train(model: BaseMLP, data, hp):
    n_train = data["train"]["X"].size(0)

    early_stopping = {
        "patience_left": hp["patience"],
        "best_eval_loss": float("inf"),
    }

    tqdm_bar = tqdm(range(hp["epochs"]))

    for _ in tqdm_bar:
        model.train()
        p = torch.randperm(n_train)

        train_loss = 0

        for i in range(0, n_train, hp["batch_size"]):
            train_loss += train_one_batch(model, data, hp, p, i)

        train_loss /= math.ceil(n_train / hp["batch_size"])
        val_loss = eval_on_val_set(model, data, early_stopping, hp)

        if early_stopping["patience_left"] == 0:
            break

        tqdm_bar.set_postfix(
            train_loss=f"{train_loss:.4f}",
            val_loss=f"{val_loss:.4f}",
            best_val_loss=f"{early_stopping["best_eval_loss"]:.4f}",
        )

    return train_loss, val_loss


@torch.no_grad()
def test(model: BaseMLP, data, hp):
    n_test = data["test"]["X"].size(0)
    idx = torch.arange(n_test)
    loss = 0

    for i in tqdm(range(0, len(idx), hp["batch_size"])):
        idx_batch = idx[i : i + hp["batch_size"]]

        X_batch = data["test"]["X"][idx_batch].to(device)
        y_batch = data["test"]["y"][idx_batch].to(device)

        loss += model.loss_fn(model(X_batch), y_batch).item()

    return loss / math.ceil(n_test / hp["batch_size"])


def main():
    data = get_data()

    return
    model = BaseMLP(data["train"]["X"].size(1)).to(device)
    hp = {
        "epochs": 100,
        "batch_size": 64,
        "patience": 10,
    }

    print("Training...")
    print("=" * 80)
    train_loss, val_loss = train(model, data, hp)
    print("=" * 80)
    print(f"Train finished. Train loss: {train_loss:.4f}, val loss: {val_loss:.4f}")

    print("Testing...")
    print("=" * 80)
    test_loss = test(model, data, hp)
    print("=" * 80)
    print(f"Test finished. Test loss: {test_loss:.4f}")


if __name__ == "__main__":
    main()
