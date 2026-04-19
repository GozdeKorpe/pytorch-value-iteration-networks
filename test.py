import argparse
import torch
import numpy as np
import torch.nn.functional as F

from dataset.dataset import *
from model import *


# =========================
# DATASET EVAL (MATCHED FORMAT)
# =========================
def dataset_eval_once(vin, config, device):

    testset = GridworldData(
        config.datafile,
        imsize=config.imsize,
        train=False
    )

    loader = torch.utils.data.DataLoader(
        testset,
        batch_size=config.batch_size,
        shuffle=False
    )

    vin.eval()

    correct = 0
    total = 0
    total_loss = 0

    # for trajectory diff approximation
    traj_diffs = []

    with torch.no_grad():
        for X, S1, S2, labels in loader:

            X, S1, S2, labels = [d.to(device) for d in [X, S1, S2, labels]]

            outputs, _ = vin(X, S1, S2, config.k)

            # ===== Prediction Loss =====
            loss = F.cross_entropy(outputs, labels, reduction='sum')
            total_loss += loss.item()

            # ===== Success Rate (same as accuracy) =====
            pred = torch.argmax(outputs, dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

            # ===== Trajectory Difference (approximation) =====
            # 0 if correct action, 1 if wrong
            diff = (pred != labels).float().cpu().numpy()
            traj_diffs.extend(diff)

    avg_loss = total_loss / total
    success_rate = correct / total
    traj_diff = np.mean(traj_diffs)

    return avg_loss, success_rate, traj_diff


# =========================
# MULTI RUN
# =========================
def dataset_eval(vin, config, device):

    all_losses = []
    all_success = []
    all_traj = []

    for run in range(config.n_runs):
        print(f"Run {run+1}/{config.n_runs}")

        loss, success, traj_diff = dataset_eval_once(vin, config, device)

        all_losses.append(loss)
        all_success.append(success)
        all_traj.append(traj_diff)

    print("\n===== FINAL AVERAGED RESULTS =====")

    print(f"Prediction Loss: {np.mean(all_losses):.4f} ")
    print(f"Success Rate: {np.mean(all_success)*100:.2f}% ")
    print(f"Trajectory Difference: {np.mean(all_traj):.4f} ")


# =========================
# MAIN
# =========================
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--datafile', type=str, required=True)

    parser.add_argument('--imsize', type=int, default=8)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--l_i', type=int, default=2)
    parser.add_argument('--l_h', type=int, default=150)
    parser.add_argument('--l_q', type=int, default=10)

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_runs', type=int, default=1)

    config = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    vin = VIN(config)
    vin.load_state_dict(torch.load(config.weights))
    vin = vin.to(device)

    dataset_eval(vin, config, device)