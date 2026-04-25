import argparse
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from dataset.dataset import *
from model import *
from domains.gridworld import *


# =========================
# ACTION VECTORS (FROM TRAINING)
# =========================
action_vecs = np.asarray([
    [-1., 0.], [1., 0.], [0., 1.], [0., -1.],
    [-1., 1.], [-1., -1.], [1., 1.], [1., -1.]
])
action_vecs[4:] = 1 / np.sqrt(2) * action_vecs[4:]


# =========================
# VISUALIZATION
# =========================
def visualize(dom, states_xy, pred_traj):
    fig, ax = plt.subplots()

    plt.imshow(dom, cmap="Greys_r")

    ax.plot(states_xy[:, 0], states_xy[:, 1], c='b', label='Optimal Path')
    ax.plot(pred_traj[:, 0], pred_traj[:, 1], '-X', c='r', label='Predicted Path')

    ax.plot(states_xy[0, 0], states_xy[0, 1], '-o', label='Start')
    ax.plot(states_xy[-1, 0], states_xy[-1, 1], '-s', label='Goal')

    ax.legend(loc='upper right', fontsize='small')

    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close(fig)


# =========================
# DATASET EVAL
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
        shuffle=True
    )

    vin.eval()

    correct = 0
    total = 0
    total_loss = 0
    traj_diffs = []

    visualized = False

    with torch.no_grad():
        for X, S1, S2, labels in loader:

            X, S1, S2, labels = [d.to(device) for d in [X, S1, S2, labels]]

            outputs, _ = vin(X, S1, S2, config.k)

            # ===== LOSS =====
            loss = F.cross_entropy(outputs, labels, reduction='sum')
            total_loss += loss.item()

            # ===== SUCCESS =====
            pred = torch.argmax(outputs, dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

            # ===== TRAJ DIFF =====
            diff = (pred != labels).float().cpu().numpy()
            traj_diffs.extend(diff)

            # =========================
            # VISUALIZATION
            # =========================
            if config.plot and not visualized:

                idx = np.random.randint(0, X.size(0))

                # ===== map (0=free, 1=obstacle for GridWorld) =====
                im = X[idx, 0].cpu().numpy()
                im = 1 - im  # invert back: 1=free, 0=obstacle

                # ===== start position =====
                start = [int(S1[idx].item()), int(S2[idx].item())]

                # ===== find goal from value map =====
                value_map = X[idx, 1].cpu().numpy()
                goal = np.unravel_index(np.argmax(value_map), value_map.shape)

                # ===== build GridWorld =====
                G = GridWorld(im, goal[0], goal[1])

                # ===== reconstruct optimal trajectory using sample_trajectory =====
                from scipy.sparse import csr_matrix
                from scipy.sparse.csgraph import dijkstra

                G_inv, W_inv = G.get_graph_inv()
                g_sparse = csr_matrix(W_inv)
                goal_s = G.map_ind_to_state(goal[0], goal[1])
                _, pred = dijkstra(g_sparse, indices=goal_s, return_predecessors=True)

                try:
                    start_s = G.map_ind_to_state(start[0], start[1])
                except IndexError:
                    start_s = goal_s

                states_xy = [start]
                curr_s = start_s
                for _ in range(50):
                    next_s = pred[curr_s]
                    if next_s < 0:
                        break
                    nr, nc = G.get_coords(np.array([next_s]))
                    nr, nc = int(nr[0]), int(nc[0])
                    states_xy.append([nr, nc])
                    if nr == goal[0] and nc == goal[1]:
                        break
                    curr_s = next_s
                states_xy = np.array(states_xy)

                # ===== predicted path =====
                pred_traj = [start]
                curr = start.copy()

                for _ in range(30):
                    s1 = torch.tensor([[curr[0]]]).float().to(device)
                    s2 = torch.tensor([[curr[1]]]).float().to(device)

                    _, preds = vin(X[idx:idx+1], s1, s2, config.k)
                    a = torch.argmax(preds, dim=1).item()

                    try:
                        s = G.map_ind_to_state(curr[0], curr[1])
                    except IndexError:
                        break

                    ns = G.sample_next_state(s, a)

                    # FIX: get_coords needs array input, returns arrays
                    nr, nc = G.get_coords(np.array([ns]))
                    nr, nc = int(nr[0]), int(nc[0])

                    curr[0] = max(0, min(nr, config.imsize - 1))
                    curr[1] = max(0, min(nc, config.imsize - 1))

                    pred_traj.append(curr.copy())

                    if nr == goal[0] and nc == goal[1]:
                        break

                pred_traj = np.array(pred_traj)
                
                # FIX: transpose image and swap col/row for correct plot orientation
                visualize(im.T, states_xy[:, [1, 0]], pred_traj[:, [1, 0]])

                visualized = True

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

    print("\n===== FINAL RESULTS =====")
    print(f"Prediction Loss: {np.mean(all_losses):.4f}")
    print(f"Success Rate: {np.mean(all_success)*100:.2f}%")
    print(f"Trajectory Difference: {np.mean(all_traj):.4f}")


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
    parser.add_argument('--n_runs', type=int, default=5)
    parser.add_argument('--plot', action='store_true', default=False)

    config = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    vin = VIN(config)
    vin.load_state_dict(torch.load(config.weights))
    vin = vin.to(device)

    dataset_eval(vin, config, device)