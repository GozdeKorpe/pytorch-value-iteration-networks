import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from dataset.dataset import *
from model import *
from domains.gridworld import *
from generators.obstacle_gen import *


# =========================
# VISUALIZATION (FIGURE 3)
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
# DATASET LOSS (SUPERVISED)
# =========================
def compute_prediction_loss(vin, config, device,
                           n_domains=100,
                           max_obs=2,
                           n_traj=1):

    vin.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():

        for _ in range(n_domains):

            goal = [np.random.randint(config.imsize),
                    np.random.randint(config.imsize)]

            obs = obstacles([config.imsize, config.imsize], goal)
            n_obs = obs.add_n_rand_obs(max_obs)
            if n_obs == 0 or not obs.add_border():
                continue

            im = obs.get_final()

            G = GridWorld(im, goal[0], goal[1])
            value_prior = G.get_reward_prior()

            states_xy, states_one_hot = sample_trajectory(G, n_traj)

            for i in range(n_traj):

                traj = states_xy[i]

                if len(traj) <= 1:
                    continue

                for t in range(len(traj) - 1):

                    curr = traj[t]

                    im_data = 1 - G.image.astype(np.int32)
                    im_data = im_data.reshape(1, 1, config.imsize, config.imsize)

                    value_data = value_prior.astype(np.int32)
                    value_data = value_data.reshape(1, 1, config.imsize, config.imsize)

                    X_in = torch.from_numpy(np.append(im_data, value_data, axis=1)).float().to(device)

                    S1_in = torch.tensor([[curr[0]]]).float().to(device)
                    S2_in = torch.tensor([[curr[1]]]).float().to(device)

                    _, predictions = vin(X_in, S1_in, S2_in, config.k)

                    # ground truth action
                    gt = states_one_hot[i][t]
                    gt_action = int(np.argmax(gt))

                    if gt_action >= predictions.shape[1]:
                        continue

                    gt_tensor = torch.tensor([gt_action]).to(device)

                    loss = F.cross_entropy(predictions, gt_tensor)

                    total_loss += loss.item()
                    total_samples += 1

    return total_loss / total_samples if total_samples > 0 else 0


# =========================
# ROLLOUT EVALUATION
# =========================
def rollout_eval(vin, config, device,
                 n_domains=100,
                 max_obs=2,
                 n_traj=1):

    vin.eval()
    correct, total = 0.0, 0.0
    traj_diffs = []

    with torch.no_grad():

        for _ in range(n_domains):

            goal = [np.random.randint(config.imsize),
                    np.random.randint(config.imsize)]

            obs = obstacles([config.imsize, config.imsize], goal)
            n_obs = obs.add_n_rand_obs(max_obs)
            if n_obs == 0 or not obs.add_border():
                continue

            im = obs.get_final()

            G = GridWorld(im, goal[0], goal[1])
            value_prior = G.get_reward_prior()

            states_xy, _ = sample_trajectory(G, n_traj)

            for i in range(n_traj):

                if len(states_xy[i]) <= 1:
                    continue

                optimal_len = len(states_xy[i])

                L = optimal_len * 3
                pred_traj = np.zeros((L, 2))
                pred_traj[0] = states_xy[i][0]

                pred_len = None

                for j in range(1, L):

                    curr = pred_traj[j - 1].astype(int)

                    im_data = 1 - G.image.astype(np.int32)
                    im_data = im_data.reshape(1, 1, config.imsize, config.imsize)

                    value_data = value_prior.astype(np.int32)
                    value_data = value_data.reshape(1, 1, config.imsize, config.imsize)

                    X_in = torch.from_numpy(np.append(im_data, value_data, axis=1)).float().to(device)

                    S1_in = torch.tensor([[curr[0]]]).float().to(device)
                    S2_in = torch.tensor([[curr[1]]]).float().to(device)

                    _, predictions = vin(X_in, S1_in, S2_in, config.k)
                    a = torch.argmax(predictions, dim=1).item()

                    s = G.map_ind_to_state(int(curr[0]), int(curr[1]))
                    ns = G.sample_next_state(s, a)

                    nr, nc = G.get_coords(np.array([ns]))
                    nr, nc = int(nr[0]), int(nc[0])

                    pred_traj[j] = [nr, nc]

                    if nr == goal[0] and nc == goal[1]:
                        pred_len = j + 1
                        break

                if pred_len is None:
                    pred_len = len(pred_traj)

                if pred_traj[j][0] == goal[0] and pred_traj[j][1] == goal[1]:
                    correct += 1

                total += 1
                traj_diffs.append(abs(pred_len - optimal_len))

                if config.plot:
                    visualize(G.image.T, states_xy[i], pred_traj)

    success_rate = correct / total if total > 0 else 0
    traj_diff = np.mean(traj_diffs) if len(traj_diffs) > 0 else 0

    return success_rate, traj_diff


# =========================
# MAIN (MULTI-RUN)
# =========================
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--weights', type=str, default='trained/vin_8x8.pth')
    parser.add_argument('--imsize', type=int, default=8)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--l_i', type=int, default=2)
    parser.add_argument('--l_h', type=int, default=150)
    parser.add_argument('--l_q', type=int, default=10)

    parser.add_argument('--plot', action='store_true', default=False)
    parser.add_argument('--n_runs', type=int, default=5)

    config = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    vin = VIN(config)
    vin.load_state_dict(torch.load(config.weights))
    vin = vin.to(device)

    all_losses = []
    all_success = []
    all_traj = []

    for run in range(config.n_runs):
        print(f"\nRun {run+1}/{config.n_runs}")

        loss = compute_prediction_loss(vin, config, device)
        success, traj_diff = rollout_eval(vin, config, device)

        all_losses.append(loss)
        all_success.append(success)
        all_traj.append(traj_diff)

    print("\n===== FINAL AVERAGED RESULTS =====")
    print(f"Prediction Loss: {np.mean(all_losses):.4f} ± {np.std(all_losses):.4f}")
    print(f"Success Rate: {np.mean(all_success)*100:.2f}% ± {np.std(all_success)*100:.2f}%")
    print(f"Trajectory Difference: {np.mean(all_traj):.2f} ± {np.std(all_traj):.2f}")