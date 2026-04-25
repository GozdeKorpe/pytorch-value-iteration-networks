import numpy as np
import sys
import argparse

sys.path.append('.')
from domains.gridworld import *
sys.path.remove('.')


def extract_action(traj):
    # Given a trajectory, outputs a 1D vector of
    #  actions corresponding to the trajectory.
    n_actions = 8
    action_vecs = np.asarray([[-1., 0.], [1., 0.], [0., 1.], [0., -1.],
                              [-1., 1.], [-1., -1.], [1., 1.], [1., -1.]])
    action_vecs[4:] = 1 / np.sqrt(2) * action_vecs[4:]
    action_vecs = action_vecs.T
    state_diff = np.diff(traj, axis=0)
    norm_state_diff = state_diff * np.tile(
        1 / np.sqrt(np.sum(np.square(state_diff), axis=1)), (2, 1)).T
    prj_state_diff = np.dot(norm_state_diff, action_vecs)
    actions_one_hot = np.abs(prj_state_diff - 1) < 0.00001
    actions = np.dot(actions_one_hot, np.arange(n_actions).T)
    return actions


def create_custom_test_data(map_array, goal_x, goal_y, n_test_samples=1000, n_traj_per_sample=1):
    """
    Create test data for a custom map.

    Parameters:
    - map_array: numpy array where 0 = free space, 1 = obstacle
    - goal_x, goal_y: goal position (row, col)
    - n_test_samples: number of test trajectories to generate
    - n_traj_per_sample: trajectories per sample (usually 1 for test)
    """

    dom_size = map_array.shape
    print(f"Creating test data for {dom_size[0]}x{dom_size[1]} map")

    # Add border to match original data format (edges become obstacles)
    bordered_map = np.ones((dom_size[0] + 2, dom_size[1] + 2), dtype=int)
    bordered_map[1:-1, 1:-1] = map_array
    
    # Adjust goal coordinates for the border
    goal_x += 1
    goal_y += 1

    X_l = []
    S1_l = []
    S2_l = []
    Labels_l = []

    # Invert map: 0 = obstacle, 1 = free in GridWorld
    im = 1 - bordered_map  # 0=free -> 1=free, 1=obstacle -> 0=obstacle

    # Create GridWorld
    G = GridWorld(im, goal_x, goal_y)

    # Get value prior
    value_prior = G.t_get_reward_prior()

    # Sample trajectories
    states_xy, states_one_hot = sample_trajectory(G, n_test_samples)

    for i in range(n_test_samples):
        if len(states_xy[i]) > 1:
            # Get optimal actions for each state
            actions = extract_action(states_xy[i])
            ns = states_xy[i].shape[0] - 1

            # Invert domain image => 0 = free, 1 = obstacle for dataset
            # Crop out the border
            image = 1 - im[1:-1, 1:-1]
            image = image.astype(np.float32)
            value_prior_cropped = value_prior[1:-1, 1:-1].astype(np.float32)

            # Resize domain and goal images and concatenate
            image_data = np.resize(image, (1, 1, dom_size[0], dom_size[1]))
            value_data = np.resize(value_prior_cropped, (1, 1, dom_size[0], dom_size[1]))
            iv_mixed = np.concatenate((image_data, value_data), axis=1)
            X_current = np.tile(iv_mixed, (ns, 1, 1, 1))

            # Resize states (adjust for border offset)
            S1_current = np.expand_dims(states_xy[i][0:ns, 0] - 1, axis=1)
            S2_current = np.expand_dims(states_xy[i][0:ns, 1] - 1, axis=1)

            # Resize labels
            Labels_current = np.expand_dims(actions, axis=1)

            # Append to output list
            X_l.append(X_current)
            S1_l.append(S1_current)
            S2_l.append(S2_current)
            Labels_l.append(Labels_current)

    # Concat all outputs
    if X_l:
        X_f = np.concatenate(X_l)
        S1_f = np.concatenate(S1_l)
        S2_f = np.concatenate(S2_l)
        Labels_f = np.concatenate(Labels_l)
        print(f"Generated {len(X_f)} test samples")
        return X_f, S1_f, S2_f, Labels_f
    else:
        print("No valid trajectories generated")
        return None, None, None, None


def create_simple_maze(size):
    """Create a simple maze for testing"""
    # Create empty map (0 = free, 1 = obstacle)
    maze = np.zeros((size, size), dtype=int)

    # Add some walls
    if size >= 8:
        # Vertical wall in the middle
        maze[:, size//2] = 1
        # Horizontal wall
        maze[size//2, :] = 1
        # Create a path
        maze[size//2, size//2] = 0
        maze[size//2-1, size//2] = 0
        maze[size//2+1, size//2] = 0

    # Ensure borders are obstacles (but GridWorld adds border)
    # maze[0, :] = 1
    # maze[-1, :] = 1
    # maze[:, 0] = 1
    # maze[:, -1] = 1

    return maze


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, required=True, help='Size of the grid (8, 16, or 28)')
    parser.add_argument('--goal_x', type=int, default=None, help='Goal row position')
    parser.add_argument('--goal_y', type=int, default=None, help='Goal col position')
    parser.add_argument('--n_samples', type=int, default=1000, help='Number of test samples')

    args = parser.parse_args()

    size = args.size
    if args.goal_x is None:
        goal_x = size - 2
    else:
        goal_x = args.goal_x

    if args.goal_y is None:
        goal_y = size - 2
    else:
        goal_y = args.goal_y

    # Create a simple custom map
    custom_map = create_simple_maze(size)

    print(f"Custom map for {size}x{size}:")
    print(custom_map)
    print(f"Goal position: ({goal_x}, {goal_y})")

    # Generate test data
    X_test, S1_test, S2_test, Labels_test = create_custom_test_data(
        custom_map, goal_x, goal_y, args.n_samples
    )

    if X_test is not None:
        # Save as npz (only test data, dummy train data)
        save_path = f"dataset/custom_gridworld_{size}x{size}.npz"
        dummy_train = np.zeros((1, 2, size, size))  # dummy train images
        dummy_s1_train = np.zeros(1)  # dummy train S1
        dummy_s2_train = np.zeros(1)  # dummy train S2
        dummy_labels_train = np.zeros(1)  # dummy train labels

        np.savez_compressed(save_path,
                           dummy_train, dummy_s1_train, dummy_s2_train, dummy_labels_train,
                           X_test, S1_test, S2_test, Labels_test)
        print("Saved custom test data to {}".format(save_path))


if __name__ == '__main__':
    main()