import numpy as np
import argparse
import os
import sys

sys.path.append('.')
from create_custom_test_data import create_custom_test_data
sys.path.remove('.')


def load_map_from_file(filepath):
    """Load a map from a text file.
    Format: 0 = free space, 1 = obstacle
    Lines represent rows, space-separated numbers."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Parse the map
    map_data = []
    for line in lines:
        line = line.strip()
        if line:
            row = [int(x) for x in line.split()]
            map_data.append(row)

    return np.array(map_data)


def create_interactive_map(size):
    """Create a map interactively"""
    print(f"Creating a {size}x{size} map interactively.")
    print("Enter each row as space-separated numbers (0=free, 1=obstacle):")

    map_data = []
    for i in range(size):
        while True:
            try:
                row_input = input(f"Row {i}: ")
                row = [int(x) for x in row_input.split()]
                if len(row) != size:
                    print(f"Error: Row must have exactly {size} elements")
                    continue
                map_data.append(row)
                break
            except ValueError:
                print("Error: Please enter only numbers separated by spaces")
                continue

    return np.array(map_data)


def main():
    parser = argparse.ArgumentParser(description='Create custom test maps for VIN')
    parser.add_argument('--size', type=int, required=True, choices=[8, 16, 28],
                       help='Size of the grid')
    parser.add_argument('--map_file', type=str,
                       help='Path to text file containing the map (optional)')
    parser.add_argument('--goal_x', type=int, help='Goal row position')
    parser.add_argument('--goal_y', type=int, help='Goal column position')
    parser.add_argument('--n_samples', type=int, default=1000,
                       help='Number of test samples to generate')

    args = parser.parse_args()

    size = args.size

    # Load or create map
    if args.map_file:
        if not os.path.exists(args.map_file):
            print("Error: Map file {} does not exist".format(args.map_file))
            return
        custom_map = load_map_from_file(args.map_file)
        if custom_map.shape != (size, size):
            print("Error: Map file must be {}x{}, got {}".format(size, size, custom_map.shape))
            return
    else:
        custom_map = create_interactive_map(size)

    print("Your map:")
    print(custom_map)

    # Set goal
    if args.goal_x is None or args.goal_y is None:
        while True:
            try:
                goal_input = input("Enter goal position (row col): ")
                goal_x, goal_y = [int(x) for x in goal_input.split()]
                if 0 <= goal_x < size and 0 <= goal_y < size:
                    if custom_map[goal_x, goal_y] == 1:
                        print("Error: Goal cannot be on an obstacle")
                        continue
                    break
                else:
                    print("Error: Goal must be within 0-{}".format(size-1))
                    continue
            except ValueError:
                print("Error: Please enter two integers separated by space")
                continue
    else:
        goal_x, goal_y = args.goal_x, args.goal_y
        if custom_map[goal_x, goal_y] == 1:
            print("Error: Goal cannot be on an obstacle")
            return

    print("Goal position: ({}, {})".format(goal_x, goal_y))

    # Now generate the test data using the previous script
    # Import and call the function from create_custom_test_data.py

    X_test, S1_test, S2_test, Labels_test = create_custom_test_data(
        custom_map, goal_x, goal_y, args.n_samples
    )

    if X_test is not None:
        # Save as npz
        save_path = "dataset/custom_gridworld_{}x{}.npz".format(size, size)
        dummy_train = np.zeros((1, 2, size, size))
        dummy_s1_train = np.zeros(1)
        dummy_s2_train = np.zeros(1)
        dummy_labels_train = np.zeros(1)

        np.savez_compressed(save_path,
                           dummy_train, dummy_s1_train, dummy_s2_train, dummy_labels_train,
                           X_test, S1_test, S2_test, Labels_test)
        print("Custom test data saved to {}".format(save_path))
        print("You can now test with: python test.py --weights trained/vin_{}x{}.pth --datafile {} --imsize {}".format(size, size, save_path, size))


if __name__ == '__main__':
    main()