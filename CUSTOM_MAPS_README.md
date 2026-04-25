# Creating Custom Maps for VIN Testing

This guide shows how to create your own custom maps for testing the Value Iteration Network (VIN) on 8x8, 16x16, and 28x28 gridworlds.

## Quick Start

### Option 1: Use Example Maps

1. Generate test data for 8x8:
```bash
python create_custom_map.py --size 8 --map_file example_map_8x8.txt --goal_x 7 --goal_y 7
```

2. Test the model:
```bash
python test.py --weights trained/vin_8x8.pth --datafile dataset/custom_gridworld_8x8.npz --imsize 8 --plot
```

The `--plot` flag will show a visualization of the predicted trajectory vs optimal path.

3. Repeat for 16x16 and 28x28:
```bash
python create_custom_map.py --size 16 --map_file example_map_16x16.txt --goal_x 15 --goal_y 15
python test.py --weights trained/vin_16x16.pth --datafile dataset/custom_gridworld_16x16.npz --imsize 16 --plot

python create_custom_map.py --size 28 --map_file example_map_28x28.txt --goal_x 27 --goal_y 27
python test.py --weights trained/vin_28x28.pth --datafile dataset/custom_gridworld_28x28.npz --imsize 28 --plot
```

### Option 2: Create Your Own Map Interactively

```bash
python create_custom_map.py --size 8
```

Follow the prompts to enter your map row by row and set the goal position.

### Option 3: Create Map from Text File

Create a text file where each line represents a row, with space-separated 0s (free) and 1s (obstacle):

```
0 0 0 0 0 0 0 0
0 1 1 1 1 1 1 0
0 1 0 0 0 0 1 0
...
```

Then run:
```bash
python create_custom_map.py --size 8 --map_file your_map.txt --goal_x 6 --goal_y 6
```

## Map Format

- **0**: Free space (agent can move here)
- **1**: Obstacle (agent cannot move here)
- The map should be square (NxN)
- Goal position should be on a free space (0)
- The GridWorld class automatically adds a border of obstacles

## Testing

After creating your custom test data, use the test.py script with:
- `--weights`: Path to the trained model (vin_8x8.pth, vin_16x16.pth, or vin_28x28.pth)
- `--datafile`: Path to your custom .npz file
- `--imsize`: Size of the grid (8, 16, or 28)
- `--plot`: Add this flag to visualize trajectories

Example:
```bash
python test.py --weights trained/vin_8x8.pth --datafile dataset/custom_gridworld_8x8.npz --imsize 8 --plot
```

## Understanding the Results

The test will output:
- **Prediction Loss**: How well the model predicts optimal actions
- **Success Rate**: Percentage of correct action predictions
- **Trajectory Difference**: Average deviation from optimal path

## Troubleshooting

- Make sure the goal position is not on an obstacle
- Ensure the map file has the correct dimensions
- Check that trained models exist in the `trained/` directory
- If no trajectories are generated, the map might be unsolvable
- Complex maps may generate fewer test samples - try simpler maps or increase `--n_samples`
- Success rates may vary depending on map complexity (e.g., 8x8 maps typically perform better than 16x16)</content>
<parameter name="filePath">c:\Users\MENESADANUR\Desktop\gozdeRL\master\CMP620\pytorch-value-iteration-networks\CUSTOM_MAPS_README.md
python create_custom_map.py --size 8 --map_file example_map_8x8.txt --goal_x 7 --goal_y 7 --n_samples 100