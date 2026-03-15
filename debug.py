import numpy as np
d = np.load('dataset/gridworld_8x8.npz')
print("Training samples:", d['arr_0'].shape[0])
print("Test samples:", d['arr_4'].shape[0])


S1 = d['arr_1']  # rows
S2 = d['arr_2']  # cols
Labels = d['arr_3']
print("S1 range:", S1.min(), S1.max())   # should be 0-7
print("S2 range:", S2.min(), S2.max())   # should be 0-7
print("Labels range:", Labels.min(), Labels.max())  # should be 0-7
print("Label distribution:", np.bincount(Labels.flatten().astype(int)))

from domains.gridworld import *
from generators.obstacle_gen import *

# Make one tiny gridworld and print a trajectory
im = np.ones((8,8)); im[0,:]=0; im[7,:]=0; im[:,0]=0; im[:,7]=0
G = GridWorld(im, 5, 5)
states_xy, _ = sample_trajectory(G, 1)
traj = states_xy[0]
print("Trajectory:", traj)
print("Diffs:", np.diff(traj, axis=0))

import numpy as np
import torch
from dataset.dataset import *

ds = GridworldData('dataset/gridworld_8x8.npz', imsize=8, train=True, transform=None)
X, S1, S2, label = ds[0]
print("X shape:", X.shape)        # should be (2, 8, 8)
print("S1 shape:", S1.shape)      # should be (1,) or scalar
print("S2 shape:", S2.shape)      # should be (1,) or scalar  
print("S1 value:", S1)            # should be 1-6
print("S2 value:", S2)            # should be 1-6
print("label:", label)            # should be 0-7
print("X[0] sample:\n", X[0])    # obstacle map, 0s and 1s
print("X[1] sample:\n", X[1])    # value prior, should have a peak at goal
S1 = d['arr_1']
S2 = d['arr_2']
labels = d['arr_3']

# Print first 10 samples
for i in range(10):
    print(f"S1={S1[i][0]}, S2={S2[i][0]}, label={labels[i][0]}")

import numpy as np
import torch
from dataset.dataset import *
from model import *
import argparse

# Load config
config = argparse.Namespace(imsize=8, l_i=2, l_h=150, l_q=10, k=10)
net = VIN(config)
net.load_state_dict(torch.load('trained/vin_8x8.pth'))
net.eval()

# Load one batch manually
ds = GridworldData('dataset/gridworld_8x8.npz', imsize=8, train=False, transform=None)
correct = 0
total = 100
for idx in range(total):
    X, s1, s2, label = ds[idx]
    X_in = X.float().unsqueeze(0)
    S1_in = torch.tensor([[s1]])
    S2_in = torch.tensor([[s2]])
    with torch.no_grad():
        _, pred = net(X_in, S1_in, S2_in, config.k)
    a = torch.argmax(pred).item()
    if a == label:
        correct += 1
    if idx < 5:
        print(f"s1={s1}, s2={s2}, predicted={a}, actual={label.item()}, correct={a==label.item()}")
        print(f"  X[1] goal location: {(X[1]==10).nonzero()}")

print(f"\nManual accuracy on 100 samples: {correct}%")

actions_extract = [[-1.,0.],[1.,0.],[0.,1.],[0.,-1.],[-1.,1.],[-1.,-1.],[1.,1.],[1.,-1.]]
actions_gridworld = list(GridWorld.ACTION.values())

print("extract_action order:")
for i, a in enumerate(actions_extract):
    print(f"  {i}: {a}")

print("\nGridWorld.ACTION order:")
for i, (name, a) in enumerate(GridWorld.ACTION.items()):
    print(f"  {i}: {name} = {a}")

print("\nMatch:", actions_extract == [list(a) for a in actions_gridworld])
from generators.obstacle_gen import *
from dataset.make_training_data import extract_action

# Build one simple gridworld with known goal
np.random.seed(42)
goal = [2, 6]
im = np.ones((8,8))
im[0,:]=0; im[7,:]=0; im[:,0]=0; im[:,7]=0  # just border walls

G = GridWorld(im, goal[0], goal[1])
states_xy, _ = sample_trajectory(G, 3)

for i, traj in enumerate(states_xy):
    if len(traj) > 1:
        actions = extract_action(traj)
        print(f"Trajectory {i}:")
        for j in range(len(traj)-1):
            r, c = traj[j]
            nr, nc = traj[j+1]
            a = int(actions[j])
            print(f"  ({r},{c}) -> ({nr},{nc}), action={a} {list(GridWorld.ACTION.keys())[a]}")
        print(f"  Goal: {goal}")
        break