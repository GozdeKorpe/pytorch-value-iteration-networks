gridworld.py

state_to_loc — wrong dimension order (n_col, n_row) → (n_row, n_col)
get_coords — unpacking c, r backwards → r, c
trace_path — (max_len, 1) shape causing numpy comparison failures → (max_len,) 1D array

dataset/make_training_data.py (indirect)

All trajectories and action labels were wrong due to the gridworld.py bugs above — fixed by regenerating the dataset

dataset/dataset.py

S1, S2, labels returned as plain int instead of tensors → wrapped in torch.tensor()

test.py

np.int deprecated crash → np.int32
Debug prints inside inner j loop shadowing outer i variable
Missing vin.eval() and torch.no_grad()
get_coords returning arrays not scalars → explicit int() extraction
Missing try/except guard on map_ind_to_state
torch.max with unnecessary keepdim=True → .item()

1. Coordinate system bug (gridworld.py)
The function that converts state indices back to grid coordinates had its row and column dimensions swapped, causing the entire navigation system to confuse rows with columns. This meant every trajectory was moving in the wrong direction — navigating east/west when it should go north/south. Fixing the dimension order made all path planning geometrically correct.
2. Path tracing bug (gridworld.py)
The shortest path tracing function used a 2D array instead of a 1D array, which caused NumPy to compare array objects instead of scalar values and silently return empty paths. This meant no valid training trajectories were ever generated, causing the zero-division crash at evaluation time.
3. Dataset corruption (indirect)
Because the coordinate and path bugs were present during dataset generation, all training trajectories and action labels in the dataset were wrong. The dataset had to be fully regenerated after the gridworld fixes for the model to learn correct navigation behavior.
4. Broken tensor types (dataset/dataset.py)
State position values S1 and S2 were returned as plain Python integers instead of PyTorch tensors, meaning they could not be properly moved to the GPU or processed by the model. This caused the model to receive malformed inputs during every training step, creating a hard ceiling on accuracy regardless of training time.
5. Missing inference settings (test.py)
The model was never switched to evaluation mode and gradient tracking was never disabled during testing, wasting memory and potentially affecting prediction consistency. Several smaller bugs including deprecated NumPy types, misplaced print statements, and a variable shadowing issue were also fixed to stabilize the evaluation pipeline.