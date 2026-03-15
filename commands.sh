python dataset/make_training_data.py --size 8 --n_domains 7000 --max_obs 10 --max_obs_size 2 --n_traj 7  
python train.py --datafile dataset/gridworld_8x8.npz --imsize 8 --epochs 80 --lr 0.005 --k 10   
python test.py --weights trained/vin_8x8.pth --imsize 8 --k 10 