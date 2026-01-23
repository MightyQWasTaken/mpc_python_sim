import numpy as np
import torch
import os

#Helper function to construct the training and testing datasets

def process_data_shuff(x0_obs, xd_obs, u_sim, data_dir, use_dagger, q=None, l_constr=None, u_constr=None):

    # Base state components
    x_components = [x0_obs, xd_obs]
  
    x_data = np.hstack(tuple(x_components))
    # Optionally append q vector and lower/upper constraint vectors (these should be per-sample)
    if q is not None:
        x_data = np.hstack((x_data, q))
    if l_constr is not None:
        x_data = np.hstack((x_data, l_constr))
    if u_constr is not None:
        x_data = np.hstack((x_data, u_constr))
    #Concatenate x_data and u_data along the second axis
    combined_data = np.hstack((x_data, u_sim))
    # Ensure data directory exists
    os.makedirs(data_dir, exist_ok=True)
    expert_data_dir = os.path.join(data_dir, 'expert_data.npy')
    

    if use_dagger:
        #Add the expert data to the combined data
        
        if os.path.exists(expert_data_dir):
            expert_data = np.load(expert_data_dir)
            combined_data = np.vstack((combined_data, expert_data))
        
    np.save(expert_data_dir, combined_data)
    
    np.random.seed(42)

    # Shuffle the combined data in-place (constraints are already appended to x_data)
    np.random.shuffle(combined_data)

    #Split the combined data back into x and u components

    x_data_shuffled = combined_data[:, :x_data.shape[1]]
    u_data_shuffled = combined_data[:, x_data.shape[1]:]

    #Split the shuffled data into training and testing sets
    train_size = int(0.8 * combined_data.shape[0])


    x_train = torch.tensor(x_data_shuffled[:train_size])
    x_test = torch.tensor(x_data_shuffled[train_size:])

    u_train = torch.tensor(u_data_shuffled[:train_size])
    u_test = torch.tensor(u_data_shuffled[train_size:])

    torch.save(x_train.float(), os.path.join(data_dir, 'x_train.pt'))
    torch.save(x_test.float(), os.path.join(data_dir, 'x_test.pt'))
    torch.save(u_train.float(), os.path.join(data_dir, 'u_train.pt'))
    torch.save(u_test.float(), os.path.join(data_dir, 'u_test.pt'))


