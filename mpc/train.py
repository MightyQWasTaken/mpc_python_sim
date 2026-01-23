# region | Imports
import torch
from torch import nn
from torch.autograd import Function, Variable
from torch.nn.parameter import Parameter
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import numpy.random as npr

import os
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import time
import pickle as pkl

import threading
from utils import model as md
import wandb
from tqdm import tqdm
# endregion


# Function to train a model on data found in data folder

# region | Device setup & data loading
# cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# If using CUDA, enable pinned memory for faster host->GPU transfer
pin_memory = True if torch.cuda.is_available() else False

# Use project-root data directory so paths don't depend on current working directory
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

# Load tensors from the specified directory
# Will want to use x_test for validation after training. Use subset of x_train for validation during testing
x_train_data = torch.load(os.path.join(data_dir, 'x_train.pt'))
x_test_data = torch.load(os.path.join(data_dir, 'x_test.pt'))
u_train_data = torch.load(os.path.join(data_dir, 'u_train.pt'))
u_test_data = torch.load(os.path.join(data_dir, 'u_test.pt'))

# endregion


#Settings

# Model selection: 'linear' (simple linear), 'lqr' (LQR-based controller), or 'nn' (neural network)
model_type = 'nn'
LoadParams = False #If true will load the initial parameters from a saved model

#Hyperparameters
hidden_size = 64
learning_rate = 5e-3
num_epochs = 500
weight_decay = 0
batch_size = 64
num_layers = 1
wandb_run_name = "NN input appended with linear cost and constraints. Constant disturbance (not sinusoid/ freq sweep). old loss function."


# region | data prep

# Create TensorDatasets
train_dataset = torch.utils.data.TensorDataset(x_train_data, u_train_data)
test_dataset = torch.utils.data.TensorDataset(x_test_data, u_test_data)

# Create DataLoaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           pin_memory=pin_memory)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          pin_memory=pin_memory)


# Get the sizes
# ~~ Not clear of the following. Add a breakpoint to see what's going on ~~
n_state, n_ctrl = x_train_data.size(1), u_train_data.size(1)
n_sc = n_state + n_ctrl
sequence_length = 0

# Save the sizes
sizes={
    'n_state': n_state, 
    'n_ctrl': n_ctrl,
    'hidden_size': hidden_size,
    'n_sc': n_sc,
    'num_layers': num_layers,
    'sequence_length': sequence_length
}

fname = os.path.join(data_dir, 'sizes.pkl')
with open(fname, 'wb') as f:
    pkl.dump(sizes, f)
# endregion


# region | Initialize the model which will be used
    
# If model_type is 'linear', use LinearController
# If model_type is 'nn', use NNController (neural network).
if model_type == 'linear':
    model = md.LinearController(n_state, n_ctrl)

elif model_type == 'nn':
    model = md.NNController(n_state, hidden_size, n_ctrl)

else:
    raise ValueError(f"Unknown model_type: {model_type}. Use 'linear', 'lqr', or 'nn'.")

# endregion

# region | loss function and model loading

criterion = md.get_loss()       # get loss function from utils

# Try to load J_mpc from the data directory if present. These
# are optional — if missing the L2 term will be skipped inside the loss.
J_mpc = None
try:
    J_path = os.path.join(data_dir, 'J_mpc.npy')
    if os.path.exists(J_path):
        J_mpc = np.load(J_path)
except Exception:
    # If load fails, continue with J_mpc = None
    J_mpc = None

# Construct the NN model
if LoadParams:                  # loads model weight if requested
    model_path = os.path.join(data_dir, 'model', 'model.ckpt')
    nnparams = torch.load(model_path)
    # support both plain state_dict and our checkpoint dict format
    if isinstance(nnparams, dict) and 'model_state' in nnparams:
        state_dict = nnparams['model_state']
    else:
        state_dict = nnparams
    model.load_state_dict(state_dict)

model.to(device)                # moves model to CPU/ GPU
# Loss and optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)       # Adam optimizer

# store the loss values (loss tracking)
loss_values = []
train_losses =[]
val_losses = []
epochs = []
# endregion

# region | test function (evaluates model on the test set and returns average loss)
# Function for testing the model
def test_model(model, test_loader):
    with torch.no_grad():
        loss_value = []
        for x_test_data, u_test_data in test_loader:
            x_test = x_test_data.to(device, non_blocking=pin_memory).float()
            u_test = u_test_data.to(device, non_blocking=pin_memory).float()
            predictions = model(x_test)
            loss_value.append(criterion(predictions, u_test).item())
    return np.mean(loss_value)
# endregion


# region | initalising wandb
run = wandb.init(
    entity="adam-gardner-univeristy-of-oxford",
    project="4YP",
    name=wandb_run_name,
    config={
        "learning_rate": learning_rate,
        "batch size": batch_size,
        "epochs": num_epochs,
    },
)
# endregion

# region | training loop setup
total_step = len(train_loader)
start = time.time()

#Start thread to monitor user input for early exit
keep_training = True
exit_event = threading.Event()

#Function to check for user input
def check_user_input():
    global keep_training
    input("Press Enter to stop training here...\n")
    keep_training = False
    exit_event.set()


input_thread = threading.Thread(target=check_user_input)
input_thread.start()
# endregion

# region | training loop
for epoch in range(num_epochs):  # episode size
    if keep_training == False:  # allows user to stop training by pressing enter (via background thread)
        break
    model.eval()
    val_loss = test_model(model, test_loader)
    val_losses.append(val_loss)
    TimeRemaining = ((time.time()-start) / (epoch+1)) * (num_epochs - epoch)
    model.train()
    print("No. iterations: ", len(train_loader))
    for i, (x_batch, u_batch) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}')):
        # Move tensors to the configured device

        # Move batch to device (non-blocking if pinned memory enabled) and cast
        x_batch = x_batch.to(device, non_blocking=pin_memory).float()
        u_batch = u_batch.to(device, non_blocking=pin_memory).float()

        optimizer.zero_grad()       # clears gradient from previous iteration
        # Forward pass
        predictions = model(x_batch)    # network computes predicted controls given the states
        # Extract per-sample control constraints from the tail of x_batch and q_vector
        # Assumes last 2*nu columns are [lower_constraints, upper_constraints]
        nu = u_batch.size(1)
        if x_batch.size(1) >= 3 * nu:
            u_upper = x_batch[:, -nu:]
            u_lower = x_batch[:, -2*nu:-nu]
            q_batch = x_batch[:, -3*nu:-2*nu]
        else:
            u_upper = None
            u_lower = None
            q_batch = None

        loss = criterion(predictions, u_batch, upper_const=u_upper, lower_const=u_lower, J=J_mpc, q_vec=q_batch)

        # Backward and optimize
        loss.backward()         # computes gradients of all parameters via backprop
        optimizer.step()        # updates all model parameters using Adam optimizer

        # store the loss value
        loss_values.append(loss.item())
        
        # Log metrics to wandb
        run.log({"loss": loss.item(), "validation loss": val_loss})
    epochs.append(epoch)
    train_losses.append(loss.item())
    print('Epoch [{}/{}], Loss: {:.4f}, Validation Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item(), val_loss))
    
# Signal the thread to exit if it hasn't already
exit_event.set()

# Wait for the input thread to finish
input_thread.join()
# endregion


# region | final test and model saving
print('Test Loss: {:.4f}'.format(test_model(model, test_loader)))

# Use project-root data directory path (not relative path)
model_dir = os.path.join(data_dir, 'model')
os.makedirs(model_dir, exist_ok=True)
# Save model artifacts in a way compatible with older loaders that expect
# a plain LinearController checkpoint (keys: `fc.weight`, `fc.bias`) in
# `model.ckpt` while preserving the full trained weights separately.
if model_type == 'linear':
    # Linear controller already uses fc.weight / fc.bias naming
    torch.save(model.state_dict(), os.path.join(model_dir, 'model.ckpt'))

elif model_type == 'lqr':
    # Export learned K into a LinearController-compatible state_dict
    linear_state = {
        'fc.weight': model.K.data.cpu(),
        'fc.bias': torch.zeros(n_ctrl, dtype=torch.float32),
    }
    torch.save(linear_state, os.path.join(model_dir, 'model.ckpt'))
    # Save full LQR state and K matrix for reproducibility
    torch.save(model.state_dict(), os.path.join(model_dir, 'model_lqr.ckpt'))
    K_matrix = model.K.data.cpu().numpy()
    np.save(os.path.join(model_dir, 'K_matrix.npy'), K_matrix)

elif model_type == 'nn':
    # Save full NN weights
    torch.save(model.state_dict(), os.path.join(model_dir, 'model.ckpt'))

    # # (Optional) Create a linear least-squares approximation u = W x + b on the training
    # # data for legacy loaders. Saved as model_linear_approx.ckpt
    # try:
    #     X = x_train_data.cpu().numpy().reshape(-1, n_state)
    #     U = u_train_data.cpu().numpy().reshape(-1, n_ctrl)
    #     X_aug = np.hstack([X, np.ones((X.shape[0], 1), dtype=X.dtype)])
    #     Beta, *_ = np.linalg.lstsq(X_aug, U, rcond=None)
    #     W = Beta[:-1, :].T.astype(np.float32)  # (n_ctrl, n_state)
    #     b = Beta[-1, :].astype(np.float32)     # (n_ctrl,)
    #     linear_state = {
    #         'fc.weight': torch.from_numpy(W),
    #         'fc.bias': torch.from_numpy(b),
    #     }
    #     torch.save(linear_state, os.path.join(model_dir, 'model_linear_approx.ckpt'))
    # except Exception as e:
    #     print('Warning: failed to compute linear fit for compatibility:', e)

else:
    # Unknown model type: save the raw state_dict
    torch.save(model.state_dict(), os.path.join(model_dir, 'model.ckpt'))

os.makedirs(os.path.join(model_dir, 'ckpt'), exist_ok=True)
os.makedirs(os.path.join(model_dir, 'losses'), exist_ok=True)

epochs = np.array(epochs)
train_losses = np.array(train_losses)
val_losses = np.array(val_losses)

np.savez(os.path.join(model_dir, 'modelloss.npz'), epochs = epochs, train_losses = train_losses, val_losses = val_losses)

#Log in storage with other models
checkpoint_meta = {
    'model_state': model.state_dict(),
    'model_type': model_type,
    'hidden_size': hidden_size,
}
torch.save(checkpoint_meta, os.path.join(model_dir, 'ckpt', f'tp-{model.type}-ns-{n_state}-hs-{hidden_size}-bs-{batch_size}.ckpt'))

np.savez(os.path.join(model_dir, 'losses', f'tp-{model.type}-ns-{n_state}-hs-{hidden_size}-bs-{batch_size}.npz'), epochs = epochs, train_losses = train_losses, val_losses = val_losses)

# endregion

# region | model plotting
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()
plt.savefig('output_train.png')
run.finish()

# endregion