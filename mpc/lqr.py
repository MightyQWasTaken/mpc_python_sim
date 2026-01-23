from scipy.io import loadmat
import numpy as np
from utils import processdata as process
from utils import model as md
import torch
from utils import sim_mpc as sim
import os

# Script to generate LQR data for the 4-state system

fname = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'systems', '4statesystemnd8.mat')
#OnlyValidforND8
mat_data = loadmat(fname)
direction = 'vertical'

if direction == 'vertical':
    kx = mat_data['Kcx_y']
    kd = mat_data['Kcd_y']
else:
    kx = mat_data['Kcx_x']
    kd = mat_data['Kcd_x']

k = np.hstack((kx, kd))
n_samples = 15000

def generate_basis(dim):
    return np.eye(dim)

def generate_training_data(n_samples, dim):
    basis_vectors = generate_basis(dim)
    coefficients = np.random.randn(n_samples, dim)  # Coefficients for linear combinations
    training_data = basis_vectors @ coefficients.T  # Generate new samples as linear combinations
    
    return training_data

# x0_obs = np.random.randn(4, n_samples)
# xd_obs = np.random.randn(4,n_samples)
x = generate_training_data(n_samples, 8)

print(x.shape)
def get_u(x):
    u = -1*k @ x
    return u


u_sim = get_u(x)
print(u_sim.shape)

x0_obs = x[0:4, :].T
xd_obs = x[4:, :].T  
u_sim = u_sim.T

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

process.process_data_shuff(x0_obs, xd_obs, u_sim, data_dir, use_dagger=False)

