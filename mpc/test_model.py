# region | imports
import torch
from torch import nn
from torch.autograd import Function, Variable
from torch.nn.parameter import Parameter
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import numpy.random as npr
 
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pickle as pkl
import scipy.sparse as sparse
from scipy.io import loadmat
from utils import sim_mpc as sim
from utils import model as md
from utils import processdata as process
from utils import diamond_I_configuration_v5 as DI

import pandas as pd
import time
# endregion

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# region | helper functions to generate random disturbance modes

#Helper function to generate random disturbance modes
def randModes(seed, RM, id_to_bpm, TOT_BPM, u_mag):
    np.random.seed(seed)    # Fix randomness with seed
    UR, SR, VR = np.linalg.svd(RM)      # SVD decomposition of response matrix
    weighted_combination = np.zeros_like(UR[:, 0])      # Initialize output vector
    doff_tmp = np.zeros((TOT_BPM, 1))   # Storage for all BPMs
    for i in range(len(SR)):
        #Generate random pertubation between -1 and 1
        pert = np.random.uniform(-1,1,1)
        #Change the ith mode by the pertubation
        weighted_combination += pert *SR[i]* UR[:, i] * u_mag
        
    doff_tmp[id_to_bpm] = weighted_combination[:, np.newaxis]       # Place in BPM indices
    doff = doff_tmp * np.ones((1,n_samples))        # Replicate across all time samples
    return doff


# Helper function to generate random sinusoidal disturbance modes
def randModesSinusoid(seed, RM, id_to_bpm, TOT_BPM, u_mag, freq=1, amp_ratio=1):
    base_doff = randModes(seed, RM, id_to_bpm, TOT_BPM, u_mag)  # shape (TOT_BPM, n_samples)

    # Time vector and sinusoid (uses globals n_samples and Ts)
    t = np.arange(0, n_samples) * Ts
    # Frequency sweep: angular frequency w = a * t  => sin(w * t) = sin(a * t^2)
    # Here `freq` acts as the sweep-rate parameter `a/(2*pi)` so that
    # the argument becomes 2*pi*freq*t^2. This keeps the previous units similar
    # while producing a time-varying frequency.
    sinusoid = 1.0 + amp_ratio * np.sin(2.0 * np.pi * freq * (t ** 2))

    # Multiply base disturbance by sinusoid in time
    doff = base_doff * sinusoid[np.newaxis, :]
    return doff


# Helper function to generate random disturbance modes with Additive White Gaussian Noise (AWGN)
def randModesAWGN(seed, RM, id_to_bpm, TOT_BPM, u_mag, noise_std=4.0):
    """
    Generates a base disturbance mode and adds time-varying Gaussian noise (AWGN).
    """
    base_doff = randModes(seed, RM, id_to_bpm, TOT_BPM, u_mag)  # shape (TOT_BPM, n_samples)

    # Generate white Gaussian noise for the entire duration
    # Since randModes sets the seed, this noise generation is also determined by 'seed'
    noise = np.random.normal(loc=0.0, scale=noise_std, size=base_doff.shape)

    # Mask to apply noise only on active BPMs
    mask = np.zeros_like(base_doff)
    mask[id_to_bpm, :] = 1.0

    # Combine base disturbance with noise
    doff = base_doff + (noise * mask)
    return doff


# Helper function to generate random sinusoidal disturbance modes with Additive White Gaussian Noise (AWGN)
def randModesSinusoidAWGN(seed, RM, id_to_bpm, TOT_BPM, u_mag, freq=1, amp_ratio=1, noise_std=4.0):
    """
    Generates a base disturbance mode with sinusoidal modulation and adds time-varying Gaussian noise (AWGN).
    """
    base_doff = randModesSinusoid(seed, RM, id_to_bpm, TOT_BPM, u_mag, freq, amp_ratio)

    # Generate white Gaussian noise for the entire duration
    # Since randModes sets the seed, this noise generation is also determined by 'seed'
    noise = np.random.normal(loc=0.0, scale=noise_std, size=base_doff.shape)

    # Mask to apply noise only on active BPMs
    mask = np.zeros_like(base_doff)
    mask[id_to_bpm, :] = 1.0

    # Combine base disturbance with noise
    doff = base_doff + (noise * mask)
    return doff

# endregion

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# region | Setup
start = time.time()     # Start timer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')       # Use GPU if available

# options
fname_RM = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'orms', 'GoldenBPMResp_DIAD.mat')        # Path to response matrix file
pick_dir = 1
dirs = ['horizontal','vertical']
pick_direction = dirs[pick_dir]      # 'horizontal' or 'vertical'
sim_IMC = False
use_FGM = False
#Simulates multiple modes of disturbance to get training data
generate_data = True
#Toggle for comparing nn performance and mpc performance
compare = False
#Toggle for using DAGGER
use_dagger = False
#Toggle for LQR limits - Note that this is incompatible with OSQP
use_lqr = False
#Define max number of samples
n_samples = 8000
#first n_include BPMs and CMs active for testing
n_include = 4


# endregion

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# region | Load pre-trained model
if use_dagger or compare:
    #Load model parameters for nn
    size_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'sizes.pkl')
    with open(size_file_path, 'rb') as f:
        sizes = pkl.load(f)

    n_state = sizes['n_state']
    n_ctrl = sizes['n_ctrl']
    hidden_size = sizes['hidden_size']
    n_sc = sizes['n_sc']
    num_layers = sizes['num_layers']
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'model', 'model.ckpt')
    nnparams = torch.load(model_path)
# endregion

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# region | Load system data
fname_correctors = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'corrector_data.csv')
correctors = pd.read_csv(fname_correctors)
hardlimits = correctors['MaxAmps'].iloc[:172]
hardlimits = hardlimits.to_numpy().reshape(172, 1)

#Configure Diamond-I storage ring
Rmat = loadmat(fname_RM)
RMorigx = Rmat['Rmat'][0][0][0]
ny_x = np.size(RMorigx, 0)
nu_x = np.size(RMorigx, 1) 
RMorigy = Rmat['Rmat'][1][1][0]
ny_y = np.size(RMorigy, 0) 
nu_y = np.size(RMorigy, 1)
assert ny_x == ny_y
assert nu_x == nu_y
TOT_BPM = np.size(RMorigx, 0)
TOT_CM = np.size(RMorigx, 1)
square_config = True
id_to_bpm_x, id_to_cm_x, id_to_bpm_y, id_to_cm_y = DI.diamond_I_configuration_v5(RMorigx, RMorigy, square_config)


id_to_bpm_x = id_to_bpm_x[:n_include]       # horizontal BPM
id_to_cm_x = id_to_cm_x[:n_include]         # horizontal CM
id_to_bpm_y = id_to_bpm_y[:n_include]       # vertical BPM
id_to_cm_y = id_to_cm_y[:n_include]         # vertical CM

RMx = RMorigx[np.ix_(id_to_bpm_x, id_to_cm_x)]      #4x4 response matrix (horizontal)
RMy = RMorigy[np.ix_(id_to_bpm_y, id_to_cm_y)]      #4x4 response matrix (vertical)


#Ensure this file is correct for n_delay and dimensions
fname = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'systems', '4statesystemnd8.mat')
#OnlyValidforND8
mat_data = loadmat(fname)

#Observer and Regulator
n_delay = 8         # System has 8-sample delay

Fs = 10000          # Sampling frequency 10 kHz
Ts = 1/Fs           # Sampling period = 0.1 ms

# Extracting data from matlab file
if pick_direction == 'vertical':
    id_to_bpm = id_to_bpm_y             # Use vertical BPM
    id_to_cm = id_to_cm_y               # Use vertical CM 
    RM = RMy                            # Use vertical response matrix
    aI_Hz = 700                         # Vertical corrector frequency
    #Observer
    Ao = mat_data['Ao_y']               # Observer state matrix
    Bo = mat_data['Bo_y']               # Observer input matrix
    Co = mat_data['Co_y']               # Observer output matrix
    Ad = mat_data['Ad_y']               # Disturbance model matrix
    Cd = mat_data['Cd_y']               # Disturbance output matrix
    #Plant with all BPMs and CMs
    Ap = mat_data['Ap_y']               # Plant state matrix
    Bp = mat_data['Bp_y']               # Plant input matrix
    Cp = mat_data['Cp_y']               # Plant output matrix

    Kfd = mat_data['Kfd_y']             # Observer gain for disturbance feedback
    Kfx = mat_data['Kfx_y']             # Observer gain for state feedback
    P_mpc = mat_data['P_y']             # Terminal cost matrix (LQR)
    Q_mpc = mat_data['Qlqr_y']          # State cost matrix
    R_mpc = mat_data['Rlqr_y']          # Input cost matrix
    #SOFB
else:
    id_to_bpm = id_to_bpm_x
    id_to_cm = id_to_cm_x
    RM = RMx
    aI_Hz = 500
    #Observer
    Ao = mat_data['Ao_x']
    Bo = mat_data['Bo_x']
    Co = mat_data['Co_x']
    Ad = mat_data['Ad_x']
    Cd = mat_data['Cd_x']
    #Plant with all BPMs and CMs
    Ap = mat_data['Ap_x']
    Bp = mat_data['Bp_x']
    Cp = mat_data['Cp_x']
    Kfd = mat_data['Kfd_x']
    Kfx = mat_data['Kfx_x']
    P_mpc = mat_data['P_x']
    Q_mpc = mat_data['Qlqr_x']
    R_mpc = mat_data['Rlqr_x']
    #SOFB

# endregion

# region | using loaded data to set up MPC and FGD (fast gradient descent)
ny = np.size(RM, 0)         # Number of outputs (BPM measurements) = 4
nu = np.size(RM, 1)         # Number of inputs (correctors) = 4
nx = nu                     # Number of states = 4 (Square so state dimension = input dimension)

#Observer
Lxd_obs = Kfd               # Disturbance observer gain
Lx8_obs = Kfx               # State observer gain
S_sp_pinv = np.linalg.pinv(np.block([[np.eye(nx) - Ao, -Bo], [Co, np.zeros((ny, nu))]]))        # Pseudo-inverse for setpoint to state conversion
S_sp_pinv = S_sp_pinv[:,nx:]        # Extract relevant portion

#MPC
horizon = 1
u_rate_scalar = 1*1000
u_rate = np.ones((nu, 1)) * u_rate_scalar
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# If u_max is never reached, then it acts like there is no u_max. Investigate; perhaps change *1000 to something lower.
u_max = hardlimits[id_to_cm] * 100                    # u_max = array([[4999], [4999], [4999], [4999]]) if hardlimits[id_to_cm] * 1000
y_max_scalar = np.inf
y_max = np.ones((id_to_bpm.size, 1)) * y_max_scalar
J_mpc = np.transpose(Bo) @ P_mpc @ Bo + R_mpc           # Hessian matrix (quadratic term) - This is the second-order cost: 0.5 * u^T @ J @ u
S_sp_pinv_x = S_sp_pinv[:nx,:]
S_sp_pinv_u = S_sp_pinv[nx:,:]
q_mat_x0 = np.transpose(Bo) @ P_mpc @ Ao                # Linear cost from state: contributes to 0.5*u^T*J*u + q^T*u
q_mat_xd = (np.hstack((Bo.T @ P_mpc, R_mpc)) @ np.vstack((S_sp_pinv_x, S_sp_pinv_u)) @ Cd)      # Linear cost from disturbance
q_mat = np.hstack((q_mat_x0, q_mat_xd))                 # Combined linear term



#Set up FGM (Fast gradient descent)
beta_fgm = 0

if use_FGM:
    eigmax = np.max(np.linalg.eigvals(J_mpc))
    eigmin = np.min(np.linalg.eigvals(J_mpc))   
    J_mpc = np.eye(J_mpc.shape[0]) - J_mpc / eigmax
    beta_fgm = (np.sqrt(eigmax) - np.sqrt(eigmin)) / (np.sqrt(eigmax) + np.sqrt(eigmin))
    q_mat = np.hstack((q_mat_x0, q_mat_xd)) / eigmax


#Rate limiter on VME processors
if pick_direction == 'vertical':
    mat_data.update(loadmat(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'awrSSy.mat')))        # Load vertical AWR dynamics

else:
    mat_data.update(loadmat(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'awrSSx.mat')))        # Load horizontal AWR dynamics


#SOFB setpoints
SOFB_setp = np.zeros((nu, 1))

# endregion

# region | generate data or simulate once if not generating data
if generate_data:
    
    
    
    
    
    
    
    # Function to check if constraints are hit
    # def check_constraints(u_sim, u_max):
    #     # u_sim: shape (n_steps, n_ctrl)
    #     # u_max: shape (n_ctrl,) or (n_ctrl, 1)
    #     hit_upper = (u_sim >= u_max.T).any(axis=1)  # True if any input hits upper bound at any time
    #     hit_lower = (u_sim <= -u_max.T).any(axis=1) # True if any input hits lower bound at any time
    #     hit_any = hit_upper | hit_lower
    #     n_hits = hit_any.sum()
    #     print(f"Constraint hits: {n_hits} out of {u_sim.shape[0]} steps")
    #     if n_hits > 0:
    #         print("Indices where constraints were hit:", np.where(hit_any)[0])
    #     else:
    #         print("No constraints were hit during simulation.")
    
    
    
    
    
    
    
    
    
    
    #Initialise array of seeds for pertubations
    n_traj = 1000 
    trainseeds = np.linspace(1, n_traj*n_include, n_traj*n_include).astype(int)
    u_mags = np.linspace(1, 100, n_traj*n_include)
    n_tests = n_traj * n_include

    #Storage for training data
    u_sim_train = np.zeros((n_samples * n_tests, n_include))
    xd_obs_train = np.zeros((n_samples * n_tests, n_include))
    x0_obs_train = np.zeros((n_samples * n_tests, n_include))
    y_awr_train = np.zeros((n_samples * n_tests, n_include))
    y_sim_train = np.zeros((n_samples * n_tests , n_include))
    lcon_train = np.zeros((n_samples * n_tests, n_include))
    ucon_train = np.zeros((n_samples * n_tests, n_include))

    k = 0
    n_complete = 0


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~## THIS IS FOR TESTING >>>>>>>>>       Plotting graph of all disturbances used in generating training data [1/3]
    #Collect disturbance traces for plotting (limit to avoid huge memory use)
    # all_doff_rms = []
    # max_plot_seeds = 4000
    # downsample_plot = 10
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~## THIS IS FOR TESTING <<<<<<<<<<<



    for seed, u_mag in zip(trainseeds, u_mags):
        print('[{}/{}]'.format(n_complete, n_tests))
        # Generate random disturbance modes based on seed
        doff = randModes(seed, RM, id_to_bpm, TOT_BPM, u_mag)

        # Generate sinusoid disturbance overlayed onto random disturbances based on seed
        # doff = randModesSinusoid(seed, RM, id_to_bpm, TOT_BPM, u_mag)

        # Generate AWGN disturbance overlayed onto random disturbances based on seed
        # doff = randModesAWGN(seed, RM, id_to_bpm, TOT_BPM, u_mag)

        # Generate Sinusoid with AWGN disturbance
        # doff = randModesSinusoidAWGN(seed, RM, id_to_bpm, TOT_BPM, u_mag)
        



        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~## THIS IS FOR TESTING >>>>>>>>>       Plotting graph of all disturbances used in generating training data [2/3]
        # Store RMS across active BPMs for plotting (limit stored seeds)
        # if len(all_doff_rms) < max_plot_seeds:
        #     # compute RMS across the active BPM indices for this seed over time
        #     rms = np.sqrt(np.mean(np.square(doff[id_to_bpm, :]), axis=0))
        #     all_doff_rms.append(rms)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~## THIS IS FOR TESTING <<<<<<<<<<<



        #Simulation
        endt = n_samples*Ts - Ts
        Lsim = n_samples*Ts
        t = np.arange(0, endt + Ts, Ts)

    
        #Initialise mpc object
        mpc = sim.Mpc(
            n_samples, n_delay, doff,
            Ap, Bp, Cp, 
            Ao, Bo, Co, Ad, Cd, Lx8_obs, Lxd_obs,
            J_mpc, q_mat, y_max,
            u_max, u_rate,
            id_to_bpm, id_to_cm,
            mat_data['A'][:n_include,:n_include], mat_data['B'][:n_include,:n_include], mat_data['C'][:n_include,:n_include], mat_data['D'][:n_include,:n_include],
            SOFB_setp, beta_fgm, use_lqr=use_lqr)

        if use_dagger:
            model = md.NNController(n_state, hidden_size, n_ctrl)
            model.load_state_dict(nnparams)
            #Simulate and store trajectory
            u_sim_expert, x0_obs_dggr, xd_obs_dggr, n_simulated = mpc.sim_dagger(model, device)
            
            try:
                assert(n_simulated < n_samples)
                u_sim_train[k:k+n_simulated, :] = u_sim_expert[:,id_to_cm]
                xd_obs_train[k:k+n_simulated, :] = xd_obs_dggr
                x0_obs_train[k:k+n_simulated, :] = x0_obs_dggr
                # compute AWR outputs for expert control (using reduced AWR matrices)
                A_awr = mat_data['A'][:n_include,:n_include]
                B_awr = mat_data['B'][:n_include,:n_include]
                C_awr = mat_data['C'][:n_include,:n_include]
                D_awr = mat_data['D'][:n_include,:n_include]
                # u_sim_expert is (n_steps, nu)
                u_local = u_sim_expert[:, id_to_cm]
                x_awr = np.zeros((A_awr.shape[0], 1))
                y_awr_seq = np.zeros((n_simulated, A_awr.shape[0]))
                for jj in range(n_simulated):
                    uvec = u_local[jj, :][:, np.newaxis]
                    x_awr = A_awr @ x_awr + B_awr @ uvec
                    yvec = C_awr @ x_awr + D_awr @ uvec
                    y_awr_seq[jj, :] = yvec.flatten()
                y_awr_train[k:k+n_simulated, :] = y_awr_seq
                # compute lower/upper constraints per sample
                u_max_local = u_max.flatten()[id_to_cm]
                sofb_local = SOFB_setp.flatten()[id_to_cm]
                u_rate_local = u_rate.flatten()[id_to_cm]
                l_seq = np.maximum(-u_max_local - sofb_local, -u_rate_local + y_awr_seq)
                u_seq = np.minimum(u_max_local - sofb_local, u_rate_local + y_awr_seq)
                lcon_train[k:k+n_simulated, :] = l_seq
                ucon_train[k:k+n_simulated, :] = u_seq
            except AssertionError as e:
                print("Did not converge")
                k -= n_simulated
        else:
            #Simulate and store trajectory
            y_sim_fgm ,u_sim_fgm, x0_obs_fgm, xd_obs_fgm, _ , n_simulated = mpc.sim_mpc(use_FGM)
            #Check that mpc has converged
            try:
                # assert(n_simulated < n_samples) # Steady state check is not valid for dynamic (sinusoidal) disturbances
                u_sim_train[k:k+n_simulated, :] = u_sim_fgm[:,id_to_cm]
                xd_obs_train[k:k+n_simulated, :] = xd_obs_fgm
                x0_obs_train[k:k+n_simulated, :] = x0_obs_fgm
                y_sim_train[k:k+n_simulated, :] = y_sim_fgm[:,id_to_bpm]
                # compute AWR outputs for FGM solution
                A_awr = mat_data['A'][:n_include,:n_include]
                B_awr = mat_data['B'][:n_include,:n_include]
                C_awr = mat_data['C'][:n_include,:n_include]
                D_awr = mat_data['D'][:n_include,:n_include]
                # u_sim_fgm is (n_steps, nu)
                u_local = u_sim_fgm[:, id_to_cm]
                x_awr = np.zeros((A_awr.shape[0], 1))
                y_awr_seq = np.zeros((n_simulated, A_awr.shape[0]))
                for jj in range(n_simulated):
                    uvec = u_local[jj, :][:, np.newaxis]
                    x_awr = A_awr @ x_awr + B_awr @ uvec
                    yvec = C_awr @ x_awr + D_awr @ uvec
                    y_awr_seq[jj, :] = yvec.flatten()
                y_awr_train[k:k+n_simulated, :] = y_awr_seq
                u_max_local = u_max.flatten()[id_to_cm]
                sofb_local = SOFB_setp.flatten()[id_to_cm]
                u_rate_local = u_rate.flatten()[id_to_cm]
                l_seq = np.maximum(-u_max_local - sofb_local, -u_rate_local + y_awr_seq)
                u_seq = np.minimum(u_max_local - sofb_local, u_rate_local + y_awr_seq)
                lcon_train[k:k+n_simulated, :] = l_seq
                ucon_train[k:k+n_simulated, :] = u_seq
            except AssertionError as e:
                print("Did not converge")
                k -= n_simulated

        k += n_simulated
        n_complete += 1

    u_sim_train = u_sim_train[:k, :]
    xd_obs_train = xd_obs_train[:k, :]
    x0_obs_train = x0_obs_train[:k, :]
    # Use project-relative data directory so saves don't depend on CWD


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~## THIS IS FOR TESTING >>>>>>>>>        Plotting graph of all disturbances used in generating training data [3/3]
    # Create and save overlay plot of collected disturbances plus a histogram showing frequency
    # if len(all_doff_rms) > 0:
    #     import matplotlib.pyplot as plt
    #     import matplotlib.cm as cm
    #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4), gridspec_kw={'width_ratios':[3,1]})
    #     t = np.arange(0, n_samples, downsample_plot) * Ts
    #     n_traces = len(all_doff_rms)
    #     cmap = cm.get_cmap('autumn')  # red -> yellow
    #     colors = cmap(np.linspace(0, 1, n_traces))
    #     for i, rms in enumerate(all_doff_rms):
    #         ax1.plot(t, rms[::downsample_plot], color=colors[i], alpha=0.9, linewidth=0.6)
    #     ax1.set_title(f'Disturbance RMS overlays (first {n_traces} seeds)')
    #     ax1.set_xlabel('Time (s)')
    #     ax1.set_ylabel('RMS disturbance (BPM units)')
    #     ax1.grid(True)

    #     # Histogram of RMS values to show frequency (downsampled across time to keep size reasonable)
    #     flat_vals = np.hstack([rms[::downsample_plot] for rms in all_doff_rms])
    #     # Choose narrow bins across observed range
    #     if flat_vals.size > 0:
    #         vmin, vmax = np.min(flat_vals), np.max(flat_vals)
    #         if np.isclose(vmin, vmax):
    #             bins = 50
    #         else:
    #             bins = np.linspace(vmin, vmax, 200)
    #         ax2.hist(flat_vals, bins=bins, color='C0', alpha=0.8)
    #     ax2.set_title('Frequency of RMS values')
    #     ax2.set_xlabel('RMS disturbance')
    #     ax2.set_ylabel('Counts')

    #     # Add a colorbar as a key from red (first seed) to yellow (last seed)
    #     sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=1, vmax=n_traces))
    #     sm.set_array([])
    #     cbar = plt.colorbar(sm, ax=ax1, pad=0.02)
    #     cbar.set_label('Seed index (red = first → yellow = last)')

    #     out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'disturbances_overlay.png')
    #     plt.tight_layout()
    #     plt.savefig(out_path)
    #     plt.close(fig)
    #     print('Saved disturbance overlay + histogram plot to:', out_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~## THIS IS FOR TESTING <<<<<<<<<<<



    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    print('Saving processed data to:', data_dir)
    # Save processed data with augmented inputs: include AWR, SOFB setpoint, linear cost q, and per-sample constraints

    # Trim preallocated constraint arrays to the actual collected length `k` before saving
    l_trim = lcon_train[:k, :]
    u_trim = ucon_train[:k, :]
    x0_trim = x0_obs_train[:k, :]
    xd_trim = xd_obs_train[:k, :]
    u_trim_sim = u_sim_train[:k, :]

    x_aug = np.hstack([x0_trim, xd_trim])
    q_trim = x_aug @ q_mat.T
    
    process.process_data_shuff(x0_trim, xd_trim, u_trim_sim, data_dir, use_dagger,
                              q=q_trim, l_constr=l_trim, u_constr=u_trim)

    # Also save J_mpc and q_vec so training can load them directly
    try:
        np.save(os.path.join(data_dir, 'J_mpc.npy'), J_mpc)
    except Exception:
        pass


else:   # If not training simulate once for comparison or evaluation
    comaprison_seed = 4200     # need to set a random seed for generating disturbances. This is used only here, to compare NN to MPC controller. It would be good if this seed is not one of the seeds used during training, which are defined by np.linspace(1, n_traj*n_include, n_traj*n_include).astype(int)
    comparison_u_mag = 100
    # Generate constant non-zero disturbance for evaluation
    doff = randModes(comaprison_seed, RM, id_to_bpm, TOT_BPM, comparison_u_mag)  # This is not a true random disturbance, because we set the seed. Seed it set to 4200

    # Generate sinusoid disturbance overlayed onto random disturbances based on seed
    # doff = randModesSinusoid(comaprison_seed, RM, id_to_bpm, TOT_BPM, comparison_u_mag)

    # Generate AWGN disturbance overlayed onto random disturbances based on seed
    # doff = randModesAWGN(comaprison_seed, RM, id_to_bpm, TOT_BPM, comparison_u_mag)
    
    # Generate Sinusoid with AWGN disturbance
    # doff = randModesSinusoidAWGN(comaprison_seed, RM, id_to_bpm, TOT_BPM, comparison_u_mag)


    #Simulation
    endt = n_samples*Ts - Ts
    Lsim = n_samples*Ts
    t = np.arange(0, endt + Ts, Ts)
    #Initialise mpc
    mpc = sim.Mpc(
        n_samples, n_delay, doff,
        Ap, Bp, Cp, 
        Ao, Bo, Co, Ad, Cd, Lx8_obs, Lxd_obs,
        J_mpc, q_mat, y_max,
        u_max, u_rate,
        id_to_bpm, id_to_cm,
        mat_data['A'][:n_include,:n_include], mat_data['B'][:n_include,:n_include], mat_data['C'][:n_include,:n_include], mat_data['D'][:n_include,:n_include],
        SOFB_setp, beta_fgm, use_lqr = use_lqr)

    #Simulate and store trajectory

    y_sim_fgm ,u_sim_fgm, x0_obs_fgm, xd_obs_fgm,x_sim, n_simulated = mpc.sim_mpc(use_FGM)


# endregion


# region | comparing to NN
if compare:
    #Initialise mpc 
    mpc = sim.Mpc(
        n_samples, n_delay, doff,
        Ap, Bp, Cp, 
        Ao, Bo, Co, Ad, Cd, Lx8_obs, Lxd_obs,
        J_mpc, q_mat, y_max,
        u_max, u_rate,
        id_to_bpm, id_to_cm,
        mat_data['A'][:n_include,:n_include], mat_data['B'][:n_include,:n_include], mat_data['C'][:n_include,:n_include], mat_data['D'][:n_include,:n_include],
        SOFB_setp, beta_fgm, use_lqr=use_lqr)
    #Load nn model
    model = md.LinearController(n_state,n_ctrl)
    model.load_state_dict(nnparams)
    #Simulate using nn model
    y_sim_nn ,u_sim_nn, x0_obs_nn, xd_obs_nn = mpc.sim_nn(model, device)
    y_nn_longterm = y_sim_nn[:, id_to_bpm]
    y_sim_nn = y_sim_nn[:n_simulated, :]
    u_sim_nn = u_sim_nn[:n_simulated, :]
    x0_obs_nn = x0_obs_nn[:n_simulated, :]
    xd_obs_nn = xd_obs_nn[:n_simulated, :]

    y_err = (y_sim_fgm - y_sim_nn) 
    u_err = (u_sim_fgm - u_sim_nn) 


# endregion



# region | Plotting

if not use_dagger:
    loss_data = np.load(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'model', 'modelloss.npz'))
    y_plt_fgm = y_sim_fgm[:, id_to_bpm]
    u_plt_fgm = u_sim_fgm[:, id_to_cm]
    if compare:
        y_plt_nn = y_sim_nn[:, id_to_bpm]
        u_plt_nn = u_sim_nn[:, id_to_cm]
        u_plt_err = u_err[:, id_to_cm]
        y_plt_err = y_err[:, id_to_bpm]

    scale_u = 0.001

    fig, axs = plt.subplots(2, 4, figsize=(15, 8))

    # Subplot 1: Disturbance
    axs[0, 0].plot(doff[id_to_bpm, :].T)
    axs[0, 0].set_title('Disturbance')


    # Subplot 2: Input
    axs[0, 1].plot(u_plt_fgm * scale_u, linestyle='-')  # solid line for u_sim_fgm
    if compare:
        axs[0, 1].plot(u_plt_nn * scale_u, linestyle='--')  # dashed line for u_sim_nn
    axs[0, 1].set_title('Input')



    start_index = int(n_simulated/4)
    n_plt = np.linspace(0, n_simulated, n_simulated)

    # # Subplot 3: % nn Steady State
    if compare:
        # axs[0, 2].plot(n_plt[start_index:],y_plt_nn[start_index:])  # dashed line for y_sim_nn
        axs[0, 2].plot(n_plt[start_index:],y_plt_nn[start_index:], linestyle='--')
        # axs[0, 2].plot(y_nn_longterm, linestyle='-.')
        axs[0, 2].set_title('NN Steady State Output')

    #Subplot 4: MPC Steady State Output
    
    axs[0, 3].plot(n_plt[start_index:],y_plt_fgm[start_index:])  # solid line for y_sim_fgm
    
    axs[0, 3].set_title('MPC Steady State Output')




    # Subplot 5: Output
    axs[1, 0].plot(y_plt_fgm, linestyle='-')  # solid line for y_sim_fgm
    if compare:
        axs[1, 0].plot(y_plt_nn, linestyle='--')  # dashed line for y_sim_nn
    axs[1, 0].set_title('Output')


    # Subplot 6: % Error in Output
    if compare:
        axs[1, 1].plot(y_plt_err, linestyle='-')  # solid line for y_sim_fgm
    axs[1, 1].set_title('Output Error')

    #Subplot 7 : Training Loss
    if compare:
        axs[1, 2].plot(loss_data['epochs'], loss_data['train_losses'])
        axs[1, 2].set_xlabel('Epoch')
        axs[1, 2].set_ylabel('Loss')
        axs[1, 2].set_title('Training Loss')
        
    #Subplot 8: Validation Loss (Only plot once model has settled)
    if compare:
        axs[1, 3].plot(loss_data['epochs'][10:], loss_data['val_losses'][10:])
        axs[1, 3].set_xlabel('Epoch')
        axs[1, 3].set_ylabel('Loss')
        axs[1, 3].set_title('Validation Loss')

        #Subplot 8: Validation Loss (Plot all epochs (good if #epochs is small ie <10))
    if compare:
        axs[1, 3].plot(loss_data['epochs'], loss_data['val_losses'])
        axs[1, 3].set_xlabel('Epoch')
        axs[1, 3].set_ylabel('Loss')
        axs[1, 3].set_title('Validation Loss')




    
    print("Time taken: ", time.time() - start)
    # # Adjust layout for better spacing
    plt.tight_layout()
    plt.savefig('output.png')
    
# endregion







    







