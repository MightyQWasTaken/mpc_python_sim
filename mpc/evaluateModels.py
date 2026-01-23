import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import torch
from torch import nn
from torch.autograd import Function, Variable
from torch.nn.parameter import Parameter
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import numpy.random as npr

import os
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

# Script to evaluate performance of models across the various disturbance modes in the Diamond Light Source storage ring



start = time.time()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# options
fname_RM = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'orms', 'GoldenBPMResp_DIAD.mat')
pick_dir = 1
dirs = ['horizontal','vertical']
pick_direction = dirs[pick_dir]
sim_IMC = False
use_FGM = True

#Toggle for LQR limits
use_lqr = True

#Hardlimits
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

#first n_include BPMs and CMs active for testing

n_include = 4


id_to_bpm_x = id_to_bpm_x[:n_include]
id_to_cm_x = id_to_cm_x[:n_include]
id_to_bpm_y = id_to_bpm_y[:n_include]
id_to_cm_y = id_to_cm_y[:n_include]

RMx = RMorigx[np.ix_(id_to_bpm_x, id_to_cm_x)]
RMy = RMorigy[np.ix_(id_to_bpm_y, id_to_cm_y)]



fname = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'systems', '4statesystemnd8.mat')
#OnlyValidforND8
mat_data = loadmat(fname)

#Observer and Regulator
n_delay = 8



Fs = 10000
Ts = 1/Fs


if pick_direction == 'vertical':
    id_to_bpm = id_to_bpm_y
    id_to_cm = id_to_cm_y
    RM = RMy
    aI_Hz = 700
    #Observer
    Ao = mat_data['Ao_y']
    Bo = mat_data['Bo_y']
    Co = mat_data['Co_y']
    Ad = mat_data['Ad_y']
    Cd = mat_data['Cd_y']
    #Plant with all BPMs and CMs
    Ap = mat_data['Ap_y']
    Bp = mat_data['Bp_y']
    Cp = mat_data['Cp_y']
    Kfd = mat_data['Kfd_y'] #Observer gain for disturbance
    Kfx = mat_data['Kfx_y'] #Observer gain for state
    P_mpc = mat_data['P_y'] #Terminal cost
    Q_mpc = mat_data['Qlqr_y'] #State weighting
    R_mpc = mat_data['Rlqr_y'] #Input weighting
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

ny = np.size(RM, 0)
nu = np.size(RM, 1)
nx = nu

#Observer
Lxd_obs = Kfd
Lx8_obs = Kfx
S_sp_pinv = np.linalg.pinv(np.block([[np.eye(nx) - Ao, -Bo], [Co, np.zeros((ny, nu))]]))
S_sp_pinv = S_sp_pinv[:,nx:]

#MPC
horizon = 1
u_rate_scalar = 1*1000
u_rate = np.ones((nu, 1)) * u_rate_scalar
u_max = hardlimits[id_to_cm] * 1000
y_max_scalar = np.inf
y_max = np.ones((id_to_bpm.size, 1)) * y_max_scalar
J_mpc = np.transpose(Bo) @ P_mpc @ Bo + R_mpc
S_sp_pinv_x = S_sp_pinv[:nx,:]
S_sp_pinv_u = S_sp_pinv[nx:,:]
q_mat_x0 = np.transpose(Bo) @ P_mpc @ Ao
q_mat_xd = (np.hstack((Bo.T @ P_mpc, R_mpc)) @ np.vstack((S_sp_pinv_x, S_sp_pinv_u)) @ Cd)
q_mat = np.hstack((q_mat_x0, q_mat_xd))

#Set up FGM
beta_fgm = 0

if use_FGM:
    eigmax = np.max(np.linalg.eigvals(J_mpc))
    eigmin = np.min(np.linalg.eigvals(J_mpc))   
    J_mpc = np.eye(J_mpc.shape[0]) - J_mpc / eigmax
    beta_fgm = (np.sqrt(eigmax) - np.sqrt(eigmin)) / (np.sqrt(eigmax) + np.sqrt(eigmin))
    q_mat = np.hstack((q_mat_x0, q_mat_xd)) / eigmax


#Rate limiter on VME processors
if pick_direction == 'vertical':
    mat_data.update(loadmat(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'awrSSy.mat')))

else:
    mat_data.update(loadmat(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'awrSSx.mat')))


n_samples = 15000

UR,SR,VR = np.linalg.svd(RM)
mag_u = 1
SOFB_setp = np.zeros((nu, 1))

n_tests = 4
u_sim_train = np.zeros((n_samples * n_tests, n_include))
xd_obs_train = np.zeros((n_samples * n_tests, n_include))
x0_obs_train = np.zeros((n_samples * n_tests, n_include))


k = 0
for imode in range(1,5):
    tmp = UR[:, imode - 1] * SR[imode - 1] * mag_u
    doff_tmp = np.zeros((TOT_BPM, 1))
    doff_tmp[id_to_bpm] = tmp[:, np.newaxis]
    doff = doff_tmp * np.ones((1,n_samples))

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
        SOFB_setp, beta_fgm, use_lqr=use_lqr)

    #Simulate and store trajectory
    y_sim_fgm ,u_sim_fgm, x0_obs_fgm, xd_obs_fgm, _ , n_simulated = mpc.sim_mpc(use_FGM)
    #Check that mpc has converged
    try:
        assert(n_simulated < n_samples)
        u_sim_train[k:k+n_simulated, :] = u_sim_fgm[:,id_to_cm]
        xd_obs_train[k:k+n_simulated, :] = xd_obs_fgm
        x0_obs_train[k:k+n_simulated, :] = x0_obs_fgm

    except AssertionError as e:
        print("Did not converge")
        k -= n_simulated
    
    k += n_simulated

u_sim_train = u_sim_train[:k, :]
xd_obs_train = xd_obs_train[:k, :]
x0_obs_train = x0_obs_train[:k, :]

udata = u_sim_train
xdata = np.hstack((x0_obs_train,xd_obs_train))




def evaluateModel(model, udata, xdata):
    model.eval()
    losses = []
    for rowindex in range(np.size(udata[1])):
        xgt = torch.tensor(xdata[rowindex,:]).float()
        ugt = torch.tensor(udata[rowindex,:]).float()
        upred = model(xgt)
        loss = torch.norm(upred - ugt)
        losses.append(loss)
    return torch.mean(torch.tensor(losses, device=device))

# Arbitary testing of various models 
root = "/home/adam/mpc_python_sim"

model = md.LinearController(8,4)
model.load_state_dict(torch.load(os.path.join(root, "data/model/model.ckpt")))

loss = evaluateModel(model, udata, xdata).cpu().detach().numpy()


# model.load_state_dict(torch.load(os.path.join(root, "500/model.ckpt")))

# loss500 = evaluateModel(model, udata, xdata).cpu().detach().numpy()


# model.load_state_dict(torch.load(os.path.join(root, "1000/model.ckpt")))

# loss1000 = evaluateModel(model, udata, xdata).cpu().detach().numpy()


# model.load_state_dict(torch.load(os.path.join(root, "2000/model.ckpt")))

# loss2000 = evaluateModel(model, udata, xdata).cpu().detach().numpy()


# model.load_state_dict(torch.load(os.path.join(root, "10000/model.ckpt")))

# loss10000 = evaluateModel(model, udata, xdata).cpu().detach().numpy()


# model.load_state_dict(torch.load(os.path.join(root, "100000/model.ckpt")))

# loss100000 = evaluateModel(model, udata, xdata).cpu().detach().numpy()


# model.load_state_dict(torch.load(os.path.join(root, "1000000/model.ckpt")))

# loss1000000 = evaluateModel(model, udata, xdata).cpu().detach().numpy()


samples = [100]
logsamples = np.log(samples)
lossdata = [loss]
labels = ["100 Samples"]

plt.bar(labels, lossdata)
plt.ylabel("Mean L2 Loss Across Dataset")
plt.show()




