import osqp
import numpy as np
import scipy.sparse as sparse
import torch.nn as nn
import torch
import torch.nn.functional as F
import random as rand
# Contains the MPC class responsible to running the simulation with a variety of controllers such as MPC, LQR and Neural Networks

# Takes parameters:
# n_samples: Number of samples to simulate
# n_delay: Delay in the system
# dist: Disturbance in the system
# Ap: Plant A matrix
# Bp: Plant B matrix
# Cp: Plant C matrix
# Ao: Observer A matrix
# Bo: Observer B matrix
# Co: Observer C matrix
# Ad: Observer A matrix
# Cd: Observer C matrix
# LxN_obs: Observer gain for xN
# Lxd_obs: Observer gain for xd
# J_MPC: Cost function matrix
# q_mat: Cost function vector
# y_max: Output constraints
# u_max: Input constraints
# u_rate: Input rate constraints
# id_to_bpm: Contains indices of active BPMs
# id_to_cm: Contains indices of active CMs
# A_awr: A matrix for AWR
# B_awr: B matrix for AWR
# C_awr: C matrix for AWR
# D_awr: D matrix for AWR
# SOFB_setp: Setpoint for SOFB
# beta_FGM: Beta value for fast gradient method
# ol_mode: Open loop mode
# dtype: Data type
# use_lqr: Use LQR controller, defaults to false

# Methods usable in the class:

# sim_mpc: 
# Simulates with MPC controller, either using OSQP or FGM to solve the quadratic program, will exit if steady state is reached
#   Inputs: use_fgm: Boolean to use FGM, if not then uses OSQP
#   Outputs: y_sim, u_sim, x0_obs, xd_obs, x_sim, n_samples (How many samples were simulated until SS)



# sim_nn: Simulates with Neural Network controller, will not exit early if steady state is reached
#   Inputs: model: Neural Network model, device: Device to run the model on
#   Outputs: y_sim, u_sim, x0_obs, xd_obs


# sim_dagger: Simulates with Neural Network controller but records expert action, exits early if steady state is reached, uses FGM
#   Inputs: model: Neural Network model, device: Device to run the model on
#  Outputs: u_sim_expert, x0_obs, xd_obs, n_samples (How many samples were simulated until SS)


class Mpc:
    def __init__(
        self, n_samples, n_delay, dist,
        Ap, Bp, Cp,
        Ao, Bo, Co, Ad, Cd, LxN_obs, Lxd_obs,
        J_MPC, q_mat, y_max,
        u_max, u_rate,
        id_to_bpm, id_to_cm,
        A_awr, B_awr, C_awr, D_awr,
        SOFB_setp,beta_FGM,
        ol_mode = False, dtype = np.float64, use_lqr = False):

        #Initialise solver parameters
        self.use_lqr = use_lqr
        self.MAX_ITER  = 20
        self.n_samples = n_samples
        self.n_delay = n_delay
        self.dist = dist
        self.dtype = dtype
        self.Ap = Ap.astype(dtype)
        self.Bp = Bp.astype(dtype)
        self.Cp = Cp.astype(dtype)
        self.Ao = Ao.astype(dtype)
        self.Bo = Bo.astype(dtype)
        self.Co = Co.astype(dtype)
        self.Ad = Ad.astype(dtype)
        self.Cd = Cd.astype(dtype)
        self.LxN_obs = LxN_obs.astype(dtype)
        self.Lxd_obs = Lxd_obs.astype(dtype)
        self.J_FGM = J_MPC.astype(dtype)
        self.J = sparse.csc_matrix(J_MPC, dtype=dtype)
        self.q_mat = q_mat.astype(dtype)
        self.beta_fgm = beta_FGM
        self.y_max = y_max.astype(dtype)
        self.u_max = u_max.astype(dtype)
        self.u_rate = u_rate.astype(dtype)
        self.id_to_bpm = id_to_bpm
        self.id_to_cm = id_to_cm
        self.A_awr = A_awr.astype(dtype)
        self.B_awr = B_awr.astype(dtype)
        self.C_awr = C_awr.astype(dtype)
        self.D_awr = D_awr.astype(dtype)
        self.SOFB_setp = SOFB_setp
        self.ol_mode = ol_mode



        #extract dimensions
        self.nx_plant, self.nu_plant = np.size(Bp,0) , np.size(Bp,1)
        self.ny_plant = np.size(Cp,0)
        self.nx_obs, self.nu_obs = np.size(Bo,0) , np.size(Bo,1)
        self.ny_obs = np.size(Co,0)

        #Variables for plant
        np.random.seed(42)
        self.x_sim_new = np.zeros((self.nx_plant, 1))
        self.x_sim_old = np.zeros((self.nx_plant, 1))
        self.y_sim = np.zeros((self.ny_plant, n_samples))
        self.u_sim = np.zeros((self.nu_plant, n_samples))
        self.x_sim = np.zeros((self.nu_plant, n_samples))
        self.u_sim_expert = np.zeros((self.nu_plant, n_samples))

        #Variables for AWR
        self.ny_awr , self.nx_awr = np.size(C_awr,0) , np.size(C_awr,1)
        self.x_awr_new = np.zeros((self.nx_awr, 1))
        self.y_awr = np.zeros((self.ny_awr, 1))

        #Variables for observer
        self.x_obs_old = np.zeros((self.nx_obs, n_delay + 1), dtype=dtype)
        self.x_obs_new = np.zeros((self.nx_obs, n_delay + 1), dtype=dtype)
        self.xd_obs_old = np.zeros((self.ny_obs, 1), dtype=dtype)
        self.x0_obs = np.zeros((self.nx_obs, n_samples), dtype=dtype)
        self.xd_obs = np.zeros((self.ny_obs, n_samples), dtype=dtype)
        self.Apow = np.zeros((self.nx_obs*(n_delay + 1), self.nx_obs), dtype=dtype)
        #FLAG
        #Compute elementwise powers in advance
        for i in range(n_delay + 1):
            self.Apow[i * self.nx_obs:(i+1) * self.nx_obs, :] = np.power(Ao, i)

        
        
        self.y_mat = (self.Co @ self.Ao).astype(dtype)

        #Setup constraints
        #OSQP A Matrix
        self.A_constr = sparse.csr_matrix(np.vstack((np.eye(self.ny_obs), Co @ Bo)), dtype=dtype)

        self.l_constr = np.vstack(( 
            np.maximum(-u_max - SOFB_setp, -u_rate + self.y_awr),
            -y_max - self.y_mat @ self.x_obs_new[:,0:1]
        ))

        self.u_constr = np.vstack((
            np.minimum(u_max - SOFB_setp, u_rate + self.y_awr),
            y_max - self.y_mat @ self.x_obs_new[:,0:1]
        ))

        if self.use_lqr:
            self.u_constr = np.ones((self.nu_obs,1)) * np.inf
            self.l_constr = np.ones((self.nu_obs,1)) * -np.inf

        #Variables for FGM
        self.z_new = np.zeros((self.nu_obs, 1), dtype=dtype)   


    #Initialises OSQP solver
    def setupOSQP(self, verbose,
        polish,
        adaptive_rho):
        
        settings = {
            'verbose': verbose,
            'polish': polish,
            'adaptive_rho': adaptive_rho,
            'max_iter': self.MAX_ITER,
            'check_termination': self.MAX_ITER
        }

        self.osqp_solver = osqp.OSQP()
        self.osqp_solver.setup(P=self.J, q=np.zeros(np.size(self.q_mat, 0)),
                       A=self.A_constr, l=self.l_constr, u=self.u_constr, **settings)


    def update_y_sim(self, k):
        if self.ol_mode:
            self.y_sim[:, k:k+1] = self.dist[:, k:k+1]
        else:
            self.y_sim[:, k:k+1] = (self.Cp @ self.x_sim_new) + self.dist[:, k:k+1]

    def update_y_meas(self, k):
        self.y_meas = self.y_sim[self.id_to_bpm, k-self.n_delay][:, np.newaxis].astype(self.dtype)

    def calc_obs_state(self, k):
        self.x_obs_new[:,0:1] = (self.Ao @ self.x_obs_old[:,0:1]) + (self.Bo @ self.u_sim[self.id_to_cm, k-1][:, np.newaxis].astype(self.dtype))
        for i in range(1, self.n_delay + 1):
            self.x_obs_new[:,i:i+1] = self.x_obs_old[:,i-1:i]
        
        self.xd_obs_new = self.Ad @ self.xd_obs_old
        

    def update_obs_measurement(self):
        delta_y = self.y_meas - self.Co @ self.x_obs_new[:,self.n_delay:self.n_delay+1] - self.Cd @ self.xd_obs_new
        delta_xN = self.LxN_obs @ delta_y
        delta_xd = self.Lxd_obs @ delta_y
        self.xd_obs_new = self.xd_obs_new + delta_xd
        self.x_obs_new = self.x_obs_new +  np.fliplr(np.reshape(self.Apow @ delta_xN, (self.nx_obs, self.n_delay+1)))

    def update_obs_state(self,k):
        self.xd_obs_old = self.xd_obs_new
        for i in range(self.n_delay + 1):
            self.x_obs_old[:,i:i+1] = self.x_obs_new[:,i:i+1]

        self.x0_obs[:,k:k+1] = self.x_obs_new[:,0:1]
        self.xd_obs[:,k:k+1] = self.xd_obs_new

    def update_q_limits(self):
        ####
        self.q = self.q_mat @ np.vstack((self.x_obs_new[:,0:1], self.xd_obs_new))
        if not self.use_lqr:
            if self.use_FGM:
                self.l_constr = np.maximum(-self.u_max - self.SOFB_setp, -self.u_rate + self.y_awr)
                self.u_constr = np.minimum(self.u_max - self.SOFB_setp, self.u_rate + self.y_awr)
                assert np.any(self.l_constr <= self.u_constr), "Lower constraint is greater than upper constraint"

            else:
            
                if self.dtype == np.float32:

                    self.l_constr = np.vstack((
                        np.maximum(-self.u_max - self.SOFB_setp.astype(self.dtype), -self.u_rate + self.y_awr.astype(self.dtype)),
                        -self.y_max - self.y_mat @ self.x_obs_new[:,0:1].astype(self.dtype)
                    ))
                    self.u_constr = np.vstack((
                        np.minimum(self.u_max - self.SOFB_setp.astype(self.dtype), self.u_rate + self.y_awr.astype(self.dtype)),
                        self.y_max - self.y_mat @ self.x_obs_new[:,0:1].astype(self.dtype)
                    ))
                else:
                    self.l_constr = np.vstack((
                        np.maximum(-self.u_max - self.SOFB_setp.astype(self.dtype), -self.u_rate + self.y_awr.astype(self.dtype)),
                        -self.y_max - self.y_mat @ self.x_obs_new[:,0:1].astype(self.dtype) - self.xd_obs_new.astype(self.dtype)
                    ))
                    self.u_constr = np.vstack((
                        np.minimum(self.u_max - self.SOFB_setp.astype(self.dtype), self.u_rate + self.y_awr.astype(self.dtype)),
                        self.y_max - self.y_mat @ self.x_obs_new[:,0:1].astype(self.dtype) - self.xd_obs_new.astype(self.dtype)
                    ))
        




    #Solves the problem using OSQP and updates the AWR state
    def solveOSQP_update_awr(self,k):
        self.osqp_solver.update(q=self.q, l=self.l_constr, u=self.u_constr)
        result = self.osqp_solver.solve()
        osqp_result = result.x[:self.nu_obs][:, np.newaxis]
        assert result.info.status_val in [1, 2], "OSQP solver did not find an optimal solution."
        self.u_sim[self.id_to_cm, k] = osqp_result.flatten()

        self.x_awr_old = self.x_awr_new
        self.x_awr_new = self.A_awr @ self.x_awr_old + self.B_awr @ osqp_result.astype(np.float64)
        self.y_awr = self.C_awr @ self.x_awr_new + self.D_awr @ osqp_result.astype(np.float64)



    def update_plant_state(self,k):
        #Make temp variable project u_sim 
        # temp = np.maximum(self.l_constr, np.minimum(self.u_constr, self.u_sim[:, k:k+1]))
        self.x_sim_old = self.x_sim_new
        self.x_sim_new = self.Ap @ self.x_sim_old + self.Bp @ self.u_sim[:, k:k+1]
        self.x_sim[:,k:k+1] = self.x_sim_old
        


    #Solves the problem using FGM and updates the AWR state
    def solveFGM_update_awr(self, k):

        out_global = self.u_sim[self.id_to_cm, k-1][:, np.newaxis]
        for i in range(self.MAX_ITER):
            self.z_old = self.z_new
            t = self.J_FGM @ out_global - self.q
            self.z_new = np.maximum(self.l_constr, np.minimum(self.u_constr, t))
            if np.any(self.z_new != t):
                print("FGM Controller is saturating")
            out_global = (1+self.beta_fgm) * self.z_new - self.beta_fgm * self.z_old
        fgm_result = self.z_new
        self.u_sim[self.id_to_cm, k] = fgm_result.flatten()

        self.x_awr_old = self.x_awr_new
        self.x_awr_new = self.A_awr @ self.x_awr_old + self.B_awr @ fgm_result.astype(np.float64)
        self.y_awr = self.C_awr @ self.x_awr_new + self.D_awr @ fgm_result.astype(np.float64)

    #Function to simulate the MPC
    def checkSS(self,k,tol):
        if np.all(abs(self.u_sim[self.id_to_cm,k] - self.u_sim[self.id_to_cm,k-1]) < tol):
            self.ss_count += 1
        else:
            self.ss_count = 0
        if self.ss_count > 10:
            print("Steady state reached")
            return True
        else:
            return False

    def terminateSS(self,k, use_dagger = False):
        self.u_sim = self.u_sim[:,0:k]
        self.y_sim = self.y_sim[:,0:k]
        self.x0_obs = self.x0_obs[:,0:k]
        self.xd_obs = self.xd_obs[:,0:k]
        if use_dagger:
            self.u_sim_expert = self.u_sim_expert[:,0:k]
        self.n_samples = k


    def sim_mpc(self, use_fgm):
        self.use_FGM = use_fgm
        self.ss_count = 0
        if not self.use_FGM:
            self.setupOSQP(False, False, False)

        for k in range(self.n_samples):
            #Measurements
            if k % 100 == 0:
                print(f"Simulation progress: {k/self.n_samples*100:.2f}%")

            self.update_y_sim(k)
            if k >= self.n_delay:
                self.update_y_meas(k)
                self.calc_obs_state(k)
                self.update_obs_measurement()
                self.update_obs_state(k)
                self.update_q_limits() 

                if use_fgm:
                    self.solveFGM_update_awr(k)

                else:
                    self.solveOSQP_update_awr(k)
            self.update_plant_state(k)
            if self.checkSS(k, 1e-4):
                self.terminateSS(k)
                break

        self.u_sim = np.transpose(self.u_sim)
        self.y_sim = np.transpose(self.y_sim)
        self.x0_obs = np.transpose(self.x0_obs)
        self.xd_obs = np.transpose(self.xd_obs)
        self.x_sim = np.transpose(self.x_sim)
        return self.y_sim, self.u_sim, self.x0_obs, self.xd_obs, self.x_sim, self.n_samples
    

    #Solves the MPC problem using nn
    def solvenn(self,k):
        # Build augmented input to match training data layout:
        # [x0_obs; xd_obs; q_vec; lower_control_constraints; upper_control_constraints]
        # x_obs_new[:,0:1] and xd_obs_new are column vectors
        x_state = np.vstack((self.x_obs_new[:,0:1], self.xd_obs_new))  # shape (n_state, 1)

        # q_vec needs to be the dynamic q(x) term: q = q_mat @ x_aug
        # x_aug for q calculation is typically [x0; xd]
        q_vec = self.q_mat @ x_state

        # Extract control-only lower/upper constraints (first nu rows correspond to control limits)
        try:
            l_ctrl = np.asarray(self.l_constr[:self.nu_obs]).reshape(self.nu_obs, 1)
            u_ctrl = np.asarray(self.u_constr[:self.nu_obs]).reshape(self.nu_obs, 1)
        except Exception:
            # Fallback: if shapes are unexpected, zero-pad to control size
            l_ctrl = np.zeros((self.nu_obs, 1))
            u_ctrl = np.zeros((self.nu_obs, 1))

        # Stack all pieces
        x_aug_np = np.vstack((x_state, q_vec, l_ctrl, u_ctrl))
        x_aug = torch.tensor(x_aug_np.T).float().to(self.device)
        # Run NN controller
        self.u_nn = self.nn_controller(x_aug).detach().cpu().numpy()
        self.u_nn = np.transpose(self.u_nn)
        t = self.u_nn
        self.u_nn = np.maximum(self.l_constr, np.minimum(self.u_constr, self.u_nn))
        if np.any(self.u_nn != t):
            print("lower constr: ", np.mean(self.l_constr))
            print("NN Controller is saturating")
        self.u_sim[self.id_to_cm,k] = self.u_nn.flatten()
    
    #Updates the AWR state using nn
    def nn_update_awr(self):
        self.x_awr_old = self.x_awr_new
        self.x_awr_new = self.A_awr @ self.x_awr_old + self.B_awr @ self.u_nn
        self.y_awr = self.C_awr @ self.x_awr_new + self.D_awr @ self.u_nn

    def solveFGM_expert(self, k):

        out_global = self.u_sim[self.id_to_cm, k-1][:, np.newaxis]
        for i in range(self.MAX_ITER):
            self.z_old = self.z_new
            t = self.J_FGM @ out_global - self.q
            self.z_new = np.maximum(self.l_constr, np.minimum(self.u_constr, t))
            out_global = (1+self.beta_fgm) * self.z_new - self.beta_fgm * self.z_old
        fgm_result = self.z_new

        self.u_sim_expert[self.id_to_cm, k] = fgm_result.flatten()


    #Simulate using nn
    def sim_nn(self,model,device):
        self.device = device
        self.nn_controller = model
        self.nn_controller.eval()
        self.nn_controller.to(torch.device(device))
        #Make sure FGM is true so that limits are updated accordingly
        self.use_FGM = True
        for k in range(self.n_samples):
            #Measurements

            if k % 100 == 0:
                print(f"Simulation progress: {k/self.n_samples*100:.2f}%")

            self.update_y_sim(k)
            if k >= self.n_delay:
                self.update_y_meas(k)
                self.calc_obs_state(k)
                self.update_obs_measurement()
                self.update_obs_state(k)
                self.update_q_limits()
                self.solvenn(k)
                self.nn_update_awr()
            self.update_plant_state(k)
        self.u_sim = np.transpose(self.u_sim)
        self.y_sim = np.transpose(self.y_sim)
        self.x0_obs = np.transpose(self.x0_obs)
        self.xd_obs = np.transpose(self.xd_obs)
        return self.y_sim, self.u_sim, self.x0_obs, self.xd_obs


    def sim_dagger(self, model, device):
        self.ss_count = 0
        self.use_FGM = True
        self.device = device
        self.nn_controller = model
        self.nn_controller.eval()
        self.nn_controller.to(torch.device(device))

        for k in range(self.n_samples):
            #Measurements
            if k % 100 == 0:
                print(f"Simulation progress: {k/self.n_samples*100:.2f}%")

            self.update_y_sim(k)
            if k >= self.n_delay:
                self.update_y_meas(k)
                self.calc_obs_state(k)
                self.update_obs_measurement()
                self.update_obs_state(k)
                self.update_q_limits()
                self.solveFGM_expert(k)
                self.solvenn(k)
                self.nn_update_awr()
            self.update_plant_state(k)
            if self.checkSS(k, 1e-4):
                self.terminateSS(k, use_dagger = True)
                break

        self.u_sim_expert = np.transpose(self.u_sim_expert)
        self.x0_obs = np.transpose(self.x0_obs)
        self.xd_obs = np.transpose(self.xd_obs)
        return self.u_sim_expert, self.x0_obs, self.xd_obs, self.n_samples



