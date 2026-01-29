import torch
from torch import nn
from torch.autograd import Function, Variable
from torch.nn.parameter import Parameter
import torch.optim as optim
import torch.nn.functional as F


# Classes to define various models used in the MPC project

# Define the arbitary neural network model

class NNController(nn.Module):
    def __init__(self, n_state, hidden_size, n_ctrl):
        super(NNController, self).__init__()
        self.type = 'LinReLux4'
        # Initialize weights and biases for all layers

        # Hidden layer 1
        self.fc1 = nn.Linear(n_state, hidden_size)
        self.act1 = nn.ReLU()

        # Hideen layer 2
        self.fc2 = nn.Linear(hidden_size, int(hidden_size/2))
        self.act2 = nn.ReLU()
        
        # Hidden layer 3
        # self.fc3 = nn.Linear(hidden_size, hidden_size)
        # self.act3 = nn.ReLU()

        # Output layer
        self.fc4 = nn.Linear(int(hidden_size/2), n_ctrl)


    def forward(self, x):  # x: (n_batch, n_state)
        out = self.fc1(x)

        out = self.act1(out)
        out = self.fc2(out)

        out = self.act2(out)
        # out = self.fc3(out)
        # out = self.act3(out)
        x = self.fc4(out)

        return x


# Define the loss function
class get_loss(nn.Module):
    """
    Improved loss that combines a supervised tracking term (L1) with
    constraint-violation penalties (ReLU-based) on upper and lower bounds.

    Usage:
      criterion = get_loss(l1_weight=1.0, l3_weight=5.0)
      loss = criterion(preds, targets, upper_const=upper, lower_const=lower)

    If `upper_const` and `lower_const` are not provided, the loss falls back
    to the simple L1 supervised tracking loss.
    """
    def __init__(self, l1_weight: float = 1.0, l2_weight: float = 0.0, l3_weight: float = 0.0, l4_weight: float = 0.0):
        super(get_loss, self).__init__()
        self.l1_weight = float(l1_weight)
        self.l2_weight = float(l2_weight)
        self.l3_weight = float(l3_weight)
        self.l4_weight = float(l4_weight)

    def forward(self, predictions, targets, upper_const=None, lower_const=None, J=None, q_vec=None):
        """
        predictions: tensor (batch, nu)
        targets: tensor (batch, nu)
        upper_const: tensor broadcastable to predictions (batch, nu) or (nu,) or None
        lower_const: tensor broadcastable to predictions (batch, nu) or (nu,) or None

        Returns: scalar loss
        """
        # 1. Supervised Tracking Loss (L1)
        loss_tracking = torch.norm(predictions - targets)

        # 2. Quadratic/Linear MPC surrogate (L2) : 0.5 * u^T J u + q^T u
        L2 = torch.tensor(0.0, dtype=predictions.dtype, device=predictions.device)
        if J is not None:
            J_t = torch.as_tensor(J, dtype=predictions.dtype, device=predictions.device)
            u = predictions
            Ju = torch.matmul(u, J_t.t())
            quad = 0.5 * torch.sum(u * Ju, dim=1)  # per-sample

            # linear term: either q_vec
            if q_vec is not None:
                qv = torch.as_tensor(q_vec, dtype=predictions.dtype, device=predictions.device)
                if qv.dim() == 1:
                    qv = qv.unsqueeze(0)
                lin = torch.sum(u * qv, dim=1)
            else:
                lin = torch.zeros_like(quad)

            L2 = torch.mean(quad + lin)

        # 3. Constraint Violation Losses (L3 and L4)
        if upper_const is None or lower_const is None:
            return loss_tracking

        # Ensure tensors are same dtype/device
        upper = upper_const.to(dtype=predictions.dtype, device=predictions.device)
        lower = lower_const.to(dtype=predictions.dtype, device=predictions.device)

        # 2. Constraint Violation Losses (L3 and L4)
        violation_upper = F.relu(predictions - upper)
        violation_lower = F.relu(-(predictions - lower))
        
        violation_upper_mean = torch.mean(violation_upper)
        violation_lower_mean = torch.mean(violation_lower)

        # Final weighted composite loss
        total_loss = (
            self.l1_weight * loss_tracking
            + self.l2_weight * L2
            + self.l3_weight * violation_upper_mean
            + self.l4_weight * violation_lower_mean
        )
        return total_loss


# Function would usually be called get_qloss - call class "q_loss"
class get_qloss(nn.Module):
    """
    Quadratic MPC loss: L(u; x) = 0.5 * u^T J u + q(x)^T u

    J should be a (nu, nu) matrix. q_mat should be provided with shape (nu, n_xaug)
    so that q(x) = q_mat @ x_aug (and x_aug has shape (batch, n_xaug)).

    The module stores J and q_mat as buffers so they move with the module/device
    and are saved with state_dict.
    """
    def __init__(self, J=None, q_mat=None, reduction='mean', dtype=torch.float32, device=None):
        super(get_qloss, self).__init__()
        self.reduction = reduction
        self.dtype = dtype
        # Register buffers so tensors are moved with the module
        if J is not None:
            J_t = torch.as_tensor(J, dtype=self.dtype)
        else:
            J_t = None
        if q_mat is not None:
            q_t = torch.as_tensor(q_mat, dtype=self.dtype)
        else:
            q_t = None

        # Use register_buffer so they are part of the module state but not parameters
        self.register_buffer('J', J_t)
        self.register_buffer('q_mat', q_t)

        # device can be None; caller can call .to(device) on the module
        if device is not None:
            self.to(device)

    def set_params(self, J=None, q_mat=None):
        """Set or update J and q_mat after construction."""
        if J is not None:
            self.J = torch.as_tensor(J, dtype=self.dtype)
        if q_mat is not None:
            self.q_mat = torch.as_tensor(q_mat, dtype=self.dtype)

    def forward(self, u_pred, x_aug=None):
        """
        u_pred: (batch, nu)
        x_aug: (batch, n_xaug) or None if q_mat is not set
        returns: scalar loss (reduction) or per-sample vector if reduction=None
        """
        if self.J is None:
            raise RuntimeError('J matrix is not set for get_qloss')

        # Ensure u_pred is float tensor
        u = u_pred.to(dtype=self.dtype)

        # Quadratic term: 0.5 * u^T J u per sample
        # compute Ju = u @ J^T (J is square)
        Ju = torch.matmul(u, self.J.t())
        quad = 0.5 * torch.sum(u * Ju, dim=1)  # (batch,)

        # Linear term
        if self.q_mat is not None:
            if x_aug is None:
                raise RuntimeError('q_mat is set but x_aug was not provided to forward')
            x = x_aug.to(dtype=self.dtype)
            # qx shape (batch, nu) = x @ q_mat.T
            qx = torch.matmul(x, self.q_mat.t())
            lin = torch.sum(qx * u, dim=1)
        else:
            lin = torch.zeros_like(quad)

        loss_vec = quad + lin

        if self.reduction == 'mean':
            return loss_vec.mean()
        elif self.reduction == 'sum':
            return loss_vec.sum()
        else:
            return loss_vec


# Define the linear controller 
class LinearController(nn.Module):
    def __init__(self, n_state, n_ctrl):
        super(LinearController, self).__init__()
        self.type = 'linear'
        self.fc = nn.Linear(n_state, n_ctrl)

    def forward(self, x):
        return self.fc(x)
    

# # Define the LQR controller
# class LQRController(nn.Module):
#     """
#     LQR-based controller that learns optimal gain matrix K.
#     Policy: u = K @ x
    
#     This controller learns a linear feedback law by fitting to data,
#     which approximates an LQR controller if the data comes from an
#     LQR-based MPC or similar optimal control system.
    
#     Args:
#         n_state: number of state dimensions
#         n_ctrl: number of control dimensions
#         dtype: data type for computations (default: torch.float32)
#     """
#     def __init__(self, n_state, n_ctrl, dtype=torch.float32):
#         super(LQRController, self).__init__()
#         self.type = 'LQR'
#         self.n_state = n_state
#         self.n_ctrl = n_ctrl
#         self.dtype = dtype
        
#         # Initialize K as a learnable parameter (n_ctrl, n_state)
#         # Random initialization from a normal distribution
#         # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#         # Rather use Identity for K_init
#         # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#         K_init = torch.randn(n_ctrl, n_state, dtype=dtype) * 0.01
#         self.K = nn.Parameter(K_init)
    
#     def forward(self, x):
#         """
#         Compute control: u = K @ x
        
#         Args:
#             x: state tensor of shape (batch, n_state)
        
#         Returns:
#             u: control tensor of shape (batch, n_ctrl)
#         """
#         # x shape: (batch, n_state)
#         # K shape: (n_ctrl, n_state)
#         # output shape: (batch, n_ctrl)
#         # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#         # Check this line:
#         # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#         u = torch.matmul(x, self.K.t())  # (batch, n_state) @ (n_state, n_ctrl)
#         return u
    
#     def get_K(self):
#         """Return the current gain matrix K."""
#         return self.K.data
    
#     def set_K(self, K_new):
#         """Set the gain matrix K from an external source."""
#         with torch.no_grad():
#             self.K.copy_(torch.as_tensor(K_new, dtype=self.dtype))
    


    