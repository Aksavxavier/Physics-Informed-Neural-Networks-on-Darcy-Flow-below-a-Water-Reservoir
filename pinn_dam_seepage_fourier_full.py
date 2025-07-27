#!/usr/bin/env python3
"""
PINN for Groundwater Head Prediction with Fourier Features and L-BFGS Optimization
"""

# ─── Imports ────────────────────────────────────────────────────────────────────
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# ─── Settings ───────────────────────────────────────────────────────────────────
torch.manual_seed(0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

# ─── Domain Geometry ────────────────────────────────────────────────────────────
reservoir_width = 100
catchment_width = 30
dam_width = 20
aquifer_height = 40
kf = 1e-6 * 60 * 60 * 24  # m/day

# ─── Load Excel Data ────────────────────────────────────────────────────────────
fname = '55_5_25.xlsx'
h_df = pd.read_excel(f"./heads/{fname}")
v_df = pd.read_excel(f"./velocities/{fname}")
h_df.columns = h_df.columns.str.strip()
v_df.columns = v_df.columns.str.strip()
reservoir_height, catchment_height, dam_depth = [int(x) for x in fname.split('.')[0].split('_')]

# ─── Extract Values ─────────────────────────────────────────────────────────────
xh_data = h_df['X'].values.astype(np.float32)
yh_data = h_df['Y'].values.astype(np.float32)
h_data = h_df['FINIT'].values.astype(np.float32)

# ─── Mesh Grid ──────────────────────────────────────────────────────────────────
x_res = np.linspace(0, reservoir_width, reservoir_width+1)
y_res = np.linspace(0, aquifer_height, aquifer_height+1)
X_res, Y_res = np.meshgrid(x_res, y_res)
xy_res = np.stack([X_res.ravel(), Y_res.ravel()], axis=1)

x_dam = np.linspace(reservoir_width, reservoir_width + dam_width, dam_width+1)
X_dam, Y_dam = np.meshgrid(x_dam, y_res)
xy_dam = np.stack([X_dam.ravel(), Y_dam.ravel()], axis=1)

x_catch = np.linspace(reservoir_width + dam_width, reservoir_width + dam_width + catchment_width, catchment_width+1)
X_catch, Y_catch = np.meshgrid(x_catch, y_res)
xy_catch = np.stack([X_catch.ravel(), Y_catch.ravel()], axis=1)

xy_all = np.vstack([xy_res, xy_dam, xy_catch])

# ─── Masks ──────────────────────────────────────────────────────────────────────
tol = 1e-6
left_bc_mask     = np.isclose(xy_all[:, 0], xh_data.min(), atol=tol)
right_bc_mask    = np.isclose(xy_all[:, 0], xh_data.max(), atol=tol)
top_res_bc_mask  = (np.isclose(xy_all[:, 1], yh_data.max(), atol=tol) & (xy_all[:, 0] <= reservoir_width + tol))
top_catch_bc_mask= (np.isclose(xy_all[:, 1], yh_data.max(), atol=tol) & (xy_all[:, 0] >= (reservoir_width + dam_width) - tol))
bottom_bc_mask   = np.isclose(xy_all[:, 1], yh_data.min(), atol=tol)
dam_left_bc_mask = (np.isclose(xy_all[:, 0], reservoir_width, atol=tol) & (xy_all[:, 1] >= (aquifer_height - dam_depth)))
dam_right_bc_mask= (np.isclose(xy_all[:, 0], reservoir_width + dam_width, atol=tol) & (xy_all[:, 1] >= (aquifer_height - dam_depth)))
dam_bot_bc_mask  = (np.isclose(xy_all[:, 1], (aquifer_height - dam_depth), atol=tol) &
                    (xy_all[:, 0] >= reservoir_width) &
                    (xy_all[:, 0] <= reservoir_width + dam_width))
dam_interior_mask= (xy_all[:, 0] > reservoir_width) & \
                   (xy_all[:, 0] < reservoir_width + dam_width) & \
                   (xy_all[:, 1] > (aquifer_height - dam_depth))

boundary_mask = (
    left_bc_mask | right_bc_mask | top_res_bc_mask | top_catch_bc_mask |
    bottom_bc_mask | dam_left_bc_mask | dam_right_bc_mask | dam_bot_bc_mask | dam_interior_mask
)
collocation_mask = ~boundary_mask

# ─── Collect Points ─────────────────────────────────────────────────────────────
col_pde = xy_all[collocation_mask]
col_left_bc = xy_all[left_bc_mask]
col_right_bc = xy_all[right_bc_mask]
col_top_res_bc = xy_all[top_res_bc_mask]
col_top_catch_bc = xy_all[top_catch_bc_mask]
col_bottom_bc = xy_all[bottom_bc_mask]
col_dam_left_bc = xy_all[dam_left_bc_mask]
col_dam_right_bc = xy_all[dam_right_bc_mask]
col_dam_bot_bc = xy_all[dam_bot_bc_mask]

# ─── Normalization ──────────────────────────────────────────────────────────────
h_star = max(h_data)
x_star = y_star = h_star

def normalize_coordinates(xy):
    return np.column_stack((xy[:, 0:1] / x_star, xy[:, 1:2] / y_star))

def to_tensor(xy_block):
    return torch.tensor(normalize_coordinates(xy_block), requires_grad=True, dtype=DTYPE, device=DEVICE)

# ─── Fourier Feature Encoder ─────────────────────────────────────────────────────
class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, scale=3.0):
        super().__init__()
        B = torch.randn((in_features, out_features)) * scale
        self.register_buffer('B', B)

    def forward(self, x):
        x_proj = 2 * torch.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

# ─── Model ──────────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, layers, activation=nn.Tanh):
        super().__init__()
        self.net = nn.Sequential()
        for i in range(len(layers) - 2):
            self.net.append(nn.Linear(layers[i], layers[i + 1]))
            self.net.append(activation())
        self.net.append(nn.Linear(layers[-2], layers[-1]))

    def forward(self, x):
        return self.net(x)

class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = FourierFeatures(2, 64, scale=3.0)  # outputs 128
        self.mlp = MLP([128, 64, 64, 64, 1])

    def forward(self, x):
        x_encoded = self.encoder(x)
        print(f"Encoded shape: {x_encoded.shape}")  # Debug line
        return self.mlp(x_encoded)

    def pde_res(self, x):
        h = self.forward(x)
        dh = torch.autograd.grad(h, x, grad_outputs=torch.ones_like(h), create_graph=True)[0]
        ddh = torch.autograd.grad(dh, x, grad_outputs=torch.ones_like(dh), create_graph=True)[0]
        return ddh[:, 0:1] + ddh[:, 1:2]

# ─── Training ───────────────────────────────────────────────────────────────────
net = PINN().to(DEVICE)
criteria = nn.MSELoss()
opt_adam = torch.optim.Adam(net.parameters(), lr=1e-3)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_adam, 'min', patience=1000)

col_pde_norm        = to_tensor(col_pde)
col_left_bc_norm    = to_tensor(col_left_bc)
col_right_bc_norm   = to_tensor(col_right_bc)
col_top_res_bc_norm = to_tensor(col_top_res_bc)
col_top_catch_bc_norm = to_tensor(col_top_catch_bc)
col_bottom_bc_norm  = to_tensor(col_bottom_bc)
col_dam_left_bc_norm = to_tensor(col_dam_left_bc)
col_dam_right_bc_norm= to_tensor(col_dam_right_bc)
col_dam_bot_bc_norm  = to_tensor(col_dam_bot_bc)

def zero_grad_norm(pred): return criteria(pred, torch.zeros_like(pred))

def compute_losses():
    pde_loss = 100 * criteria(net.pde_res(col_pde_norm), torch.zeros_like(col_pde_norm[:, :1]))
    bc_h1 = net(col_top_res_bc_norm)
    bc_h2 = net(col_top_catch_bc_norm)
    bc_loss = (
        50 * criteria(bc_h1, torch.ones_like(bc_h1) * reservoir_height / h_star) +
        50 * criteria(bc_h2, torch.ones_like(bc_h2) * catchment_height / h_star) +
        zero_grad_norm(torch.autograd.grad(net(col_left_bc_norm), col_left_bc_norm, grad_outputs=torch.ones_like(net(col_left_bc_norm)), create_graph=True)[0][:, 0:1]) +
        zero_grad_norm(torch.autograd.grad(net(col_right_bc_norm), col_right_bc_norm, grad_outputs=torch.ones_like(net(col_right_bc_norm)), create_graph=True)[0][:, 0:1]) +
        zero_grad_norm(torch.autograd.grad(net(col_bottom_bc_norm), col_bottom_bc_norm, grad_outputs=torch.ones_like(net(col_bottom_bc_norm)), create_graph=True)[0][:, 1:2]) +
        zero_grad_norm(torch.autograd.grad(net(col_dam_left_bc_norm), col_dam_left_bc_norm, grad_outputs=torch.ones_like(net(col_dam_left_bc_norm)), create_graph=True)[0][:, 0:1]) +
        zero_grad_norm(torch.autograd.grad(net(col_dam_right_bc_norm), col_dam_right_bc_norm, grad_outputs=torch.ones_like(net(col_dam_right_bc_norm)), create_graph=True)[0][:, 0:1]) +
        zero_grad_norm(torch.autograd.grad(net(col_dam_bot_bc_norm), col_dam_bot_bc_norm, grad_outputs=torch.ones_like(net(col_dam_bot_bc_norm)), create_graph=True)[0][:, 1:2])
    )
    return pde_loss, bc_loss

for epoch in range(50000):
    opt_adam.zero_grad()
    pde_loss, bc_loss = compute_losses()
    total_loss = pde_loss + bc_loss
    total_loss.backward()
    opt_adam.step()
    sched.step(total_loss)
    if epoch % 1000 == 0:
        print(f"Epoch {epoch:6d} | Total: {total_loss.item():.2e} | PDE: {pde_loss.item():.2e} | BC: {bc_loss.item():.2e}")

print(" Fine-tuning with L-BFGS...")
opt_lbfgs = torch.optim.LBFGS(net.parameters(), max_iter=500, tolerance_grad=1e-9, tolerance_change=1e-9)
def closure():
    opt_lbfgs.zero_grad()
    pde_loss, bc_loss = compute_losses()
    loss = pde_loss + bc_loss
    loss.backward()
    return loss
opt_lbfgs.step(closure)

# ─── Prediction and Evaluation ──────────────────────────────────────────────────
x_test = xh_data.reshape(-1, 1)
y_test = yh_data.reshape(-1, 1)
test_input = np.concatenate((x_test, y_test), axis=1)
test_input = torch.tensor(normalize_coordinates(test_input), dtype=DTYPE, device=DEVICE, requires_grad=True)
h_pred = net(test_input).cpu().detach().numpy() * h_star
r2 = r2_score(h_data.flatten(), h_pred.flatten())
rmse = np.sqrt(np.mean((h_pred.flatten() - h_data.flatten()) ** 2))
print(f" R² Score: {r2:.4f}")
print(f" RMSE: {rmse:.4f} m")

# ─── Visualization ──────────────────────────────────────────────────────────────
plt.figure(figsize=(21, 6))
vmin = min(h_pred.min(), h_data.min())
vmax = max(h_pred.max(), h_data.max())
error = np.abs(h_pred.flatten() - h_data.flatten())

plt.subplot(1, 3, 1)
sc1 = plt.scatter(xh_data, yh_data, c=h_pred, cmap='viridis', s=40, vmin=vmin, vmax=vmax)
plt.colorbar(sc1, label='Predicted Head [m]')
plt.xlabel('X [m]'); plt.ylabel('Y [m]')
plt.title('PINN Prediction'); plt.axis('equal')

plt.subplot(1, 3, 2)
sc2 = plt.scatter(xh_data, yh_data, c=h_data, cmap='viridis', s=40, vmin=vmin, vmax=vmax)
plt.colorbar(sc2, label='Measured Head [m]')
plt.xlabel('X [m]'); plt.ylabel('Y [m]')
plt.title('Ground Truth'); plt.axis('equal')

plt.subplot(1, 3, 3)
sc3 = plt.scatter(xh_data, yh_data, c=error, cmap='inferno', s=40)
plt.colorbar(sc3, label='|Prediction − Truth| [m]')
plt.xlabel('X [m]'); plt.ylabel('Y [m]')
plt.title(f'Absolute Error\nR² = {r2:.4f}, RMSE = {rmse:.4f} m')
plt.axis('equal')
plt.tight_layout()
plt.savefig("pinn_fourier_head_comparison.png", dpi=300)
plt.show()

