Physics-Informed Neural Networks for Darcy Flow Below a Water Reservoir

This repository contains a Physics-Informed Neural Network (PINN) implementation to estimate water pressure distribution and Darcy velocity in a porous medium located beneath a dam reservoir. The project is part of the ENHANCE BIP summer school task (July 2022).

Objective

Predict the pressure head distribution in an aquifer beneath a dam using a PINN.
Estimate the Darcy velocity and flow rate across the domain.
Integrate dam depth and hydraulic head levels into the model as variable inputs.

Features

Solve steady-state Darcy flow using PINNs.
Flexible to different dam depths (hd) and reservoir/catchment levels (h1, h2).
Estimate both pressure and velocity fields.
Support for multiple geometrical and boundary configurations.

Dataset

Datasets provided in Excel format follow the naming convention:

h1_h2_hd.xlsx
where:

h1: Reservoir level
h2: Catchment level
hd: Dam embedding depth
Each dataset contains a 2D grid of reference pressure values.

Tasks

Reproduce pressure head using a PINN trained on one dataset.
Predict Darcy velocity and compute flow rate.
Incorporate dam depth as a geometric feature in the model.
Allow head levels (h1, h2) as input features for generalization.

Requirements

Python 3.8+
PyTorch
NumPy
Pandas
Matplotlib
SciPy
Install dependencies via:

pip install -r requirements.txt

 Output

Pressure distribution heatmaps
Darcy velocity vector fields
Estimated flow rates beneath the dam

References

J.R. Philip, Flow in Porous Media, Ann. Rev. Fluid Mech., 1970.# Physics-Informed-Neural-Networks-on-Darcy-Flow-below-a-Water-Reservoir
