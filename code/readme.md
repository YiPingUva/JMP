This repository provides the codes for replicating the results presented in the paper. Follow the steps below to train neural networks and simulate data for the collateral economy and bond economy.

Training Neural Networks for the Collateral Economy
Step 1: Solve for Steady-State Variables
	1	Navigate to the folder Steady_state_solver_matlab/collateral_economy.
	2	Run the MATLAB file steady_state_solver_be.mlx.
	3	The steady-state data will be saved as xuss_ce.mat.
Step 2: Initialize the Training with Steady-State Values
	1	Use the steady-state portfolio holdings of housing, debt contracts, and consumption as the starting state vector for training.
	2	Navigate to the folder Collateral_economy/baseline.py.
	3	Copy and paste the corresponding variables from xuss_ce.mat to construct x0 in the Python code (lines 122–127).
Step 3: Train Neural Networks
	1	Run the Python script to start the first training schedule: > python baseline.py --train_from_scratch
	1	The results will be saved in Collateral_economy/output/1st_baseline.
Step 4: Continue Training
	1	After completing the first training schedule:
	◦	Uncomment lines 1294–1301 in baseline.py to start the second training schedule.
	2	Results will be saved as follows:
	◦	Intermediate results: Collateral_economy/output/2nd_baseline.
	◦	Final results: Collateral_economy/output/final_baseline (after completing episodes specified on line 1296).
Note: Monitor the cost function and adjust hyperparameters of neural networks as necessary to determine the appropriate stopping point for training.

Simulating Data for the Collateral Economy
	1	To simulate data using trained neural networks saved in Collateral_economy/output/final_baseline, set plot_epi_length to the desired number of simulation periods. For example, the paper uses 500,000 periods (edit this on line 672 of baseline.py).
	2	Run the script: >python baseline.py
	3	Results will be saved in Collateral_economy/output/restart_baseline.

Replicating Results for the Bond Economy
To replicate results for the bond economy, follow the same steps as above, but use the files located in:
	•	Steady_state_solver_matlab/bond_economy
	•	Bond_economy

Questions
If you have any questions about the code, please reach out to Yi Ping at yp7ec@virginia.edu or yi.ping@eruni.org.
