# NTNU IDUN Slurm & NN Training Guidelines

When generating Slurm scripts, batch jobs, or Python training code for the NTNU IDUN cluster, you must adhere strictly to these constraints:

## 1. Connection & Session Management
*   **Network Access:** IDUN is only accessible from within the NTNU network. Ensure the user is either on campus or connected to the NTNU VPN before attempting to connect.
*   **Login Nodes:** Connect via SSH to one of the login nodes (e.g., `ssh username@idun-login1.hpc.ntnu.no` or `idun-login2.hpc.ntnu.no`). 
*   **Terminal Stability:** Never run heavy computations directly on the login nodes. For interactive Slurm jobs (e.g., using `srun`) or scripts that require active terminal monitoring, strongly recommend launching them inside a `screen` or `tmux` session first. This prevents the training run from aborting if the SSH connection or VPN drops.

## 2. Slurm Job Configuration
*   **Partitions:** IDUN uses a simplified partition system. Use `GPUQ` for all Neural Network training. Do not use legacy partitions (e.g., WORKQ, EPIC) as they have been removed.
*   **Time Limits:** The maximum time limit on `GPUQ` is 14 days. Format as `#SBATCH --time=DD-HH:MM:SS`.
*   **GPU Requests:** Explicitly request the GPU architecture. IDUN supports `H100`, `A100`, `V100`, and `P100`. Format: `#SBATCH --gres=gpu:A100:1`.
*   **Account:** Remind the user to insert their group account. Prioritize `share-*` accounts for higher priority if applicable.

## 3. Environment & Module Loading
*   Before running Python scripts, always include `module purge` followed by the necessary module loads (e.g., `module load Anaconda3` or `module load Python`).
*   Assume the user might be using Apptainer (Singularity) containers, which is common for GenAI workloads on IDUN. Ask for confirmation before generating complex Python virtual environment setups.

## 4. NN Training Best Practices
*   **WandB / Logging:** Ensure offline mode is toggled if compute nodes lack direct external internet access, or configure the standard Weights & Biases environment variables.
*   **Checkpointing:** Save frequent model checkpoints. IDUN jobs can be preempted or run out of time. Ensure `train.py` supports resuming from the latest `.pt` or `.safetensors` file.
