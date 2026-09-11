# NTNU IDUN Slurm & NN Training Guidelines

When generating Slurm scripts, batch jobs, or Python training code for the NTNU IDUN cluster, you must adhere strictly to these constraints:

## 1. Connection & Session Management
*   **Network Access:** IDUN is only accessible from within the NTNU network. Ensure the user is either on campus or connected to the NTNU VPN before attempting to connect.
*   **Login Nodes:** Connect via SSH to one of the login nodes (e.g., `ssh username@idun-login1.hpc.ntnu.no`). 
*   **Terminal Stability:** Never run heavy computations directly on the login nodes. For interactive Slurm jobs (e.g., using `srun`) or scripts that require active terminal monitoring, strongly recommend launching them inside a `screen` or `tmux` session first to prevent dropped VPN connections from killing the job.

## 2. Storage & File Synchronization
*   **Source Code (Git):** Do not use file transfer protocols for code. Keep project source files in a Git repository (GitHub, GitLab, or NTNU GitLab). Push from the local machine, and `git pull` from the IDUN login node.
*   **Datasets & Checkpoints (rsync):** For syncing heavy data (image datasets, `.pt` model weights), use `rsync -avzP` over SSH. Do not upload large datasets to Git. 
*   **Directories (Work vs. Home):** 
    *   `/cluster/home/<username>/`: Small quota. Use *only* for source code, small config files, and Apptainer/Conda setups.
    *   `/cluster/work/<username>/`: Large quota. Always point Python training scripts here for dataset loading, WandB offline caching, and saving model checkpoints.

## 3. Slurm Job Configuration
*   **Partitions:** IDUN uses a simplified partition system. Use `GPUQ` for all Neural Network training. Do not use legacy partitions (e.g., WORKQ, EPIC) as they have been removed.
*   **Time Limits:** The maximum time limit on `GPUQ` is 14 days. Format as `#SBATCH --time=DD-HH:MM:SS`.
*   **GPU Requests:** Explicitly request the GPU architecture. IDUN supports `h100`, `a100`, `v100`, and `p100`. Format: `#SBATCH --gres=gpu:a100:1`.
*   **Account:** Remind the user to insert their group account. Prioritize `share-*` accounts for higher priority if applicable.

## 4. Environment & Module Loading
*   **Module Setup:** Before running Python scripts or batch jobs, always run `module purge` followed by `module load Anaconda3` (or relevant modules).
*   **Conda Environment:** Use the existing Conda environment named **`lonnx`** (`conda activate lonnx`).
*   **PyTorch Version Invariance:** **Never modify or upgrade the PyTorch package version in the `lonnx` environment.** It must remain at the exact version currently installed:
    ```
    torch==2.12.1+cu126
    ```
    This specific build is critical to maintain CUDA compatibility with older GPU architectures present on IDUN (such as `V100` and `P100`). Avoid installing or updating any package that could pull in or trigger a PyTorch version change.
*   **Container Usage:** If Apptainer (Singularity) containers are used on IDUN, verify container bindings and module interactions with the user before generating execution scripts.
*   **JAX with CUDA:** Check the compute node with `nvidia-smi` before selecting a wheel. The current IDUN A100 nodes report CUDA 12.9, so install compatible CUDA 12 packages with `python -m pip install --upgrade "jax[cuda12]"`. Do not use the default CPU-only `jaxlib` installation for GPU jobs. Verify GPU execution inside an allocated Slurm job (`srun`) with `python -c "import jax; print(jax.devices())"`.

## 5. Directory & Workspace Structure
*   **Repository Location:** When cloning or setting up repositories on IDUN, always place them in subdirectories under:
    ```
    /cluster/home/yamanu/sandbox
    ```
    (e.g., `/cluster/home/yamanu/sandbox/project-odin`, `/cluster/home/yamanu/sandbox/torchlogix`).

## 6. NN Training Best Practices
*   **WandB / Logging:** Ensure offline mode is toggled if compute nodes lack direct external internet access, or configure the standard Weights & Biases environment variables (`WANDB_MODE=offline`). Point all log directories to `/cluster/work/`.
*   **Checkpointing:** Save frequent model checkpoints to `/cluster/work/`. IDUN jobs can be preempted or run out of time. Ensure `train.py` supports resuming from the latest `.pt` or `.safetensors` file.
