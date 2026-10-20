---
name: ntnu-idun-skills
description: >-
  Generates Slurm batch scripts, configures Python environments (venv/pip or Conda),
  and manages storage and training workflows for the NTNU IDUN HPC cluster.
  Use when writing Slurm job scripts, executing neural network training on IDUN,
  configuring venvs or conda environments, or handling data synchronization between local machines and IDUN.
---

# NTNU IDUN Slurm & Neural Network Training

Guidelines, constraints, and standard workflows for running Slurm jobs, managing environments (pip/venv or Conda), and training neural networks on the NTNU IDUN HPC cluster.

See [the Slurm batch job template](assets/slurm_template.sbatch) for a starter script supporting both virtual environments and Conda.

---

## 1. Connection & Session Management

* **Network Access:** IDUN is only accessible within the NTNU network. Ensure connection to campus Wi-Fi/Ethernet or the NTNU VPN before connecting.
* **Login Nodes:** Connect via SSH to an IDUN login node:
  ```bash
  ssh <username>@idun-login1.hpc.ntnu.no
  ```
* **Interactive Terminal Stability:** Never run heavy computations directly on login nodes. When running interactive Slurm sessions (`srun`) or scripts requiring terminal monitoring, always run inside `tmux` or `screen` so that a dropped VPN/network connection does not terminate the job:
  ```bash
  tmux new -s training
  ```

---

## 2. Storage & File Synchronization

| Filesystem | Path | Quota | Purpose |
| :--- | :--- | :--- | :--- |
| **Home** | `/cluster/home/<username>/` | Small | Git repositories, source code, small config files. |
| **Work** | `/cluster/work/<username>/` | Large | Datasets, virtual environments, conda envs, pip cache, checkpoints, logs. |

* **Source Code (Git):** Keep project source code in Git (GitHub, GitLab, NTNU GitLab). Push from local and run `git pull` on the IDUN login node.
* **Large Files & Datasets (rsync):** Do not track large files in Git. Sync datasets and model checkpoints (`.pt`, `.safetensors`) using `rsync` over SSH:
  ```bash
  rsync -avzP ./dataset/ <username>@idun-login1.hpc.ntnu.no:/cluster/work/<username>/datasets/
  ```

---

## 3. Slurm Job Configuration

### Partitions
* `GPUQ`: Required for all GPU and neural network training jobs. Maximum time limit is 14 days (`--time=14-00:00:00`).
* `CPUQ`: Use for non-GPU preprocessing or CPU computations.
* **Deprecated Partitions:** Do not use legacy partitions such as `WORKQ` or `EPIC` (they have been removed).

### Resource Directives
* **GPU Architectures:** Explicitly specify GPU architecture. Supported models: `h100`, `a100`, `v100`, `p100`.
  ```bash
  #SBATCH --gres=gpu:a100:1
  ```
* **`torch.compile` (Triton) vs. Tesla P100:**
  > [!WARNING]
  > **Avoid requesting or allowing Tesla P100 (`p100`) GPUs if the job uses `torch.compile`:**
  > PyTorch 2.x Inductor uses Triton as its GPU code generation backend. Triton strictly requires NVIDIA CUDA Compute Capability $\ge 7.0$ (Volta `v100`, Ampere `a100`, Hopper `h100`).
  > The Tesla P100 is Pascal architecture (Compute Capability 6.0) and will immediately fail during JIT compilation with `torch._inductor.exc.GPUTooOldForTriton`.
  > 
  > For jobs involving `torch.compile`:
  > - Explicitly request modern GPUs: `#SBATCH --gres=gpu:a100:1` or `#SBATCH --gres=gpu:v100:1`.
  > - If requesting generic GPUs with `#SBATCH --gres=gpu:1`, add a feature constraint:
  >   ```bash
  >   #SBATCH --constraint="a100|h100|v100"
  >   ```
  >   to prevent Slurm from allocating any of IDUN's P100 nodes (`idun-11-[02-14]`).
* **Accounts:** Specify group account. Prioritize `share-*` accounts for higher queue priority:
  ```bash
  #SBATCH --account=share-<group>
  ```
* **Time Format:** Format time limits strictly as `DD-HH:MM:SS`:
  ```bash
  #SBATCH --time=02-12:00:00
  ```

---

## 4. Environment & Package Management (pip/venv & Conda)

To keep projects portable across IDUN and other development machines, projects should rely on standard dependency manifests (`requirements.txt` or `pyproject.toml` for pip, or `environment.yml` for Conda).

### Filesystem & Quota Hygiene
> [!IMPORTANT]
> **Always place environments and package caches on `/cluster/work/`:**
> Installing packages into `/cluster/home/` quickly exhausts your strict inode and disk quota.
> * **Pip / venv:** Store venvs at `/cluster/work/$USER/venvs/<project-name>` and redirect pip cache:
>   ```bash
>   export PIP_CACHE_DIR="/cluster/work/$USER/.cache/pip"
>   ```
> * **Conda:** Store Conda envs and pkgs on `/cluster/work/` (e.g. `conda create --prefix /cluster/work/$USER/conda-envs/<project-name>` or configure `envs_dirs` in `~/.condarc`).

### Option A: Standard Python `venv` + `pip` (Recommended for cross-machine portability)
```bash
# 1. Load cluster Python or Anaconda base module
module purge
module load Anaconda3/2025.06-1   # or a specific Python module

# 2. Create venv on the work filesystem (run once)
mkdir -p /cluster/work/$USER/venvs
python3 -m venv /cluster/work/$USER/venvs/<project-name>

# 3. Activate and install dependencies
source /cluster/work/$USER/venvs/<project-name>/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Option B: Conda Environment
```bash
# 1. Module setup (do NOT use 'set -u' before loading modules)
module purge
module load Anaconda3/2025.06-1

# 2. Activate project environment
conda activate /cluster/work/$USER/conda-envs/<project-name>
# Or named environment if configured in ~/.condarc:
# conda activate <project-name>
```

### PyTorch Architecture Compatibility (Lowest Common Denominator)
> [!TIP]
> **Cross-GPU & Cross-Machine Portability:**
> IDUN hosts a heterogeneous GPU fleet ranging from older nodes (`P100` [Pascal, SM 6.0], `V100` [Volta, SM 7.0]) to modern nodes (`A100` [Ampere, SM 8.0], `H100` [Hopper, SM 9.0]).
> 
> To avoid maintaining separate environment configurations for each experiment or getting runtime failures when allocated a different GPU:
> * Pin PyTorch to a **lowest-common-denominator build** that retains backward compatibility across older compute capabilities (e.g., `torch==2.12.1+cu126` or a CUDA 11.8/12.1 build with SM 6.0/7.0 support) in your `requirements.txt`.
> * Avoid unpinned `pip install torch` or indiscriminate package upgrades that install newer wheels compiled only for SM 8.0+, which will crash with `CUDA error: no kernel image is available for execution on the device` if scheduled on a V100 or P100.

### Shell Strictness & Module Loading Gotcha
> [!WARNING]
> Do **not** use `set -u` or `set -eux` before `module purge` or `module load`. IDUN's environment initialization scripts reference optional unset variables (e.g. `QT_XCB_GL_INTEGRATION`), causing strict shells to abort. Use `set -eo pipefail` around module setup, and enable nounset only afterward if needed.

### JAX with CUDA
Check the node CUDA capability with `nvidia-smi` before selecting wheels. For CUDA 12 nodes:
```bash
python -m pip install --upgrade "jax[cuda12]"
```
Never use CPU-only `jaxlib` for GPU jobs. Verify inside an interactive Slurm job (`srun`):
```bash
python -c "import jax; print(jax.devices())"
```

---

## 5. Neural Network Training Best Practices

1. **Pre-training Diagnostics:** Always print diagnostics to Slurm logs before training loops start:
   ```bash
   echo "Host: $(hostname)"
   echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-none}"
   command -v nvidia-smi >/dev/null && nvidia-smi
   ```
2. **Unbuffered Output:** Run Python scripts with unbuffered output (`python -u` or `export PYTHONUNBUFFERED=1`) so print/log statements stream in real-time to Slurm log files.
3. **WandB Offline Mode:** IDUN compute nodes may lack external internet access. Enable offline logging and point directory to `/cluster/work/`:
   ```bash
   export WANDB_MODE=offline
   export WANDB_DIR="/cluster/work/$USER/wandb"
   ```
4. **Checkpoint Resumption:** Always save frequent model checkpoints to `/cluster/work/<username>/checkpoints/` and ensure training scripts can resume from the latest `.pt` or `.safetensors` file.

---

## 6. Slurm Job Monitoring for Agentic AI

When pair-programming or delegating long-running training jobs to an agentic AI assistant, avoid active polling loops in chat (which burn tokens and context window). Use one of two primary monitoring patterns:

### Approach A: Reactive Background Watcher (Best for Short-to-Medium Runs < 2h)
The agent executes an asynchronous background task on the local machine that discretely queries the remote Slurm queue (e.g. `slurm_watcher.py` or a shell loop with keepalive). Because queries are discrete or use keepalives, it is immune to idle TCP drops:
```python
# distill-experiments/slurm_watcher.py
python3 distill-experiments/slurm_watcher.py <slurm_job_id> --interval 60
```
Or via shell with SSH keepalive flags (`-o ServerAliveInterval=15 -o ServerAliveCountMax=3`):
```bash
python3 -c '
import subprocess, time, sys
job_id = sys.argv[1]
while subprocess.run(["ssh", "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=15", "username@idun-login1.hpc.ntnu.no", f"squeue -j {job_id} -h"], capture_output=True, text=True).stdout.strip():
    time.sleep(60)
print(f"SLURM_JOB_DONE:{job_id}")
' <slurm_job_id>
```
* **Reactive Wakeup:** The agentic runtime automatically triggers an event callback waking up the agent the moment the background watcher exits, so the agent can immediately parse results, inspect checkpoints, and continue the workflow without burning context tokens while waiting.

### Approach B: Filesystem Sentinel Hook with Scheduled Heartbeats (Best for Long Sweeps & Multi-Day Runs)
For long jobs or multi-job arrays where network/VPN drops can interrupt persistent SSH sessions:
1. **Add an exit trap to the `.sbatch` script**:
   ```bash
   STATUS_DIR="/cluster/work/$USER/job_status"
   mkdir -p "$STATUS_DIR"

   cleanup() {
       EXIT_CODE=$?
       echo "{\"job_id\": $SLURM_JOB_ID, \"exit_code\": $EXIT_CODE, \"finished_at\": \"$(date -Iseconds)\"}" \
           > "$STATUS_DIR/${SLURM_JOB_ID}.status"
   }
   trap cleanup EXIT
   ```
2. **Scheduled Inspection**: The agent sets a periodic schedule or timer (e.g. every 15–30 minutes) to inspect the tail of `slurm-%j.out` for intermediate progress (loss, epoch metrics) and verify if `${SLURM_JOB_ID}.status` has appeared. This is fully resilient to VPN resets and login node restarts.

