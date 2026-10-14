# NTNU IDUN Agent Skill

An [Agent Skills](https://agentskills.io/) specification-compliant skill providing instructions, Slurm templates, environment guidelines, and best practices for developing and training neural networks on the **NTNU IDUN HPC cluster**.

---

## What This Skill Covers

* **Connection & Session Management:** VPN access, login node SSH guidelines, and `tmux`/`screen` session preservation.
* **Storage & Synchronization:** Work (`/cluster/work/`) vs. Home (`/cluster/home/`) quotas, Git code sync, and `rsync` dataset/weights transfer.
* **Slurm Job Configuration:** `GPUQ` vs `CPUQ` partitions, time limits (up to 14 days), GPU architecture flags (`h100`, `a100`, `v100`, `p100`), and account allocation.
* **Environment Setup:** Support for standard Python `venv`/`pip` (portable across machines) and Conda environments, quota protection via `/cluster/work/`, cross-GPU PyTorch baseline compatibility (`P100` through `H100`), CUDA 12 JAX, and shell strictness safeguards.
* **NN Training Best Practices:** Unbuffered logging, pre-run hardware diagnostics, offline WandB configuration, and persistent checkpointing.

---

## Skill Structure

```text
ntnu-idun-skills/
├── SKILL.md                  # Main Agent Skills definition (frontmatter + guidelines)
├── assets/
│   └── slurm_template.sbatch # Ready-to-use Slurm batch script template
└── README.md                 # Repository documentation
```

---

## How to Use This Skill

### In Antigravity / Gemini CLI / Codex / Claude Code / VS Code / Cursor

#### 1. Project-Level (Workspace)
To enable this skill in your project repository:
```bash
mkdir -p .agents/skills/
git clone git@github.com:maltanar/ntnu-idun-skills.git .agents/skills/ntnu-idun-skills
```
*(or add it as a git submodule)*:
```bash
git submodule add git@github.com:maltanar/ntnu-idun-skills.git .agents/skills/ntnu-idun-skills
```

#### 2. Global (Personal Machine)
To make this skill available across all your local workspaces:
* **Antigravity / Gemini:**
  ```bash
  git clone git@github.com:maltanar/ntnu-idun-skills.git ~/.gemini/config/skills/ntnu-idun-skills
  ```
* **Claude Code / Agent Skills global directory:**
  ```bash
  git clone git@github.com:maltanar/ntnu-idun-skills.git ~/.claude/skills/ntnu-idun-skills
  ```

---

## Validation

This skill conforms to the [Agent Skills Specification](https://agentskills.io/specification) and can be validated with the official reference tool:

```bash
python3 -m skills_ref.cli validate .
```

