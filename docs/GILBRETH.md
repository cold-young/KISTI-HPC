<div align="center">

# gilbreth-cli

**Workflow CLI for the Purdue RCAC Gilbreth GPU cluster**

[![ENG](https://img.shields.io/badge/ENG-2B41D8?style=for-the-badge)](GILBRETH.md)
[![KOR](https://img.shields.io/badge/KOR-9AA4B2?style=for-the-badge)](GILBRETH.ko.md)

</div>

> The cluster is named **Gilbreth** (after Prof. Lillian Gilbreth) and lives at `gilbreth.rcac.purdue.edu`.
> Everything here follows the [official RCAC Gilbreth guide](https://docs.rcac.purdue.edu/userguides/gilbreth/).

---

## Contents

1. [Gilbreth at a glance](#1-gilbreth-at-a-glance)
2. [Install and set up](#2-install-and-set-up)
3. [Daily workflow (Quick Start)](#3-daily-workflow-quick-start)
4. [Command reference](#4-command-reference)
5. [experiment.yaml schema](#5-experimentyaml-schema)
6. [Scenarios](#6-scenarios)
7. [Live server test plan](#7-live-server-test-plan)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Gilbreth at a glance

What differs from the previous cluster this tool family started on.

| Item | Gilbreth | Notes |
|------|----------|-------|
| Login | `ssh <id>@gilbreth.rcac.purdue.edu` | Purdue career password + MFA push. **SSH keys are allowed** (no MFA afterwards) |
| Home | `/home/<id>` (25 GB) | Snapshot backups. Too small for conda envs |
| Scratch | `/scratch/gilbreth/<id>` (100 TB, 2M files) | No backup. **Files untouched for 60 days are purged** |
| Group storage | `/depot/<group>` | Long-term, purchased per group |
| Partitions | `a10`, `a30`, `a100-40gb`, `a100-80gb`, `h100`, `training` | **GPU type = partition**. No `--gres=gpu:a100` strings |
| QOS | `normal` (charged, up to 14 d) / `standby` (free, 4 h max, low priority) / `training` (training partition only, 24 h) | Set with `-q` |
| Required sbatch flags | `-A <account> -p <partition> -q <qos> --time --gpus-per-node=N` | Default walltime is 30 min. Also set `--mem` |
| Accounts | `slist` | RCAC calls a queue an "account" |
| Login nodes | Each has one A30 | OK for short GPU smoke tests, not for real work |
| Compute node internet | **Yes** (pypi, HuggingFace verified) | pip install, online wandb and HF downloads all work inside jobs |
| Python | `module load conda` (Miniforge) or `anaconda` | `conda-env-mod` style also supported |
| Containers | `apptainer` (ex-singularity) preinstalled | `/scratch`, `/depot`, `/apps` bind automatically |
| Interactive | `sinteractive -A ... -p ... --gpus-per-node=1 -t 1:00:00` | |
| Web | [gateway.gilbreth.rcac.purdue.edu](https://gateway.gilbreth.rcac.purdue.edu/) (Open OnDemand: Jupyter, file browser) / [desktop.gilbreth.rcac.purdue.edu](https://desktop.gilbreth.rcac.purdue.edu/) (ThinLinc) | |
| Bulk transfer | Globus collection `Purdue Gilbreth Cluster - Scratch` | rsync goes through the login host |

Node inventory (2026): A30 ×3 (B, D), A100-40GB ×2 (G), A10 ×3 (H), A100-80GB ×2 (I, K), A100-80GB ×4 NVLink (J), H100 ×2 (L), A100-40GB ×4 NVLink (N). 100 Gbps InfiniBand between nodes.

### 1.1 What the live cluster actually looks like (2026-10-14, account `shey`)

| Item | Observed |
|------|----------|
| Login node | `gilbreth-fe00` (Rocky Linux 9.8, 128 cores, 503 GB, one A30, internet access) |
| My account | `shey` only. QOS `normal`, `standby`, `training` all allowed |
| GPUs I can use | **A30 ×8** (partition a30), **A100-40GB ×4** (partition a100-40gb, shared by the group). a10 / a100-80gb / h100 are N/A (`slist`) |
| Load | 60+ of 72 A30s busy, about 490 jobs pending in standby. Even `normal` can wait minutes to tens of minutes |
| a30 nodes | 24 cores / 190 GB / 3× A30 (B nodes), 16 cores / 190 GB / 3× A30 (D nodes). **Memory runs out first**: request 32 to 60 GB per GPU |
| a100-40gb nodes | G: 128 cores / 510 GB / 2× A100, N: 48 cores / 1 TB / 4× A100 NVLink (`--constraint=nvlink`) |
| Default walltime | 30 min (`DefaultTime=00:30:00`). Without `time` the job dies at 30 min |
| QOS limits | normal 14 d, standby 4 h, training 24 h (`sacctmgr show qos`) |
| Home | `/home/ahn205`, 25 GB |
| Scratch | `/scratch/gilbreth/ahn205` (`$RCAC_SCRATCH`, `$CLUSTER_SCRATCH`), **200 TB** / 2M files |
| Depot | no group depot (`/depot/shey` does not exist) |
| Modules loaded at login | `gcc/11.5.0 openmpi/4.1.6 cuda/12.6.0 xalt external rcac`. **After `module purge`, run `module load rcac` first** or conda/anaconda/cuda-13 are invisible |
| conda | `module load conda` gives Miniforge (conda 25.3, conda-forge channel). Versions `conda/2024.09 2025.02 2025.09 (default) 2026.03`. `anaconda/2025.06-py313` and `conda-env-mod` also exist |
| CUDA / cuDNN | `cuda/12.1.1`, `cuda/12.6.0` (login default), `cuda/13.1.0` (what plain `module load cuda` picks). `cudnn/9.2.0.82-12`. Driver 595 |
| System python | `/usr/bin/python3` = 3.9. No `uv` (pip-installable) |
| Containers | `apptainer 1.5.3` (= `singularity`). **NGC PyTorch container modules**: `module load ngc/default pytorch/25.01-py3` turns `python` into `singularity run --nv nvcr.io/nvidia/pytorch:25.01-py3`. Images live in `/apps/ngc/images/*.sif` |
| Interactive | `sinteractive` is a `salloc` wrapper; every option passes through (`--gpus-per-node`, `--cpus-per-gpu` fine) |
| Archive | `hsi`, `htar` (Fortress) installed |
| Caveat: conda cache | `module load conda` pins `CONDA_PKGS_DIRS` and `CONDA_ENVS_PATH` to **home**. gilbreth-cli re-exports them to scratch after the module load (`gilbreth-env setup` puts envs in `/scratch/gilbreth/<id>/envs/` and the package cache in `/scratch/.../.conda/pkgs`). Do the same order by hand |

`gilbreth-env setup --python 3.11 --name rcac_test` took about a minute and created `/scratch/gilbreth/ahn205/envs/rcac_test` (181 MB).

**Combinations that work for this account**

```bash
# quick experiments / debugging: A30 (24 GB)
gilbreth-run -p a30 -g 1 --mem 32G -t 02:00:00 python train.py
# bigger models: A100-40GB (4 shared by the group, usually 1 or 2 free)
gilbreth-run -p a100-40gb -g 1 --mem 64G -t 08:00:00 python train.py
# a 4-GPU NVLink node
gilbreth-run -p a100-40gb -g 4 -C nvlink --mem 240G -t 08:00:00 --torchrun train.py
```

`-q standby` is free but the queue currently holds close to 500 jobs, so expect hours. Use `normal` when it matters.

---

## 2. Install and set up

### 2.1 Install

```bash
cd <this-repo>
pip install -e .
gilbreth-exp --help
```

### 2.2 Username and SLURM account

```bash
# ~/.zshrc or ~/.bashrc
export GILBRETH_USER="purdue_id"        # career account
export GILBRETH_ACCOUNT="mygroup"       # from slist (leave empty until step 2.4 if unknown)
```

You can also put them in `configs/gilbreth.yaml` (`ssh.user`, `slurm.account`). Precedence: `CLI option > experiment.yaml > env var > gilbreth.yaml`.

### 2.3 SSH config and first connection

**MFA prerequisite (once).** Since 2026-05-11 RCAC uses **Microsoft Authenticator** instead of Duo.
1. Install Microsoft Authenticator on your phone (Google Play / App Store).
2. Sign in at https://mysignins.microsoft.com/security-info with `<id>@purdue.edu` and your career password → "Add sign-in method" → Microsoft Authenticator → scan the QR code.
3. On the same page set the default method to **"App based authentication - notification"** (SSH and ThinLinc only work with push). Add a backup method such as SMS.
4. At the SSH password prompt type only your password; a push request appears on the phone. If it does not, open the app and approve the pending request (a known issue RCAC has announced).

```bash
gilbreth-connect --setup-config     # adds/refreshes a 'Host gilbreth' block in ~/.ssh/config
gilbreth-connect                    # password -> approve MFA on your phone -> session lives 12 h
gilbreth-status                     # Gilbreth: CONNECTED
```

While the session is alive every `gilbreth-*` command, `ssh gilbreth` and `rsync` reuse it without re-authenticating.

**To drop MFA entirely** (Gilbreth accepts SSH keys):

```bash
ssh-keygen -t ed25519               # only if you have no key yet
gilbreth-connect --setup-key        # ssh-copy-id, one last MFA
```

### 2.4 Find your account and run the doctor

```bash
gilbreth-info accounts              # slist: accounts you can charge and their GPU balance
export GILBRETH_ACCOUNT="<one of them>"
gilbreth-info doctor                # config, session, scratch, module conda/cuda, apptainer, sinteractive
```

All OK means you are ready.

---

## 3. Daily workflow (Quick Start)

```bash
# 0. once a day
gilbreth-connect

# 1. Python env (once) - created under /scratch/gilbreth/<id>/envs/<name>
gilbreth-env setup --from environment.yml           # name: taken from the yml
gilbreth-env setup --python 3.11 --name myenv -r requirements.txt   # no yml

# 2. ship the code
gilbreth-sync push ./my_project/                    # -> /scratch/gilbreth/<id>/projects/my_project
gilbreth-sync watch ./my_project/                   # auto-push on every save

# 3-a. fastest path: one command becomes one batch job
cd my_project
gilbreth-run -g 1 -t 02:00:00 python train.py --lr 1e-4
gilbreth-run -g 1 -t 00:30:00 -q standby python train.py --epochs 1     # free standby smoke test

# 3-b. sweeps / pipelines: experiment.yaml
gilbreth-exp init                                   # write a template
gilbreth-exp launch experiment.yaml --dry-run       # render scripts only
gilbreth-exp launch experiment.yaml                 # submit

# 4. monitor
gilbreth-exp status                                 # squeue
gilbreth-exp logs <JOB_ID>                          # tail -f
gilbreth-exp watch <JOB_ID> --pull ./my_project     # pull experiments/ when done
gilbreth-exp history                                # sacct, last 3 days

# 5. fetch results
gilbreth-sync pull ./my_project/                    # experiments/, results/, *.json, *.csv
gilbreth-sync pull ./my_project/ --all-files

# 6. debug on a GPU node
gilbreth-shell -g 1 -t 01:00:00                     # sinteractive, starts in the project dir
gilbreth-shell --login                              # login-node shell (one A30 for quick tests)
```

---

## 4. Command reference

### 4.1 Session (`gilbreth-connect` / `-status` / `-disconnect`)

| Command | What it does |
|---------|--------------|
| `gilbreth-connect` | Open the ControlMaster session (password + MFA once, 12 h) |
| `gilbreth-connect --setup-config` | Create/refresh the `Host gilbreth` block in `~/.ssh/config` |
| `gilbreth-connect --setup-key [-i key.pub]` | Install your public key (`ssh-copy-id`) |
| `gilbreth-status` | Session state |
| `gilbreth-disconnect` | Close the session |

### 4.2 Sync (`gilbreth-sync`)

```bash
gilbreth-sync push [PATH] [-p PROJECT] [--delete] [--dry-run]    # local -> scratch/projects/PROJECT
gilbreth-sync pull [PATH] [-p PROJECT] [--all-files] [--dry-run] # scratch -> local (results only by default)
gilbreth-sync watch [PATH] [-p PROJECT] [-i 2]                   # auto-push on change
gilbreth-sync push-data ./data/ [--dest REMOTE_DIR]              # -> scratch/data/<name>
```

Exclude patterns (`.git/`, `__pycache__/`, `wandb/`, `*.pt`, `outputs/` ...) and pull patterns live under `sync:` in `configs/gilbreth.yaml`.

### 4.3 Environments (`gilbreth-env`)

```bash
gilbreth-env setup --from environment.yml [-n NAME] [-r requirements.txt]
gilbreth-env setup --python 3.11 --name NAME [-r requirements.txt]
gilbreth-env list                     # scratch envs + conda env list
gilbreth-env activate NAME            # prints the 3 activate lines to paste in a shell
gilbreth-env export NAME -o env.yml   # remote env -> local yml
gilbreth-env remove NAME
gilbreth-env modules [QUERY]          # module avail / module spider QUERY
```

Envs are created with `--prefix /scratch/gilbreth/<id>/envs/<NAME>` to protect the 25 GB home quota. An env idle for 60 days can be purged, so keep `environment.yml` in the repo.

**Installing PyTorch** (conda-forge channel by default; pip wheels are simplest):

```bash
gilbreth-env setup --python 3.11 --name torch_env
gilbreth-shell --login
module load conda; export CONDA_PKGS_DIRS=$RCAC_SCRATCH/.conda/pkgs PIP_CACHE_DIR=$RCAC_SCRATCH/.cache/pip
eval "$(conda shell.bash hook)"; conda activate $RCAC_SCRATCH/envs/torch_env
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print(torch.cuda.is_available())"   # the login node's A30 answers immediately
```

Or keep a `requirements.txt` and run `gilbreth-env setup --python 3.11 --name torch_env -r requirements.txt`.

**Skip the env entirely with the NGC PyTorch container** (fastest when you need no extra packages):

```yaml
environment:
  type: apptainer
  container: /apps/ngc/images/nvcr.io_nvidia_pytorch:25.01-py3.sif
```
```bash
gilbreth-run --container /apps/ngc/images/nvcr.io_nvidia_pytorch:25.01-py3.sif -p a30 -g 1 -t 01:00:00 python train.py
```

### 4.4 Experiments (`gilbreth-exp`)

```bash
gilbreth-exp init [-o experiment.yaml]
gilbreth-exp launch experiment.yaml [--dry-run] [-n N] [-A acct] [-p part] [-q qos] [--gpus N] [--time T] [--mem M] [--no-sync]
gilbreth-exp status [--all-users] [-p PARTITION]
gilbreth-exp history [-d DAYS]
gilbreth-exp logs JOB_ID [--no-follow] [-n LINES]
gilbreth-exp watch [JOB_ID ...] [-i SEC] [--pull DIR] [--project NAME]
gilbreth-exp cancel JOB_ID | all
gilbreth-exp attach JOB_ID            # shell into a running job's node (srun --jobid)
```

`launch` first rsyncs the directory containing `experiment.yaml` (skip with `--no-sync`), uploads configs and scripts, then submits with `sbatch --parsable`. Scripts are kept locally in `~/.gilbreth/runs/<project>_<timestamp>/` and remotely in `projects/<project>/scripts/<timestamp>/`.

### 4.5 One-off jobs and interactive shells (`gilbreth-run`, `gilbreth-shell`)

```bash
gilbreth-run [options] COMMAND...     # options go BEFORE the command; the command's own -t/--lr are passed through untouched
  -A acct  -p part  -q qos  -g GPUS  --cpus-per-gpu N  --mem 32G  -t HH:MM:SS  -C nvlink
  --project NAME     # remote dir (default: current directory name)
  --env NAME         # conda env  (--env-type conda|conda-env-mod|apptainer|venv|none)
  --container X.sif  # run inside apptainer
  --torchrun         # wrap with torchrun --nproc_per_node=<GPUS>
  --no-sync          # skip pushing the current directory
  --dry-run

gilbreth-shell [-g 1] [-t 01:00:00] [-p a30] [-q standby] [--cd PROJECT]   # sinteractive
gilbreth-shell --login                                                     # login node
```

### 4.6 Cluster info (`gilbreth-info`)

```bash
gilbreth-info accounts        # slist
gilbreth-info partitions      # partition table + live sinfo when connected
gilbreth-info nodes [-p PART] # sfeatures
gilbreth-info quota           # myquota
gilbreth-info purge-check [-d 50]   # files close to the 60-day purge
gilbreth-info doctor          # config / session / module checks
```

---

## 5. experiment.yaml schema

```yaml
project: my_model                 # remote: /scratch/gilbreth/<id>/projects/my_model

slurm:                            # missing keys fall back to slurm: in configs/gilbreth.yaml
  account: mygroup                # default GILBRETH_ACCOUNT
  partition: a100-80gb            # a10 | a30 | a100-40gb | a100-80gb | h100 | training
  qos: normal                     # normal | standby | training (forced for the training partition)
  nodes: 1
  gpus: 1                         # --gpus-per-node
  cpus_per_gpu: 8
  mem: 64G
  time: "08:00:00"
  constraint: nvlink              # optional

environment:
  type: conda                     # conda | conda-env-mod | apptainer | venv | none
  name: train_env                 # conda: envs/<name> or absolute path
                                  # conda-env-mod: the name printed by conda-env-mod (e.g. my_env-py3.12.3)
                                  # venv: envs/<name> or absolute path
  container: /scratch/gilbreth/<id>/img.sif   # for apptainer
  modules: [cuda]                 # extra module loads (conda/anaconda is added from type)
  launcher: python                # python | torchrun

env_vars:                         # exported in the job (merged with env_vars in gilbreth.yaml)
  TOKENIZERS_PARALLELISM: "false"

script: train.py                  # single-stage entry point
args: "--config {config_path} --lr {learning_rate}"
  # placeholders: base_params keys, sweep keys, {config_path}, {exp_id}

base_params: {epochs: 10, seed: 42}

sweep:
  method: grid                    # grid | random (count: N, seed: S)
  mode: array                     # array: one sbatch --array | separate: one sbatch per combination
  parallel: 3                     # max concurrent array tasks (%3)
  parameters:
    learning_rate: [1.0e-4, 3.0e-4]

pipeline:                         # when present, replaces script/args; stages chain with afterok
  - name: train
    script: train.py
    args: "--config {config_path}"
  - name: eval
    script: evaluate.py
    args: "--run {exp_id}"
```

Inside the job you get `PROJECT`, `EXPERIMENT_ID`, `CONFIG_PATH` (per-combination YAML incl. `_exp_id`), `STAGE_NAME`, `MASTER_ADDR`, `MASTER_PORT` and the usual `SLURM_*`. Logs go to `/scratch/gilbreth/<id>/logs/<jobname>_<jobid>.out|err` (array jobs: `<jobname>_<arrayid>_<task>.out`); completions are appended to `logs/completed.log`.

Note: PyYAML reads `1e-4` as a string. Write floats as `1.0e-4`. `launch` pushes the whole directory containing `experiment.yaml`, so keep that file at the project root.

---

## 6. Scenarios

### A. Run one training script fast

```bash
gilbreth-connect
cd my_project
gilbreth-env setup --python 3.11 --name myenv -r requirements.txt   # once
gilbreth-run -g 1 -t 04:00:00 --env myenv python train.py --lr 3e-4
gilbreth-exp logs <JOB_ID>
```

### B. Hyper-parameter sweep (job array)

```bash
gilbreth-exp init
# edit sweep.parameters (6 combinations)
gilbreth-exp launch experiment.yaml --dry-run      # inspect ~/.gilbreth/runs/.../scripts/job_array.sh
gilbreth-exp launch experiment.yaml                # a single sbatch --array=0-5%3
gilbreth-exp watch --pull ./my_project
```

### C. Multi-GPU DDP (torchrun)

```yaml
slurm: {partition: a100-80gb, gpus: 4, mem: 240G, time: "12:00:00", constraint: nvlink}
environment: {type: conda, name: myenv, launcher: torchrun}
script: train_ddp.py
```

With `nodes: 2` the template switches to `srun torchrun --nnodes=$SLURM_NNODES --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT ...`. Jobs needing more than 4 GPUs belong on the `training` partition (`-p training`; the QOS is forced to `training`, 24 h), which RCAC grants on request.

### D. Apptainer container (Isaac Lab, Genesis, heavy deps)

```bash
apptainer build sim.sif containers/genesis-sim.def
gilbreth-sync push-data ./sim.sif --dest /scratch/gilbreth/$GILBRETH_USER/images

gilbreth-run -g 1 -t 02:00:00 --container /scratch/gilbreth/$GILBRETH_USER/images/sim.sif python run_sim.py
```

The template runs `apptainer exec --nv --bind /scratch/gilbreth/<id> [--bind /depot/<group>]`. Set `paths.depot` to bind the depot as well.

### E. Interactive development / Jupyter

```bash
gilbreth-shell -g 1 -t 02:00:00 -q standby     # free standby GPU shell
# on the node
module load conda; eval "$(conda shell.bash hook)"; conda activate /scratch/gilbreth/$USER/envs/myenv
```

For Jupyter use Open OnDemand at [gateway.gilbreth.rcac.purdue.edu](https://gateway.gilbreth.rcac.purdue.edu/), picking account and partition in the form. Install `ipykernel` in the scratch env to use it as a kernel.

### F. Free compute with standby

Anything under 4 hours can run with `-q standby` without charging your account. Queue times are longer, so use it for debugging and short runs.

```bash
gilbreth-run -q standby -g 1 -t 03:59:00 python train.py --epochs 5
```

---

## 7. Live server test plan

| # | Test | Command | Pass criteria |
|---|------|---------|---------------|
| 1 | Login | `gilbreth-connect --setup-config && gilbreth-connect && gilbreth-status && ssh gilbreth hostname` | hostname printed with no extra prompt after one MFA |
| 2 | Doctor | `gilbreth-info accounts && gilbreth-info doctor` | accounts listed, doctor all OK (adjust `gilbreth.yaml` if module names differ) |
| 3 | Sync | `mkdir -p /tmp/gtest && echo "print('hi')" > /tmp/gtest/t.py && gilbreth-sync push /tmp/gtest && ssh gilbreth ls /scratch/gilbreth/$GILBRETH_USER/projects/gtest` | t.py listed |
| 4 | Env | `gilbreth-env setup --python 3.11 --name gtest_env && gilbreth-env list` | envs/gtest_env exists |
| 5 | One-off job | `cd /tmp/gtest && gilbreth-run -q standby -g 1 -t 00:05:00 -p a30 --env gtest_env python -c "import torch,os;print(os.environ['CUDA_VISIBLE_DEVICES'])"` | job id printed, `gilbreth-exp logs ID` shows a GPU index |
| 6 | Sweep | `gilbreth-exp init && gilbreth-exp launch experiment.yaml --dry-run && gilbreth-exp launch experiment.yaml -q standby -p a30 --time 00:05:00` | array job submitted and visible in `gilbreth-exp status` |
| 7 | Results | `gilbreth-exp watch <ID> --pull /tmp/gtest && gilbreth-exp history` | pulled after completion, sacct shows COMPLETED |
| 8 | Interactive | `gilbreth-shell -q standby -p a30 -g 1 -t 00:10:00` then `nvidia-smi` | shell on a GPU node |

### 7.1 Results on the live cluster (2026-10-14, ahn205 / shey)

```
[x] Test 1: login         gilbreth-fe00, one MFA then ControlMaster reuse, ssh gilbreth hostname OK
[x] Test 2: doctor        slist / scratch / module conda / cuda / apptainer / sinteractive / account all OK
[x] Test 3: sync          gilbreth-run pushed to /scratch/gilbreth/ahn205/projects/gtest
[x] Test 4: env           gilbreth-env setup --python 3.11 --name rcac_test -> about 1 min, 181 MB on scratch
[x] Test 5: one-off job   gilbreth-run -p a30 -g 1 --mem 8G -t 00:05:00 -> Job 11950925, started 4 min after submit, COMPLETED
                          compute node gilbreth-b011: A30 24 GB, CUDA_VISIBLE_DEVICES=0, cuda/12.6.0 loaded,
                          pypi.org / huggingface.co returned 200 (compute nodes have internet), scratch writable, /tmp 193 GB, apptainer 1.5.3
[x] Test 6: monitoring    gilbreth-exp status / logs / history / cancel verified
[x] Test 7: sweep (array) gilbreth-exp launch (2 combinations, conda env rcac_test) -> Job 11951056 array, both tasks COMPLETED
                          quoted args ('exp {exp_id}') arrived intact, gilbreth-exp watch --pull fetched eval_results.json automatically
[ ] Test 8: interactive   only the sinteractive option pass-through was checked (salloc wrapper)
```

Bug caught on the live cluster: `module purge` in the job script also unloaded the `external` module, after which `module load conda` failed with "exist but cannot be loaded". The template now runs `module load rcac` (`environment.base_module`) right after the purge to restore the login stack before loading conda/cuda. Hand-written sbatch scripts need the same order.

Observations: `-q standby` did not start within our window (about 490 jobs queued) so the test switched to `normal`; `module purge` prints a harmless warning about the sticky `xalt` module. Compute nodes have internet, so `WANDB_MODE=offline` is not required (removed from the defaults).

---

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Host key verification failed` / rsync 255 | no ControlMaster session | `gilbreth-status` then `gilbreth-connect` |
| MFA push keeps coming | session expired (12 h) or stale socket | `gilbreth-disconnect; rm -f ~/.ssh/sockets/*gilbreth*; gilbreth-connect`, or `--setup-key` |
| `sbatch: error: Invalid account` | `-A` missing or misspelled | `gilbreth-info accounts`, fix `GILBRETH_ACCOUNT` |
| `Invalid qos specification` | `normal` on the training partition, or standby on a partition without it | `-q training` or `-q normal` |
| Job dies at 30 min | no `time` set (Gilbreth default) | set `slurm.time` |
| `module: command not found` / module missing | module renamed | check `gilbreth-env modules conda` / `cuda`, update `gilbreth.yaml` |
| `conda: command not found` inside the job | wrong `conda_module` | set `environment.conda_module: anaconda` |
| Disk quota exceeded (home) | 25 GB home full (pip/HF caches) | keep envs and caches in scratch; default `env_vars` set `HF_HOME`, `XDG_CACHE_HOME`; `pip cache purge` |
| Scratch files vanished | 60-day inactivity purge | `gilbreth-info purge-check` early, keep copies via `gilbreth-sync pull`, Depot or Fortress |
| Pending forever | partition busy | `gilbreth-info partitions` for idle nodes, switch to `-p a30` or `-q standby` |
| OOM | `mem` too low | raise `--mem` (nodes have 192 GB to 2 TB) |
| No log yet | not started | check the Reason column in `gilbreth-exp status` |

Support: rcac-help@purdue.edu

**Sources**
- [Gilbreth User Guide](https://docs.rcac.purdue.edu/userguides/gilbreth/)
- [Using AI Agents (partition/QOS/module summary)](https://docs.rcac.purdue.edu/userguides/gilbreth/using_ai_agents/)
- [Queues](https://docs.rcac.purdue.edu/userguides/gilbreth/run_jobs/queues/) · [Submit script](https://docs.rcac.purdue.edu/userguides/gilbreth/run_jobs/submit_script/) · [GPU jobs](https://docs.rcac.purdue.edu/userguides/gilbreth/run_jobs/gpu_jobs/) · [Custom ML env](https://docs.rcac.purdue.edu/userguides/gilbreth/run_jobs/learning/customml/) · [Apptainer](https://docs.rcac.purdue.edu/userguides/gilbreth/run_jobs/apptainer_example/)
- [Storage quota](https://docs.rcac.purdue.edu/userguides/gilbreth/storage/storage_quota/) · [Globus](https://docs.rcac.purdue.edu/userguides/gilbreth/storage/globus/)
- [ML-Purdue cnn-workshop (real-world example)](https://github.com/ML-Purdue/cnn-workshop)
