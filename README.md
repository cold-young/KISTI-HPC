<div align="center">

# neuron-cli

**Workflow automation CLI for the KISTI Neuron HPC cluster**

[![ENG](https://img.shields.io/badge/ENG-2B41D8?style=for-the-badge)](README.md)
[![KOR](https://img.shields.io/badge/KOR-9AA4B2?style=for-the-badge)](README.ko.md)

</div>

---

## Contents

1. [Install and set up](#1-install-and-set-up)
2. [Daily workflow (Quick Start)](#2-daily-workflow-quick-start)
3. [Command reference](#3-command-reference)
4. [Scenarios](#4-scenarios)
5. [Live server test plan (Phase 5)](#5-live-server-test-plan-phase-5)
6. [Troubleshooting](#6-troubleshooting)

---

## 1. Install and set up

### 1.1 Install the package

```bash
cd KISTI-HPC
pip install -e .
```

### 1.2 Set your KISTI user ID

Pick one of the two:

**Option A) Environment variable (recommended)**
```bash
# add to ~/.zshrc or ~/.bashrc
export NEURON_USER="your_kisti_id"
```

**Option B) Edit the config file (recommended)**
```bash
# configs/neuron.yaml
ssh:
  user: "your_kisti_id"   # <-- put it here
```

### 1.3 Generate the SSH config

```bash
neuron-connect --setup-config
```

This appends the ControlMaster settings to `~/.ssh/config`.
To do it by hand, add this block to `~/.ssh/config` yourself:

```
Host neuron
    HostName neuron.ksc.re.kr
    User <YOUR_ID>
    ControlMaster auto
    ControlPath ~/.ssh/sockets/%r@%h-%p
    ControlPersist 4h
    ServerAliveInterval 60
```

---

## 2. Daily workflow (Quick Start)

The loop you repeat every research day.

### Step 1: Connect (once a day)

```bash
neuron-connect
# enter the OTP code -> the session stays open for 4 hours
# no OTP needed for any command after this
```

### Step 2: Sync your code

```bash
# push everything at once
neuron-sync push ./my_project/

# watch mode (re-syncs automatically on every file save)
neuron-sync watch ./my_project/
```

### Step 3: Run the experiment

```bash
# after writing experiment.yaml
neuron-exp launch experiment.yaml

# check status
neuron-exp status

# stream logs live
neuron-exp logs <JOB_ID>
```

### Step 4: Review results

```bash
neuron-results results my_project --sort accuracy
neuron-results best my_project --metric accuracy --top 5
```

### Step 5: Download results

```bash
neuron-sync pull ./my_project/         # result files only
neuron-sync pull ./my_project/ --all-files  # everything
```

---

## 3. Command reference

### 3.1 SSH session

| Command | What it does |
|--------|------|
| `neuron-connect` | Open the SSH session (OTP entered once) |
| `neuron-connect --setup-config` | Generate `~/.ssh/config` |
| `neuron-status` | Check connection state |
| `neuron-disconnect` | Close the session |

### 3.2 Code sync (`neuron-sync`)

```bash
neuron-sync push [PATH]                     # local -> Neuron
neuron-sync push . --delete                 # also delete remote files missing locally
neuron-sync push . --dry-run                # preview without transferring
neuron-sync pull [PATH]                     # Neuron -> local (results only)
neuron-sync pull . --all-files              # pull every file
neuron-sync watch [PATH]                    # auto-sync on file change
neuron-sync watch . --interval 5            # 5-second debounce
neuron-sync push-data ./data/ --via dm      # large data, routed through the DM node
```

**Excluded from sync** (managed in `configs/defaults.yaml`):
- `.git/`, `__pycache__/`, `wandb/`, `*.pt`, `*.pth`, `outputs/`, `logs/`

### 3.3 Environments (`neuron-env`)

```bash
neuron-env setup --from environment.yml     # create or update a Conda env
neuron-env setup --from env.yml --name myenv  # name it explicitly
neuron-env activate myenv                   # print the activate command (to copy)
neuron-env export myenv                     # remote env -> environment.yml
neuron-env list                             # list Conda envs on Neuron
```

### 3.4 Experiment manager (`neuron-exp`)

```bash
neuron-exp launch experiment.yaml           # submit the experiment
neuron-exp launch experiment.yaml -n 3      # only the first 3 combinations
neuron-exp launch experiment.yaml --dry-run # generate the script only
neuron-exp launch experiment.yaml --partition cas_v100nv_8  # override partition
neuron-exp launch experiment.yaml --gpus 4  # override GPU count
neuron-exp status                           # list running jobs
neuron-exp logs <JOB_ID>                    # stream logs (Ctrl+C to stop)
neuron-exp cancel <JOB_ID>                  # cancel one job
neuron-exp cancel all                       # cancel everything
```

### 3.5 Result comparison (`neuron-results`)

```bash
neuron-results results <PROJECT> --sort accuracy    # all results, sorted
neuron-results compare <PROJECT> exp1 exp2          # compare two runs
neuron-results best <PROJECT> --metric accuracy -t 5  # top 5
neuron-results sync-wandb <PROJECT>                 # W&B offline -> local
```

### 3.6 Partition picker (`neuron-partition`)

```bash
neuron-partition --gpus 2                   # best partition for 2 GPUs
neuron-partition --gpus 4 --prefer-cost     # weight cost efficiency higher
```

Sample output:
```
Rank  Partition          GPU    Idle   Mix   Perf   Cost   Score
1     amd_a100nv_8       A100   3      2     1.0    Med    63.0  <-- recommended
2     cas_v100nv_8       V100   5      1     0.5    Low    58.0
...
```

### 3.7 Containers (`neuron-container`)

```bash
neuron-container build containers/pytorch-base.def            # build locally
neuron-container build containers/genesis-sim.def --remote-build  # build on Neuron
neuron-container push pytorch-base.sif                        # upload via the DM node
neuron-container run /scratch/$USER/.neuron-tools/pytorch-base.sif python train.py
neuron-container list                                         # list remote images
```

### 3.8 File retention (`neuron-keepalive`)

```bash
neuron-keepalive check --days 12            # files due for deletion within 12 days
neuron-keepalive touch                      # refresh atime on important files
neuron-keepalive touch --dry-run            # preview
neuron-keepalive setup-cron --interval 7    # run automatically every 7 days
```

### 3.9 Notifications (`neuron-notify`)

```bash
neuron-notify setup                         # create the notify.yaml template
neuron-notify test --method slack           # send a test notification
neuron-notify test --method email           # send a test email
```

Called automatically from the SLURM script (you rarely run it yourself):
```bash
neuron-notify send --status $? --job-name $SLURM_JOB_NAME --exp-id $EXPERIMENT_ID
```

### 3.10 Simulators (`neuron-sim`)

```bash
neuron-sim list                             # simulator / GPU compatibility table
neuron-sim compat genesis -p amd_a100nv_8   # check compatibility
neuron-sim check-env genesis                # is it installed on Neuron?
neuron-sim init-pipeline genesis --env manipulation  # generate experiment.yaml
neuron-sim scaffold genesis --env manipulation       # generate starter code
```

---

## 4. Scenarios

### Scenario A: Architecture search

```bash
# 1. connect
neuron-connect

# 2. build the environment
neuron-env setup --from environment.yml

# 3. sync code (background watcher)
neuron-sync watch ./src/ &

# 4. write experiment.yaml
cat > experiment.yaml << 'EOF'
project: arch_search
slurm:
  partition: amd_a100nv_8
  gpus: 2
  time: "12:00:00"
  comment: pytorch
environment:
  name: my_env
  modules: [cuda/12.4.1]
base_params:
  dataset: /scratch/$USER/data/my_data
  epochs: 50
  batch_size: 64
  seed: 42
sweep:
  method: grid
  parameters:
    learning_rate: [1e-4, 3e-4, 1e-3]
    model_arch: [resnet50, vit_base, vit_large]
    hidden_dim: [256, 512]
pipeline:
  - name: train
    script: train.py
    args: "--config {config_path}"
  - name: evaluate
    script: evaluate.py
    depends_on: train
    args: "--checkpoint checkpoints/best_model.pth"
EOF

# 5. dry-run first
neuron-exp launch experiment.yaml --dry-run

# 6. launch (18 combinations = 3 lr * 3 arch * 2 dim)
neuron-exp launch experiment.yaml

# 7. read the results
neuron-exp status
neuron-results best arch_search --metric accuracy --top 5
```

### Scenario B: Genesis simulation pipeline

```bash
# 1. check compatibility
neuron-sim compat genesis -p amd_a100nv_8

# 2. generate starter code + pipeline
neuron-sim scaffold genesis --env manipulation -o ./src/
neuron-sim init-pipeline genesis --env manipulation -o experiment_sim.yaml

# 3. fill in the generated TODOs
# src/genesis_datagen.py  - data generation
# src/train.py            - training
# src/genesis_eval.py     - evaluation

# 4. prepare the container
neuron-container build containers/genesis-sim.def --remote-build
# or build locally, then upload
neuron-container build containers/genesis-sim.def
neuron-container push genesis-sim.sif

# 5. run the pipeline (datagen -> train -> eval chains automatically)
neuron-exp launch experiment_sim.yaml
```

### Scenario C: Migrating existing Isaac Gym code

```bash
# 1. pin the Isaac Gym environment (Singularity)
neuron-container build containers/isaacgym-preview.def --remote-build

# 2. check compatibility
neuron-sim compat isaac_gym -p amd_a100nv_8

# 3. sync and run the existing code
neuron-sync push ./isaacgym_project/
neuron-exp launch experiment_isaacgym.yaml

# Note: Isaac Gym is deprecated
# Start new projects on Genesis
neuron-sim list  # compare compatibility
```

---

## 5. Live server test plan (Phase 5)

How the CLI gets validated against the real Neuron cluster, one step at a time.

### 5.1 Before you start

| Item | Check | Command |
|------|----------|--------|
| KISTI account | ID/PW plus a registered Any OTP app | - |
| Local install | `pip install -e .` succeeded | `neuron-exp --help` |
| User ID | Environment variable set | `echo $NEURON_USER` |

### 5.2 Step-by-step tests

#### Test 1: SSH connection (5 min)

```bash
# 1-1. generate the SSH config
neuron-connect --setup-config

# 1-2. connect (enter OTP)
neuron-connect

# 1-3. confirm
neuron-status
# expected: "Neuron: CONNECTED"

# 1-4. confirm the session persists (no second OTP)
ssh neuron 'hostname'
# expected: neuron01, neuron02, or neuron03

# 1-5. disconnect
neuron-disconnect
neuron-status
# expected: "Neuron: DISCONNECTED"
```

**Pass criteria**: SSH commands run without further authentication after a single OTP entry

#### Test 2: File sync (10 min)

```bash
# 2-1. create a test project
mkdir -p /tmp/neuron_test && echo "print('hello')" > /tmp/neuron_test/test.py

# 2-2. push
neuron-connect
neuron-sync push /tmp/neuron_test/ -p neuron_test

# 2-3. verify on the remote
ssh neuron 'ls /scratch/$USER/projects/neuron_test/'
# expected: test.py

# 2-4. pull
ssh neuron 'echo "result=42" > /scratch/$USER/projects/neuron_test/eval_results.json'
neuron-sync pull /tmp/neuron_test/ -p neuron_test --all-files

# 2-5. watch mode (separate terminal)
neuron-sync watch /tmp/neuron_test/ -p neuron_test
# edit a file from another terminal -> confirm it syncs automatically
```

**Pass criteria**: push and pull both succeed, watch mode picks up changes on its own

#### Test 3: Environment setup (15 min)

```bash
# 3-1. create a test environment.yml
cat > /tmp/environment.yml << 'EOF'
name: neuron_test_env
dependencies:
  - python=3.10
  - pytorch
  - pip:
    - wandb
    - rich
EOF

# 3-2. create the environment
neuron-env setup --from /tmp/environment.yml

# 3-3. verify
neuron-env list

# 3-4. check the activate command
neuron-env activate neuron_test_env
```

**Pass criteria**: the Conda environment is created and appears in `neuron-env list`

#### Test 4: SLURM submission (20 min)

```bash
# 4-1. write a minimal training script
cat > /tmp/neuron_test/train.py << 'PYEOF'
import os, json, time
config_path = os.environ.get("CONFIG_PATH", "")
exp_id = os.environ.get("EXPERIMENT_ID", "unknown")
print(f"Running experiment: {exp_id}")
print(f"Config: {config_path}")
print(f"GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'none')}")
# simple simulation
time.sleep(10)
results = {"accuracy": 0.85, "loss": 0.32, "exp_id": exp_id}
os.makedirs("experiments/" + exp_id, exist_ok=True)
with open(f"experiments/{exp_id}/eval_results.json", "w") as f:
    json.dump(results, f)
print("Done!")
PYEOF

# 4-2. write experiment.yaml
cat > /tmp/neuron_test/experiment.yaml << 'EOF'
project: neuron_test
slurm:
  partition: cas_v100_2
  nodes: 1
  gpus: 1
  cpus_per_gpu: 4
  time: "00:10:00"
  comment: pytorch
environment:
  name: neuron_test_env
  modules:
    - cuda/12.4.1
base_params:
  epochs: 1
  batch_size: 32
  seed: 42
sweep:
  method: grid
  parameters:
    learning_rate: [1e-4, 3e-4]
EOF

# 4-3. sync
neuron-sync push /tmp/neuron_test/ -p neuron_test

# 4-4. dry-run
neuron-exp launch /tmp/neuron_test/experiment.yaml --dry-run

# 4-5. submit for real (small partition for the test)
neuron-exp launch /tmp/neuron_test/experiment.yaml --partition cas_v100_2 --gpus 1

# 4-6. status
neuron-exp status

# 4-7. logs
neuron-exp logs <JOB_ID>
```

**Pass criteria**: sbatch submission succeeds, the job shows up in squeue, log streaming works

#### Test 5: Result collection (10 min)

```bash
# 5-1. once the job finishes
neuron-results results neuron_test --sort accuracy
neuron-results best neuron_test --metric accuracy

# 5-2. download
neuron-sync pull /tmp/neuron_test/ -p neuron_test
```

**Pass criteria**: `eval_results.json` parses correctly, sort/compare output renders

#### Test 6: Partition, retention, notifications (10 min)

```bash
# 6-1. partition recommendation
neuron-partition --gpus 2

# 6-2. retention status
neuron-keepalive check --days 12
neuron-keepalive touch --dry-run

# 6-3. notification test (if Slack is configured)
neuron-notify test --method slack
```

**Pass criteria**: sinfo parsing and scoring print correctly, keepalive file scan completes

#### Test 7: Simulator environment (optional, 30 min)

```bash
# 7-1. is Genesis installed?
neuron-sim check-env genesis

# 7-2. if not, build the container
neuron-container build containers/genesis-sim.def --remote-build

# 7-3. import test
neuron-container run /scratch/$USER/.neuron-tools/genesis-sim.sif \
    python -c "import genesis; print('Genesis OK')"
```

**Pass criteria**: Genesis imports successfully (from Conda or inside Singularity)

### 5.3 Checklist (verified 2026-05-07)

```
[x] Test 1: SSH connection (connect/status/disconnect) - glogin01, single OTP
[x] Test 2: File sync (push/pull) - rsync + ControlPath
[ ] Test 3: Environment setup (setup/list/activate) - conda unusable directly on Neuron, containers recommended
[x] Test 4: SLURM submission (launch/status/logs/cancel) - Jobs 717737, 717738 succeeded
[x] Test 5: Result collection (results/best/compare) - 3 experiments retrieved
[x] Test 6: Partition / retention - H200 recommended, at-risk files detected
[ ] Test 7: Simulator container build (optional)
```

### 5.4 Issues found on the live server

| Issue | Cause | Resolution |
|------|------|----------|
| conda unusable directly on Neuron | `conda init` not run in the base environment | Fixed - the template skips `source activate` when the env is `base` |
| `module purge` kills the job instantly | It strips the SLURM environment variables too | Fixed - `module purge` removed from the template |
| scp does not support ControlMaster | scp cannot multiplex over the socket | Fixed - scp replaced with rsync everywhere |
| `$USER` not expanded in SLURM scripts | KISTI environment quirk | Fixed - replaced with the Jinja2 variable `{{ user }}` |
| sbatch only runs from scratch | KISTI policy | Fixed - `_run_remote` wraps commands in `cd /scratch/$USER &&` |

### 5.5 Other issues to expect

| Issue | Cause | What to do |
|-----------|------|----------|
| OTP session expired | Past the 4-hour ControlPersist window | Run `neuron-connect` again |
| rsync connection refused | No SSH socket | Check `neuron-status`, then `neuron-connect` |
| sbatch fails | Partition full, or a bad option | Find an alternative with `neuron-partition`, inspect the script with `--dry-run` |
| module load fails | Module name changed | Run `ssh neuron 'module avail cuda'`, then update `configs/defaults.yaml` |
| Files deleted from /scratch | Untouched for 15 days | Set up `neuron-keepalive setup-cron` right away |
| Compute nodes have no network | pip and wandb cannot reach the internet | Use a Singularity container, set `WANDB_MODE=offline` |

---

## 6. Troubleshooting

### SSH connection problems

```bash
# check the socket
ls -la ~/.ssh/sockets/

# clear a dead socket
neuron-disconnect
rm -f ~/.ssh/sockets/*neuron*
neuron-connect
```

### SLURM job never starts

```bash
# check the queue
neuron-partition --gpus 2

# move to an emptier partition
neuron-exp launch experiment.yaml --partition cas_v100_2 --gpus 1
```

### No result files

```bash
# read the logs
neuron-exp logs <JOB_ID>

# look on the remote directly
ssh neuron 'ls /scratch/$USER/projects/<PROJECT>/experiments/'
```

### Watch mode is not syncing

```bash
# check the exclude patterns (sync.exclude in configs/defaults.yaml)
neuron-sync push . --dry-run

# force a sync
neuron-sync push .
```
