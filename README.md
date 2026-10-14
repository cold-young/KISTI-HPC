<div align="center">

# rcac-cli

**A local command line for running GPU experiments on Purdue RCAC's Gilbreth cluster.**

[English](README.md) · [한국어](README.ko.md)

</div>

`rcac-cli` keeps you on your laptop. It opens one authenticated SSH session to a
Gilbreth front-end node, then pushes code, builds environments, renders and
submits `sbatch` scripts, watches jobs and pulls results back through that same
session — so Purdue Login prompts you once, not once per command.

It targets **Gilbreth only**. Every partition, QOS and filesystem fact it
encodes is cited to `docs.rcac.purdue.edu` in [§2](#2-gilbreth-in-facts), and
everything it could not confirm from those pages is listed as an open row in
[§7](#7-live-verification-checklist) rather than guessed at.

## Also here: gilbreth-cli (live-verified sibling)

A second implementation, `gilbreth_cli` (`gilbreth-connect` / `gilbreth-status` / `gilbreth-disconnect`, `gilbreth-sync`, `gilbreth-env`, `gilbreth-exp`, `gilbreth-run`, `gilbreth-shell`, `gilbreth-info`),
was written independently and verified on the real cluster on 2026-10-14 (account `shey`: A30 and A100-40GB jobs, scratch conda
envs, job-array sweeps). Its guide, including what the live cluster actually looks like, is in
[docs/GILBRETH.md](docs/GILBRETH.md) / [docs/GILBRETH.ko.md](docs/GILBRETH.ko.md). The two tools do not share code; see
[§9](#9-project-history) for how they relate.

## Contents

1. [Install and set up](#1-install-and-set-up)
2. [Gilbreth, in facts](#2-gilbreth-in-facts)
3. [The submission triple](#3-the-submission-triple)
4. [Daily workflow (Quick Start)](#4-daily-workflow-quick-start)
5. [Command reference](#5-command-reference)
6. [Scenarios](#6-scenarios)
7. [Live-verification checklist](#7-live-verification-checklist)
8. [Troubleshooting](#8-troubleshooting)
9. [Project history](#9-project-history)

---

## 1. Install and set up

### 1.1 Install the package

```bash
git clone <this-repo> && cd <this-repo>
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Twelve console scripts land on your `PATH`; [§5](#5-command-reference)
documents all twelve. Nothing is installed on Gilbreth — the cluster side is
plain `ssh`, `rsync` and `sbatch`.

### 1.2 Tell rcac-cli who you are

Two values, and **the environment is the primary path**:

```bash
export RCAC_USER=<your Purdue career account id>
export RCAC_ACCOUNT=<the Slurm account you may charge>
```

`RCAC_USER` and `RCAC_ACCOUNT` **always beat** `configs/rcac.yaml`. That file is
tracked by git and ships with the placeholders `YOUR_PURDUE_ID` and
`YOUR_QUEUE`; the environment path exists so nobody has to edit a tracked file
to use the tool. Put the two `export` lines wherever you keep your shell
environment.

Do not know your account yet? Run `rcac-partition` — it works with an
unresolved user and prints the accounts the cluster says you may charge. On the
cluster itself, `slist` prints the same thing.

Anything that genuinely needs a resolved value — a scratch path, an `sbatch -A`
— fails at the point of use with a message naming the environment variable and
the config file it looked in. **No credential and no path is ever silently
substituted**, and neither is `-p` or `--time`: `configs/defaults.yaml` ships no
partition (which partitions you may use depends on your account) and no
walltime (Gilbreth's own default is 30 minutes, which kills essentially any
real job).

Two *resource requests* do carry defaults, and say so. Omit `--mem` and it is
sized as `gpus_per_node × 32G`; omit `--cpus-per-gpu` and it is 8. Neither
number comes from an RCAC page — `configs/defaults.yaml` records that beside
each — and both are printed in the `#SBATCH` header `--dry-run` shows you, so
neither is hidden. The asymmetry is deliberate: a wrong resource request queues
longer or is OOM-killed, while a wrong account charges somebody else's
allocation.

### 1.3 Generate the SSH config

```bash
rcac-connect --setup-config   # writes one managed Host block to ~/.ssh/config
rcac-connect                  # authenticate once; the session stays open
```

This matters more on Gilbreth than on a single-headed cluster. The name
`gilbreth.rcac.purdue.edu` resolves to **four front-end nodes**, round-robin, so
without multiplexing your `sbatch` and your `squeue` can land on different
hosts. The managed block sets `ControlMaster auto` with a persistent
`ControlPath` — OpenSSH's mechanism for **pinning every later `rcac-*` command
to the one front-end node the master session authenticated against**, and for
prompting Purdue Login once per `ControlPersist` window instead of once per
command.

That is what the configuration asks for, not a verified fact about Gilbreth.
No page checked says whether its front ends accept SSH control sockets; a site
can refuse them. Row **L2** of §7.2 is the single command that settles it, and
it blocks the merge for exactly this reason — so until it has been run, treat
every claim below about the socket being reused as the intended behaviour
rather than the observed one.

```bash
rcac-status                    # is the master socket alive?
rcac-disconnect                # close it
rcac-disconnect --remove-config   # remove the managed block from ~/.ssh/config
```

The managed block is delimited by begin/end markers, so `--remove-config`
removes exactly what `--setup-config` wrote and leaves the rest of your
`~/.ssh/config` byte-for-byte intact.

### 1.4 Keep your Purdue ID out of git

```bash
git config core.hooksPath .githooks
```

That one command installs a `pre-commit` hook that parses every staged
`configs/*.yaml` and **refuses the commit** if `ssh.user` or `slurm.account`
holds anything other than a placeholder. CI runs the same check on every push,
but the hook is what stops a credential **before** it reaches a commit — CI can
only tell you it already did. Install it before your first commit.

`git commit --no-verify` bypasses it. Do that only when you know why.

---

## 2. Gilbreth, in facts

Everything in this section is quoted from RCAC's own documentation, and every
row carries the URL it came from. Facts that RCAC does **not** publish are not
here — they are open rows in [§7](#7-live-verification-checklist).

### 2.1 Access and the four front-end nodes

<!-- facts:begin -->
| Fact | Value | [Source](https://docs.rcac.purdue.edu/userguides/gilbreth/accounts/) |
|---|---|---|
| Login | `ssh <your-id>@gilbreth.rcac.purdue.edu` | [accounts](https://docs.rcac.purdue.edu/userguides/gilbreth/accounts/) |
| Authentication | Purdue's MFA two-factor (approve the notification) **or** SSH keys | [accounts](https://docs.rcac.purdue.edu/userguides/gilbreth/accounts/) |
| Front-end nodes | **4 nodes**, 64 cores, 512 GB memory and 1× A30 (24 GB) each | [overview](https://docs.rcac.purdue.edu/userguides/gilbreth/overview/) |
| Web portal | Open OnDemand at `gateway.gilbreth.rcac.purdue.edu` | [gateway](https://docs.rcac.purdue.edu/userguides/gilbreth/gateway/) |
<!-- facts:end -->

Because one hostname fronts four nodes, a session that is not multiplexed is a
session that changes machines under you. That is what §1.3's ControlMaster
block is meant to prevent — pending L2.

### 2.2 Partitions — `-p` selects the GPU type

<!-- facts:begin -->
| `-p` | GPU | Nodes | GPUs/node | Walltime | [Source](https://docs.rcac.purdue.edu/userguides/gilbreth/run_jobs/queues/) |
|---|---|---|---|---|---|
| `a10` | A10, 24 GB | 16 | 3 | 2 weeks | [queues](https://docs.rcac.purdue.edu/userguides/gilbreth/run_jobs/queues/) |
| `a30` | A30, 24 GB | 24 | 3 | 2 weeks | [queues](https://docs.rcac.purdue.edu/userguides/gilbreth/run_jobs/queues/) |
| `a100-40gb` | A100, 40 GB | 32 (12 G-nodes + 20 N-nodes) | 2 on G, 4 on N | 2 weeks | [queues](https://docs.rcac.purdue.edu/userguides/gilbreth/run_jobs/queues/) |
| `a100-80gb` | A100, 80 GB | 57 | 2 | 2 weeks | [queues](https://docs.rcac.purdue.edu/userguides/gilbreth/run_jobs/queues/) |
| `training` | mixed (V100 / A100 / H100) | 8 | 4 | 24 hours | [queues](https://docs.rcac.purdue.edu/userguides/gilbreth/run_jobs/queues/) |
| `h100` | **contradicted — see below** | — | — | — | [faqs](https://docs.rcac.purdue.edu/userguides/gilbreth/faqs/) vs [queues](https://docs.rcac.purdue.edu/userguides/gilbreth/run_jobs/queues/) |
<!-- facts:end -->

**On `h100`, this project does not pick a side.** RCAC's FAQ page says "if you
wanted to run on an H100 GPU… specify `--partition=h100`", and its sample
`slist` output prints an H100 column. RCAC's queues page lists no `h100`
partition at all and places H100 hardware inside `training`. Both pages are
current. `rcac-cli` therefore carries `h100` as an **unverified** entry: it will
not silently drop it, and it will not treat it as real. Submitting to it warns
and names the contradiction. One `slist` on the cluster settles it — that is
[L1](#72-the-checklist).

The N nodes of `a100-40gb` are the NVLinked ones:

<!-- facts:begin -->
| Constraint | When | Rule | [Source](https://docs.rcac.purdue.edu/userguides/gilbreth/run_jobs/queues/) |
|---|---|---|---|
| `--constraint='nvlink'` | `-p a100-40gb` only | use it when **multiple GPUs serve a single task**, to land on the NVLinked Gilbreth-N nodes | [queues](https://docs.rcac.purdue.edu/userguides/gilbreth/run_jobs/queues/) |
<!-- facts:end -->

It is the only node feature RCAC documents for Gilbreth. `rcac-cli` will not
emit any other `--constraint` value for you.

### 2.3 QOS — `-q` is a QOS, never an account

<!-- facts:begin -->
| `-q` | Cap | Charged against your allocation? | Notes | [Source](https://docs.rcac.purdue.edu/userguides/gilbreth/run_jobs/queues/) |
|---|---|---|---|---|
| `normal` | partition maximum | yes | the default; high priority, adds no extra limits | [queues](https://docs.rcac.purdue.edu/userguides/gilbreth/run_jobs/queues/) |
| `standby` | **4 hours** | **no** | low priority on idle resources; request with `-q standby` / `--qos=standby` | [queues](https://docs.rcac.purdue.edu/userguides/gilbreth/run_jobs/queues/) |
| `training` | 24 hours | yes | **required** with `-p training` | [queues](https://docs.rcac.purdue.edu/userguides/gilbreth/run_jobs/queues/) |
| `-A standby` | — | — | **abolished**: the `standby` QOS "replaces the previous `-A standby` style of submission" | [queues](https://docs.rcac.purdue.edu/userguides/gilbreth/run_jobs/queues/) |
<!-- facts:end -->

`standby` is the single most useful thing on this page: **four uncharged hours**
is enough to compile, smoke-test a training loop, or debug a data pipeline
without spending allocation. Use it for everything that is not a real run.

`rcac-cli` rejects `-A standby` with a message pointing at `-q standby`, and
rejects `-p training` without `-q training` rather than silently rewriting your
submission into a 24-hour cap you did not budget for.

### 2.4 Submission flags

<!-- facts:begin -->
| Flag | Required | Meaning | [Source](https://docs.rcac.purdue.edu/userguides/gilbreth/run_jobs/submit_script/) |
|---|---|---|---|
| `-A <account>` | yes | "SLURM requires each job that is submitted to be associated with an **Account**" | [submit_script](https://docs.rcac.purdue.edu/userguides/gilbreth/run_jobs/submit_script/) |
| `-p <partition>` | yes | "we also require each job to be associated with a specific **Partition**, which specifies what type of GPU to use" | [submit_script](https://docs.rcac.purdue.edu/userguides/gilbreth/run_jobs/submit_script/) |
| `-q <qos>` | no (defaults to `normal`) | `normal`, `standby` or `training` | [queues](https://docs.rcac.purdue.edu/userguides/gilbreth/run_jobs/queues/) |
| `--gpus-per-node=N` | yes for GPU jobs | "GPU jobs must specify the number of GPUs using the `--gpus-per-node` option" | [submit_script](https://docs.rcac.purdue.edu/userguides/gilbreth/run_jobs/submit_script/) |
| `--mem=<size>` | yes | requested alongside the GPU count | [submit_script](https://docs.rcac.purdue.edu/userguides/gilbreth/run_jobs/submit_script/) |
| `--time=<walltime>` | in practice, yes | "By default, jobs receive 30 minutes of wall time" | [submit_script](https://docs.rcac.purdue.edu/userguides/gilbreth/run_jobs/submit_script/) |
<!-- facts:end -->

**There is no documented `--gres` GPU-type string on Gilbreth.** The partition
already chose the GPU; `--gpus-per-node` chooses how many. `rcac-cli` never
generates a `--gres` GPU-type request, and the test suite fails the build if one
appears in a template.

### 2.5 Filesystems and the 60-day purge

<!-- facts:begin -->
| Path or command | What it is | [Source](https://docs.rcac.purdue.edu/userguides/gilbreth/storage/) |
|---|---|---|
| `/home/<your-id>` | your home directory; quota-limited, with nightly/weekly/monthly snapshots kept for roughly two to three months | [home_directory](https://docs.rcac.purdue.edu/userguides/gilbreth/storage/home_directory/) |
| `/scratch/gilbreth/<your-id>` | large parallel scratch; **"Files in scratch directories are not backed up"** | [scratch_space](https://docs.rcac.purdue.edu/userguides/gilbreth/storage/scratch_space/) |
| purge policy | **"Files that have not been accessed or had content modified in 60 days are purged"** | [scratch_space](https://docs.rcac.purdue.edu/userguides/gilbreth/storage/scratch_space/) |
| `$RCAC_SCRATCH` | "your scratch directory path… may change without warning, but this variable will remain current" — use it in scripts | [scratch_space](https://docs.rcac.purdue.edu/userguides/gilbreth/storage/scratch_space/) |
| `findscratch` | prints your scratch path, e.g. `/scratch/gilbreth/myusername` | [scratch_space](https://docs.rcac.purdue.edu/userguides/gilbreth/storage/scratch_space/) |
| `myquota` | "To check the current quotas of your home and scratch directories use the `myquota` command" | [storage_quota](https://docs.rcac.purdue.edu/userguides/gilbreth/storage/storage_quota/) |
<!-- facts:end -->

**Read your quota from `myquota`, not from this README.** RCAC's example output
shows a 25 GB home limit and a 100 TB scratch limit, but those are the numbers
in a published example, not a promise about your account. `rcac-keepalive check`
runs `myquota` for you and prints what it actually said.

Sixty days is measured from the **later** of access time and content
modification time, so a dataset you merely read still counts as fresh.
`rcac-keepalive touch` refreshes access times on the files you nominate;
`rcac-keepalive setup-cron` does it on a schedule.

### 2.6 Moving data

<!-- facts:begin -->
| Method | Use it for | [Source](https://docs.rcac.purdue.edu/userguides/gilbreth/storage/globus/) |
|---|---|---|
| Globus at `https://transfer.rcac.purdue.edu` | bulk transfers; "works on any operating system that is connected to the internet" | [globus](https://docs.rcac.purdue.edu/userguides/gilbreth/storage/globus/) |
| collection **"Gilbreth Cluster Collection"** | reaches your Gilbreth home and scratch; searching for "gilbreth" suggests it | [globus](https://docs.rcac.purdue.edu/userguides/gilbreth/storage/globus/) |
| SCP / SFTP | small and medium transfers, over the login host | [storage](https://docs.rcac.purdue.edu/userguides/gilbreth/storage/) |
<!-- facts:end -->

**Gilbreth documents no dedicated data-transfer node.** There is no separate
transfer host to configure, and `rcac-cli` has no setting for one. `rcac-sync
push` and `pull` go over the login host through the existing ControlMaster
socket; `rcac-sync push-data` does the same and prints the Globus route above a
size threshold, where a web transfer that survives a closed laptop beats an
`rsync` that does not.

### 2.7 Software: modules, environments, containers

<!-- facts:begin -->
| Topic | What RCAC documents | [Source](https://docs.rcac.purdue.edu/userguides/gilbreth/software/) |
|---|---|---|
| Module system | Lmod; discover with `module spider` / `module avail`, load with `module load` | [software](https://docs.rcac.purdue.edu/userguides/gilbreth/software/) |
| Python environments | `module purge` → `module load anaconda` → `conda-env-mod create -n <name>` → `module load use.own` → `module load conda-env/<name>-py<X.Y>` | [customml](https://docs.rcac.purdue.edu/userguides/gilbreth/run_jobs/learning/customml/) |
| GPU libraries | `module load cuda` and `module load cudnn`, only if your framework does not bundle them | [customml](https://docs.rcac.purdue.edu/userguides/gilbreth/run_jobs/learning/customml/) |
| Containers | Apptainer; RCAC overlays "your home directory, scratch, Data Depot, and application file systems" — `/home/$USER`, `/apps`, `/scratch`, `/depot` | [apptainer_example](https://docs.rcac.purdue.edu/userguides/gilbreth/run_jobs/apptainer_example/) |
| Accounts and GPU access | `slist` "will print a table of the access to each GPU type for each account you are a member of" | [faqs](https://docs.rcac.purdue.edu/userguides/gilbreth/faqs/) |
<!-- facts:end -->

Three consequences worth stating outright:

- **`conda-env-mod`, not bare `conda`.** RCAC is explicit that loading the
  modules `conda-env-mod` generates "has different behavior" from
  `conda create` plus `source activate`, and recommends the module route. The
  environment's identity is therefore a *module name*, `conda-env/<name>-pyX.Y`.
- **Apptainer, and it needs no `module load`.** It is not Docker and it is not
  the older runtime it replaced. Because RCAC overlays `/home/$USER`, `/apps`,
  `/scratch` and `/depot` for you, re-binding them with `-B` is at best a no-op.
- **Do not invent module names or versions.** RCAC's own guidance, and the
  reason this README names module *families* (`anaconda`, `cuda`, `cudnn`,
  `use.own`) and no version strings. `rcac-env modules` asks the cluster; a
  `module load` of something that does not exist fails your job at line one.

NGC containers are available on RCAC systems through `module load ngc`. The
image directory `/apps/ngc/images/` is documented for **Gautschi**, not for
Gilbreth, so this project never assumes that path exists here — see
[L18](#72-the-checklist).

---

## 3. The submission triple

This is the one thing to carry away from the whole document. A Gilbreth
submission takes **four parts, not one**:

```
sbatch -A <account>  -p <partition>  -q <qos>  --time=<walltime>  ...
        └ who pays    └ which GPU     └ which  └ or you get
                                        rules    30 minutes
```

Other clusters fold all of this into `-p`. Gilbreth does not: `-p` is the **GPU
type**, `-A` is the **allocation being billed**, and `-q` is the **QOS**. Miss
`-A` or `-p` and the job is rejected; miss `--time` and it is killed after 30
minutes.

`rcac-cli` builds all four from one place, so `rcac-exp launch`, the keepalive
sweep and the simulator's generated manifests cannot disagree about them:

```bash
rcac-exp launch experiment.yaml --dry-run \
  -A <account> -p a100-40gb -q normal --time 04:00:00 --mem 64G
```

`--dry-run` renders the script and prints the exact `sbatch` command **without
contacting the cluster at all**. Read it before you submit. Precedence runs
CLI flag → the manifest's `slurm:` block → `configs/defaults.yaml`.

---

## 4. Daily workflow (Quick Start)

### Step 1: Connect

```bash
rcac-connect          # once per ControlPersist window; two-factor once
rcac-status
```

### Step 2: Sync your code

```bash
cd ~/work/my_project
rcac-sync push                 # local -> /scratch/gilbreth/<you>/projects/my_project
rcac-sync push --dry-run       # see what would move first
rcac-sync watch                # keep pushing on every save
```

### Step 3: Launch

```bash
rcac-exp launch experiment.yaml --dry-run -p a100-40gb --time 04:00:00
rcac-exp launch experiment.yaml -p a100-40gb --time 04:00:00
```

Start with `--dry-run` every time you change the `slurm:` block. It is free and
it is the only place a bad `-p`/`-q` pair is cheap to find.

### Step 4: Watch

```bash
rcac-exp status
rcac-exp logs <job-id>
rcac-exp watch my_project --auto-pull ./results
```

### Step 5: Pull results

```bash
rcac-sync pull
rcac-results best my_project -m val_accuracy
rcac-results compare my_project exp_001 exp_007
```

---

## 5. Command reference

Twelve console scripts, all installed by `pip install -e .`:

| Script | Purpose |
|---|---|
| `rcac-connect` | open the ControlMaster session; write the SSH config |
| `rcac-status` | is the session alive? |
| `rcac-disconnect` | close the session; remove the managed SSH block |
| `rcac-sync` | push/pull code and data |
| `rcac-env` | `conda-env-mod` environments |
| `rcac-exp` | render, submit, watch and cancel jobs |
| `rcac-results` | read and compare experiment output |
| `rcac-partition` | which accounts and GPUs can you use? |
| `rcac-container` | Apptainer images |
| `rcac-keepalive` | survive the 60-day scratch purge |
| `rcac-notify` | Slack / email job notifications |
| `rcac-sim` | simulator compatibility and pipeline scaffolding |

### 5.1 SSH session — `rcac-connect`, `rcac-status`, `rcac-disconnect`

```bash
rcac-connect --setup-config       # write the managed Host block
rcac-connect                      # authenticate; hold the master socket
rcac-status                       # report on the socket
rcac-disconnect                   # close it
rcac-disconnect --remove-config   # remove the managed block
```

One `Host gilbreth` stanza is written, carrying `ControlMaster`, `ControlPath`
and `ControlPersist`, plus a comment explaining the four front ends. Everything
else in your `~/.ssh/config` is left alone.

### 5.2 Code sync — `rcac-sync`

```bash
rcac-sync push [PATH] [-p PROJECT] [--dry-run] [--delete]
rcac-sync pull [PATH] [-p PROJECT] [--all-files] [--dry-run]
rcac-sync watch [PATH] [-p PROJECT] [-i SECONDS]
rcac-sync push-data <LOCAL_DATA_PATH> [--dry-run]
```

Exclusions and pull filters live in `configs/defaults.yaml` under `sync:`.
Every transfer is routed over the ControlMaster socket, so a sync should not
trigger a second two-factor prompt — intended behaviour, unverified until §7.2's
L2 and L15. `push-data` prints the Globus route for anything
large enough that a browser-driven transfer is the better tool.

### 5.3 Environments — `rcac-env`

```bash
rcac-env modules [PATTERN]        # ask the cluster what exists
rcac-env setup --name train_env --py 3.12 [--from environment.yml] [--gpu]
rcac-env list
rcac-env activate train_env [--py 3.12]
rcac-env export train_env -o environment.yml
```

`rcac-env setup --name train_env --py 3.12` runs the documented sequence and
leaves you with a module named **`conda-env/train_env-py3.12`**. The `-py3.12`
suffix is part of the name, which matters in one specific place:

```yaml
environment:
  type: conda
  name: train_env-py3.12      # the FULL module name, suffix included
```

A manifest's `environment.name` is what the generated job script passes to
`module load conda-env/<name>`, so dropping the suffix produces a job that dies
on its first line. `rcac-env activate` prints exactly the two lines the job
script uses — `module load use.own` and `module load conda-env/<name>-pyX.Y` —
and nothing else, so `$(rcac-env activate train_env)` is safe to embed.

`--py` is discovered from the cluster's `anaconda` module when you omit it, and
never guessed.

### 5.4 Experiments — `rcac-exp`

```bash
rcac-exp launch <experiment.yaml> [--dry-run] [-n MAX_JOBS]
    [-A ACCOUNT] [-p PARTITION] [-q QOS] [--time T] [--mem M]
    [--gpus N] [--constraint nvlink] [--no-discover] [--no-cache]
rcac-exp status
rcac-exp logs <JOB_ID>
rcac-exp watch <PROJECT> [--auto-pull DIR] [-i SECONDS] [-j JOB_IDS]
rcac-exp cancel <JOB_ID|all>
```

A manifest declares the project, the environment, the sweep and a `slurm:`
block; `examples/experiment_simple.yaml` and
`examples/experiment_genesis.yaml` are working starting points and both parse
through `--dry-run` as shipped.

Note what the examples deliberately omit: **`-A` is never written into a
manifest.** An account is a credential, and a manifest is a file you might
commit.

### 5.5 Results — `rcac-results`

```bash
rcac-results results <PROJECT> [-s METRIC] [-l LIMIT]
rcac-results best <PROJECT> [-m METRIC] [-t N]
rcac-results compare <PROJECT> <EXP_ID>...
rcac-results sync-wandb <PROJECT> [--exp-id ID]
```

`sync-wandb` pulls offline W&B runs down and syncs them locally. That is the
usual shape when compute nodes have no outbound route to the internet; no page
checked says whether Gilbreth's do, so offline mode is the safe default here
rather than a documented necessity.

### 5.6 Partition picker — `rcac-partition`

```bash
rcac-partition [-g GPUS] [--prefer-cost] [--no-discover] [--no-cache]
```

Asks the cluster which accounts you may charge and which GPU types each of them
reaches — the `slist` matrix — then ranks the partitions for the job you
described. **It exits 0 whether or not the cluster answered**, because advice
that aborts is worse than advice with a caveat on it. When it falls back to the
built-in table, it says so under a banner, and unverified entries such as
`h100` stay labelled unverified.

This is the command to run before you have an `RCAC_ACCOUNT`.

### 5.7 Containers — `rcac-container`

```bash
rcac-container build <DEF_FILE> [-n NAME] [--remote-build]
rcac-container push <IMAGE.sif> [-d REMOTE_DIR]
rcac-container list
rcac-container run <REMOTE_IMAGE.sif> [COMMAND...] [-B PATH] [--nv]
```

Apptainer throughout. `--nv` is on by default for `run`. `-B` is for paths RCAC
does **not** already overlay — `/home`, `/apps`, `/scratch` and `/depot` are
mounted for you. Uploaded images default to `<paths.scratch>/.rcac-tools/`.

`--remote-build` uploads the definition file and builds on a front-end node
without requesting an allocation, because RCAC documents Apptainer as rootless.
If that build fails, the tool prints the interactive-allocation fallback rather
than assuming it was needed — see [L17](#72-the-checklist).

Four definition files ship under `containers/`.

### 5.8 Scratch retention — `rcac-keepalive`

```bash
rcac-keepalive check [-d DAYS] [--no-discover]
rcac-keepalive touch [--dry-run] [-v]
rcac-keepalive setup-cron [-i DAYS] [--batch]
```

`check` gives three answers, most authoritative first: what the cluster says is
already staged for deletion, what `myquota` reports about the space itself, and
which files a local `find -atime` scan puts near the window. The default warning
threshold is 45 days, which keeps a quarter of the 60-day window in hand.

`setup-cron` runs the sweep **on a front-end node** by default. The sweep is a
metadata-only `find … -exec touch -a`, far below any reasonable front-end
budget, whereas a batch job would probably have to hold a GPU it never uses:
every partition RCAC documents for Gilbreth selects a GPU type (§2.2), and no
page checked names a CPU-only one. That is the reason for the default, not a
rule the cluster is known to enforce.
`--batch` submits it as a job anyway, for sweeps large enough to deserve one or
sites whose local policy forbids the front-end route.

### 5.9 Notifications — `rcac-notify`

```bash
rcac-notify setup                 # write a notify.yaml template
rcac-notify test [-m slack|email] [--message TEXT]
rcac-notify send --status 0 --job-name NAME --exp-id ID [--metric M] [-m METHOD]
```

`notify.yaml` resolves through the same three tiers as every other config:
`$RCAC_CONFIG_DIR` → `./configs/` → the copy packaged inside `rcac_cli`.
`send` is what generated job scripts call on completion.

### 5.10 Simulators — `rcac-sim`

```bash
rcac-sim list
rcac-sim compat <genesis|mujoco|isaac_gym|isaac_sim|pybullet> [-p PARTITION]
rcac-sim check-env <genesis|mujoco|isaac_gym|pybullet>
rcac-sim init-pipeline <genesis|mujoco|isaac_gym> [-p PARTITION] [--env NAME] [--episodes N] [-o FILE]
rcac-sim scaffold <genesis|mujoco|isaac_gym> [--env NAME] [-o DIR]
```

`init-pipeline` validates the `slurm:` block **as it writes it**, not later at
launch. A generated manifest with an illegal partition/QOS pair would otherwise
survive happily until someone submitted it, possibly on another machine, days
later, having assumed a generated file was a valid one.

---

## 6. Scenarios

### Scenario A: an architecture sweep on A100s

```bash
rcac-connect
cd ~/work/my_model && rcac-sync push

rcac-env setup --name train_env --py 3.12 --from environment.yml --gpu
# -> module conda-env/train_env-py3.12 ; put that FULL name in the manifest

rcac-partition -g 2                          # which accounts reach a100-40gb?
rcac-exp launch experiment.yaml --dry-run \
    -p a100-40gb -q normal --time 08:00:00 --mem 64G --gpus 2
rcac-exp launch experiment.yaml -p a100-40gb -q normal --time 08:00:00 --mem 64G --gpus 2

rcac-exp watch my_model --auto-pull ./results
rcac-results best my_model -m val_accuracy
```

Multiple GPUs feeding a **single** task? Add `--constraint nvlink` to land on
the NVLinked Gilbreth-N nodes. Independent replicas do not need it.

### Scenario B: a Genesis simulation pipeline in a container

```bash
rcac-container build containers/genesis-sim.def --remote-build
rcac-container push genesis-sim.sif            # -> <scratch>/.rcac-tools/
rcac-container run <scratch>/.rcac-tools/genesis-sim.sif nvidia-smi

rcac-sim compat genesis -p a100-80gb
rcac-sim init-pipeline genesis -p a100-80gb --episodes 10000 -o experiment.yaml
rcac-exp launch experiment.yaml --dry-run --time 24:00:00
```

`examples/experiment_genesis.yaml` is the same pipeline written out by hand.

### Scenario C: four free hours on `standby`

```bash
rcac-exp launch experiment.yaml -n 1 \
    -p a30 -q standby --time 04:00:00 --mem 32G
```

Uncharged, low priority, hard 4-hour cap. This is the right way to find out
that your data loader path is wrong — and note that `-A standby` is **not** how
you ask for it; that form was abolished.

---

## 7. Live-verification checklist

`rcac-cli` was built entirely offline against RCAC's published documentation.
The rows below are the things those pages could **not** settle. None of them
blocks normal use; each is a one-line command on a login node that turns a
"probably" into a fact. Record the output, then delete the row.

### 7.1 Before you start

You need a Gilbreth account, an allocation to charge, and a working
`rcac-connect`. Run the whole list in one session so the ControlMaster socket is
shared. Where a row replaces a test fixture, the target path is given.

### 7.2 The checklist

| # | Open question | Command | What it settles |
|---|---|---|---|
| **L0** | Connectivity | `ssh <id>@gilbreth.rcac.purdue.edu hostname` | which of the 4 front ends answered |
| **L1** | Does an `h100` **partition** exist? | `slist` ; `sinfo -s` ; `sinfo -o '%P %G'` | the FAQ-vs-queues contradiction in §2.2 |
| **L2** | Does ControlMaster actually multiplex here? | `rcac-connect --setup-config && rcac-connect`, then `ssh gilbreth hostname` twice | the whole non-interactive design; **blocks merge** |
| **L3** | Does `apptainer --nv` work? | `apptainer exec --nv <img>.sif nvidia-smi` | whether `--nv` should stay the default |
| **L4** | Do `sacct` / `showpartitions` exist? | `which sacct showpartitions ; sacct -j <id>` | whether `rcac-exp status` can use `sacct` |
| **L5** | `sfeatures` output format | `sfeatures \| tee tests/fixtures/sfeatures.txt` | replaces a doc-derived fixture; **`sfeatures` itself is unconfirmed** |
| **L5b** | Confirm `slist` is account-scoped | `slist \| tee tests/fixtures/slist.txt` | **Already acted on.** RCAC's FAQ says `slist` prints the GPU types *each account you belong to* can reach, so the discovered partition list is taken from `sinfo` alone — otherwise a real partition you merely cannot charge would be reported as "does not exist on this cluster". Capture the real output to confirm the shape, and to replace the doc-derived fixture |
| **L6** | Is there a `/depot/<group>` for you? | `ls -d /depot/*/ ; echo $RCAC_SCRATCH ; findscratch` | fills the optional `paths.depot`; undocumented for Gilbreth |
| **L7** | Per-account job limits | `sacctmgr show assoc user=$USER format=Account,MaxJobs,MaxSubmit` | bounds how large a sweep may be |
| **L8** | Real module names and versions | `module spider anaconda ; module avail` ; `module spider ml-toolkit` | **write down no name this does not print.** No RCAC Gilbreth page names an `ml-toolkit*` module or a specific `anaconda` version |
| **L9** | Retention and quota | `myquota` ; `purgelist \| head` | your real quota. **`purgelist` is not named on any Gilbreth page checked** — confirm it exists before relying on `rcac-keepalive check`'s first answer |
| **L10** | `conda-env-mod` round trip | `rcac-env setup --from environment.yml --py 3.12 && rcac-env activate <name>` | that the generated module name matches what the job script loads |
| **L11** | A real submission on `normal` | `rcac-exp launch … -p <verified> --time 00:10:00` | end-to-end proof of the `-A`/`-p`/`-q` triple; **blocks merge** |
| **L12** | A real submission on `standby` | `… -q standby --time 00:10:00` | the uncharged path; **blocks merge** |
| **L13** | The abolished form is really rejected | `sbatch -A standby …` by hand | that the error message points the right way |
| **L14** | NVLink | `-p a100-40gb --gpus-per-node 2 --constraint nvlink` | that the constraint lands the job on the N nodes |
| **L15** | Container round trip | `rcac-container push … && rcac-container run … nvidia-smi` | that no second two-factor prompt appears |
| **L16** | Do the A10s expose RT cores? | in `-p a10`: `nvidia-smi -q \| grep -i rt` | the Isaac Sim compatibility entry, currently unverified |
| **L17** | Does rootless `apptainer build` need an allocation? | on a front end: `apptainer build /tmp/t.sif docker://alpine` | whether `--remote-build`'s fallback hint stays |
| **L18** | Is `/apps/ngc/images/` present on Gilbreth? | `module load ngc ; ls /apps/ngc/images/ 2>&1` | that path is documented for Gautschi only |

### 7.3 What blocks a merge

**L2, L11 and L12.** Multiplexing, a real charged submission and a real
uncharged one. Everything else is *recorded*: it changes a table or deletes a
caveat, not the design.

---

## 8. Troubleshooting

### Two-factor prompts on every command

The ControlMaster socket is not up, so each command authenticates on its own —
and each one may land on a different front-end node.

```bash
rcac-status
rcac-connect                 # re-establish
ls -l ~/.ssh/sockets/        # the socket should be here
```

If the socket path does not exist, create the directory or re-run
`rcac-connect --setup-config`.

### `sbatch` rejects the submission

Check the four parts before anything else:

```bash
rcac-exp launch experiment.yaml --dry-run    # read the generated header
slist                                        # on the cluster: what may you charge?
```

- `-A` missing or wrong → `slist` names your accounts.
- `-A standby` → abolished. Use `-q standby` and a real account.
- `-p training` without `-q training` → rejected on purpose, not rewritten.
- `-p h100` → RCAC's pages contradict each other; run [L1](#72-the-checklist).
- A partition your account cannot reach → `rcac-partition -g <n>`.

### The job never starts

`standby` is low priority by design, and a 4-hour cap on a busy partition can
sit for a while. Ask for fewer GPUs, or a partition your account has more of:

```bash
squeue -u $USER --start
rcac-partition -g 1 --prefer-cost
```

### No result files came back

```bash
rcac-exp logs <job-id>
rcac-sync pull --dry-run
```

`pull` filters to the result patterns in `configs/defaults.yaml`; use
`--all-files` to bypass that. If files are missing from scratch entirely,
suspect the 60-day purge and run `rcac-keepalive check`.

### `rcac-partition` prints the offline table

It could not reach the cluster, so it fell back to the built-in hints and said
so. Bring the session up (`rcac-connect`) and re-run. The fallback is advice,
not a claim about your entitlements.

### A command sits for 40 seconds before doing anything

That is runtime discovery on a machine that cannot reach Purdue. Each probe has
a 20-second timeout and there can be two of them. Turn it off:

```bash
rcac-exp launch experiment.yaml --dry-run --no-discover
export RCAC_NO_DISCOVER=1        # or set it once for the whole shell
```

With discovery off, an unrecognised partition warns instead of failing, and the
offline hint table is used under its banner. Use `--no-cache` for the opposite
problem: forcing a re-probe within a single process.

---

## 9. Project history

<!-- history:begin -->

This tool began life as **neuron-cli**, a launcher for KISTI's Neuron cluster,
and was ported to Purdue RCAC Gilbreth in a series of commits on the `rcac`
branch. The port was not a rename. The clusters differ in the things that
decide whether a job runs at all:

- KISTI Neuron took a single `-p` for the queue; Gilbreth needs `-A` **and**
  `-p` **and** `-q`, where `-p` is the GPU type.
- Neuron's scratch purged at 15 days; Gilbreth's purges at 60, counted from the
  later of access and modification time.
- Neuron authenticated with an OTP; Gilbreth uses Purdue's MFA or SSH keys.
- Neuron's home was `home01` and bulk data moved through a dedicated DM node;
  Gilbreth has neither — no dedicated transfer host is documented, and bulk
  data goes over Globus.
- Neuron ran Singularity; Gilbreth runs Apptainer, system-installed.

Nothing above describes the cluster this tool now targets. It is kept only so
that a returning Neuron user recognises what changed and why. `rcac-connect
--remove-config` still removes the old `neuron` block from `~/.ssh/config` for
exactly that reason.

<!-- history:end -->

---

Contributions welcome. Please run `pytest -q` and install the credential hook
(`git config core.hooksPath .githooks`) before you commit.
