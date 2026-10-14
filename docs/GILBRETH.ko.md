<div align="center">

# gilbreth-cli

**Purdue RCAC Gilbreth GPU 클러스터용 워크플로우 CLI**

[![ENG](https://img.shields.io/badge/ENG-9AA4B2?style=for-the-badge)](GILBRETH.md)
[![KOR](https://img.shields.io/badge/KOR-2B41D8?style=for-the-badge)](GILBRETH.ko.md)

</div>

> 클러스터의 정식 이름은 **Gilbreth** (Lillian Gilbreth 교수 이름) 입니다. 접속 주소는 `gilbreth.rcac.purdue.edu` 입니다.
> 이 도구는 [RCAC 공식 Gilbreth 가이드](https://docs.rcac.purdue.edu/userguides/gilbreth/) 를 기준으로 만들었습니다.

---

## 목차

1. [Gilbreth 한눈에 보기](#1-gilbreth-한눈에-보기)
2. [설치 및 초기 설정](#2-설치-및-초기-설정)
3. [일상 워크플로우 (Quick Start)](#3-일상-워크플로우-quick-start)
4. [명령어 레퍼런스](#4-명령어-레퍼런스)
5. [experiment.yaml 스키마](#5-experimentyaml-스키마)
6. [실전 시나리오](#6-실전-시나리오)
7. [실제 서버 테스트 계획](#7-실제-서버-테스트-계획)
8. [트러블슈팅](#8-트러블슈팅)

---

## 1. Gilbreth 한눈에 보기

Neuron(KISTI)과 다른 점만 먼저 보세요.

| 항목 | Gilbreth | 비고 |
|------|----------|------|
| 접속 | `ssh <id>@gilbreth.rcac.purdue.edu` | Purdue career 계정 비밀번호 + MFA 푸시 승인. **SSH 키 등록 가능** (이후 MFA 불필요) |
| 홈 | `/home/<id>` (25 GB) | 스냅샷 백업 있음. conda env 두기엔 좁음 |
| 스크래치 | `/scratch/gilbreth/<id>` (100 TB, 200만 파일) | 백업 없음. **60일 미접근 파일 자동 삭제** |
| 그룹 저장소 | `/depot/<group>` | 장기 보관, 구매한 그룹만 |
| 파티션 | `a10`, `a30`, `a100-40gb`, `a100-80gb`, `h100`, `training` | **GPU 종류 = 파티션**. `--gres=gpu:a100` 식 지정 안 함 |
| QOS | `normal` (과금, 최대 14일) / `standby` (무료, 최대 4시간, 저우선순위) / `training` (training 파티션 전용, 24시간) | `-q` 로 지정 |
| 필수 sbatch 옵션 | `-A <account> -p <partition> -q <qos> --time --gpus-per-node=N` | 기본 walltime 30분. `--mem` 도 명시 권장 |
| 계정 조회 | `slist` | "queue" 가 아니라 "account" 라고 부름 |
| 로그인 노드 | A30 1장 포함 | 짧은 GPU 코드 테스트 가능 (긴 작업 금지) |
| 컴퓨트 노드 인터넷 | **됨** (pypi, HuggingFace 확인) | pip install, wandb online, HF 다운로드 모두 작업 안에서 가능 |
| 파이썬 | `module load conda` (Miniforge) 또는 `anaconda` | `conda-env-mod` 방식도 지원 |
| 컨테이너 | `apptainer` (옛 singularity) 기본 설치 | `/scratch`, `/depot`, `/apps` 자동 bind |
| 대화형 | `sinteractive -A ... -p ... --gpus-per-node=1 -t 1:00:00` | |
| 웹 | [gateway.gilbreth.rcac.purdue.edu](https://gateway.gilbreth.rcac.purdue.edu/) (Open OnDemand: Jupyter, 파일 탐색기) / [desktop.gilbreth.rcac.purdue.edu](https://desktop.gilbreth.rcac.purdue.edu/) (ThinLinc) | |
| 대용량 전송 | Globus 컬렉션 `Purdue Gilbreth Cluster - Scratch` | rsync 는 로그인 노드 경유 |

노드 구성 (2026 기준): A30 ×3 (B, D), A100-40GB ×2 (G), A10 ×3 (H), A100-80GB ×2 (I, K), A100-80GB ×4 NVLink (J), H100 ×2 (L), A100-40GB ×4 NVLink (N). 노드 간 100 Gbps InfiniBand.

### 1.1 실제 접속해서 확인한 내용 (2026-10-14, account `shey`)

| 항목 | 확인 결과 |
|------|-----------|
| 로그인 노드 | `gilbreth-fe00` (Rocky Linux 9.8, 128 코어, 503 GB, A30 1장, 인터넷 됨) |
| 내 account | `shey` 하나. QOS `normal`, `standby`, `training` 모두 허용 |
| 쓸 수 있는 GPU | **A30 ×8** (a30 파티션), **A100-40GB ×4** (a100-40gb 파티션, 그룹 공유). a10 / a100-80gb / h100 은 N/A (`slist`) |
| 파티션 혼잡도 | a30 72장 중 60장 이상 사용 중, standby 대기열 약 490개. `normal` 도 몇 분~수십 분 대기 가능 |
| a30 노드 | 24 코어 / 190 GB / A30 ×3 (B 노드), 16 코어 / 190 GB / A30 ×3 (D 노드). **메모리가 먼저 바닥남**: GPU 1장당 `--mem` 32~60 GB 권장 |
| a100-40gb 노드 | G: 128 코어 / 510 GB / A100 ×2, N: 48 코어 / 1 TB / A100 ×4 NVLink (`--constraint=nvlink`) |
| 기본 walltime | 30분 (`DefaultTime=00:30:00`). `time` 안 적으면 30분에 죽음 |
| QOS 상한 | normal 14일, standby 4시간, training 24시간 (`sacctmgr show qos`) |
| 홈 | `/home/ahn205` 25 GB |
| 스크래치 | `/scratch/gilbreth/ahn205` (`$RCAC_SCRATCH`, `$CLUSTER_SCRATCH`), **200 TB** / 200만 파일 |
| Depot | 그룹 depot 없음 (`/depot/shey` 없음) |
| 로그인 시 자동 로드 모듈 | `gcc/11.5.0 openmpi/4.1.6 cuda/12.6.0 xalt external rcac`. **`module purge` 후에는 `module load rcac` 를 먼저** 해야 conda/anaconda/cuda-13 이 보임 |
| conda | `module load conda` → Miniforge (conda 25.3, conda-forge 채널). 버전 `conda/2024.09 2025.02 2025.09(기본) 2026.03`. `anaconda/2025.06-py313` 도 있음. `conda-env-mod` 도 설치됨 |
| CUDA / cuDNN | `cuda/12.1.1`, `cuda/12.6.0`(로그인 기본), `cuda/13.1.0`(`module load cuda` 기본). `cudnn/9.2.0.82-12`. 드라이버 595 |
| 시스템 python | `/usr/bin/python3` = 3.9. `uv` 없음 (pip 으로 설치 가능) |
| 컨테이너 | `apptainer 1.5.3` (= `singularity`). **NGC PyTorch 컨테이너 모듈 제공**: `module load ngc/default pytorch/25.01-py3` 하면 `python` 이 `singularity run --nv nvcr.io/nvidia/pytorch:25.01-py3` 로 감싸짐. 이미지는 `/apps/ngc/images/*.sif` |
| 대화형 | `sinteractive` = `salloc` 래퍼. 모든 옵션 그대로 전달됨 (`--gpus-per-node`, `--cpus-per-gpu` OK) |
| 아카이브 | `hsi`, `htar` (Fortress) 설치됨 |
| 주의: conda 캐시 | `module load conda` 가 `CONDA_PKGS_DIRS` 와 `CONDA_ENVS_PATH` 를 **홈으로** 고정함. gilbreth-cli 는 모듈 로드 후 scratch 로 다시 덮어씀 (`gilbreth-env setup` 은 env 를 `/scratch/gilbreth/<id>/envs/` 에, 패키지 캐시를 `/scratch/.../.conda/pkgs` 에 둠). 손으로 쓸 때도 같은 순서로 하세요 |

`gilbreth-env setup --python 3.11 --name rcac_test` 는 약 1분 걸렸고 `/scratch/gilbreth/ahn205/envs/rcac_test` (181 MB) 가 생겼습니다.

**이 계정에서 실제로 쓸 조합**

```bash
# 빠른 실험 / 디버깅: A30 (24 GB)
gilbreth-run -p a30 -g 1 --mem 32G -t 02:00:00 python train.py
# 큰 모델: A100-40GB (그룹 4장 공유, 보통 1~2장 비어 있음)
gilbreth-run -p a100-40gb -g 1 --mem 64G -t 08:00:00 python train.py
# 4장 NVLink 노드가 필요하면
gilbreth-run -p a100-40gb -g 4 -C nvlink --mem 240G -t 08:00:00 --torchrun train.py
```

`-q standby` 는 무료지만 현재 대기열이 500개 가까이 쌓여 있어서 몇 시간 기다릴 수 있습니다. 급하면 `normal` 을 쓰세요.

---

## 2. 설치 및 초기 설정

### 2.1 설치

```bash
cd KISTI-HPC
pip install -e .
gilbreth-exp --help
```

### 2.2 사용자 ID 와 SLURM 계정 설정

```bash
# ~/.zshrc 또는 ~/.bashrc
export GILBRETH_USER="purdue_id"        # career account
export GILBRETH_ACCOUNT="mygroup"       # slist 로 확인 (아직 모르면 비워두고 2.4 로)
```

또는 `configs/gilbreth.yaml` 의 `ssh.user`, `slurm.account` 에 적어도 됩니다. 우선순위는 `CLI 옵션 > experiment.yaml > 환경변수 > gilbreth.yaml` 입니다.

### 2.3 SSH 설정 생성 및 접속

**MFA 준비 (처음 한 번).** 2026-05-11 부터 RCAC 는 Duo 대신 **Microsoft Authenticator** 를 씁니다.
1. 휴대폰에 Microsoft Authenticator 앱 설치 (Google Play / App Store).
2. https://mysignins.microsoft.com/security-info 에 `<id>@purdue.edu` + career 비밀번호로 로그인 → "Add sign-in method" → Microsoft Authenticator → 앱으로 QR 스캔.
3. 같은 페이지에서 기본 방법을 **"App based authentication - notification"** 으로 설정 (SSH/ThinLinc 는 푸시 알림만 지원). 백업 방법(SMS 등)도 하나 더 추가 권장.
4. SSH 로그인 시 비밀번호만 입력하면 휴대폰에 승인 요청이 옵니다. 알림이 안 뜨면 앱을 직접 열어 대기 중인 요청을 승인하세요 (RCAC 공지된 알려진 문제).

```bash
gilbreth-connect --setup-config     # ~/.ssh/config 에 'Host gilbreth' 블록 추가 (재실행 시 갱신)
gilbreth-connect                    # 비밀번호 입력 -> 휴대폰 MFA 승인 -> 12시간 유지
gilbreth-status                     # Gilbreth: CONNECTED
```

접속이 유지되는 동안 모든 `gilbreth-*` 명령과 `ssh gilbreth`, `rsync` 가 추가 인증 없이 동작합니다.

**MFA 를 아예 없애고 싶다면** (Gilbreth 는 SSH 키를 허용합니다):

```bash
ssh-keygen -t ed25519               # 키가 없을 때만
gilbreth-connect --setup-key        # ssh-copy-id, 마지막으로 한 번 MFA
```

### 2.4 계정 확인 및 점검

```bash
gilbreth-info accounts              # slist: 내가 쓸 수 있는 account 와 GPU 잔량
export GILBRETH_ACCOUNT="<위에서 고른 account>"
gilbreth-info doctor                # 설정, 접속, scratch, module conda/cuda, apptainer, sinteractive 점검
```

`doctor` 가 전부 OK 면 준비 끝입니다.

---

## 3. 일상 워크플로우 (Quick Start)

```bash
# 0. (하루 한 번) 접속
gilbreth-connect

# 1. 파이썬 환경 (처음 한 번) - /scratch/gilbreth/<id>/envs/<name> 에 생성
gilbreth-env setup --from environment.yml           # name: 은 yml 에서
gilbreth-env setup --python 3.11 --name myenv -r requirements.txt   # yml 없이

# 2. 코드 올리기
gilbreth-sync push ./my_project/                    # -> /scratch/gilbreth/<id>/projects/my_project
gilbreth-sync watch ./my_project/                   # 저장할 때마다 자동 push

# 3-a. 가장 빠른 길: 명령 한 줄을 그대로 배치 작업으로
cd my_project
gilbreth-run -g 1 -t 02:00:00 python train.py --lr 1e-4
gilbreth-run -g 1 -t 00:30:00 -q standby python train.py --epochs 1     # 무료 standby 로 빠른 테스트

# 3-b. 스윕 / 파이프라인: experiment.yaml
gilbreth-exp init                                   # 템플릿 생성
gilbreth-exp launch experiment.yaml --dry-run       # 스크립트만 생성해서 확인
gilbreth-exp launch experiment.yaml                 # 제출

# 4. 모니터링
gilbreth-exp status                                 # squeue
gilbreth-exp logs <JOB_ID>                          # tail -f
gilbreth-exp watch <JOB_ID> --pull ./my_project     # 끝나면 experiments/ 자동 다운로드
gilbreth-exp history                                # 최근 3일 sacct

# 5. 결과 가져오기
gilbreth-sync pull ./my_project/                    # experiments/, results/, *.json, *.csv
gilbreth-sync pull ./my_project/ --all-files

# 6. 디버깅은 GPU 노드에서 직접
gilbreth-shell -g 1 -t 01:00:00                     # sinteractive, 프로젝트 디렉토리에서 시작
gilbreth-shell --login                              # 로그인 노드 쉘 (A30 1장으로 짧은 테스트)
```

---

## 4. 명령어 레퍼런스

### 4.1 접속 (`gilbreth-connect` / `-status` / `-disconnect`)

| 명령 | 설명 |
|------|------|
| `gilbreth-connect` | ControlMaster 세션 열기 (비밀번호 + MFA 1회, 12시간 유지) |
| `gilbreth-connect --setup-config` | `~/.ssh/config` 에 `Host gilbreth` 블록 생성/갱신 |
| `gilbreth-connect --setup-key [-i key.pub]` | 공개키 등록 (`ssh-copy-id`) |
| `gilbreth-status` | 세션 상태 |
| `gilbreth-disconnect` | 세션 종료 |

### 4.2 동기화 (`gilbreth-sync`)

```bash
gilbreth-sync push [PATH] [-p PROJECT] [--delete] [--dry-run]    # local -> scratch/projects/PROJECT
gilbreth-sync pull [PATH] [-p PROJECT] [--all-files] [--dry-run] # scratch -> local (기본: 결과 파일만)
gilbreth-sync watch [PATH] [-p PROJECT] [-i 2]                   # 변경 감지 자동 push
gilbreth-sync push-data ./data/ [--dest REMOTE_DIR]              # -> scratch/data/<name>
```

제외 패턴(`.git/`, `__pycache__/`, `wandb/`, `*.pt`, `outputs/` …)과 pull 포함 패턴은 `configs/gilbreth.yaml` 의 `sync:` 에서 수정합니다.

### 4.3 환경 (`gilbreth-env`)

```bash
gilbreth-env setup --from environment.yml [-n NAME] [-r requirements.txt]
gilbreth-env setup --python 3.11 --name NAME [-r requirements.txt]
gilbreth-env list                     # scratch envs + conda env list
gilbreth-env activate NAME            # 셸에 붙여넣을 activate 3줄 출력
gilbreth-env export NAME -o env.yml   # 원격 env -> 로컬 yml
gilbreth-env remove NAME
gilbreth-env modules [QUERY]          # module avail / module spider QUERY
```

env 는 `/scratch/gilbreth/<id>/envs/<NAME>` 에 `--prefix` 로 만들어 홈 25 GB 를 아낍니다. 60일 이상 쓰지 않으면 purge 될 수 있으니 `environment.yml` 을 레포에 같이 두세요.

**PyTorch 설치 예시** (conda-forge 채널 기본, pip 휠이 가장 간단):

```bash
gilbreth-env setup --python 3.11 --name torch_env
gilbreth-shell --login
module load conda; export CONDA_PKGS_DIRS=$RCAC_SCRATCH/.conda/pkgs PIP_CACHE_DIR=$RCAC_SCRATCH/.cache/pip
eval "$(conda shell.bash hook)"; conda activate $RCAC_SCRATCH/envs/torch_env
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print(torch.cuda.is_available())"   # 로그인 노드 A30 에서 바로 확인 가능
```

또는 `requirements.txt` 를 두고 `gilbreth-env setup --python 3.11 --name torch_env -r requirements.txt` 한 줄로.

**env 없이 NGC PyTorch 컨테이너 쓰기** (패키지 추가 설치가 필요 없을 때 가장 빠름):

```yaml
environment:
  type: apptainer
  container: /apps/ngc/images/nvcr.io_nvidia_pytorch:25.01-py3.sif
```
```bash
gilbreth-run --container /apps/ngc/images/nvcr.io_nvidia_pytorch:25.01-py3.sif -p a30 -g 1 -t 01:00:00 python train.py
```

### 4.4 실험 (`gilbreth-exp`)

```bash
gilbreth-exp init [-o experiment.yaml]
gilbreth-exp launch experiment.yaml [--dry-run] [-n N] [-A acct] [-p part] [-q qos] [--gpus N] [--time T] [--mem M] [--no-sync]
gilbreth-exp status [--all-users] [-p PARTITION]
gilbreth-exp history [-d DAYS]
gilbreth-exp logs JOB_ID [--no-follow] [-n LINES]
gilbreth-exp watch [JOB_ID ...] [-i SEC] [--pull DIR] [--project NAME]
gilbreth-exp cancel JOB_ID | all
gilbreth-exp attach JOB_ID            # 실행 중인 작업 노드에 쉘 붙기 (srun --jobid)
```

`launch` 는 기본적으로 `experiment.yaml` 이 있는 디렉토리를 먼저 rsync 한 뒤 (`--no-sync` 로 생략) 설정/스크립트를 업로드하고 `sbatch --parsable` 로 제출합니다. 생성된 스크립트는 로컬 `~/.gilbreth/runs/<project>_<timestamp>/` 와 원격 `projects/<project>/scripts/<timestamp>/` 에 남습니다.

### 4.5 단발 작업 / 대화형 (`gilbreth-run`, `gilbreth-shell`)

```bash
gilbreth-run [옵션] COMMAND...        # 옵션은 COMMAND 앞에. COMMAND 의 -t/--lr 등은 그대로 작업에 전달됨
  -A acct  -p part  -q qos  -g GPUS  --cpus-per-gpu N  --mem 32G  -t HH:MM:SS  -C nvlink
  --project NAME     # 원격 디렉토리 (기본: 현재 디렉토리 이름)
  --env NAME         # conda env  (--env-type conda|conda-env-mod|apptainer|venv|none)
  --container X.sif  # apptainer 로 실행
  --torchrun         # torchrun --nproc_per_node=<GPUS> 로 감싸기
  --no-sync          # 현재 디렉토리 push 생략
  --dry-run

gilbreth-shell [-g 1] [-t 01:00:00] [-p a30] [-q standby] [--cd PROJECT]   # sinteractive
gilbreth-shell --login                                                     # 로그인 노드
```

### 4.6 클러스터 정보 (`gilbreth-info`)

```bash
gilbreth-info accounts        # slist
gilbreth-info partitions      # 파티션 표 + (접속 시) sinfo 실시간 상태
gilbreth-info nodes [-p PART] # sfeatures
gilbreth-info quota           # myquota
gilbreth-info purge-check [-d 50]   # 60일 purge 임박 파일 목록
gilbreth-info doctor          # 설정/접속/모듈 점검
```

---

## 5. experiment.yaml 스키마

```yaml
project: my_model                 # 원격: /scratch/gilbreth/<id>/projects/my_model

slurm:                            # 생략한 키는 configs/gilbreth.yaml 의 slurm: 값
  account: mygroup                # 생략 시 GILBRETH_ACCOUNT
  partition: a100-80gb            # a10 | a30 | a100-40gb | a100-80gb | h100 | training
  qos: normal                     # normal | standby | training (training 파티션이면 자동)
  nodes: 1
  gpus: 1                         # --gpus-per-node
  cpus_per_gpu: 8
  mem: 64G
  time: "08:00:00"
  constraint: nvlink              # 선택

environment:
  type: conda                     # conda | conda-env-mod | apptainer | venv | none
  name: train_env                 # conda: envs/<name> 또는 절대경로
                                  # conda-env-mod: conda-env-mod 가 출력한 이름 (예: my_env-py3.12.3)
                                  # venv: envs/<name> 또는 절대경로
  container: /scratch/gilbreth/<id>/img.sif   # apptainer 일 때
  modules: [cuda]                 # 추가 module load (conda/anaconda 모듈은 type 에 따라 자동)
  launcher: python                # python | torchrun

env_vars:                         # 작업에 export (gilbreth.yaml 의 env_vars 와 병합)
  TOKENIZERS_PARALLELISM: "false"

script: train.py                  # 단일 단계 실행 파일
args: "--config {config_path} --lr {learning_rate}"
  # 사용 가능한 플레이스홀더: base_params 키, sweep 키, {config_path}, {exp_id}

base_params: {epochs: 10, seed: 42}

sweep:
  method: grid                    # grid | random (count: N, seed: S)
  mode: array                     # array: sbatch --array 하나 | separate: 조합마다 sbatch
  parallel: 3                     # array 동시 실행 상한 (%3)
  parameters:
    learning_rate: [1.0e-4, 3.0e-4]

pipeline:                         # 있으면 script/args 대신 단계별 실행, afterok 로 연결
  - name: train
    script: train.py
    args: "--config {config_path}"
  - name: eval
    script: evaluate.py
    args: "--run {exp_id}"
```

작업 안에서 쓸 수 있는 환경변수: `PROJECT`, `EXPERIMENT_ID`, `CONFIG_PATH` (조합별 YAML, `_exp_id` 포함), `STAGE_NAME`, `MASTER_ADDR`, `MASTER_PORT`, `SLURM_*`. 로그는 `/scratch/gilbreth/<id>/logs/<jobname>_<jobid>.out|err` (array 는 `<jobname>_<arrayid>_<task>.out`), 완료 기록은 `logs/completed.log`.

주의: YAML 에서 `1e-4` 는 PyYAML 이 문자열로 읽습니다. 숫자로 쓰려면 `1.0e-4` 로 적으세요. `launch` 는 `experiment.yaml` 이 있는 디렉토리 전체를 push 하므로, `experiment.yaml` 은 프로젝트 루트에 두세요.

---

## 6. 실전 시나리오

### A. 학습 스크립트 하나 빨리 돌리기

```bash
gilbreth-connect
cd my_project
gilbreth-env setup --python 3.11 --name myenv -r requirements.txt   # 처음 한 번
gilbreth-run -g 1 -t 04:00:00 --env myenv python train.py --lr 3e-4
gilbreth-exp logs <JOB_ID>
```

### B. 하이퍼파라미터 스윕 (job array)

```bash
gilbreth-exp init
# experiment.yaml 의 sweep.parameters 수정 (6 조합)
gilbreth-exp launch experiment.yaml --dry-run      # ~/.gilbreth/runs/.../scripts/job_array.sh 확인
gilbreth-exp launch experiment.yaml                # sbatch --array=0-5%3 한 번
gilbreth-exp watch --pull ./my_project
```

### C. 멀티 GPU DDP (torchrun)

```yaml
slurm: {partition: a100-80gb, gpus: 4, mem: 240G, time: "12:00:00", constraint: nvlink}
environment: {type: conda, name: myenv, launcher: torchrun}
script: train_ddp.py
```

`gpus: 4, nodes: 2` 면 `srun torchrun --nnodes=$SLURM_NNODES --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT …` 로 자동 전환됩니다. 4 GPU 를 넘는 작업은 `training` 파티션(`-p training`, QOS 는 자동으로 `training`, 24시간) 을 RCAC 에 신청하세요.

### D. Apptainer 컨테이너 (Isaac Lab / Genesis 등 복잡한 의존성)

```bash
# 로컬 또는 Gilbreth 에서 빌드 후 scratch 에 두기
apptainer build sim.sif containers/genesis-sim.def
gilbreth-sync push-data ./sim.sif --dest /scratch/gilbreth/$GILBRETH_USER/images

gilbreth-run -g 1 -t 02:00:00 --container /scratch/gilbreth/$GILBRETH_USER/images/sim.sif python run_sim.py
```

템플릿은 `apptainer exec --nv --bind /scratch/gilbreth/<id> [--bind /depot/<group>]` 로 실행합니다. `paths.depot` 을 설정하면 depot 도 bind 됩니다.

### E. 대화형 개발 / Jupyter

```bash
gilbreth-shell -g 1 -t 02:00:00 -q standby     # 무료 standby 로 GPU 노드 쉘
# 노드 안에서
module load conda; eval "$(conda shell.bash hook)"; conda activate /scratch/gilbreth/$USER/envs/myenv
```

Jupyter 는 [gateway.gilbreth.rcac.purdue.edu](https://gateway.gilbreth.rcac.purdue.edu/) 의 Open OnDemand 에서 account/partition 을 고르고 띄우는 것이 가장 간단합니다. scratch env 를 커널로 쓰려면 env 안에 `ipykernel` 을 설치하세요.

### F. 무료 standby 활용

4시간 이하 작업은 `-q standby` 로 제출하면 account 잔량을 쓰지 않습니다. 대기 시간은 길 수 있으니 디버깅/짧은 실험에 쓰세요.

```bash
gilbreth-run -q standby -g 1 -t 03:59:00 python train.py --epochs 5
```

---

## 7. 실제 서버 테스트 계획

| # | 테스트 | 명령 | 통과 기준 |
|---|--------|------|-----------|
| 1 | 접속 | `gilbreth-connect --setup-config && gilbreth-connect && gilbreth-status && ssh gilbreth hostname` | MFA 1회 후 추가 인증 없이 hostname 출력 |
| 2 | 점검 | `gilbreth-info accounts && gilbreth-info doctor` | account 표시, doctor 전부 OK (conda/cuda 모듈 이름이 다르면 `gilbreth.yaml` 수정) |
| 3 | 동기화 | `mkdir -p /tmp/gtest && echo "print('hi')" > /tmp/gtest/t.py && gilbreth-sync push /tmp/gtest && ssh gilbreth ls /scratch/gilbreth/$GILBRETH_USER/projects/gtest` | t.py 표시 |
| 4 | 환경 | `gilbreth-env setup --python 3.11 --name gtest_env && gilbreth-env list` | envs/gtest_env 생성 |
| 5 | 단발 작업 | `cd /tmp/gtest && gilbreth-run -q standby -g 1 -t 00:05:00 -p a30 --env gtest_env python -c "import torch,os;print(os.environ['CUDA_VISIBLE_DEVICES'])"` | job id 출력, `gilbreth-exp logs ID` 에 GPU 번호 |
| 6 | 스윕 | `gilbreth-exp init && gilbreth-exp launch experiment.yaml --dry-run && gilbreth-exp launch experiment.yaml -q standby -p a30 --time 00:05:00` | array job 제출, `gilbreth-exp status` 에 표시 |
| 7 | 결과 | `gilbreth-exp watch <ID> --pull /tmp/gtest && gilbreth-exp history` | 완료 후 pull, sacct 에 COMPLETED |
| 8 | 대화형 | `gilbreth-shell -q standby -p a30 -g 1 -t 00:10:00` 후 `nvidia-smi` | GPU 노드 쉘 |

### 7.1 실제 서버 검증 결과 (2026-10-14, ahn205 / shey)

```
[x] Test 1: 접속          gilbreth-fe00, MFA 1회 후 ControlMaster 재사용, ssh gilbreth hostname OK
[x] Test 2: doctor        slist / scratch / module conda / cuda / apptainer / sinteractive / account 전부 OK
[x] Test 3: 동기화        gilbreth-run 이 /scratch/gilbreth/ahn205/projects/gtest 로 push
[x] Test 4: 환경          gilbreth-env setup --python 3.11 --name rcac_test -> 약 1분, 181 MB (scratch)
[x] Test 5: 단발 작업     gilbreth-run -p a30 -g 1 --mem 8G -t 00:05:00 -> Job 11950925, 제출 4분 후 시작, COMPLETED
                          컴퓨트 노드 gilbreth-b011: A30 24 GB, CUDA_VISIBLE_DEVICES=0, cuda/12.6.0 로드,
                          pypi.org / huggingface.co 접속 200 (컴퓨트 노드도 인터넷 됨), scratch 쓰기 OK, /tmp 193 GB, apptainer 1.5.3
[x] Test 6: 모니터링      gilbreth-exp status / logs / history / cancel 동작 확인
[x] Test 7: 스윕 (array)  gilbreth-exp launch (2 조합, conda env rcac_test) -> Job 11951056 array, 두 task 모두 COMPLETED
                          args 의 따옴표('exp {exp_id}') 가 그대로 전달됨, gilbreth-exp watch --pull 로 eval_results.json 자동 수신
[ ] Test 8: 대화형        sinteractive 옵션 전달 방식만 확인 (salloc 래퍼)
```

실서버에서 잡은 버그: 작업 스크립트의 `module purge` 가 `external` 모듈까지 내려서 그 뒤 `module load conda` 가 "exist but cannot be loaded" 로 실패했습니다. 지금은 purge 직후 `module load rcac` (`environment.base_module`) 로 로그인 기본 스택을 복원한 뒤 conda/cuda 를 로드합니다. 손으로 sbatch 스크립트를 쓸 때도 같은 순서가 필요합니다.

관찰된 주의점: `-q standby` 는 대기열 약 490개로 당장 시작되지 않았고 (`normal` 로 바꿔서 통과), `module purge` 가 sticky 모듈 `xalt` 경고를 stderr 에 남기지만 무해합니다. 컴퓨트 노드에서 인터넷이 되므로 `WANDB_MODE=offline` 은 필수가 아닙니다 (기본값에서 제거함).

---

## 8. 트러블슈팅

| 증상 | 원인 | 해결 |
|------|------|------|
| `Host key verification failed` / rsync 255 | ControlMaster 세션 없음 | `gilbreth-status` → `gilbreth-connect` |
| MFA 푸시가 계속 옴 | 세션 만료(12h) 또는 소켓 깨짐 | `gilbreth-disconnect; rm -f ~/.ssh/sockets/*gilbreth*; gilbreth-connect` 또는 `--setup-key` |
| `sbatch: error: Invalid account` | `-A` 미설정/오타 | `gilbreth-info accounts` 후 `GILBRETH_ACCOUNT` 수정 |
| `Invalid qos specification` | `training` 파티션에 `normal`, 또는 standby 미지원 파티션 | `-q training` 또는 `-q normal` |
| 작업이 30분에 죽음 | `time` 미지정 (Gilbreth 기본 30분) | `slurm.time` 설정 |
| `module: command not found` / 모듈 없음 | 모듈 이름 변경 | `gilbreth-env modules conda`, `gilbreth-env modules cuda` 로 확인 후 `gilbreth.yaml` 수정 |
| `conda: command not found` in job | `conda_module` 이름 틀림 | `environment.conda_module: anaconda` 로 변경 |
| Disk quota exceeded (home) | 25 GB 홈 초과 (pip/HF 캐시) | env 와 캐시를 scratch 로: 기본 `env_vars` 의 `HF_HOME`, `XDG_CACHE_HOME` 참고, `pip cache purge` |
| scratch 파일이 사라짐 | 60일 미접근 purge | `gilbreth-info purge-check` 로 미리 확인, `gilbreth-sync pull` / Depot / Fortress 로 보관 |
| Pending 이 너무 김 | 파티션 혼잡 | `gilbreth-info partitions` 로 idle 노드 확인, `-p a30`/`-q standby` 로 변경 |
| OOM | `mem` 부족 | `--mem 128G` 등 상향 (노드당 192 GB~2 TB) |
| 로그 없음 | 아직 시작 전 | `gilbreth-exp status` 의 Reason 확인 (Priority, Resources 등) |

문의: rcac-help@purdue.edu

**참고 자료**
- [Gilbreth User Guide](https://docs.rcac.purdue.edu/userguides/gilbreth/)
- [Using AI Agents (파티션/QOS/모듈 요약)](https://docs.rcac.purdue.edu/userguides/gilbreth/using_ai_agents/)
- [Queues](https://docs.rcac.purdue.edu/userguides/gilbreth/run_jobs/queues/) · [Submit script](https://docs.rcac.purdue.edu/userguides/gilbreth/run_jobs/submit_script/) · [GPU jobs](https://docs.rcac.purdue.edu/userguides/gilbreth/run_jobs/gpu_jobs/) · [Custom ML env](https://docs.rcac.purdue.edu/userguides/gilbreth/run_jobs/learning/customml/) · [Apptainer](https://docs.rcac.purdue.edu/userguides/gilbreth/run_jobs/apptainer_example/)
- [Storage quota](https://docs.rcac.purdue.edu/userguides/gilbreth/storage/storage_quota/) · [Globus](https://docs.rcac.purdue.edu/userguides/gilbreth/storage/globus/)
- [ML-Purdue cnn-workshop (실사용 예시)](https://github.com/ML-Purdue/cnn-workshop)
