# KISTI Neuron HPC 연구 자동화 계획서

## 로보틱스 AI 모델 개발을 위한 HPC 워크플로우 자동화

---

## 1. 현황 분석

### 1.1 KISTI Neuron 시스템 요약

| 항목 | 내용 |
|------|------|
| 스케줄러 | SLURM |
| 접속 | SSH + OTP (Any OTP 앱) |
| 로그인 노드 | neuron01~03.ksc.re.kr (CPU Limit 2시간) |
| 데이터 전송 노드 | neuron-dm.ksc.re.kr (CPU Limit 없음) |
| 작업 디렉터리 | /scratch/$USER (15일 미접근 파일 자동 삭제) |
| 홈 디렉터리 | 용량 제한, 계산 작업 금지 |
| 노드 정책 | 공유 노드 (하나의 노드에 복수 작업 동시 수행) |
| Wall time | 기본 48시간, Interactive 8시간 |
| 필수 옵션 | `--comment` (application 이름 명시 의무) |

### 1.2 활용 가능 GPU 파티션 (로보틱스 AI 학습 추천순)

| 파티션 | GPU | GPU 수/노드 | 총 Core | 추천 용도 |
|--------|-----|------------|---------|----------|
| amd_a100nv_8 | A100 (NVLink) | 8 | 64 | 대규모 학습, 멀티 GPU 분산학습 |
| eme_h200nv_8 | H200 (NVLink) | 8 | 96 | 최고성능 학습 (높은 과금) |
| amd_h200nv_8 | H200 (NVLink) | 8 | 96 | 최고성능 학습 (높은 과금) |
| cas_v100nv_8 | V100 (NVLink) | 8 | 32 | 중간 규모 학습 |
| cas_v100_4 | V100 | 4 | 40 | 일반 학습, 실험 |
| cas_v100_2 | V100 | 2 | 32 | 소규모 실험/디버깅 |

### 1.3 시뮬레이터 GPU 호환성 조사 결과

| 시뮬레이터 | V100 | A100 | H200 | 비고 |
|-----------|------|------|------|------|
| **Isaac Sim** | ❌ | ❌ | ❌ | RT Core 필수. Neuron GPU 전부 미지원 |
| **Isaac Gym (Preview)** | ✅ | ✅ | ✅ | A100에서 검증됨. 단, **deprecated** (더 이상 업데이트 없음). Isaac Lab으로 마이그레이션 권장되나, Isaac Lab은 Isaac Sim 기반이므로 역시 Neuron에서 불가 |
| **Genesis** | ✅ | ✅ | ✅ | 순수 CUDA 기반, RT Core 불필요. A100에서 Isaac Gym 대비 10~80배 빠른 시뮬레이션 속도 보고. 100% Python. **Neuron에서 가장 적합한 시뮬레이터** |
| **MuJoCo** | ✅ | ✅ | ✅ | 범용 물리 시뮬레이터. GPU 가속 (MJX) 지원 |
| **PyBullet** | ✅ | ✅ | ✅ | 경량 시뮬레이터, GPU 가속 제한적 |

> **결론**: Isaac Sim은 Neuron에서 사용 불가. 시뮬레이션이 필요할 경우 Genesis → MuJoCo 순으로 추천. Vision 기반 manipulation 학습에는 Genesis의 photo-realistic 렌더링이 유리.

### 1.4 현재 Pain Points

1. **반복적 코드 동기화**: 로컬 수정 → Neuron 전송 → 확인 사이클이 수동
2. **환경 재설정**: 매번 Conda 활성화, module load 등 반복
3. **OTP 인증 반복**: SSH 접속 시마다 OTP 입력 필요
4. **실험 관리 부재**: 파라미터/아키텍처별 결과를 체계적으로 추적하기 어려움
5. **HPC 디버깅 불편**: 로컬과 분리된 환경에서 코드 확인 및 디버깅 필요
6. **파일 삭제 정책 리스크**: /scratch 15일 미접근 시 자동 삭제

---

## 2. 자동화 아키텍처 설계

### 2.1 전체 시스템 구조

```
┌─────────────────────────────────────────────────────────────┐
│                    로컬 개발 머신                              │
│                                                             │
│  [코드 편집] → [neuron-cli] → [rsync/scp] ──────────────┐   │
│       ↑                                                 │   │
│       └──── [결과 수신/시각화] ← [결과 자동 다운로드] ←──┘   │
└─────────────────────────────────────────────────────────────┘
                          │ SSH + OTP (ControlMaster)
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   Neuron 로그인 노드                          │
│                                                             │
│  [neuron-runner] ──→ [SLURM Job 제출]                       │
│       │                    │                                │
│       │              ┌─────┴──────────────────┐             │
│       │              ▼          ▼              ▼            │
│       │         [실험 1]   [실험 2] ...   [실험 N]           │
│       │         (Job Array / Dependency Chain)               │
│       │              │          │              │            │
│       │              └──────────┴──────────────┘            │
│       │                         │                           │
│       │                    [결과 수집]                        │
│       │                         │                           │
│       └──── [Slack/Email 알림] ←┘                           │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 핵심 컴포넌트 5가지

| # | 컴포넌트 | 역할 | 구현 위치 |
|---|---------|------|----------|
| A | **SSH 연결 관리자** | OTP 세션 유지, 멀티플렉싱 | 로컬 |
| B | **코드 동기화 엔진** | 로컬↔Neuron 양방향 파일 동기화 | 로컬 |
| C | **실험 매니저** | 파라미터/아키텍처 조합 → SLURM 스크립트 자동 생성 및 의존성 체이닝 | 로컬+Neuron |
| D | **환경 부트스트래퍼** | Conda/Singularity 환경 자동 구축 및 복원 | Neuron |
| E | **결과 수집기** | 학습 결과 자동 수집, 비교, 알림 | 양쪽 |

---

## 3. 단계별 개발 계획

### Phase 1: 기반 인프라 (1~2주)

**목표**: SSH 연결 자동화 + 코드 동기화 + 환경 설정 원클릭화

#### 3.1.1 SSH ControlMaster 설정 (OTP 입력 최소화)

SSH ControlMaster를 활용하면 최초 1회 OTP 인증 후, 이후 접속은 기존 세션을 재사용하여 OTP 입력 없이 접속 가능합니다.

```
# ~/.ssh/config
Host neuron
    HostName neuron.ksc.re.kr
    User <USER_ID>
    Port 22
    ControlMaster auto
    ControlPath ~/.ssh/sockets/%r@%h-%p
    ControlPersist 4h          # 4시간 유지 (Interactive walltime 기준)
    ServerAliveInterval 60     # 60초 간격 keep-alive
    ServerAliveCountMax 10

Host neuron-dm
    HostName neuron-dm.ksc.re.kr
    User <USER_ID>
    Port 22
    ControlMaster auto
    ControlPath ~/.ssh/sockets/%r@%h-%p
    ControlPersist 8h          # DM 노드는 CPU limit 없음
    ServerAliveInterval 60
```

**개발 항목**:
- `neuron-connect`: SSH ControlMaster 세션 초기화 스크립트 (OTP 1회 입력)
- `neuron-status`: 현재 연결 상태 확인
- `neuron-disconnect`: 세션 정리

#### 3.1.2 코드 동기화 도구 (`neuron-sync`)

```python
# 핵심 기능
neuron-sync push              # 로컬 → Neuron (rsync, .gitignore 기반 필터)
neuron-sync pull              # Neuron → 로컬 (결과 파일만)
neuron-sync watch             # 파일 변경 감지 → 자동 push (fswatch/watchdog)
```

**동기화 규칙**:
- 대상 경로: `/scratch/$USER/projects/<project_name>/`
- 제외 대상: `.git/`, `__pycache__/`, `*.pyc`, `wandb/`, 대용량 데이터셋
- 데이터셋은 별도 관리 (neuron-dm 노드를 통해 1회 전송)

#### 3.1.3 환경 부트스트래퍼 (`neuron-env`)

```python
# Conda 환경 자동 생성/복원
neuron-env setup              # environment.yml 기반 Conda 환경 구축
neuron-env activate           # 환경 활성화 + module load
neuron-env export             # 현재 환경 → environment.yml 내보내기
neuron-env singularity-build  # Phase 3에서 구현
```

**자동 생성되는 module load 체인**:
```bash
module purge
module load cuda/12.4.1       # PyTorch 호환 CUDA
# Conda 환경 활성화
source activate <env_name>
```

---

### Phase 2: 실험 자동화 (2~3주)

**목표**: 파라미터 탐색 자동화 + 의존성 기반 파이프라인 + 결과 관리

#### 3.2.1 실험 설정 시스템 (`experiment.yaml`)

```yaml
# experiment.yaml 예시
project: manipulation_transformer
description: "Vision-based manipulation model - architecture search"

# 기본 SLURM 설정
slurm:
  partition: amd_a100nv_8       # 추천: A100 NVLink
  nodes: 1
  gpus: 2
  cpus_per_gpu: 8               # A100 노드 기본값
  time: "24:00:00"
  comment: pytorch              # KISTI 필수 옵션

# 환경 설정
environment:
  type: conda                    # conda | singularity
  name: manipulation_env
  modules:
    - cuda/12.4.1

# 기본 학습 파라미터
base_params:
  dataset: /scratch/$USER/data/manipulation_v2
  epochs: 100
  batch_size: 64
  optimizer: adamw
  seed: 42

# 파라미터/아키텍처 탐색 설정
sweep:
  method: grid                   # grid | random | bayesian
  parameters:
    learning_rate: [1e-4, 3e-4, 1e-3]
    model_arch: [resnet50, vit_base, vit_large]
    hidden_dim: [256, 512]
    dropout: [0.1, 0.3]

# 파이프라인 설정
pipeline:
  - name: train
    script: train.py
    args: "--config {config_path}"
  - name: evaluate
    script: evaluate.py
    depends_on: train            # train 완료 후 실행
    args: "--checkpoint {train.best_model} --test-split test"
  - name: export
    script: export_model.py
    depends_on: evaluate
    condition: "evaluate.accuracy > 0.85"  # 조건부 실행
```

#### 3.2.2 실험 매니저 CLI (`neuron-exp`)

```python
# 실험 실행 명령어
neuron-exp launch experiment.yaml        # 전체 sweep 실행
neuron-exp launch experiment.yaml -n 3   # 상위 3개 조합만 실행
neuron-exp status                        # 모든 실험 상태 조회
neuron-exp logs <exp_id>                 # 특정 실험 로그 스트리밍
neuron-exp cancel <exp_id>               # 실험 취소
neuron-exp results                       # 전체 결과 테이블 출력
neuron-exp compare <exp1> <exp2>         # 실험 비교
neuron-exp best                          # 최고 성능 실험 조회
```

**내부 동작 흐름**:

```
experiment.yaml
      │
      ▼
[파라미터 조합 생성] ← grid/random/bayesian
      │
      ├── config_001.yaml (lr=1e-4, arch=resnet50, ...)
      ├── config_002.yaml (lr=1e-4, arch=vit_base, ...)
      └── config_N.yaml
      │
      ▼
[SLURM 스크립트 자동 생성]
      │
      ├── job_001_train.sh
      │     └── sbatch 제출 → JOB_ID_001
      ├── job_001_eval.sh
      │     └── sbatch --dependency=afterok:JOB_ID_001 → JOB_ID_001_eval
      ├── job_002_train.sh
      │     └── sbatch 제출 → JOB_ID_002
      └── ...
      │
      ▼
[결과 수집] → results/<exp_id>/summary.json
```

#### 3.2.3 SLURM 스크립트 자동 생성기

자동 생성되는 SLURM 스크립트 템플릿:

```bash
#!/bin/bash
#SBATCH -J {project}_{exp_id}
#SBATCH -p {partition}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:{gpus}
#SBATCH --cpus-per-gpu={cpus_per_gpu}
#SBATCH --time={time}
#SBATCH --comment {comment}
#SBATCH -o /scratch/$USER/logs/%x_%j.out
#SBATCH -e /scratch/$USER/logs/%x_%j.err

# === 환경 설정 (자동 생성) ===
module purge
module load {modules}
source activate {conda_env}

# === 작업 디렉터리 ===
cd /scratch/$USER/projects/{project}

# === 실험 실행 ===
export EXPERIMENT_ID="{exp_id}"
export CONFIG_PATH="{config_path}"
export CUDA_VISIBLE_DEVICES=0,1

python {script} {args}

# === 결과 기록 ===
echo "EXPERIMENT_DONE: $EXPERIMENT_ID" >> /scratch/$USER/logs/completed.log
```

#### 3.2.4 결과 추적 및 비교

```
/scratch/$USER/
├── projects/
│   └── manipulation_transformer/
│       ├── src/                          # 소스 코드
│       ├── configs/                      # 자동 생성된 실험 설정
│       └── experiments/
│           ├── exp_20260507_001/
│           │   ├── config.yaml           # 사용된 파라미터
│           │   ├── train_log.json        # 학습 메트릭
│           │   ├── checkpoints/          # 모델 체크포인트
│           │   └── eval_results.json     # 평가 결과
│           └── summary.csv              # 전체 실험 비교 테이블
├── data/                                # 데이터셋 (별도 관리)
└── logs/                                # SLURM 로그
```

**Weights & Biases (W&B) 오프라인 모드 연동**:

Neuron 계산 노드에서 외부 네트워크가 제한될 수 있으므로, W&B 오프라인 모드로 로깅 후 로컬에서 sync하는 방식을 지원합니다.

```python
# train.py 내부
import wandb
wandb.init(mode="offline", project="manipulation", config=config)
# ... 학습 ...
wandb.finish()

# 로컬에서 동기화
# neuron-exp sync-wandb <exp_id>
```

---

### Phase 3: 고도화 (3~4주)

**목표**: Singularity 컨테이너화 + 스마트 스케줄링 + 모니터링 대시보드

#### 3.3.1 Singularity 컨테이너 관리 (`neuron-container`)

```python
neuron-container build Singularityfile   # 이미지 빌드
neuron-container push <image.sif>        # Neuron으로 전송
neuron-container run <image.sif> ...     # 컨테이너 내 실행
```

**Singularity Definition 파일 템플릿**:

```singularity
Bootstrap: docker
From: nvcr.io/nvidia/pytorch:24.01-py3

%post
    pip install wandb hydra-core omegaconf
    pip install genesis-world   # Genesis 시뮬레이터 (확장 시)
    pip install timm einops     # Vision Transformer 관련

%environment
    export WANDB_MODE=offline

%runscript
    python "$@"
```

**장점**: 환경 재현성 보장, Conda 환경 깨짐 방지, 노드 간 완벽 동일 환경

#### 3.3.2 스마트 파티션 선택기

GPU 유형, 대기열 상태, 과금 효율을 종합 판단하여 최적 파티션을 자동 추천합니다.

```python
neuron-exp launch experiment.yaml --auto-partition

# 내부 로직:
# 1. sinfo로 각 파티션 idle/mix 노드 확인
# 2. 요청 GPU 수 vs 파티션 가용 GPU 비교
# 3. 과금 요율 대비 성능 비교 (A100이 V100 대비 ~2배 빠르나 과금도 높음)
# 4. 최적 파티션 자동 선택 or 추천 목록 제시
```

#### 3.3.3 파일 보존 자동화 (Anti-purge)

/scratch의 15일 삭제 정책에 대응:

```python
# crontab 또는 SLURM recurring job으로 실행
neuron-keepalive                          # 중요 파일 atime 갱신
neuron-keepalive --dry-run                # 삭제 대상 파일 미리 확인
```

```bash
# 중요 디렉터리의 atime을 주기적으로 갱신
find /scratch/$USER/data -type f -exec touch -a {} +
find /scratch/$USER/projects -name "*.pth" -exec touch -a {} +
find /scratch/$USER/projects -name "*.yaml" -exec touch -a {} +
```

#### 3.3.4 알림 시스템

```python
# 작업 완료/실패 시 알림
neuron-exp launch experiment.yaml --notify slack    # Slack webhook
neuron-exp launch experiment.yaml --notify email    # 이메일
```

SLURM 스크립트 끝에 자동 추가:

```bash
# 작업 완료 시 알림 전송
python /scratch/$USER/.neuron-tools/notify.py \
    --status $? \
    --job-name $SLURM_JOB_NAME \
    --exp-id $EXPERIMENT_ID \
    --metric "$(cat eval_results.json | python -c 'import sys,json; print(json.load(sys.stdin).get(\"accuracy\",\"N/A\"))')"
```

---

### Phase 4: 시뮬레이터 확장 (필요 시)

#### 3.4.1 Genesis 시뮬레이터 연동

Genesis는 RT Core 없이도 Neuron의 A100/H200에서 실행 가능하며, 로보틱스 manipulation 환경 구축에 적합합니다.

```yaml
# experiment.yaml에 시뮬레이터 단계 추가
pipeline:
  - name: generate_data
    script: genesis_datagen.py
    args: "--env manipulation --num_episodes 10000"
  - name: train
    script: train.py
    depends_on: generate_data
    args: "--dataset {generate_data.output_path}"
  - name: evaluate_sim
    script: genesis_eval.py
    depends_on: train
    args: "--checkpoint {train.best_model}"
```

#### 3.4.2 Isaac Gym Preview 활용 (한시적)

Isaac Gym Preview 4는 deprecated이지만 A100에서 정상 작동하며, 기존 연구 코드와의 호환성이 필요한 경우 활용 가능합니다. Singularity로 환경을 고정해두는 것을 권장합니다.

---

## 4. 프로젝트 구조

```
neuron-automation/
├── pyproject.toml                    # 패키지 설정
├── README.md
├── neuron_cli/                       # CLI 도구 (pip install -e .)
│   ├── __init__.py
│   ├── connect.py                    # SSH 연결 관리
│   ├── sync.py                       # 코드 동기화
│   ├── env.py                        # 환경 관리
│   ├── experiment.py                 # 실험 매니저
│   ├── scheduler.py                  # SLURM 스크립트 생성
│   ├── results.py                    # 결과 수집/비교
│   ├── container.py                  # Singularity 관리
│   ├── keepalive.py                  # 파일 보존
│   ├── notify.py                     # 알림
│   └── partition_selector.py         # 스마트 파티션 선택
├── templates/                        # SLURM 스크립트 템플릿
│   ├── single_gpu.sh.j2
│   ├── multi_gpu.sh.j2
│   ├── job_array.sh.j2
│   └── pipeline.sh.j2
├── configs/                          # 기본 설정 파일
│   ├── neuron.yaml                   # 접속 정보, 경로 설정
│   └── defaults.yaml                 # SLURM 기본 옵션
├── containers/                       # Singularity 정의 파일
│   ├── pytorch-base.def
│   └── genesis-sim.def
└── tests/
    └── ...
```

---

## 5. 사용 시나리오 (End-to-End)

### 시나리오: 새로운 Vision Manipulation 모델 개발

```bash
# 1. 최초 1회: OTP 인증으로 SSH 세션 열기
$ neuron-connect
🔐 OTP 번호를 입력하세요: 123456
✅ Neuron 연결 완료 (4시간 유지)

# 2. 프로젝트 초기화 및 환경 구축
$ neuron-env setup --from environment.yml
✅ Conda 환경 'manipulation_env' 생성 완료 on Neuron

# 3. 데이터셋 업로드 (DM 노드 경유, 최초 1회)
$ neuron-sync push-data ./data/ --via dm
📦 데이터 전송 중... 15GB → neuron-dm:/scratch/user/data/

# 4. 코드 동기화 (자동 감시 모드)
$ neuron-sync watch ./src/
👀 파일 변경 감시 시작...
   [12:03:15] model.py 변경 → 동기화 완료 (0.3s)
   [12:05:22] train.py 변경 → 동기화 완료 (0.2s)

# 5. 실험 정의 및 실행
$ neuron-exp launch experiment.yaml
📋 파라미터 조합 12개 생성
   - lr=1e-4, arch=resnet50, dim=256, drop=0.1  → job_001
   - lr=1e-4, arch=resnet50, dim=256, drop=0.3  → job_002
   - ...
🚀 SLURM 제출 완료: Job IDs [10001-10012]
   각 job에 evaluate 단계 연결 (dependency=afterok)

# 6. 실시간 모니터링
$ neuron-exp status
╭──────────┬──────────────┬─────────┬───────┬──────────╮
│ Exp ID   │ Parameters   │ Status  │ GPU   │ Epoch    │
├──────────┼──────────────┼─────────┼───────┼──────────┤
│ job_001  │ resnet50/256 │ RUNNING │ A100×2│ 45/100   │
│ job_002  │ resnet50/512 │ RUNNING │ A100×2│ 38/100   │
│ job_003  │ vit_base/256 │ PENDING │ -     │ -        │
│ ...      │ ...          │ ...     │ ...   │ ...      │
╰──────────┴──────────────┴─────────┴───────┴──────────╯

# 7. 결과 비교
$ neuron-exp results --sort accuracy
╭──────────┬──────────────┬──────────┬──────────┬──────────╮
│ Exp ID   │ Architecture │ LR       │ Accuracy │ Time(h)  │
├──────────┼──────────────┼──────────┼──────────┼──────────┤
│ job_007  │ vit_base/512 │ 3e-4     │ 0.923    │ 6.2      │
│ job_004  │ vit_base/256 │ 1e-4     │ 0.918    │ 4.8      │
│ job_001  │ resnet50/256 │ 1e-4     │ 0.895    │ 3.1      │
╰──────────┴──────────────┴──────────┴──────────┴──────────╯

# 8. 최적 모델 다운로드
$ neuron-sync pull-results job_007
📥 체크포인트 다운로드: best_model.pth (245MB)
📥 평가 결과: eval_results.json
```

---

## 6. 기술 스택

| 영역 | 기술 | 이유 |
|------|------|------|
| CLI 프레임워크 | Python (Click/Typer) | 로보틱스 연구 환경과 통일, 빠른 개발 |
| 설정 관리 | YAML + Hydra/OmegaConf | 계층적 설정 오버라이드, PyTorch 생태계 표준 |
| 템플릿 엔진 | Jinja2 | SLURM 스크립트 동적 생성 |
| 파일 동기화 | rsync (subprocess) | 증분 전송, SSH 위에서 동작 |
| 파일 감시 | watchdog | 크로스 플랫폼 파일시스템 이벤트 감지 |
| 실험 추적 | W&B (offline) + 자체 CSV/JSON | 오프라인 환경 대응 |
| 컨테이너 | Singularity | KISTI 공식 지원, rootless |
| 알림 | Slack webhook / SMTP | 경량, 의존성 없음 |

---

## 7. 일정 요약

| 주차 | 마일스톤 | 산출물 |
|------|---------|--------|
| 1주 | SSH 자동 연결 + 코드 동기화 | `neuron-connect`, `neuron-sync` |
| 2주 | 환경 자동 구축 + SLURM 템플릿 | `neuron-env`, 템플릿 .j2 파일들 |
| 3주 | 실험 매니저 (파라미터 sweep) | `neuron-exp launch/status/results` |
| 4주 | 파이프라인 의존성 + 결과 비교 | dependency chain, 결과 대시보드 |
| 5주 | Singularity 컨테이너화 | 컨테이너 빌드/배포 자동화 |
| 6주 | 스마트 스케줄링 + 알림 + 파일보존 | partition selector, keepalive |
| 7주+ | Genesis 시뮬레이터 연동 (필요 시) | 데이터 생성 → 학습 파이프라인 |

---

## 8. 리스크 및 대응

| 리스크 | 영향 | 대응 방안 |
|--------|------|----------|
| OTP 세션 만료 (ControlPersist 초과) | SSH 세션 끊김, 동기화 실패 | 세션 만료 전 자동 재연결 프롬프트 + 배치 작업은 제출 후 독립 실행 |
| /scratch 파일 자동 삭제 (15일) | 데이터/체크포인트 손실 | keepalive 크론잡 + 중요 결과는 로컬에 자동 pull |
| 계산 노드 네트워크 제한 | W&B, pip install 불가 | 오프라인 모드 기본 + Singularity로 사전 패키징 |
| SLURM 대기열 지연 | 실험 전체 지연 | 자동 파티션 폴백 (A100 대기 길면 V100으로 전환) |
| Conda 환경 깨짐 | 재현 불가, 디버깅 시간 낭비 | Phase 3에서 Singularity 전환으로 근본 해결 |

---

## 9. 다음 단계

이 계획서를 기반으로 **Phase 1부터 순차적으로 구현**을 시작할 수 있습니다.

우선 구현할 항목:
1. `~/.ssh/config` 설정 파일 생성
2. `neuron-sync` CLI 도구 (rsync 래퍼)
3. 기본 SLURM 템플릿 생성기

각 Phase의 코드 구현을 원하시면 말씀해 주세요.
