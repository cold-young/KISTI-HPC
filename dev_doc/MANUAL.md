# neuron-cli 사용 매뉴얼

KISTI Neuron HPC 워크플로우 자동화 CLI 도구

---

## 목차

1. [설치 및 초기 설정](#1-설치-및-초기-설정)
2. [일상 워크플로우 (Quick Start)](#2-일상-워크플로우-quick-start)
3. [명령어 상세 레퍼런스](#3-명령어-상세-레퍼런스)
4. [실전 시나리오](#4-실전-시나리오)
5. [실제 서버 테스트 계획 (Phase 5)](#5-실제-서버-테스트-계획-phase-5)
6. [트러블슈팅](#6-트러블슈팅)

---

## 1. 설치 및 초기 설정

### 1.1 패키지 설치

```bash
cd KISTI-HPC
pip install -e .
```

### 1.2 KISTI 사용자 ID 설정

두 가지 방법 중 택 1:

**방법 A) 환경변수 (권장)**
```bash
# ~/.zshrc 또는 ~/.bashrc에 추가
export NEURON_USER="your_kisti_id"
```

**방법 B) 설정 파일 직접 수정**
```bash
# configs/neuron.yaml
ssh:
  user: "your_kisti_id"   # <-- 여기에 입력
```

### 1.3 SSH Config 자동 생성

```bash
neuron-connect --setup-config
```

이 명령은 `~/.ssh/config`에 ControlMaster 설정을 자동으로 추가합니다.
수동으로 하려면 `~/.ssh/config`에 아래 내용을 직접 추가:

```
Host neuron
    HostName neuron.ksc.re.kr
    User <YOUR_ID>
    ControlMaster auto
    ControlPath ~/.ssh/sockets/%r@%h-%p
    ControlPersist 4h
    ServerAliveInterval 60
```

### 1.4 알림 설정 (선택)

```bash
neuron-notify setup          # configs/notify.yaml 템플릿 생성
# 편집: Slack webhook URL 또는 Email 설정 입력
neuron-notify test --method slack   # 테스트 알림 전송
```

---

## 2. 일상 워크플로우 (Quick Start)

아래는 매일 연구할 때 반복하는 전형적인 흐름입니다.

### Step 1: 접속 (1일 1회)

```bash
neuron-connect
# OTP 번호 입력 → 4시간 세션 유지
# 이후 모든 명령에서 OTP 재입력 불필요
```

### Step 2: 코드 동기화

```bash
# 한번에 올리기
neuron-sync push ./my_project/

# 자동 감시 모드 (파일 저장할 때마다 자동 동기화)
neuron-sync watch ./my_project/
```

### Step 3: 실험 실행

```bash
# experiment.yaml 작성 후
neuron-exp launch experiment.yaml

# 상태 확인
neuron-exp status

# 로그 실시간 보기
neuron-exp logs <JOB_ID>
```

### Step 4: 결과 확인

```bash
neuron-results results my_project --sort accuracy
neuron-results best my_project --metric accuracy --top 5
```

### Step 5: 결과 다운로드

```bash
neuron-sync pull ./my_project/         # 결과 파일만 가져오기
neuron-sync pull ./my_project/ --all-files  # 전체 가져오기
```

---

## 3. 명령어 상세 레퍼런스

### 3.1 SSH 연결 관리

| 명령어 | 설명 |
|--------|------|
| `neuron-connect` | SSH 세션 열기 (OTP 1회 입력) |
| `neuron-connect --setup-config` | ~/.ssh/config 자동 생성 |
| `neuron-status` | 연결 상태 확인 |
| `neuron-disconnect` | 세션 종료 |

### 3.2 코드 동기화 (`neuron-sync`)

```bash
neuron-sync push [PATH]                     # 로컬 → Neuron
neuron-sync push . --delete                 # 원격에 없는 파일도 삭제 동기화
neuron-sync push . --dry-run                # 실제 전송 없이 미리보기
neuron-sync pull [PATH]                     # Neuron → 로컬 (결과만)
neuron-sync pull . --all-files              # 전체 파일 가져오기
neuron-sync watch [PATH]                    # 파일 변경 자동 감시+동기화
neuron-sync watch . --interval 5            # 5초 간격 debounce
neuron-sync push-data ./data/ --via dm      # 대용량 데이터 (DM 노드 경유)
```

**동기화 제외 대상** (`configs/defaults.yaml`에서 관리):
- `.git/`, `__pycache__/`, `wandb/`, `*.pt`, `*.pth`, `outputs/`, `logs/`

### 3.3 환경 관리 (`neuron-env`)

```bash
neuron-env setup --from environment.yml     # Conda 환경 생성/업데이트
neuron-env setup --from env.yml --name myenv  # 이름 지정
neuron-env activate myenv                   # 활성화 명령 출력 (복사용)
neuron-env export myenv                     # 원격 환경 → environment.yml
neuron-env list                             # Neuron의 conda 환경 목록
```

### 3.4 실험 매니저 (`neuron-exp`)

```bash
neuron-exp launch experiment.yaml           # 실험 제출
neuron-exp launch experiment.yaml -n 3      # 상위 3개 조합만
neuron-exp launch experiment.yaml --dry-run # 스크립트만 생성
neuron-exp launch experiment.yaml --partition cas_v100nv_8  # 파티션 오버라이드
neuron-exp launch experiment.yaml --gpus 4  # GPU 수 오버라이드
neuron-exp status                           # 실행중인 작업 목록
neuron-exp logs <JOB_ID>                    # 로그 스트리밍 (Ctrl+C로 종료)
neuron-exp cancel <JOB_ID>                  # 작업 취소
neuron-exp cancel all                       # 전체 취소
```

### 3.5 결과 비교 (`neuron-results`)

```bash
neuron-results results <PROJECT> --sort accuracy    # 전체 결과 (정렬)
neuron-results compare <PROJECT> exp1 exp2          # 두 실험 비교
neuron-results best <PROJECT> --metric accuracy -t 5  # 상위 5개
neuron-results sync-wandb <PROJECT>                 # W&B 오프라인 → 로컬 동기화
```

### 3.6 파티션 추천 (`neuron-partition`)

```bash
neuron-partition --gpus 2                   # 2 GPU 기준 최적 파티션
neuron-partition --gpus 4 --prefer-cost     # 비용 효율 우선
```

출력 예시:
```
Rank  Partition          GPU    Idle   Mix   Perf   Cost   Score
1     amd_a100nv_8       A100   3      2     1.0    Med    63.0  <-- recommended
2     cas_v100nv_8       V100   5      1     0.5    Low    58.0
...
```

### 3.7 컨테이너 관리 (`neuron-container`)

```bash
neuron-container build containers/pytorch-base.def            # 로컬 빌드
neuron-container build containers/genesis-sim.def --remote-build  # Neuron에서 빌드
neuron-container push pytorch-base.sif                        # DM 노드로 업로드
neuron-container run /scratch/$USER/.neuron-tools/pytorch-base.sif python train.py
neuron-container list                                         # 원격 이미지 목록
```

### 3.8 파일 보존 (`neuron-keepalive`)

```bash
neuron-keepalive check --days 12            # 12일 내 삭제될 파일 확인
neuron-keepalive touch                      # 중요 파일 atime 갱신
neuron-keepalive touch --dry-run            # 미리보기
neuron-keepalive setup-cron --interval 7    # 7일 주기 자동 실행 설정
```

### 3.9 알림 (`neuron-notify`)

```bash
neuron-notify setup                         # notify.yaml 템플릿 생성
neuron-notify test --method slack            # 테스트 알림
neuron-notify test --method email            # 테스트 이메일
```

SLURM 스크립트에서 자동 호출 (직접 사용할 일은 적음):
```bash
neuron-notify send --status $? --job-name $SLURM_JOB_NAME --exp-id $EXPERIMENT_ID
```

### 3.10 시뮬레이터 (`neuron-sim`)

```bash
neuron-sim list                             # 시뮬레이터 GPU 호환성 표
neuron-sim compat genesis -p amd_a100nv_8   # 호환성 확인
neuron-sim check-env genesis                # Neuron에 설치 여부 확인
neuron-sim init-pipeline genesis --env manipulation  # experiment.yaml 자동 생성
neuron-sim scaffold genesis --env manipulation       # 스타터 코드 생성
```

---

## 4. 실전 시나리오

### 시나리오 A: 새로운 모델 아키텍처 서치

```bash
# 1. 접속
neuron-connect

# 2. 환경 구축
neuron-env setup --from environment.yml

# 3. 코드 동기화 (백그라운드 자동 감시)
neuron-sync watch ./src/ &

# 4. experiment.yaml 작성
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

# 5. 먼저 dry-run으로 확인
neuron-exp launch experiment.yaml --dry-run

# 6. 실행 (18개 조합 = 3 lr * 3 arch * 2 dim)
neuron-exp launch experiment.yaml

# 7. 결과 확인
neuron-exp status
neuron-results best arch_search --metric accuracy --top 5
```

### 시나리오 B: Genesis 시뮬레이션 파이프라인

```bash
# 1. 호환성 확인
neuron-sim compat genesis -p amd_a100nv_8

# 2. 스타터 코드 + 파이프라인 자동 생성
neuron-sim scaffold genesis --env manipulation -o ./src/
neuron-sim init-pipeline genesis --env manipulation -o experiment_sim.yaml

# 3. 생성된 코드 수정 (TODO 부분 구현)
# src/genesis_datagen.py  - 데이터 생성 로직
# src/train.py            - 학습 로직
# src/genesis_eval.py     - 평가 로직

# 4. 컨테이너 준비
neuron-container build containers/genesis-sim.def --remote-build
# 또는 로컬 빌드 후 업로드
neuron-container build containers/genesis-sim.def
neuron-container push genesis-sim.sif

# 5. 파이프라인 실행 (datagen → train → eval 자동 체이닝)
neuron-exp launch experiment_sim.yaml
```

### 시나리오 C: 기존 Isaac Gym 코드 마이그레이션

```bash
# 1. Isaac Gym 환경 고정 (Singularity)
neuron-container build containers/isaacgym-preview.def --remote-build

# 2. 호환성 확인
neuron-sim compat isaac_gym -p amd_a100nv_8

# 3. 기존 코드 동기화 및 실행
neuron-sync push ./isaacgym_project/
neuron-exp launch experiment_isaacgym.yaml

# 참고: Isaac Gym은 deprecated
# 신규 프로젝트는 Genesis 사용 권장
neuron-sim list  # 호환성 비교표 확인
```

---

## 5. 실제 서버 테스트 계획 (Phase 5)

개발한 CLI 도구를 실제 Neuron 서버에서 검증하기 위한 단계별 테스트 계획입니다.

### 5.1 사전 준비

| 항목 | 확인 사항 | 명령어 |
|------|----------|--------|
| KISTI 계정 | ID/PW + Any OTP 앱 등록 완료 | - |
| 로컬 설치 | `pip install -e .` 성공 | `neuron-exp --help` |
| 사용자 ID | 환경변수 설정 | `echo $NEURON_USER` |

### 5.2 단계별 테스트

#### Test 1: SSH 연결 (5분)

```bash
# 1-1. SSH config 생성
neuron-connect --setup-config

# 1-2. 연결 테스트 (OTP 입력)
neuron-connect

# 1-3. 연결 확인
neuron-status
# 예상: "Neuron: CONNECTED"

# 1-4. 세션 유지 확인 (OTP 재입력 없이)
ssh neuron 'hostname'
# 예상: neuron01, neuron02, 또는 neuron03

# 1-5. 종료 테스트
neuron-disconnect
neuron-status
# 예상: "Neuron: DISCONNECTED"
```

**통과 기준**: OTP 1회 입력 후 추가 인증 없이 SSH 명령 실행 가능

#### Test 2: 파일 동기화 (10분)

```bash
# 2-1. 테스트 프로젝트 생성
mkdir -p /tmp/neuron_test && echo "print('hello')" > /tmp/neuron_test/test.py

# 2-2. Push 테스트
neuron-connect
neuron-sync push /tmp/neuron_test/ -p neuron_test

# 2-3. 원격 확인
ssh neuron 'ls /scratch/$USER/projects/neuron_test/'
# 예상: test.py

# 2-4. Pull 테스트
ssh neuron 'echo "result=42" > /scratch/$USER/projects/neuron_test/eval_results.json'
neuron-sync pull /tmp/neuron_test/ -p neuron_test --all-files

# 2-5. Watch 모드 테스트 (별도 터미널)
neuron-sync watch /tmp/neuron_test/ -p neuron_test
# 다른 터미널에서 파일 수정 → 자동 동기화 확인
```

**통과 기준**: push/pull 양방향 동기화 성공, watch 모드 자동 감지

#### Test 3: 환경 설정 (15분)

```bash
# 3-1. 테스트용 environment.yml 생성
cat > /tmp/environment.yml << 'EOF'
name: neuron_test_env
dependencies:
  - python=3.10
  - pytorch
  - pip:
    - wandb
    - rich
EOF

# 3-2. 환경 생성
neuron-env setup --from /tmp/environment.yml

# 3-3. 환경 확인
neuron-env list

# 3-4. 활성화 명령 확인
neuron-env activate neuron_test_env
```

**통과 기준**: Conda 환경 생성 완료, `neuron-env list`에 표시

#### Test 4: SLURM 작업 제출 (20분)

```bash
# 4-1. 간단한 학습 스크립트 준비
cat > /tmp/neuron_test/train.py << 'PYEOF'
import os, json, time
config_path = os.environ.get("CONFIG_PATH", "")
exp_id = os.environ.get("EXPERIMENT_ID", "unknown")
print(f"Running experiment: {exp_id}")
print(f"Config: {config_path}")
print(f"GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'none')}")
# 간단한 시뮬레이션
time.sleep(10)
results = {"accuracy": 0.85, "loss": 0.32, "exp_id": exp_id}
os.makedirs("experiments/" + exp_id, exist_ok=True)
with open(f"experiments/{exp_id}/eval_results.json", "w") as f:
    json.dump(results, f)
print("Done!")
PYEOF

# 4-2. experiment.yaml 작성
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

# 4-3. 동기화
neuron-sync push /tmp/neuron_test/ -p neuron_test

# 4-4. dry-run 확인
neuron-exp launch /tmp/neuron_test/experiment.yaml --dry-run

# 4-5. 실제 제출 (작은 파티션으로 테스트)
neuron-exp launch /tmp/neuron_test/experiment.yaml --partition cas_v100_2 --gpus 1

# 4-6. 상태 확인
neuron-exp status

# 4-7. 로그 확인
neuron-exp logs <JOB_ID>
```

**통과 기준**: sbatch 제출 성공, squeue에 표시, 로그 스트리밍 동작

#### Test 5: 결과 수집 (10분)

```bash
# 5-1. 작업 완료 대기 후 결과 확인
neuron-results results neuron_test --sort accuracy
neuron-results best neuron_test --metric accuracy

# 5-2. 결과 다운로드
neuron-sync pull /tmp/neuron_test/ -p neuron_test
```

**통과 기준**: eval_results.json 파싱 성공, 정렬/비교 출력 정상

#### Test 6: 파티션/보존/알림 (10분)

```bash
# 6-1. 파티션 추천
neuron-partition --gpus 2

# 6-2. 파일 보존 상태 확인
neuron-keepalive check --days 12
neuron-keepalive touch --dry-run

# 6-3. 알림 테스트 (Slack 설정한 경우)
neuron-notify test --method slack
```

**통과 기준**: sinfo 파싱 + 스코어링 출력, keepalive 파일 스캔 성공

#### Test 7: 시뮬레이터 환경 검증 (선택, 30분)

```bash
# 7-1. Genesis 설치 확인
neuron-sim check-env genesis

# 7-2. 미설치 시 컨테이너 빌드
neuron-container build containers/genesis-sim.def --remote-build

# 7-3. Genesis import 테스트
neuron-container run /scratch/$USER/.neuron-tools/genesis-sim.sif \
    python -c "import genesis; print('Genesis OK')"
```

**통과 기준**: Genesis import 성공 (Conda 또는 Singularity 내)

### 5.3 테스트 체크리스트

```
[ ] Test 1: SSH 연결 (connect/status/disconnect)
[ ] Test 2: 파일 동기화 (push/pull/watch)
[ ] Test 3: 환경 설정 (setup/list/activate)
[ ] Test 4: SLURM 제출 (launch/status/logs/cancel)
[ ] Test 5: 결과 수집 (results/best/compare)
[ ] Test 6: 파티션/보존/알림
[ ] Test 7: 시뮬레이터 (선택)
```

### 5.4 예상 이슈 및 대응

| 예상 이슈 | 원인 | 대응 방법 |
|-----------|------|----------|
| OTP 세션 만료 | ControlPersist 4h 초과 | `neuron-connect` 재실행 |
| rsync 연결 거부 | SSH 소켓 없음 | `neuron-status` 확인 → `neuron-connect` |
| sbatch 실패 | 파티션 풀, 옵션 오류 | `neuron-partition`으로 대안 확인, `--dry-run`으로 스크립트 검토 |
| module load 실패 | 모듈명 변경 | `ssh neuron 'module avail cuda'`로 확인, `configs/defaults.yaml` 수정 |
| conda create 실패 | 디스크 쿼터 초과 | `ssh neuron 'du -sh /scratch/$USER'` 확인 |
| /scratch 파일 삭제 | 15일 미접근 | `neuron-keepalive setup-cron` 즉시 설정 |
| 계산 노드 네트워크 차단 | pip/wandb 외부 접속 불가 | Singularity 컨테이너 사용, `WANDB_MODE=offline` |

---

## 6. 트러블슈팅

### SSH 연결 문제

```bash
# 소켓 상태 확인
ls -la ~/.ssh/sockets/

# 죽은 소켓 정리
neuron-disconnect
rm -f ~/.ssh/sockets/*neuron*
neuron-connect
```

### SLURM 작업 안 잡힘

```bash
# 대기열 확인
neuron-partition --gpus 2

# 더 빈 파티션으로 변경
neuron-exp launch experiment.yaml --partition cas_v100_2 --gpus 1
```

### 결과 파일 없음

```bash
# 로그 확인
neuron-exp logs <JOB_ID>

# 직접 원격 확인
ssh neuron 'ls /scratch/$USER/projects/<PROJECT>/experiments/'
```

### watch 모드 동기화 안 됨

```bash
# 제외 패턴 확인 (configs/defaults.yaml의 sync.exclude)
neuron-sync push . --dry-run

# 강제 동기화
neuron-sync push .
```
