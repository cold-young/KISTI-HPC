# KISTI Neuron 자동화 개발 현황

> 마지막 업데이트: 2026-05-07 (Phase 5 실서버 테스트 완료)

---

## 전체 진행률

| Phase | 상태 | 진행률 |
|-------|------|--------|
| Phase 1: 기반 인프라 | 완료 | 100% |
| Phase 2: 실험 자동화 | 완료 | 100% |
| Phase 3: 고도화 | 완료 | 100% |
| Phase 4: 시뮬레이터 확장 | 완료 | 100% |
| Phase 5: 실서버 테스트 | 완료 | 100% |

---

## 프로젝트 구조

```
KISTI-HPC/
├── pyproject.toml              # 패키지 설정, 12개 CLI 엔트리포인트
├── neuron_cli/                 # 메인 CLI 패키지 (13개 모듈)
│   ├── __init__.py
│   ├── config.py               # YAML 설정 로더
│   ├── connect.py              # SSH ControlMaster 세션 관리
│   ├── sync.py                 # rsync 기반 코드 동기화
│   ├── env.py                  # Conda 환경 관리
│   ├── scheduler.py            # SLURM 스크립트 Jinja2 생성기
│   ├── experiment.py           # 실험 매니저 (launch/status/logs/cancel)
│   ├── results.py              # 결과 수집/비교/랭킹
│   ├── container.py            # Singularity 컨테이너 관리
│   ├── partition_selector.py   # 스마트 파티션 선택기
│   ├── keepalive.py            # /scratch 파일 보존 자동화
│   ├── notify.py               # Slack/Email 알림
│   └── simulator.py            # 시뮬레이터 호환성/파이프라인 생성
├── templates/                  # Jinja2 SLURM 템플릿 (4개)
├── configs/                    # neuron.yaml, defaults.yaml
├── containers/                 # Singularity .def (3개)
├── examples/                   # experiment YAML 예제 (2개)
├── tests/                      # pytest 테스트
└── dev_doc/                    # 계획서, 매뉴얼, 현황
```

---

## CLI 엔트리포인트 현황

| 명령어 | 모듈 | 실서버 테스트 |
|--------|------|:---:|
| `neuron-connect` | `connect:connect` | PASS |
| `neuron-status` | `connect:status` | PASS |
| `neuron-disconnect` | `connect:disconnect` | PASS |
| `neuron-sync` | `sync:cli` | PASS |
| `neuron-env` | `env:cli` | - |
| `neuron-exp` | `experiment:cli` | PASS |
| `neuron-results` | `results:cli` | PASS |
| `neuron-container` | `container:cli` | PASS |
| `neuron-keepalive` | `keepalive:cli` | PASS |
| `neuron-notify` | `notify:cli` | - |
| `neuron-partition` | `partition_selector:select_partition` | PASS |
| `neuron-sim` | `simulator:cli` | PASS |

---

## Phase 5 실서버 테스트 결과

### 테스트 환경
- 서버: KISTI Neuron (`neuron.ksc.re.kr`)
- 사용자: x3320a02
- 테스트 파티션: cas_v100_2 (V100 GPU)
- 테스트 노드: gpu26

### 테스트 결과

| 테스트 | 결과 | 실행된 Job ID |
|--------|:----:|:---:|
| SSH ControlMaster 연결 | PASS | - |
| neuron-sync push/pull | PASS | - |
| neuron-exp launch (2-job sweep) | PASS | 717737, 717738 |
| neuron-exp status | PASS | - |
| neuron-results (결과 조회/랭킹) | PASS | - |
| neuron-partition (파티션 추천) | PASS | - |
| neuron-keepalive check | PASS | - |
| neuron-container list | PASS | - |
| neuron-sim list/compat/check-env | PASS | - |

### 실험 결과 (실제 Neuron에서 실행)

```
_exp_id              accuracy    loss        learning_rate
20260507_172406_001  0.9099      0.0926      0.0003         <- Best
20260507_172406_000  0.8599      0.1426      0.0001
20260507_172028_000  0.8599      0.1426      0.0001
```

### 실서버 테스트 중 발견 & 수정한 버그 (5건)

| # | 발견 사항 | 원인 | 수정 내용 |
|---|----------|------|----------|
| S1 | rsync 인증 실패 | ControlMaster 소켓 미전달 | `_run_rsync`에 `-e "ssh -o ControlPath=..."` 추가 |
| S2 | sbatch 제출 실패 | KISTI는 scratch에서만 sbatch 허용 | `experiment.py _run_remote`에 `cd /scratch/$USER &&` 래핑 |
| S3 | scp 파일 업로드 실패 | scp가 ControlMaster 미지원 | scp → rsync 전면 교체 (`experiment.py`, `env.py`) |
| S4 | SLURM 작업 즉시 취소 (exit 53) | 템플릿의 `$USER`가 해석 안 됨 | `$USER` → Jinja2 `{{ user }}` 교체, `module purge` 제거, `source activate` 조건부 |
| S5 | neuron-results 결과 0건 | `-exec sh -c` 이스케이핑 깨짐 | 2단계 접근(find → echo+cat)으로 변경 |

---

## 버그 이력 (전체 #1~S5 해결 완료)

### Phase 1 버그 (#1~#4): 모두 해결
- env.py 이중 SSH, 환경 파싱, 쉘 인젝션, sync.py 패턴 하드코딩

### Phase 2 버그 (#5~#12): 모두 해결
- multi_gpu torchrun, exit code 통일, job_array 인덱싱, scheduler job_array 함수, experiment dead code/multi-GPU/upload

### Phase 3 버그 (#13~#24): 모두 해결
- pyproject.toml 엔트리포인트, SLURM 옵션, shlex, keepalive find 패턴, partition 안내

### Phase 5 버그 (#S1~#S5): 모두 해결
- rsync ControlPath, sbatch scratch, scp→rsync, $USER→{{ user }}, results 파싱

**미해결 버그: 0건**

---

## 다음 단계 (향후 개선)

1. **테스트 커버리지 확대**: multi_gpu, job_array, simulator 테스트 추가
2. **neuron-env 실서버 테스트**: Conda/Singularity 환경 구축 검증
3. **neuron-notify 실서버 테스트**: Slack webhook 설정 후 알림 검증
4. **neuron-container build --remote-build 테스트**: 대용량 컨테이너 빌드
5. **멀티 노드 분산 학습 테스트**: multi_gpu 템플릿 실제 검증
6. **Genesis 시뮬레이터 E2E**: genesis-sim.sif 빌드 → 데이터 생성 → 학습 파이프라인
