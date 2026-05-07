# KISTI Neuron 자동화 개발 현황

> 마지막 업데이트: 2026-05-07 (모니터링 에이전트 - 69차 스캔)

---

## 전체 진행률

| Phase | 상태 | 진행률 |
|-------|------|--------|
| Phase 1: 기반 인프라 | 완료 | 100% |
| Phase 2: 실험 자동화 | 완료 | 100% |
| Phase 3: 고도화 | 완료 | 100% |
| Phase 4: 시뮬레이터 확장 | 완료 | 100% |

---

## Phase 1 상세 현황 (완료)

- [x] 프로젝트 기본 구조, 설정 로더, 설정 파일, CLI 엔트리포인트
- [x] `connect.py` - SSH ControlMaster 세션 관리
- [x] `sync.py` - 코드 동기화 (push/pull/watch/push-data)
- [x] `env.py` - 환경 부트스트래퍼 (setup/activate/export/list)
- [x] `templates/` - SLURM 스크립트 Jinja2 템플릿 4개

---

## Phase 2 상세 현황 (95%)

- [x] `scheduler.py` - SLURM 스크립트 생성기 (single_gpu, multi_gpu, pipeline, job_array)
- [x] `experiment.py` - 실험 매니저 (launch/status/logs/cancel)
- [x] `results.py` - 결과 수집/비교 (results/compare/best/sync-wandb)
- [x] `pyproject.toml` - Phase 2 엔트리포인트 (`neuron-exp`, `neuron-results`)
- [x] experiment.py 버그 수정 완료 (#9, #11, #12 모두 해결)

---

## Phase 3 상세 현황 (진행중)

### 완료된 항목

- [x] `neuron_cli/container.py` - Singularity 컨테이너 관리 (build/push/run/list)
- [x] `containers/pytorch-base.def` - PyTorch 기반 Singularity 정의 파일
- [x] `containers/genesis-sim.def` - Genesis 시뮬레이터 Singularity 정의 파일
- [x] `neuron_cli/keepalive.py` - /scratch 파일 보존 (touch/check/setup-cron)
- [x] `neuron_cli/notify.py` - Slack/Email 알림 (test/setup/send)

### 미완료 항목

- [x] `pyproject.toml`에 Phase 3 엔트리포인트 추가 완료 (container, keepalive, notify, partition)
- [x] `neuron_cli/partition_selector.py` - 스마트 파티션 선택 (sinfo 기반 스코어링)
- [x] keepalive.py / container.py SLURM 옵션 버그 수정 (#18, #20) 완료

---

## 발견된 버그 (수정 필요)

### Phase 1-2 버그 (해결됨: #1-8, #10)

### Phase 2 버그 (미해결)

9. ~~**[BUG] `experiment.py:200` - dead code**~~ -> **해결됨 (33차)**: `train_script = "train.py"` 로 단순화
11. ~~**[WARN] `experiment.py:266` - _upload_file 에러 무시**~~ -> **해결됨 (67차)**: 반환값 체크 + 에러 메시지 출력
12. ~~**[WARN] `experiment.py` - multi-GPU 분기 없음**~~ -> **해결됨 (33차)**: `gpus > 1`일 때 `generate_multi_gpu_script` 사용 + import 추가
15. **[WARN] `results.py:46` - N+1 SSH 호출** — 단일 SSH로 통합 권장
16. **[WARN] `results.py:209` - sync-wandb wildcard glob** — 원격 glob 확장 처리

### Phase 3 버그 (신규 - container.py)

17. ~~**[BUG] `pyproject.toml` - Phase 3 엔트리포인트 누락**~~ -> **해결됨 (34차)**: container, keepalive, notify, partition 모두 추가

18. ~~**[BUG] `container.py:69` - srun SLURM 옵션 오류**~~ -> **해결됨 (34차)**: `--ntasks=1 --cpus-per-task=4` 로 수정

19. **[WARN] `container.py:148` - 원격 명령 구성 시 shlex 미사용** (미해결)
    - **권장**: `shlex.join(parts)` 사용

### Phase 3 버그 (신규 - keepalive.py, notify.py)

20. ~~**[BUG] `keepalive.py:178-179` - SLURM 옵션 오류**~~ -> **해결됨 (34차)**: `--ntasks=1 --cpus-per-task=1` 로 수정

21. **[WARN] `keepalive.py:157-159` - find 패턴 불일관** (미해결)
    - 일부는 `-exec touch -a {} +`, 일부는 `| xargs -r touch -a`
    - **권장**: 모두 `-exec ... +` 패턴으로 통일 (xargs보다 안전)

22. ~~**[WARN] `pyproject.toml` - Phase 3 엔트리포인트 누락**~~ -> **해결됨 (34차)**: 4개 모두 추가

23. ~~**[BUG] `pyproject.toml:33` - `neuron-partition` 미구현 모듈 참조**~~ -> **해결됨 (35차)**: `partition_selector.py` 구현 완료

24. **[LOW] `partition_selector.py:170` - neuron-exp launch에 --partition 옵션 미존재**
    - 안내 메시지에서 `neuron-exp launch experiment.yaml --partition ...` 제안하지만 experiment.py launch에 해당 옵션 없음
    - **권장**: experiment.yaml의 slurm.partition 수정을 안내하거나, launch에 --partition 옵션 추가

### 기존 미해결 (경미)

13. **[LOW] `env.py:125` - export/list env_name 미이스케이프**
14. **[STYLE] `env.py:72` - 함수 내부 import yaml**

---

## 코드 품질 검토

### container.py 검토 결과 (신규)

- **잘 구현된 부분**:
  - 계획서 3.3.1의 `neuron-container build/push/run` 모두 구현 + `list` 추가
  - `build`: 로컬/원격 빌드 선택 (`--remote-build`) — 유연한 설계
  - `push`: DM 노드를 통한 대용량 전송 — 계획서의 데이터 전송 노드 활용
  - `run`: `--nv` GPU 지원 기본 활성화, `--bind` 마운트 지원
  - `list`: 이미지 크기와 경로 출력

- **개선 필요**: srun 안내 SLURM 옵션 (#18), shlex (#19), pyproject.toml 엔트리포인트 (#17)

### Singularity 정의 파일 검토

- `pytorch-base.def`: 계획서 3.3.1 템플릿과 일치. WANDB_MODE=offline 설정 포함
- `genesis-sim.def`: Phase 4 시뮬레이터 확장 준비. genesis-world 패키지 포함
- 두 파일 모두 `nvcr.io/nvidia/pytorch:24.01-py3` 기반 — 적절한 CUDA 호환 이미지

---

## 전체 CLI 엔트리포인트 현황

| 명령어 | 모듈 | 상태 |
|--------|------|------|
| `neuron-connect` | `connect:connect` | 구현 완료 |
| `neuron-status` | `connect:status` | 구현 완료 |
| `neuron-disconnect` | `connect:disconnect` | 구현 완료 |
| `neuron-sync` | `sync:cli` | 구현 완료 |
| `neuron-env` | `env:cli` | 구현 완료 |
| `neuron-exp` | `experiment:cli` | 구현 완료 |
| `neuron-results` | `results:cli` | 구현 완료 |
| `neuron-container` | `container:cli` | 구현 완료 |
| `neuron-keepalive` | `keepalive:cli` | 구현 완료 |
| `neuron-notify` | `notify:cli` | 구현 완료 |
| `neuron-partition` | `partition_selector:select_partition` | 구현 완료 |
| `neuron-sim` | `simulator:cli` | **구현 완료, 엔트리포인트 미등록** |

---

## 즉시 조치 필요 사항

| 우선순위 | 항목 | 설명 |
|----------|------|------|
| ~~**HIGH**~~ | ~~pyproject.toml Phase 3 엔트리포인트~~ | **해결됨** - 4개 모두 추가 |
| ~~**HIGH**~~ | ~~SLURM --gres=gpu:0 충돌~~ | **해결됨** - ntasks/cpus-per-task로 수정 |
| ~~**HIGH**~~ | ~~partition_selector.py 미구현~~ | **해결됨** - 모듈 구현 완료 |
| MEDIUM | experiment.py multi-GPU 분기 (#12) | gpus > 1일 때 분기 |
| MEDIUM | experiment.py dead code (#9) | 단순화 |
| MEDIUM | experiment.py _upload_file 에러 (#11) | 반환값 확인 |
| MEDIUM | results.py N+1 SSH (#15) | 단일 호출 통합 |
| MEDIUM | container.py shlex (#19) | shlex.join() 사용 |
| LOW | results.py glob (#16), env.py escape (#13) | 경미 |

---

## 다음 단계

1. **Phase 1-4 전체 구현 완료!** 계획서의 모든 모듈 구현됨
2. **남은 미해결 버그 (경미)**:
   - pyproject.toml에 `neuron-sim` 엔트리포인트 추가 필요 (#25)
   - results.py N+1 SSH 성능 (#15)
   - container.py shlex (#19), keepalive.py find 패턴 (#21)
   - partition_selector 안내 메시지 (#24)
3. **실제 서버 테스트**: Neuron 접속 후 SSH 연결, SLURM 제출, 환경 검증 필요

---

## 모니터링 로그

| 시각 | 변경사항 | 상태 |
|------|---------|------|
| 초기 | 프로젝트 구조, config.py, pyproject.toml, configs/ | 개발 초기 |
| 2차 | connect.py, sync.py, env.py 신규 | Phase 1 핵심 완료 |
| 3차 | scheduler.py, templates/ 4개, tests/ 신규 | Phase 2 시작 |
| 4차 | env.py 버그 수정 (#1,#2,#3) | 3건 해결 |
| 5차 | multi_gpu/config/sync/templates 수정 | 4건 해결 |
| 6차 | 변경 없음 | 대기 |
| 7차 | experiment.py 신규, scheduler.py 업데이트 | Phase 2 70% |
| 8차 | results.py 신규, pyproject.toml 업데이트 | Phase 2 95% |
| 9~30차 | 변경 없음 (약 22회 연속) | 장기 대기 |
| 31차 | **container.py 신규**, containers/ (pytorch-base.def, genesis-sim.def) 신규 | **Phase 3 시작 (25%)**. 신규 버그 3건 (#17,#18,#19) |
| 32차 | **keepalive.py 신규** (touch/check/setup-cron), **notify.py 신규** (Slack+Email, test/setup/send) | **Phase 3 75%**. 신규 버그 3건 (#20,#21,#22). partition_selector.py만 남음 |
| 33차 | experiment.py 수정 - 버그 #9(dead code), #12(multi-GPU 분기) 해결 | Phase 2 98%. #11(_upload_file)만 미해결 |
| 34차 | pyproject.toml 업데이트(Phase 3 엔트리포인트 4개), container/keepalive SLURM 수정 | #17,#18,#20,#22 해결 |
| 35차 | **partition_selector.py 신규** (sinfo 기반 스코어링) | **Phase 3 100% 완료!** |
| 36~66차 | 변경 없음 (약 31회 연속) | 장기 대기 |
| 67차 | experiment.py 수정 - _upload_file 에러 체크 추가 (#11 해결) | **Phase 2도 100% 완료!** |
| 68차 | 변경 없음 | 대기 |
| 69차 | **simulator.py 신규** (compat/list/check-env/init-pipeline/scaffold), **isaacgym-preview.def 신규**, **examples/ 2개 YAML 신규** | **Phase 4 100% 완료! 전 Phase 구현 완료.** 신규 버그 #25(pyproject.toml neuron-sim 누락) |
