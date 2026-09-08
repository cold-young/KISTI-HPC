<div align="center">

# rcac-cli

**퍼듀 RCAC Gilbreth 클러스터에서 GPU 실험을 돌리기 위한 로컬 커맨드라인 도구.**

[English](README.md) · [한국어](README.ko.md)

</div>

`rcac-cli`는 여러분을 노트북 앞에 그대로 둡니다. Gilbreth 프런트엔드 노드로
인증된 SSH 세션을 하나만 열어 두고, 그 세션을 통해 코드를 올리고 환경을 만들고
`sbatch` 스크립트를 생성·제출하고 작업을 지켜보다가 결과를 다시 내려받습니다.
그래서 Purdue Login 인증은 명령마다가 아니라 세션마다 한 번만 뜹니다.

이 도구는 **Gilbreth 전용**입니다. 여기에 들어 있는 파티션·QOS·파일시스템 사실은
모두 [§2](#2-사실로-보는-gilbreth)에서 `docs.rcac.purdue.edu` URL로 출처가 붙어
있고, 그 문서로 확인할 수 없었던 것은 추측하지 않고
[§7](#7-실서버-검증-체크리스트)의 미결 항목으로 남겨 두었습니다.

## 목차

1. [설치와 초기 설정](#1-설치와-초기-설정)
2. [사실로 보는 Gilbreth](#2-사실로-보는-gilbreth)
3. [제출의 네 조각](#3-제출의-네-조각)
4. [일상 워크플로우 (Quick Start)](#4-일상-워크플로우-quick-start)
5. [명령어 레퍼런스](#5-명령어-레퍼런스)
6. [시나리오](#6-시나리오)
7. [실서버 검증 체크리스트](#7-실서버-검증-체크리스트)
8. [트러블슈팅](#8-트러블슈팅)
9. [프로젝트 연혁](#9-프로젝트-연혁)

---

## 1. 설치와 초기 설정

### 1.1 패키지 설치

```bash
git clone <this-repo> && cd <this-repo>
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

콘솔 스크립트 12개가 `PATH`에 설치됩니다. 12개 전부는
[§5](#5-명령어-레퍼런스)에 정리되어 있습니다. Gilbreth 쪽에는 아무것도 설치하지
않습니다. 클러스터에서 쓰는 것은 순수한 `ssh`, `rsync`, `sbatch` 뿐입니다.

### 1.2 내가 누구인지 알려주기

값은 두 개, 그리고 **환경변수가 기본 경로**입니다.

```bash
export RCAC_USER=<퍼듀 career account ID>
export RCAC_ACCOUNT=<과금 가능한 Slurm 계정>
```

`RCAC_USER`와 `RCAC_ACCOUNT`는 **항상** `configs/rcac.yaml`보다 우선합니다. 그
파일은 git이 추적하는 파일이고 `YOUR_PURDUE_ID`, `YOUR_QUEUE`라는 자리표시자
상태로 배포됩니다. 환경변수 경로가 존재하는 이유가 바로 이것입니다. 추적되는
파일을 고치지 않고도 도구를 쓸 수 있어야 하기 때문입니다. 위 두 줄은 평소 셸
환경을 관리하는 곳에 넣어 두세요.

계정 이름을 아직 모른다면 `rcac-partition`을 실행하세요. 사용자 값이 없어도
동작하며, 과금 가능한 계정 목록을 클러스터에서 받아 출력합니다. 클러스터에
직접 들어가 있다면 `slist`가 같은 표를 보여 줍니다.

스크래치 경로나 `sbatch -A`처럼 값이 실제로 필요한 지점에서는, 어떤 환경변수와
어떤 설정 파일을 찾아봤는지 이름을 밝히며 실패합니다. 조용히 기본값을 끼워
넣는 일은 없습니다.

### 1.3 SSH config 생성

```bash
rcac-connect --setup-config   # ~/.ssh/config 에 관리 블록 하나를 기록
rcac-connect                  # 한 번 인증하면 세션이 유지됨
```

이 단계는 프런트엔드가 하나인 클러스터보다 Gilbreth에서 훨씬 중요합니다.
`gilbreth.rcac.purdue.edu`라는 이름은 **프런트엔드 노드 4대**로 라운드로빈
분산되기 때문에, 멀티플렉싱이 없으면 `sbatch`와 `squeue`가 서로 다른 호스트에
떨어질 수 있습니다. 생성되는 관리 블록은 `ControlMaster auto`와 유지되는
`ControlPath`를 설정해서, **이후의 모든 `rcac-*` 명령을 마스터 세션이 인증한 그
한 대의 프런트엔드 노드에 고정**합니다. 덤으로 Purdue Login 인증은 명령마다가
아니라 `ControlPersist` 구간마다 한 번만 뜹니다.

```bash
rcac-status                    # 마스터 소켓이 살아 있나?
rcac-disconnect                # 닫기
rcac-disconnect --remove-config   # ~/.ssh/config 에서 관리 블록 제거
```

관리 블록은 시작·종료 마커로 감싸져 있어서 `--remove-config`는
`--setup-config`가 쓴 만큼만 정확히 지우고, `~/.ssh/config`의 나머지는 바이트
단위로 그대로 둡니다.

### 1.4 퍼듀 ID를 git에 올리지 않기

```bash
git config core.hooksPath .githooks
```

이 한 줄이 `pre-commit` 훅을 설치합니다. 스테이징된 모든 `configs/*.yaml`을
파싱해서 `ssh.user` 또는 `slurm.account`가 자리표시자가 아니면 **커밋을
거부**합니다. 이 저장소에는 CI가 없습니다. `.github/` 아래 워크플로우는 고아가
된 문서 작업입니다. 따라서 **이 훅이 유일한 자동 방어선**입니다. 첫 커밋 전에
설치하세요.

`git commit --no-verify`로 우회할 수 있습니다. 이유를 알 때만 쓰세요.

---

## 2. 사실로 보는 Gilbreth

이 절의 내용은 전부 RCAC 공식 문서에서 인용한 것이고, 모든 행에 출처 URL이
붙어 있습니다. RCAC가 **공개하지 않은** 사실은 여기에 없습니다. 그런 것들은
[§7](#7-실서버-검증-체크리스트)의 미결 항목으로 가 있습니다.

### 2.1 접속과 프런트엔드 노드 4대

<!-- facts:begin -->
| 항목 | 값 | [출처](https://docs.rcac.purdue.edu/userguides/gilbreth/accounts/) |
|---|---|---|
| 로그인 | `ssh <your-id>@gilbreth.rcac.purdue.edu` | [accounts](https://docs.rcac.purdue.edu/userguides/gilbreth/accounts/) |
| 인증 | 퍼듀 MFA 2단계 인증(알림 승인) **또는** SSH 키 | [accounts](https://docs.rcac.purdue.edu/userguides/gilbreth/accounts/) |
| 프런트엔드 노드 | **4대**, 각 64코어 · 512 GB 메모리 · A30(24 GB) 1장 | [overview](https://docs.rcac.purdue.edu/userguides/gilbreth/overview/) |
| 웹 포털 | Open OnDemand, `gateway.gilbreth.rcac.purdue.edu` | [gateway](https://docs.rcac.purdue.edu/userguides/gilbreth/gateway/) |
<!-- facts:end -->

호스트 이름 하나가 노드 4대를 대표하기 때문에, 멀티플렉싱하지 않은 세션은
사용자 모르게 기계를 갈아타는 세션입니다. §1.3의 ControlMaster 블록이 바로 그
문제를 막습니다.

### 2.2 파티션 — `-p`가 GPU 종류를 고른다

<!-- facts:begin -->
| `-p` | GPU | 노드 수 | 노드당 GPU | 최대 walltime | [출처](https://docs.rcac.purdue.edu/userguides/gilbreth/run_jobs/queues/) |
|---|---|---|---|---|---|
| `a10` | A10, 24 GB | 16 | 3 | 2주 | [queues](https://docs.rcac.purdue.edu/userguides/gilbreth/run_jobs/queues/) |
| `a30` | A30, 24 GB | 24 | 3 | 2주 | [queues](https://docs.rcac.purdue.edu/userguides/gilbreth/run_jobs/queues/) |
| `a100-40gb` | A100, 40 GB | 32 (G 노드 12 + N 노드 20) | G는 2, N은 4 | 2주 | [queues](https://docs.rcac.purdue.edu/userguides/gilbreth/run_jobs/queues/) |
| `a100-80gb` | A100, 80 GB | 57 | 2 | 2주 | [queues](https://docs.rcac.purdue.edu/userguides/gilbreth/run_jobs/queues/) |
| `training` | 혼합 (V100 / A100 / H100) | 8 | 4 | 24시간 | [queues](https://docs.rcac.purdue.edu/userguides/gilbreth/run_jobs/queues/) |
| `h100` | **모순 — 아래 설명 참조** | — | — | — | [faqs](https://docs.rcac.purdue.edu/userguides/gilbreth/faqs/) 대 [queues](https://docs.rcac.purdue.edu/userguides/gilbreth/run_jobs/queues/) |
<!-- facts:end -->

**`h100`에 대해 이 프로젝트는 어느 쪽 편도 들지 않습니다.** RCAC의 FAQ 페이지는
"H100 GPU에서 돌리고 싶으면 `--partition=h100`을 지정하라"고 적고 있고, 샘플
`slist` 출력에도 H100 열이 있습니다. 반면 queues 페이지에는 `h100` 파티션이
아예 없고 H100 하드웨어는 `training` 안에 들어 있습니다. 두 페이지 모두
현행입니다. 그래서 `rcac-cli`는 `h100`을 **unverified** 항목으로 들고 갑니다.
조용히 지우지도 않고, 실재한다고 단정하지도 않습니다. 여기로 제출하면 경고와
함께 이 모순을 알려 줍니다. 클러스터에서 `slist` 한 번이면 끝나는 문제이고,
그것이 [L1](#72-체크리스트)입니다.

`a100-40gb`의 N 노드가 NVLink가 달린 쪽입니다.

<!-- facts:begin -->
| 제약 조건 | 적용 대상 | 규칙 | [출처](https://docs.rcac.purdue.edu/userguides/gilbreth/run_jobs/queues/) |
|---|---|---|---|
| `--constraint='nvlink'` | `-p a100-40gb` 에서만 | **여러 GPU가 하나의 태스크를 위해 쓰일 때** 지정해 NVLink가 달린 Gilbreth-N 노드로 보냄 | [queues](https://docs.rcac.purdue.edu/userguides/gilbreth/run_jobs/queues/) |
<!-- facts:end -->

RCAC가 Gilbreth용으로 문서화한 유일한 노드 feature입니다. `rcac-cli`는 다른
`--constraint` 값을 대신 만들어 주지 않습니다.

### 2.3 QOS — `-q`는 QOS이지 계정이 아니다

<!-- facts:begin -->
| `-q` | 상한 | 할당량 차감? | 비고 | [출처](https://docs.rcac.purdue.edu/userguides/gilbreth/run_jobs/queues/) |
|---|---|---|---|---|
| `normal` | 파티션 상한 | 예 | 기본값. 높은 우선순위, 추가 제한 없음 | [queues](https://docs.rcac.purdue.edu/userguides/gilbreth/run_jobs/queues/) |
| `standby` | **4시간** | **아니오** | 유휴 자원에서 낮은 우선순위. `-q standby` / `--qos=standby` 로 명시 요청 | [queues](https://docs.rcac.purdue.edu/userguides/gilbreth/run_jobs/queues/) |
| `training` | 24시간 | 예 | `-p training` 에서 **필수** | [queues](https://docs.rcac.purdue.edu/userguides/gilbreth/run_jobs/queues/) |
| `-A standby` | — | — | **폐지됨**: `standby` QOS가 "이전의 `-A standby` 제출 방식을 대체한다" | [queues](https://docs.rcac.purdue.edu/userguides/gilbreth/run_jobs/queues/) |
<!-- facts:end -->

이 페이지에서 가장 쓸모 있는 항목은 `standby`입니다. **과금 없는 4시간**이면
컴파일, 학습 루프 스모크 테스트, 데이터 파이프라인 디버깅에 충분합니다. 진짜
실험이 아닌 모든 것은 여기서 돌리세요.

`rcac-cli`는 `-A standby`를 거부하면서 `-q standby`를 가리키는 메시지를 내고,
`-q training` 없는 `-p training`도 거부합니다. 조용히 고쳐서 사용자가 예산에
넣지도 않은 24시간 상한 아래로 밀어 넣지 않습니다.

### 2.4 제출 플래그

<!-- facts:begin -->
| 플래그 | 필수 | 의미 | [출처](https://docs.rcac.purdue.edu/userguides/gilbreth/run_jobs/submit_script/) |
|---|---|---|---|
| `-A <account>` | 예 | "SLURM은 제출되는 모든 작업이 **Account**와 연결되기를 요구한다" | [submit_script](https://docs.rcac.purdue.edu/userguides/gilbreth/run_jobs/submit_script/) |
| `-p <partition>` | 예 | "Gilbreth에서는 각 작업이 특정 **Partition**과 연결될 것도 요구하며, 이는 어떤 GPU를 쓸지를 지정한다" | [submit_script](https://docs.rcac.purdue.edu/userguides/gilbreth/run_jobs/submit_script/) |
| `-q <qos>` | 아니오 (기본 `normal`) | `normal`, `standby`, `training` 중 하나 | [queues](https://docs.rcac.purdue.edu/userguides/gilbreth/run_jobs/queues/) |
| `--gpus-per-node=N` | GPU 작업이면 예 | "GPU 작업은 `--gpus-per-node` 옵션으로 GPU 개수를 반드시 지정해야 한다" | [submit_script](https://docs.rcac.purdue.edu/userguides/gilbreth/run_jobs/submit_script/) |
| `--mem=<size>` | 예 | GPU 개수와 함께 요청 | [submit_script](https://docs.rcac.purdue.edu/userguides/gilbreth/run_jobs/submit_script/) |
| `--time=<walltime>` | 사실상 예 | "기본적으로 작업에는 30분의 wall time이 주어진다" | [submit_script](https://docs.rcac.purdue.edu/userguides/gilbreth/run_jobs/submit_script/) |
<!-- facts:end -->

**Gilbreth에는 문서화된 `--gres` GPU 종류 문자열이 없습니다.** GPU 종류는 이미
파티션이 골랐고, 개수는 `--gpus-per-node`가 고릅니다. `rcac-cli`는 `--gres`로
GPU 종류를 요청하는 코드를 생성하지 않으며, 템플릿에 그런 문자열이 들어오면
테스트가 빌드를 실패시킵니다.

### 2.5 파일시스템과 60일 purge

<!-- facts:begin -->
| 경로 또는 명령 | 무엇인가 | [출처](https://docs.rcac.purdue.edu/userguides/gilbreth/storage/) |
|---|---|---|
| `/home/<your-id>` | 홈 디렉터리. 용량 쿼터가 있고, 야간·주간·월간 스냅샷이 약 2~3개월간 보관됨 | [home_directory](https://docs.rcac.purdue.edu/userguides/gilbreth/storage/home_directory/) |
| `/scratch/gilbreth/<your-id>` | 대용량 병렬 스크래치. **"스크래치 디렉터리의 파일은 백업되지 않는다"** | [scratch_space](https://docs.rcac.purdue.edu/userguides/gilbreth/storage/scratch_space/) |
| purge 정책 | **"60일 동안 접근되지 않았거나 내용이 수정되지 않은 파일은 삭제된다"** | [scratch_space](https://docs.rcac.purdue.edu/userguides/gilbreth/storage/scratch_space/) |
| `$RCAC_SCRATCH` | "스크래치 경로는 예고 없이 바뀔 수 있지만 이 변수는 항상 최신이다" — 스크립트에서는 이 변수를 쓸 것 | [scratch_space](https://docs.rcac.purdue.edu/userguides/gilbreth/storage/scratch_space/) |
| `findscratch` | 스크래치 경로를 출력. 예: `/scratch/gilbreth/myusername` | [scratch_space](https://docs.rcac.purdue.edu/userguides/gilbreth/storage/scratch_space/) |
| `myquota` | "홈과 스크래치의 현재 쿼터를 확인하려면 `myquota` 명령을 쓴다" | [storage_quota](https://docs.rcac.purdue.edu/userguides/gilbreth/storage/storage_quota/) |
<!-- facts:end -->

**쿼터 숫자는 이 README가 아니라 `myquota`에서 읽으세요.** RCAC의 예시 출력에는
홈 25 GB, 스크래치 100 TB가 찍혀 있지만 그것은 공개된 예시의 숫자이지 여러분
계정에 대한 약속이 아닙니다. `rcac-keepalive check`가 `myquota`를 대신 실행해
실제 응답을 보여 줍니다.

60일은 접근 시각과 내용 수정 시각 중 **나중 것**을 기준으로 셉니다. 그래서 읽기만
한 데이터셋도 여전히 "최근"으로 취급됩니다. `rcac-keepalive touch`는 지정한
파일의 접근 시각을 갱신하고, `rcac-keepalive setup-cron`은 그것을 주기 작업으로
만듭니다.

### 2.6 데이터 옮기기

<!-- facts:begin -->
| 방법 | 쓰임새 | [출처](https://docs.rcac.purdue.edu/userguides/gilbreth/storage/globus/) |
|---|---|---|
| Globus, `https://transfer.rcac.purdue.edu` | 대용량 전송. "인터넷에 연결된 어떤 운영체제에서도 동작한다" | [globus](https://docs.rcac.purdue.edu/userguides/gilbreth/storage/globus/) |
| 컬렉션 **"Gilbreth Cluster Collection"** | Gilbreth 홈과 스크래치에 접근. "gilbreth"로 검색하면 제안됨 | [globus](https://docs.rcac.purdue.edu/userguides/gilbreth/storage/globus/) |
| SCP / SFTP | 소·중 규모 전송. 로그인 호스트 경유 | [storage](https://docs.rcac.purdue.edu/userguides/gilbreth/storage/) |
<!-- facts:end -->

**Gilbreth에는 문서화된 전용 데이터 전송 노드가 없습니다.** 따로 설정할 전송
호스트가 없고, `rcac-cli`에도 그런 설정 항목이 없습니다. `rcac-sync push`와
`pull`은 이미 열려 있는 ControlMaster 소켓을 통해 로그인 호스트로 나가고,
`rcac-sync push-data`도 마찬가지이되 일정 크기를 넘으면 위의 Globus 경로를 안내
합니다. 노트북을 닫아도 살아남는 웹 전송이 그렇지 못한 `rsync`보다 나은
지점이기 때문입니다.

### 2.7 소프트웨어: 모듈, 환경, 컨테이너

<!-- facts:begin -->
| 주제 | RCAC 문서가 말하는 것 | [출처](https://docs.rcac.purdue.edu/userguides/gilbreth/software/) |
|---|---|---|
| 모듈 시스템 | Lmod. `module spider` / `module avail` 로 찾고 `module load` 로 적재 | [software](https://docs.rcac.purdue.edu/userguides/gilbreth/software/) |
| 파이썬 환경 | `module purge` → `module load anaconda` → `conda-env-mod create -n <name>` → `module load use.own` → `module load conda-env/<name>-py<X.Y>` | [customml](https://docs.rcac.purdue.edu/userguides/gilbreth/run_jobs/learning/customml/) |
| GPU 라이브러리 | 프레임워크가 자체 포함하지 않는 경우에만 `module load cuda`, `module load cudnn` | [customml](https://docs.rcac.purdue.edu/userguides/gilbreth/run_jobs/learning/customml/) |
| 컨테이너 | Apptainer. RCAC가 "홈 디렉터리, 스크래치, Data Depot, 애플리케이션 파일시스템"을 오버레이 — `/home/$USER`, `/apps`, `/scratch`, `/depot` | [apptainer_example](https://docs.rcac.purdue.edu/userguides/gilbreth/run_jobs/apptainer_example/) |
| 계정과 GPU 접근 권한 | `slist`는 "여러분이 속한 각 계정별로 각 GPU 종류에 대한 접근 권한 표를 출력한다" | [faqs](https://docs.rcac.purdue.edu/userguides/gilbreth/faqs/) |
<!-- facts:end -->

여기서 곧바로 따라 나오는 결론 세 가지:

- **맨 `conda`가 아니라 `conda-env-mod`.** RCAC는 `conda-env-mod`가 만든 모듈을
  적재하는 것이 `conda create` + `source activate`와 "동작이 다르다"고 명시하고
  모듈 방식을 권합니다. 따라서 환경의 정체성은 *모듈 이름*,
  즉 `conda-env/<name>-pyX.Y` 입니다.
- **Apptainer이고, `module load`가 필요 없습니다.** Docker도 아니고, 그것이
  대체한 이전 런타임도 아닙니다. RCAC가 `/home/$USER`, `/apps`, `/scratch`,
  `/depot`을 알아서 오버레이해 주므로 `-B`로 다시 바인드하는 것은 잘해야 무의미
  합니다.
- **모듈 이름이나 버전을 지어내지 마세요.** RCAC 자신의 지침이며, 이 README가
  모듈 *계열* 이름(`anaconda`, `cuda`, `cudnn`, `use.own`)만 적고 버전 문자열은
  하나도 적지 않은 이유입니다. `rcac-env modules`가 클러스터에 직접 물어봅니다.
  존재하지 않는 모듈을 `module load` 하면 작업은 첫 줄에서 죽습니다.

NGC 컨테이너는 RCAC 시스템에서 `module load ngc`로 쓸 수 있습니다. 다만 이미지
디렉터리 `/apps/ngc/images/`는 **Gautschi** 문서에만 있고 Gilbreth 문서에는
없습니다. 그래서 이 프로젝트는 그 경로가 여기 있다고 절대 가정하지 않습니다.
[L18](#72-체크리스트)을 보세요.

---

## 3. 제출의 네 조각

문서 전체에서 하나만 가져간다면 이것입니다. Gilbreth 제출은 **한 조각이 아니라
네 조각**으로 이루어집니다.

```
sbatch -A <account>  -p <partition>  -q <qos>  --time=<walltime>  ...
        └ 누가 낸다    └ 어떤 GPU      └ 어떤   └ 없으면
                                        규칙     30분
```

다른 클러스터는 이것들을 전부 `-p` 하나에 접어 넣습니다. Gilbreth는 그렇지
않습니다. `-p`는 **GPU 종류**, `-A`는 **과금되는 할당**, `-q`는 **QOS**입니다.
`-A`나 `-p`를 빠뜨리면 작업이 거부되고, `--time`을 빠뜨리면 30분 뒤에 죽습니다.

`rcac-cli`는 이 네 가지를 한 곳에서 만듭니다. 그래서 `rcac-exp launch`와
keepalive 스윕과 시뮬레이터가 생성한 매니페스트가 서로 다른 말을 할 수 없습니다.

```bash
rcac-exp launch experiment.yaml --dry-run \
  -A <account> -p a100-40gb -q normal --time 04:00:00 --mem 64G
```

`--dry-run`은 스크립트를 렌더링하고 실제 `sbatch` 명령줄을 출력하되 **클러스터에
전혀 접속하지 않습니다**. 제출 전에 읽어 보세요. 우선순위는 CLI 플래그 →
매니페스트의 `slurm:` 블록 → `configs/defaults.yaml` 순입니다.

---

## 4. 일상 워크플로우 (Quick Start)

### Step 1: 접속

```bash
rcac-connect          # ControlPersist 구간마다 한 번, 2단계 인증도 한 번
rcac-status
```

### Step 2: 코드 동기화

```bash
cd ~/work/my_project
rcac-sync push                 # 로컬 -> /scratch/gilbreth/<you>/projects/my_project
rcac-sync push --dry-run       # 무엇이 옮겨질지 먼저 확인
rcac-sync watch                # 저장할 때마다 자동 push
```

### Step 3: 실행

```bash
rcac-exp launch experiment.yaml --dry-run -p a100-40gb --time 04:00:00
rcac-exp launch experiment.yaml -p a100-40gb --time 04:00:00
```

`slurm:` 블록을 고칠 때마다 `--dry-run`부터 하세요. 공짜이고, 잘못된 `-p`/`-q`
조합을 싸게 잡을 수 있는 유일한 지점입니다.

### Step 4: 관찰

```bash
rcac-exp status
rcac-exp logs <job-id>
rcac-exp watch my_project --auto-pull ./results
```

### Step 5: 결과 내려받기

```bash
rcac-sync pull
rcac-results best my_project -m val_accuracy
rcac-results compare my_project exp_001 exp_007
```

---

## 5. 명령어 레퍼런스

`pip install -e .` 이 설치하는 콘솔 스크립트 12개 전부:

| 스크립트 | 용도 |
|---|---|
| `rcac-connect` | ControlMaster 세션 열기, SSH config 작성 |
| `rcac-status` | 세션이 살아 있는지 확인 |
| `rcac-disconnect` | 세션 종료, 관리 SSH 블록 제거 |
| `rcac-sync` | 코드·데이터 push/pull |
| `rcac-env` | `conda-env-mod` 환경 관리 |
| `rcac-exp` | 작업 생성·제출·관찰·취소 |
| `rcac-results` | 실험 결과 조회와 비교 |
| `rcac-partition` | 어떤 계정으로 어떤 GPU를 쓸 수 있나 |
| `rcac-container` | Apptainer 이미지 |
| `rcac-keepalive` | 60일 스크래치 purge 대비 |
| `rcac-notify` | Slack / 이메일 작업 알림 |
| `rcac-sim` | 시뮬레이터 호환성과 파이프라인 스캐폴딩 |

### 5.1 SSH 세션 — `rcac-connect`, `rcac-status`, `rcac-disconnect`

```bash
rcac-connect --setup-config       # 관리 Host 블록 기록
rcac-connect                      # 인증하고 마스터 소켓 유지
rcac-status                       # 소켓 상태 보고
rcac-disconnect                   # 종료
rcac-disconnect --remove-config   # 관리 블록 제거
```

기록되는 `Host gilbreth` 스탠자는 하나뿐이며 `ControlMaster`, `ControlPath`,
`ControlPersist`와 프런트엔드 4대를 설명하는 주석을 담고 있습니다.
`~/.ssh/config`의 나머지는 건드리지 않습니다.

### 5.2 코드 동기화 — `rcac-sync`

```bash
rcac-sync push [PATH] [-p PROJECT] [--dry-run] [--delete]
rcac-sync pull [PATH] [-p PROJECT] [--all-files] [--dry-run]
rcac-sync watch [PATH] [-p PROJECT] [-i SECONDS]
rcac-sync push-data <LOCAL_DATA_PATH> [--dry-run]
```

제외 패턴과 pull 필터는 `configs/defaults.yaml`의 `sync:` 아래에 있습니다. 모든
전송이 ControlMaster 소켓을 재사용하므로 동기화가 두 번째 2단계 인증을 부르지
않습니다. `push-data`는 브라우저 기반 전송이 더 나은 크기부터 Globus 경로를
안내합니다.

### 5.3 환경 — `rcac-env`

```bash
rcac-env modules [PATTERN]        # 무엇이 있는지 클러스터에 질의
rcac-env setup --name train_env --py 3.12 [--from environment.yml] [--gpu]
rcac-env list
rcac-env activate train_env [--py 3.12]
rcac-env export train_env -o environment.yml
```

`rcac-env setup --name train_env --py 3.12` 는 문서화된 순서를 그대로 실행하고
**`conda-env/train_env-py3.12`** 라는 모듈을 남깁니다. `-py3.12` 접미사도 이름의
일부이며, 이것이 문제가 되는 지점이 정확히 한 곳 있습니다.

```yaml
environment:
  type: conda
  name: train_env-py3.12      # 접미사를 포함한 전체 모듈 이름
```

매니페스트의 `environment.name`은 생성된 작업 스크립트가
`module load conda-env/<name>` 에 넘기는 값입니다. 접미사를 빠뜨리면 작업은 첫
줄에서 죽습니다. `rcac-env activate`는 작업 스크립트가 쓰는 두 줄 —
`module load use.own` 과 `module load conda-env/<name>-pyX.Y` — 만 출력하고 그
외에는 아무것도 내보내지 않으므로 `$(rcac-env activate train_env)` 형태로 그대로
끼워 넣어도 안전합니다.

`--py`를 생략하면 클러스터의 `anaconda` 모듈에서 알아내며, 추측하지 않습니다.

### 5.4 실험 — `rcac-exp`

```bash
rcac-exp launch <experiment.yaml> [--dry-run] [-n MAX_JOBS]
    [-A ACCOUNT] [-p PARTITION] [-q QOS] [--time T] [--mem M]
    [--gpus N] [--constraint nvlink] [--no-discover] [--no-cache]
rcac-exp status
rcac-exp logs <JOB_ID>
rcac-exp watch <PROJECT> [--auto-pull DIR] [-i SECONDS] [-j JOB_IDS]
rcac-exp cancel <JOB_ID|all>
```

매니페스트는 프로젝트, 환경, 스윕, `slurm:` 블록을 선언합니다.
`examples/experiment_simple.yaml` 과 `examples/experiment_genesis.yaml` 이 바로
쓸 수 있는 출발점이고, 배포 상태 그대로 `--dry-run`을 통과합니다.

예제가 일부러 빼놓은 것에 주목하세요. **`-A`는 매니페스트에 절대 쓰지
않습니다.** 계정은 자격증명이고, 매니페스트는 커밋될 수 있는 파일입니다.

### 5.5 결과 — `rcac-results`

```bash
rcac-results results <PROJECT> [-s METRIC] [-l LIMIT]
rcac-results best <PROJECT> [-m METRIC] [-t N]
rcac-results compare <PROJECT> <EXP_ID>...
rcac-results sync-wandb <PROJECT> [--exp-id ID]
```

`sync-wandb`는 오프라인 W&B 실행 기록을 내려받아 로컬에서 동기화합니다. 계산
노드가 인터넷에 나가지 못하는 클러스터에서 흔한 형태입니다.

### 5.6 파티션 추천 — `rcac-partition`

```bash
rcac-partition [-g GPUS] [--prefer-cost] [--no-discover] [--no-cache]
```

과금 가능한 계정과 각 계정이 닿는 GPU 종류(`slist` 행렬)를 클러스터에 물어본 뒤,
설명한 작업에 맞게 파티션 순위를 매깁니다. **클러스터가 응답했든 아니든 종료
코드는 0입니다.** 중단되는 조언은 단서가 붙은 조언보다 나쁘기 때문입니다.
내장 표로 후퇴할 때는 배너로 그 사실을 밝히고, `h100` 같은 미확인 항목은 미확인
표시를 유지합니다.

`RCAC_ACCOUNT`를 아직 모를 때 실행할 명령이 바로 이것입니다.

### 5.7 컨테이너 — `rcac-container`

```bash
rcac-container build <DEF_FILE> [-n NAME] [--remote-build]
rcac-container push <IMAGE.sif> [-d REMOTE_DIR]
rcac-container list
rcac-container run <REMOTE_IMAGE.sif> [COMMAND...] [-B PATH] [--nv]
```

전부 Apptainer입니다. `run`의 `--nv`는 기본 켜짐입니다. `-B`는 RCAC가 이미
오버레이해 주지 **않는** 경로용입니다. `/home`, `/apps`, `/scratch`, `/depot`은
이미 마운트되어 있습니다. 업로드된 이미지는 기본적으로
`<paths.scratch>/.rcac-tools/` 로 갑니다.

`--remote-build`는 정의 파일을 올려 프런트엔드 노드에서 빌드하며 할당을 요청하지
않습니다. RCAC가 Apptainer를 rootless로 문서화하고 있기 때문입니다. 그 빌드가
실패하면 대화형 할당 대안을 가정하지 않고 힌트로 출력합니다.
[L17](#72-체크리스트)을 보세요.

정의 파일 네 개가 `containers/` 아래에 있습니다.

### 5.8 스크래치 보존 — `rcac-keepalive`

```bash
rcac-keepalive check [-d DAYS] [--no-discover]
rcac-keepalive touch [--dry-run] [-v]
rcac-keepalive setup-cron [-i DAYS] [--batch]
```

`check`는 신뢰도가 높은 순서로 세 가지 답을 줍니다. 클러스터가 이미 삭제 대상
으로 잡아 둔 것, `myquota`가 말하는 공간 자체의 상태, 그리고 로컬
`find -atime` 스캔이 경계에 가깝다고 본 파일들입니다. 기본 경고 임계치는
45일이고, 60일 창의 4분의 1을 여유로 남깁니다.

`setup-cron`은 기본적으로 스윕을 **프런트엔드 노드에서** 돌립니다. 스윕은
메타데이터만 건드리는 `find … -exec touch -a` 이라 어떤 합리적인 프런트엔드
예산에도 한참 못 미치는 반면, Gilbreth에서 배치 작업으로 돌리면 쓰지도 않을
GPU를 잡아야 합니다. 모든 파티션이 GPU 파티션이고 `--gpus-per-node`의 하한이
1이기 때문입니다. `--batch`는 그래도 배치로 제출합니다. 스윕이 프런트엔드에
두기엔 클 때, 혹은 기관 정책이 그것을 금할 때 쓰세요.

### 5.9 알림 — `rcac-notify`

```bash
rcac-notify setup                 # notify.yaml 템플릿 생성
rcac-notify test [-m slack|email] [--message TEXT]
rcac-notify send --status 0 --job-name NAME --exp-id ID [--metric M] [-m METHOD]
```

`notify.yaml`은 다른 모든 설정과 똑같은 3단계로 해석됩니다.
`$RCAC_CONFIG_DIR` → `./configs/` → `rcac_cli` 패키지 안에 동봉된 사본.
`send`는 생성된 작업 스크립트가 종료 시점에 호출하는 명령입니다.

### 5.10 시뮬레이터 — `rcac-sim`

```bash
rcac-sim list
rcac-sim compat <genesis|mujoco|isaac_gym|isaac_sim|pybullet> [-p PARTITION]
rcac-sim check-env <genesis|mujoco|isaac_gym|pybullet>
rcac-sim init-pipeline <genesis|mujoco|isaac_gym> [-p PARTITION] [--env NAME] [--episodes N] [-o FILE]
rcac-sim scaffold <genesis|mujoco|isaac_gym> [--env NAME] [-o DIR]
```

`init-pipeline`은 `slurm:` 블록을 나중에 launch 시점이 아니라 **쓰는 그 순간**에
검증합니다. 그러지 않으면 잘못된 파티션/QOS 조합이 담긴 생성 파일이 누군가
제출할 때까지, 어쩌면 며칠 뒤 다른 기계에서, "생성된 파일이니 유효하겠지"라는
가정과 함께 멀쩡히 살아남습니다.

---

## 6. 시나리오

### 시나리오 A: A100에서 아키텍처 스윕

```bash
rcac-connect
cd ~/work/my_model && rcac-sync push

rcac-env setup --name train_env --py 3.12 --from environment.yml --gpu
# -> 모듈 conda-env/train_env-py3.12 ; 이 전체 이름을 매니페스트에 넣을 것

rcac-partition -g 2                          # 어떤 계정이 a100-40gb에 닿나?
rcac-exp launch experiment.yaml --dry-run \
    -p a100-40gb -q normal --time 08:00:00 --mem 64G --gpus 2
rcac-exp launch experiment.yaml -p a100-40gb -q normal --time 08:00:00 --mem 64G --gpus 2

rcac-exp watch my_model --auto-pull ./results
rcac-results best my_model -m val_accuracy
```

여러 GPU가 **하나의** 태스크를 먹이는 구조인가요? `--constraint nvlink`를 붙여
NVLink가 달린 Gilbreth-N 노드로 보내세요. 독립적인 복제 실행에는 필요 없습니다.

### 시나리오 B: 컨테이너 안의 Genesis 시뮬레이션 파이프라인

```bash
rcac-container build containers/genesis-sim.def --remote-build
rcac-container push genesis-sim.sif            # -> <scratch>/.rcac-tools/
rcac-container run <scratch>/.rcac-tools/genesis-sim.sif nvidia-smi

rcac-sim compat genesis -p a100-80gb
rcac-sim init-pipeline genesis -p a100-80gb --episodes 10000 -o experiment.yaml
rcac-exp launch experiment.yaml --dry-run --time 24:00:00
```

`examples/experiment_genesis.yaml` 이 같은 파이프라인을 손으로 적어 둔 것입니다.

### 시나리오 C: `standby`로 공짜 4시간

```bash
rcac-exp launch experiment.yaml -n 1 \
    -p a30 -q standby --time 04:00:00 --mem 32G
```

과금 없음, 낮은 우선순위, 4시간 하드 상한. 데이터 로더 경로가 틀렸다는 사실을
알아내기에 딱 맞는 방식입니다. 그리고 `-A standby`는 이것을 요청하는 방법이
**아닙니다**. 그 형태는 폐지되었습니다.

---

## 7. 실서버 검증 체크리스트

`rcac-cli`는 RCAC 공개 문서만 보고 전부 오프라인으로 만들어졌습니다. 아래 항목은
그 문서로는 결론이 나지 않은 것들입니다. 일상적인 사용을 막는 것은 하나도
없습니다. 각 줄은 로그인 노드에서 한 줄짜리 명령으로 "아마도"를 "사실"로 바꿉니다.
결과를 기록한 다음 그 줄을 지우세요.

### 7.1 시작하기 전에

Gilbreth 계정, 과금 가능한 할당, 그리고 동작하는 `rcac-connect`가 필요합니다.
ControlMaster 소켓을 공유하도록 한 세션 안에서 전부 실행하세요. 테스트 픽스처를
대체하는 항목은 대상 경로를 함께 적어 두었습니다.

### 7.2 체크리스트

| # | 미결 질문 | 명령 | 무엇이 정해지나 |
|---|---|---|---|
| **L0** | 접속 | `ssh <id>@gilbreth.rcac.purdue.edu hostname` | 프런트엔드 4대 중 어느 것이 응답했나 |
| **L1** | `h100` **파티션**이 존재하나? | `slist` ; `sinfo -s` ; `sinfo -o '%P %G'` | §2.2의 FAQ 대 queues 모순 |
| **L2** | 여기서 ControlMaster가 실제로 멀티플렉싱하나? | `rcac-connect --setup-config && rcac-connect` 후 `ssh gilbreth hostname` 두 번 | 비대화형 설계 전체. **머지 차단 항목** |
| **L3** | `apptainer --nv` 가 동작하나? | `apptainer exec --nv <img>.sif nvidia-smi` | `--nv` 기본값 유지 여부 |
| **L4** | `sacct` / `showpartitions` 가 있나? | `which sacct showpartitions ; sacct -j <id>` | `rcac-exp status`가 `sacct`를 쓸 수 있는지 |
| **L5** | `sfeatures` 출력 형식 | `sfeatures \| tee tests/fixtures/sfeatures.txt` | 문서 기반 픽스처를 대체. **`sfeatures` 존재 자체가 미확인** |
| **L5b** | `slist`는 계정 범위로만 출력되나? | `slist \| tee tests/fixtures/slist.txt` | RCAC FAQ는 **내 계정이 닿는** GPU 종류만 찍힌다는 뜻으로 읽힙니다. 그렇다면 발견된 파티션 목록은 `sinfo` 만으로 만들어야 합니다. 아니면 과금 권한이 없을 뿐 실재하는 파티션이 "이 클러스터에 없음"으로 보고됩니다 |
| **L6** | 나에게 `/depot/<group>` 이 있나? | `ls -d /depot/*/ ; echo $RCAC_SCRATCH ; findscratch` | 선택 항목 `paths.depot` 을 채움. Gilbreth 문서에는 없음 |
| **L7** | 계정별 작업 수 제한 | `sacctmgr show assoc user=$USER format=Account,MaxJobs,MaxSubmit` | 스윕 크기의 상한 |
| **L8** | 실제 모듈 이름과 버전 | `module spider anaconda ; module avail` ; `module spider ml-toolkit` | **여기서 출력되지 않은 이름은 적지 말 것.** 어떤 RCAC Gilbreth 문서도 `ml-toolkit*` 모듈이나 특정 `anaconda` 버전을 명시하지 않습니다 |
| **L9** | 보존 정책과 쿼터 | `myquota` ; `purgelist \| head` | 실제 쿼터. **확인한 어떤 Gilbreth 문서도 `purgelist` 를 언급하지 않습니다.** `rcac-keepalive check`의 첫 번째 답에 의존하기 전에 존재를 확인하세요 |
| **L10** | `conda-env-mod` 왕복 | `rcac-env setup --from environment.yml --py 3.12 && rcac-env activate <name>` | 생성된 모듈 이름이 작업 스크립트가 적재하는 이름과 일치하는지 |
| **L11** | `normal` 로 실제 제출 | `rcac-exp launch … -p <verified> --time 00:10:00` | `-A`/`-p`/`-q` 삼중항의 종단 간 증명. **머지 차단 항목** |
| **L12** | `standby` 로 실제 제출 | `… -q standby --time 00:10:00` | 과금 없는 경로. **머지 차단 항목** |
| **L13** | 폐지된 형태가 실제로 거부되나 | 손으로 `sbatch -A standby …` | 오류 메시지가 올바른 방향을 가리키는지 |
| **L14** | NVLink | `-p a100-40gb --gpus-per-node 2 --constraint nvlink` | 제약 조건이 작업을 N 노드로 보내는지 |
| **L15** | 컨테이너 왕복 | `rcac-container push … && rcac-container run … nvidia-smi` | 두 번째 2단계 인증이 뜨지 않는지 |
| **L16** | A10에 RT 코어가 노출되나? | `-p a10` 안에서 `nvidia-smi -q \| grep -i rt` | 현재 미확인인 Isaac Sim 호환성 항목 |
| **L17** | rootless `apptainer build` 에 할당이 필요한가? | 프런트엔드에서 `apptainer build /tmp/t.sif docker://alpine` | `--remote-build` 대안 힌트의 존치 여부 |
| **L18** | Gilbreth에 `/apps/ngc/images/` 가 있나? | `module load ngc ; ls /apps/ngc/images/ 2>&1` | 그 경로는 Gautschi 문서에만 있음 |

### 7.3 무엇이 머지를 막나

**L2, L11, L12.** 멀티플렉싱, 과금되는 실제 제출, 과금 없는 실제 제출. 나머지는
전부 *기록* 항목입니다. 표를 바꾸거나 단서를 지울 뿐, 설계를 바꾸지 않습니다.

---

## 8. 트러블슈팅

### 명령마다 2단계 인증이 뜬다

ControlMaster 소켓이 떠 있지 않아 각 명령이 따로 인증하고 있습니다. 그리고 각
명령이 서로 다른 프런트엔드 노드에 떨어질 수 있습니다.

```bash
rcac-status
rcac-connect                 # 다시 연결
ls -l ~/.ssh/sockets/        # 여기에 소켓이 있어야 함
```

소켓 경로가 없으면 디렉터리를 만들거나 `rcac-connect --setup-config`를 다시
실행하세요.

### `sbatch` 가 제출을 거부한다

무엇보다 먼저 네 조각을 확인하세요.

```bash
rcac-exp launch experiment.yaml --dry-run    # 생성된 헤더를 읽어 볼 것
slist                                        # 클러스터에서: 무엇에 과금할 수 있나
```

- `-A` 누락 또는 오타 → `slist`가 계정 이름을 알려 줍니다.
- `-A standby` → 폐지됨. `-q standby` 와 실제 계정을 쓰세요.
- `-q training` 없는 `-p training` → 일부러 거부합니다. 고쳐 주지 않습니다.
- `-p h100` → RCAC 문서끼리 모순됩니다. [L1](#72-체크리스트)을 실행하세요.
- 계정이 닿지 않는 파티션 → `rcac-partition -g <n>`.

### 작업이 시작되지 않는다

`standby`는 설계상 우선순위가 낮고, 붐비는 파티션에서 4시간 상한 작업은 한참
기다릴 수 있습니다. GPU를 덜 요청하거나, 계정이 더 많이 가진 파티션을 쓰세요.

```bash
squeue -u $USER --start
rcac-partition -g 1 --prefer-cost
```

### 결과 파일이 안 돌아온다

```bash
rcac-exp logs <job-id>
rcac-sync pull --dry-run
```

`pull`은 `configs/defaults.yaml`의 결과 패턴으로 걸러 냅니다. 그것을 무시하려면
`--all-files`를 쓰세요. 스크래치에서 파일 자체가 사라졌다면 60일 purge를
의심하고 `rcac-keepalive check`를 돌리세요.

### `rcac-partition` 이 오프라인 표를 출력한다

클러스터에 닿지 못해 내장 힌트로 후퇴했고, 그 사실을 밝힌 것입니다. 세션을
살리고(`rcac-connect`) 다시 실행하세요. 후퇴한 표는 조언이지 권한에 대한 주장이
아닙니다.

### 아무 일도 하기 전에 40초를 멈춰 있는다

퍼듀에 닿지 못하는 기계에서 런타임 탐색이 도는 중입니다. 프로브 하나당 타임아웃이
20초이고 프로브는 두 개일 수 있습니다. 꺼 두세요.

```bash
rcac-exp launch experiment.yaml --dry-run --no-discover
export RCAC_NO_DISCOVER=1        # 또는 셸 전체에 한 번만 설정
```

탐색을 끄면 인식되지 않는 파티션은 실패 대신 경고가 되고, 오프라인 힌트 표가
배너와 함께 쓰입니다. 반대 문제, 즉 한 프로세스 안에서 다시 조회하게 만들려면
`--no-cache`를 쓰세요.

---

## 9. 프로젝트 연혁

<!-- history:begin -->

이 도구는 KISTI Neuron 클러스터용 런처 **neuron-cli** 로 시작했고, `rcac`
브랜치의 일련의 커밋을 거쳐 퍼듀 RCAC Gilbreth로 포팅되었습니다. 이 포팅은 이름
바꾸기가 아니었습니다. 두 클러스터는 작업이 아예 도느냐 마느냐를 결정하는
지점에서 다릅니다.

- KISTI Neuron은 큐를 `-p` 하나로 받았지만, Gilbreth는 `-A` **와** `-p` **와**
  `-q`가 모두 필요하고 그 중 `-p`는 GPU 종류입니다.
- Neuron의 스크래치는 15일에 purge 되었고, Gilbreth는 접근 시각과 수정 시각 중
  나중 것을 기준으로 60일입니다.
- Neuron은 OTP로 인증했고, Gilbreth는 퍼듀 MFA 또는 SSH 키를 씁니다.
- Neuron의 홈은 home01 이었고 대용량 데이터는 전용 DM node를 거쳤습니다.
  Gilbreth에는 둘 다 없습니다. 문서화된 전용 전송 호스트가 없고, 대용량 데이터는
  Globus로 갑니다.
- Neuron은 Singularity를 썼고, Gilbreth는 시스템에 설치된 Apptainer를 씁니다.

위 내용 중 어느 것도 이 도구가 지금 대상으로 하는 클러스터를 설명하지 않습니다.
Neuron을 쓰던 사람이 무엇이 왜 바뀌었는지 알아볼 수 있도록 남겨 둔 것뿐입니다.
`rcac-connect --remove-config` 가 `~/.ssh/config` 에서 예전 neuron 블록까지 아직
제거하는 이유도 똑같습니다.

<!-- history:end -->

---

기여를 환영합니다. 커밋 전에 `pytest -q` 를 돌리고 자격증명 훅
(`git config core.hooksPath .githooks`)을 설치해 주세요.
