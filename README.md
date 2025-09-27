# daboa_AI 코드 설명

이 저장소는 LSTM 기반의 비정상 행동 감지를 위해 작성된 실험용 노트북과 학습된 가중치를 포함합니다. 아래에서는 주요 노트북에서 정의된 모듈과 함수들을 중심으로 전체 흐름을 설명합니다.

## 프로젝트 개요

- **lstm_ae.ipynb**: LSTM 오토인코더 모델 정의, 데이터 전처리용 `AbnormalDataset`, 학습 루프 `train` 함수와 실행 진입점(`main`)이 포함되어 있습니다.
- **model_predict.ipynb**: YOLOv8 포즈 추정으로 특징을 추출하고, 학습된 오토인코더를 활용해 이상 행동을 감지하는 Flask 기반 추론 서버가 구현되어 있습니다.
- **LSTMAutoEncoder/**: 학습된 체크포인트(`*.pth`)가 저장되어 있는 디렉터리입니다.

## 모델 구조 (`lstm_ae.ipynb`)

| 구성 요소 | 함수/모듈 | 설명 |
| --- | --- | --- |
| 인코더 | `Encoder` 클래스 | `nn.LSTM`을 사용해 입력 시퀀스의 은닉 상태 `(hn, cn)`를 생성합니다. `forward` 내부에서 `initHidden`을 호출하여 배치 크기에 맞춰 은닉 상태를 초기화합니다.
| 디코더 | `Decoder` 클래스 | 인코더의 은닉 상태를 받아 단일 타임스텝씩 복원 출력을 생성합니다. LSTM 출력은 `nn.Linear`를 거쳐 원래 특징 차원으로 투영됩니다.
| 오토인코더 | `LSTMAutoEncoder` 클래스 | 인코더-디코더를 결합해 입력 시퀀스를 역순으로 재구성합니다. `forward`는 디코더를 반복 호출해 전체 시퀀스 길이만큼 출력을 채웁니다.

## 데이터셋 구성

- **`AbnormalDataset` 클래스**: CSV로 저장된 키포인트 시퀀스를 불러와 최소 길이를 보장하며, JSON 주석을 읽어 프레임별 정상/비정상 라벨을 생성합니다. `__getitem__`은 `(sequence, target_labels)` 튜플을 반환하고, `find_real_idx`는 겹치는 윈도우 인덱스를 실제 행 인덱스로 변환합니다.

## 학습 파이프라인

- **하이퍼파라미터 파싱**: `parse_args`는 데이터 경로, 디바이스, 배치 크기, 학습률, 에폭 수, 검증 주기, 얼리 스톱 조건, W&B 설정 등을 CLI 인자로 받습니다.
- **재현성 설정**: `set_seed`는 PyTorch, NumPy, Python 난수 시드를 고정해 실험 재현성을 확보합니다.
- **`train` 함수**:
  - `NormalDataset`을 로드해 학습/검증 세트를 `random_split`으로 나눕니다.
  - `LSTMAutoEncoder`를 초기화하고 `MSELoss`로 학습합니다.
  - 학습 진행 중 주기적으로 체크포인트(`*_latest.pth`)와 최고 성능 모델(`*_best.pth`)을 저장하며, `val_interval`에 맞춰 검증 손실을 기록합니다.
  - `torch.optim.lr_scheduler.MultiStepLR`과 W&B 로깅(`wandb.log`)을 사용합니다.
  - 검증 손실이 개선되지 않는 경우 `patience` 기준으로 얼리 스톱합니다.
- **`main` 함수**: `parse_args` 결과를 언패킹하여 `train`을 호출하는 실행 진입점입니다.

## 추론 및 서비스 (`model_predict.ipynb`)

| 구성 요소 | 함수/모듈 | 설명 |
| --- | --- | --- |
| 특징 추출 | `YOLO("yolov8n-pose.pt")` | 업로드된 영상을 프레임 단위로 처리하며, 포즈 키포인트와 바운딩 박스를 추적 ID별로 수집합니다.
| 스케일 정규화 | `MinMaxScaler` | 각 추적 ID의 최근 20프레임 특징을 0~1 범위로 정규화해 오토인코더 입력 분포를 맞춥니다.
| 재구성 오차 계산 | `calculate_mse` 내부 함수 | 오토인코더 출력과 실제 관측치를 비교하여 프레임 단위 MSE를 구합니다.
| 이상 탐지 로직 | `process_video` 함수 | MSE가 동적으로 계산된 기준(이동 평균과 임계값 조합)을 넘으면 이상 시점으로 판단해, 연속 구간을 `anomaly_times`에 기록하고 필요 시 비디오 클립을 저장합니다.
| 웹 서비스 | Flask `@app.route('/process-video/', methods=['POST'])` | 업로드된 영상을 처리해 이상 구간을 JSON 형태로 반환하며, `flask_ngrok`과 `pyngrok`을 사용해 외부에서 접속 가능합니다.

## 체크포인트

- `LSTMAutoEncoder/` 폴더에는 학습 과정에서 저장된 최신/최고 성능 가중치 파일(`*_latest.pth`, `*_best.pth`)이 포함되어 있어, 재학습 없이도 추론 파이프라인에서 바로 불러올 수 있습니다.

## 참고

- 노트북 기반 실험 코드이므로, 실제 서비스 배포 시에는 공통 모듈을 `.py` 파일로 분리하고 환경 의존적인 경로/토큰을 환경 변수로 관리하는 것이 좋습니다.

## 실행 방법

> 아래 절차는 CUDA가 탑재된 Linux 환경을 기준으로 작성되었습니다. GPU가 없더라도 CPU 모드로 동작하지만 학습 시간이 크게 증가할 수 있습니다.

### 1. 환경 준비

1. Python 3.9 이상 버전을 권장합니다.
2. 가상 환경을 생성하고 필요한 패키지를 설치합니다.

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install -r requirements.txt  # requirements.txt가 없는 경우 노트북 첫 셀을 참고해 수동 설치
   ```

### 2. 학습 실행 (`lstm_ae.ipynb`)

1. JupyterLab 또는 VS Code의 노트북 인터페이스에서 `lstm_ae.ipynb`를 열고 상단부터 순서대로 셀을 실행합니다.
2. 커맨드라인에서 일괄 실행하고 싶은 경우 `papermill`을 활용할 수 있습니다.

   ```bash
   papermill lstm_ae.ipynb lstm_ae_out.ipynb \
     -p data_dir /path/to/dataset \
     -p device cuda:0 \
     -p batch_size 64 \
     -p max_epoch 100
   ```

3. 학습이 완료되면 `LSTMAutoEncoder/` 디렉터리에 체크포인트가 저장됩니다.

### 3. 추론/서비스 실행 (`model_predict.ipynb`)

1. 학습 시 생성된 체크포인트를 `LSTMAutoEncoder/` 폴더에 배치합니다.
2. 노트북을 열고 Flask 앱 초기화 셀까지 실행하면, `ngrok` 또는 로컬호스트를 통해 `/process-video/` 엔드포인트가 활성화됩니다.
3. 터미널에서 다음 명령으로 테스트 영상을 업로드할 수 있습니다.

   ```bash
   curl -X POST http://127.0.0.1:5000/process-video/ \
     -F "video_file=@/path/to/video.mp4"
   ```

4. 응답 JSON에 이상 구간(`anomaly_times`)과 저장된 결과 경로가 포함됩니다.

> **TIP:** 지속적으로 서비스를 운영하려면 `model_predict.ipynb`를 `.py` 스크립트로 변환한 뒤 `gunicorn`과 같은 WSGI 서버에서 실행하는 것을 권장합니다.
