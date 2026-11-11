## 0. 프로젝트 정보

- 주제: YOLO 기반 작업장 개인보호구 미착용 및 위험구역 침입 탐지 서비스
- 단계: 모델 test 평가 완료, 서비스 프로토타입 구현 및 검증 중
- 데이터 출처: [Ultralytics Construction-PPE](https://docs.ultralytics.com/datasets/detect/construction-ppe)
- 데이터 라이선스: CC BY 4.0 (Ultralytics team, 2026)

## 1. 문제 정의

작업장 이미지에서 작업자와 안전모·안전조끼 착용 상태를 탐지하고, 지정한 위험구역 진입 여부를 표시하는 교육용 서비스 프로토타입을 만든다. 정량 목표는 baseline 결과를 확인한 뒤 확정한다.

## 2. 데이터

- 원본: 이미지 1,000장, 객체 박스 14,863개, 12개 클래스. 원본 train 800장 / val 200장.
- 선정 이유: `Person`, `helmet`, `no_helmet`, `vest`, `no_vest`가 모두 라벨로 제공된다.
- 학습용 5개 클래스: `0 Person`, `1 helmet`, `2 no_helmet`, `3 vest`, `4 no_vest`.
- 원본 12개 클래스의 박스 중 7,215개를 유지했다. 제외 클래스 박스는 제거하고 이미지는 유지했다.
- 새 분할: train 800장 / val 100장 / test 100장 (80/10/10). 원본 train은 유지하고 원본 val만 파일명 정렬 후 Python `random.Random(42)`로 섞어 앞의 100장을 test로 배정했다.
- test는 모델 선택과 조정에 사용하지 않는다.
- 클래스별 박스 수는 아래 표와 같다.

| Split | Images | Objects | Person | helmet | no_helmet | vest | no_vest |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 800 | 5,791 | 1,962 | 1,282 | 634 | 1,476 | 437 |
| val | 100 | 726 | 246 | 146 | 94 | 170 | 70 |
| test | 100 | 698 | 235 | 148 | 85 | 170 | 60 |

`data/construction-ppe-5class/split_manifest.csv`에 이미지별 분할이 기록되어 있고, `class_mapping.csv`에 원본 ID와 새 ID가 기록되어 있다. 원본 데이터는 별도로 보존한다.

## 3. 데이터 전처리

클래스 재배치와 이미지 경계 밖으로 조금 나간 박스 69개를 경계에 맞게 수정했다. 8장은 핵심 5개 클래스가 없어 빈 라벨을 가진 배경 이미지로 유지했다.

`preprocessing_preview.py`에 다음 전처리를 구현하고 같은 **train** 이미지의 전·후 비교를 저장했다. 그림은 전처리 검증용이다. 실제 학습에 사용할 증강 설정과 효과는 학습 단계에서 따로 기록한다. val/test에는 무작위 증강을 적용하지 않는다.

| 기법 | 설정 | 작업장 이미지에서의 이유 | 박스 처리 |
|---|---|---|---|
| 비율 유지 크기 조정 | 640×640, 여백 추가 | 모델 입력 크기를 맞추면서 사람 모양의 왜곡을 줄임 | 크기와 여백에 맞춰 좌표 변경 |
| 좌우 반전 | 수평 반전 | 작업자 방향 변화에 대응 | `x_center`를 `1 - x_center`로 변경 |
| 밝기·대비 조정 | 밝기 0.65, 대비 1.15 | 실내·그늘 등 조명 변화 예시 | 위치가 안 바뀌므로 좌표 유지 |

비교 그림: [근접 작업자 예시](results/figures/preprocessing_ultralytics-ppe-nov2026-009.jpg), [넓은 작업장 예시](results/figures/preprocessing_source-8c4bd5367bc76d76cd877bc769fbd70a.jpg). 넓은 작업장 예시는 640×640으로 줄이면 사람이 작아지는 문제가 보여, 작은 객체 성능을 추후 별도로 확인해야 한다.

## 4. 모델

YOLO11n baseline과 YOLO26n을 각각 60 epochs, 640px, batch 8, seed 42로 직접 파인튜닝했다. 둘 다 Tesla T4, Ultralytics 8.4.208, PyTorch 2.14.1+cu130에서 실행했다.

## 5. 평가 결과

YOLO11n의 저장된 `best.pt`를 검증한 결과 (val 100장, 객체 726개): Precision 0.927, Recall 0.761, mAP50 0.815, mAP50-95 0.604. 클래스별 Recall은 Person 0.895, helmet 0.767, no_helmet 0.574, vest 0.868, no_vest 0.700이다. 이 수치는 validation 결과이며 test 결과가 아니다. 원본 학습 노트북의 출력에서 확인했다.

YOLO26n의 val 결과는 Precision 0.947, Recall 0.751, mAP50 0.823, mAP50-95 0.620이며 no_helmet Recall 0.574, no_vest Recall 0.674다. 두 실험의 best.pt 검증 출력 기준이다. 두 모델의 no_helmet Recall이 낮아 검증 데이터에서 오류와 기준값을 분석했고 이후 개선 실험을 진행했다. `notebooks/03_val_analysis_colab.ipynb`의 검증 분석을 실행했다.

검증 기준값별 no_helmet 분석을 실행했다 (정답 94개, IoU 0.5 매칭). YOLO11n은 기준값 0.25에서 TP 59, FP 14, FN 35 (Recall 0.6277), 0.10에서 TP 62, FP 44, FN 32 (Recall 0.6596)였다. YOLO26n은 각각 0.25에서 TP 57, FP 9, FN 37 (Recall 0.6064), 0.10에서 TP 63, FP 29, FN 31 (Recall 0.6702)였다. 낮은 기준값은 미탐을 일부 줄이는 대신 오탐을 늘렸다. 기준값별 결과는 기존 Ultralytics 요약 수치와 산출 방식이 달라 직접 비교하지 않는다.

추가 확인: YOLO11n 기준값 0.25에서 미탐 정답을 포함한 이미지 14장을 검토했다. 여러 장에서 작업자와 머리가 멀리 작게 나타났다. 이는 작은 객체가 어려울 수 있다는 관찰이며, 개별 미탐의 원인을 확정한 것은 아니다. 입력 크기 640을 960으로 바꾸는 `notebooks/05_train_yolo11n_img960_colab.ipynb`을 준비했다. 나머지 주요 학습 설정은 baseline과 같다. test는 사용하지 않았다.

이후 960 실험을 실행했다. val에서 Precision 0.925, Recall 0.806, mAP50 0.865, mAP50-95 0.653, no_helmet Recall 0.633, no_vest Recall 0.743이었다. 동일한 학습 조건의 640 baseline보다 전체 mAP와 두 미착용 Recall이 높았다. 학습 시간은 960에서 약 38.5분, 640에서 약 29.6분이었다. val 결과를 바탕으로 test 확인 전에 YOLO11n 960을 최종 모델로 선택한다. 성능 차이를 일반화하려면 추가 검증이 필요하다.

최종 test 평가 (100장, 698개 객체)를 실행했다. YOLO11n 640 / YOLO26n 640 / YOLO11n 960의 mAP50은 각각 0.842 / 0.851 / 0.889이고, mAP50-95는 0.622 / 0.635 / 0.677이다. 미리 선택한 YOLO11n 960의 test Precision은 0.921, Recall은 0.805이며 no_helmet Recall은 0.682, no_vest Recall은 0.633이다. test 결과는 선택 후 보고한 결과이며 모델이나 임계값 조정에 사용하지 않는다. 실제 서비스의 고정 confidence 기준 성능을 뜻하지 않는다.

## 6. 서비스 프로토타입

Streamlit 이미지 업로드 프로토타입 `app.py`를 구현했다. 직접 학습한 YOLO11n 960 가중치 `models/best.pt`를 읽어 탐지한다. 사용자는 이미지와 사각형 위험구역을 지정하고, 작업자 박스 하단 중심점이 영역 안에 있는지 확인할 수 있다. PPE 박스 중심이 사람 박스 안에 있으면 해당 사람에게 연결하며, 겹치는 사람 중 가장 작은 박스에 연결하는 단순 규칙이다. 탐지되지 않은 장비는 '확인 불가'로 처리한다. 영역 미설정, 작업자 미탐지, 연결 실패도 확인 필요로 표시한다. 위험구역 침입은 DANGER, PPE 미착용은 WARNING으로 표시한다. Colab에서 train 이미지 한 장으로 실제 모델 추론과 연결 함수를 실행해 작업자 두 명을 탐지하고 미착용을 표시했다. Streamlit 화면에서의 실제 추론 및 정량적인 사람별 연결·위험구역 판정 평가는 아직 진행 중이다.

## 7. 배포

배포 전.

## 8. 실행 방법

원본 ZIP을 풀어 얻은 `construction-ppe` 폴더에서 다음 명령으로 데이터셋을 다시 만들 수 있다.

```bash
python prepare_data.py /path/to/construction-ppe
```

이 명령은 원본을 수정하지 않는다. 결과는 `data/construction-ppe-5class/`에 만들어진다. 결과 폴더가 이미 차 있으면 덮어쓰지 않고 중단한다. 모델 학습 환경과 설치 명령은 학습 단계에서 기록한다.

서비스 실행 (Python 환경에서): `pip install -r requirements.txt` 후 `streamlit run app.py`. `models/best.pt`가 필요하다. 이 환경에서는 YOLO 패키지를 설치할 수 없어 Streamlit 화면에서의 실제 추론은 아직 검증하지 못했다.

GPU 학습 준비: `data/construction-ppe-5class.zip`을 Google Drive의 내 드라이브 최상위 폴더에 올린 후, Colab에서 `notebooks/01_train_baseline_colab.ipynb`를 열고 위에서 아래로 실행한다. 실행 결과는 Drive의 `ppe-project/runs/baseline_yolo11n_seed42/`에 저장된다. ZIP과 학습 결과는 GitHub 저장소에 올리지 않는다.

## 9. Lessons Learned

실험과 서비스 구현 후 실제 경험을 바탕으로 작성한다.
