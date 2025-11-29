# 미국 교통사고 심각도 예측 AI 프로젝트 (US Accidents Severity Prediction)

이 프로젝트는 미국 교통사고 데이터(US Accidents Dataset)를 활용하여 사고의 심각도(Severity 1~4)를 예측하는 AI 모델을 개발하는 연구 프로젝트입니다. 특히, **치명적인 사고(Severity 4)의 재현율(Recall)을 극대화**하여 안전 중심의 예측 시스템을 구축하는 것을 목표로 합니다.

## 📁 디렉토리 구조 (Directory Structure)

프로젝트는 다음과 같은 구조로 정리되어 있습니다.

```
project_root/
├── data/                   # 데이터 로드 및 전처리 관련 코드
│   ├── preprocess_us_accidents.py  # 전처리 파이프라인
│   └── raw/                        # 원본 데이터 (gitignored)
├── models/                 # 모델 정의 코드
│   ├── crash_severity_net.py       # 메인 모델 (Late Fusion MLP)
│   └── tab_transformer.py          # 실험적 모델 (TabTransformer)
├── scripts/                # 실행 스크립트 (학습, 평가, 시각화 등)
│   ├── train.py                    # 모델 학습
│   ├── evaluate_kfold.py           # K-Fold 교차 검증
│   ├── calibrate.py                # 임계값 보정 (Threshold Calibration)
│   ├── plot_pr_curve.py            # PR 곡선 생성
│   ├── plot_thesis_figures.py      # 논문용 그래프 생성
│   └── generate_report.py          # 결과 요약 리포트 생성
├── experiments/            # 실험용 스크립트
│   ├── find_best_model.py          # 최적 모델 탐색
│   └── generate_weights_plot.py    # 가중치 시각화
├── analysis/               # 분석 스크립트
│   └── explain_model.py            # SHAP 기반 설명 가능성 분석
├── thesis_materials/       # 논문 관련 자료 (결과, 그래프, 방법론)
│   ├── figures/                    # 생성된 그래프 이미지
│   ├── results_summary.md          # 결과 요약
│   └── methodology_details.md      # 방법론 상세
├── requirements.txt        # 의존성 패키지 목록
└── README.md               # 프로젝트 설명서
```

## 🚀 설치 방법 (Installation)

Python 3.8 이상 환경에서 실행하는 것을 권장합니다.

1. **가상환경 생성 및 활성화**
   ```bash
   python -m venv .venv
   # Windows
   .\.venv\Scripts\activate
   # Mac/Linux
   source .venv/bin/activate
   ```

2. **의존성 패키지 설치**
   ```bash
   pip install -r requirements.txt
   ```

## 💻 실행 방법 (Usage)

모든 스크립트는 프로젝트 루트 디렉토리에서 실행해야 합니다.

### 1. 모델 학습 (Training)
기본 `CrashSeverityNet` 모델을 학습합니다.
```bash
python scripts/train.py --model_type crash_severity_net --epochs 10 --batch_size 256
```
*옵션:*
- `--loss_type`: `ce` (CrossEntropy), `focal` (Focal Loss), `ce_weighted` (Weighted CE)
- `--split_strategy`: `time` (시간순 분할), `random` (무작위 분할)

### 2. 모델 평가 (Evaluation)
K-Fold 교차 검증을 수행하여 모델의 일반화 성능을 평가합니다.
```bash
python scripts/evaluate_kfold.py --folds 5 --model_type rf
```

### 3. 임계값 보정 (Threshold Calibration)
Severity 4(치명적 사고)의 재현율을 높이기 위해 최적의 결정 임계값을 찾습니다.
```bash
python scripts/calibrate.py
```

### 4. 시각화 및 리포트 생성 (Visualization)
논문에 사용할 그래프(Confusion Matrix, Recall Comparison 등)를 생성합니다.
```bash
python scripts/plot_thesis_figures.py
```

## 📊 논문 자료 (Thesis Materials)
모든 실험 결과와 그래프는 `thesis_materials/` 디렉토리에 저장됩니다.
- **결과 요약**: [thesis_materials/results_summary.md](thesis_materials/results_summary.md)
- **주요 그래프**: `thesis_materials/figures/`

## 📝 라이선스
이 프로젝트는 학술 연구 목적으로 작성되었습니다.
