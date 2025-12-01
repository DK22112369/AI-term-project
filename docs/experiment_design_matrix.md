# Experiment Design Matrix & Research Plan

**목표**: IF 7~10급 저널(T-ITS, ESWA) 투고를 위한 체계적인 실험 설계

---

## 1. Experiment Matrix (Ablation Studies)

`experiments/run_experiment_matrix.py`를 통해 다음 5가지 핵심 실험을 자동화하여 수행한다.

| ID | Experiment Name | Model | Loss Function | Split Strategy | Hypothesis (가설) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | `crash_latefusion_weighted_time` | **CrashSeverityNet** | **Weighted CE** | **Time** | **(Main)** Late Fusion과 가중 손실함수가 불균형 데이터에서 최적의 성능(Recall)을 낼 것이다. |
| 2 | `crash_latefusion_ce_time` | CrashSeverityNet | Standard CE | Time | 가중치 없는 일반 CE는 다수 클래스(Severity 2)에 편향될 것이다. |
| 3 | `crash_latefusion_focal_time` | CrashSeverityNet | **Focal Loss** | Time | Focal Loss는 Hard Example(Severity 4) 학습에 도움을 줄 것이다. |
| 4 | `crash_earlyfusion_weighted_time` | **Early MLP** | Weighted CE | Time | (Baseline) 단순 결합(Early Fusion)은 그룹별 특징을 충분히 살리지 못할 것이다. |
| 5 | `crash_latefusion_sampler_time` | CrashSeverityNet | Standard CE | Time | (Alternative) 오버샘플링(Sampler)은 가중 손실함수와 유사한 효과를 내거나, 과적합될 수 있다. |

---

## 2. Advanced Metrics (평가 지표)

단순 정확도(Accuracy)를 넘어, 안전 중심의 지표를 도입하였다 (`utils/metrics.py`).

- **Macro F2-Score**: Recall에 Precision보다 2배의 가중치를 둔 지표. 인명 안전이 중요한 본 연구에서 핵심 지표로 활용.
- **ROC-AUC (One-vs-Rest)**: 임계값(Threshold) 변화에 따른 모델의 변별력을 종합적으로 평가.
- **Safety Factor**: Baseline(CatBoost) 대비 Severity 4 Recall의 향상 비율.

---

## 3. Cross-Dataset Validation Plan (Future Work)

현재는 US Accidents 데이터셋만 사용하지만, 연구의 일반화(Generalization) 성능 입증을 위해 타 데이터셋 확장을 계획한다.

### 3.1 Target Datasets
- **UK Accidents (Stats19)**: 영국 교통사고 데이터. 도로 환경(좌측 통행)과 기상 조건이 미국과 다름.
- **Canada NCDB**: 캐나다 국가 충돌 데이터베이스.

### 3.2 Implementation Strategy
1.  **Data Adapter**: 각 데이터셋의 컬럼을 US Accidents의 표준 포맷(`Temporal`, `Weather`, `Road`, `Spatial`)으로 매핑하는 어댑터 작성.
2.  **Transfer Learning**:
    - US 데이터로 Pre-train된 Encoder 가중치를 고정(Freeze)하거나 미세 조정(Fine-tune).
    - 타 데이터셋에 대해 Few-shot Learning 수행.

---

## 4. How to Run

```bash
# 전체 실험 매트릭스 실행 (순차적)
python experiments/run_experiment_matrix.py
```

실행 결과는 `results/` 디렉토리에 JSON 로그와 Confusion Matrix 이미지로 자동 저장된다.
