# Imbalance & Fail-Safe Experiment Plan

**목표**: 클래스 불균형(Class Imbalance) 문제를 해결하고, 안전 중심(Safety-Critical)의 Fail-Safe 성능을 극대화하기 위한 실험 설계.

---

## 1. 실험 요인 (Experimental Factors)

### 1.1 Loss Functions (손실 함수)
- **Cross Entropy (CE)**: Baseline. 다수 클래스(Severity 2)에 편향될 가능성 높음.
- **Weighted CE**: 클래스 빈도의 역수를 가중치로 부여. 소수 클래스(Severity 3, 4)의 Recall 향상 기대.
- **Focal Loss**: $\gamma$ 값을 조절하여 "학습하기 어려운 샘플(Hard Example)"에 집중.
    - $\gamma=2.0$ (Standard)
    - $\gamma=5.0$ (Strong focus on hard examples)

### 1.2 Sampling Strategies (샘플링)
- **None**: 원본 데이터 분포 사용.
- **WeightedRandomSampler**: 배치(Batch) 구성 시 소수 클래스를 더 자주 추출. (Over-sampling 효과)
- **SMOTE**: Feature Space에서 소수 클래스 데이터를 합성 (Synthetic Minority Over-sampling).

### 1.3 Decision Rules (결정 규칙)
- **Argmax**: $ \hat{y} = \text{argmax}_k P(y=k|x) $. (기본)
- **Threshold Calibration (Fail-Safe)**:
    - Fatal Class(Severity 4)의 확률 $P(y=4|x)$가 임계값 $\tau$를 넘으면 무조건 4로 예측.
    - $$ \hat{y} = 4 \quad \text{if} \quad P(y=4|x) > \tau \quad \text{else} \quad \text{argmax}_{k \neq 4} P(y=k|x) $$

---

## 2. Threshold Sweep Experiment

**목표**: Accuracy를 일부 희생하더라도 Fatal Recall을 극대화할 수 있는 최적의 $\tau^*$ 탐색.

### 2.1 실험 방법
1.  학습된 `CrashSeverityNet` 모델 로드.
2.  Test Set에 대한 예측 확률(Logits/Softmax) 추출.
3.  $\tau \in [0.0, 1.0]$ 범위에서 0.05 간격으로 변화시키며 다음 지표 계산:
    - **Fatal Recall**: 실제 사망 사고 중 예측 성공 비율.
    - **Fatal Precision**: 사망 사고로 예측한 것 중 실제 비율.
    - **Fatal F2-Score**: Recall에 가중치를 둔 조화 평균.
    - **Overall Accuracy**: 전체 정확도.
4.  **Trade-off Curve** 시각화.

### 2.2 실행 스크립트
- `experiments/threshold_sweep_fatal.py`

---

## 3. Expected Outcomes (예상 결과)

- **Loss**: Weighted CE가 가장 안정적인 성능을 보일 것으로 예상되나, Focal Loss가 극단적인 불균형(Sev 4)에서 더 나은 Recall을 보일 수도 있음.
- **Threshold**: $\tau$가 낮아질수록 Recall은 급격히 상승하고 Precision/Accuracy는 하락함.
- **Optimal Point**: F2-Score가 최대가 되거나, Recall이 목표치(예: 80%)를 달성하는 지점을 $\tau^*$로 선정.

---

## 4. Future Work
- **Cost-Sensitive Learning**: 오분류 비용 행렬(Cost Matrix)을 정의하여 손실 함수에 반영.
- **Ensemble**: 서로 다른 Loss로 학습된 모델들의 앙상블.
