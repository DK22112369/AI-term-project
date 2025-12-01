# Research Roadmap: IF 7+ Journal Publication (T-ITS, ESWA, AAP)

**목표**: 단순 졸업논문을 넘어, **Impact Factor 7~10급 국제 저널**에 투고 가능한 수준의 연구 플랫폼 구축.
**핵심 전략**: 실험의 **양(Volume)**과 **깊이(Depth)**를 동시에 확장하여, 제안 방법론(CrashSeverityNet)의 우수성과 일반화 능력을 입증.

---

## 1. 현재 상태 요약 (Current Status)

### 1.1 Core Model
- **CrashSeverityNet**: Group-wise Late Fusion MLP (Temporal, Weather, Road, Spatial).
- **Key Mechanism**: Feature Grouping → Independent Encoders → Concatenation → Classification.

### 1.2 Official Experiment Results (Baseline)
- **Settings**: Weighted Cross Entropy Loss, Time-based Split (60/20/20), 20 Epochs.
- **Performance**:
    - **Test Accuracy**: 59.2% (Trade-off accepted)
    - **Severity 4 (Fatal) Recall**: **68.6%**
    - **Safety Improvement Factor**: Baseline(CatBoost) 대비 **22.9배** 향상.

---

## 2. 확장 연구 축 (Research Axes)

### Axis A: 베이스라인 & 아키텍처 비교 (Baselines & Architecture)
**목표**: "딥러닝이 트리 모델보다 못한가?"라는 질문에 대해, **"안전(Safety) 관점에서는 딥러닝(Late Fusion)이 압도적임"**을 증명.

- **비교 대상**:
    - **Tree Models**: CatBoost (SOTA), XGBoost, LightGBM, Random Forest.
    - **DL Models**: Early Fusion MLP (Simple Concat), TabTransformer (Attention-based).
- **실험 계획**:
    - 동일한 전처리 및 Time-based Split 적용.
    - **지표**: Accuracy, Macro F1, **Fatal Recall**, **Fatal F2-Score**.

### Axis B: 불균형 & Fail-Safe 실험 (Imbalance & Fail-Safe)
**목표**: 제안 모델이 클래스 불균형을 극복하고 **Fail-Safe(안전장치)** 역할을 수행함을 심층 분석.

- **실험 요인 (Factors)**:
    - **Loss Functions**: Cross Entropy vs Weighted CE vs Focal Loss ($\gamma=2, 5$).
    - **Sampling**: None vs WeightedRandomSampler vs SMOTE.
    - **Decision Rule**: Argmax vs **Threshold Calibration** (Safety-Critical Thresholding).
- **핵심 분석**:
    - **Threshold Sweep**: 임계값 변화에 따른 Fatal Recall vs Accuracy Trade-off 곡선 분석.

### Axis C: 일반화 & Cross-Dataset (Generalization)
**목표**: US Accidents 외의 데이터셋에서도 작동함을 보여 **모델의 강건성(Robustness)** 입증.

- **Target Datasets**:
    - **FARS (Fatality Analysis Reporting System)**: 미국 치명 사고 전용 데이터.
    - **UK Stats19**: 영국 교통사고 데이터 (환경/법규 차이).
- **실험 시나리오**:
    - **Scenario 1**: Train on US → Test on FARS (Zero-shot / Transfer).
    - **Scenario 2**: Train on FARS → Test on US (Severe Subset).

### Axis D: XAI & 정책 제언 (Explainability)
**목표**: 블랙박스 모델의 판단 근거를 제시하고, 이를 **교통 안전 정책**과 연결.

- **확장 분석**:
    - **Severity-wise SHAP**: "치명 사고(Sev 4)"와 "경미 사고(Sev 2)"의 주요 요인 차이 분석.
    - **Group Importance**: 4가지 피처 그룹(시간/날씨/도로/공간) 중 무엇이 가장 결정적인가?
- **정책 연결**:
    - 예: "심야 시간대(Temporal) 특정 구간(Spatial)에서 치명 사고 확률이 급증하므로, 가변 속도 제한 필요."

---

## 3. 실행 단위 (Execution Tasks)

### Task 1: 실험 자동화 (Experiment Driver)
- **파일**: `experiments/run_model_grid.py`
- **역할**: JSON 설정을 읽어 다수의 모델/설정을 일괄 실행.
- **결과물**: `results/{exp_id}_metrics.json`, `results/{exp_id}_confmat.png`

### Task 2: 임계값 최적화 (Threshold Sweep)
- **파일**: `experiments/threshold_sweep_fatal.py`
- **역할**: 학습된 모델의 Logits를 로드하여 $\tau \in [0.1, 0.9]$ 스윕.
- **결과물**: `results/threshold_sweep_fatal.png` (Recall-Accuracy Trade-off Curve).

### Task 3: 타 데이터셋 준비 (Cross-Dataset Scaffolding)
- **파일**: `data/preprocess_fars.py` (Skeleton)
- **역할**: FARS 데이터를 US Accidents 포맷으로 변환하는 로직 설계.
- **문서**: `docs/cross_dataset_plan.md`

### Task 4: XAI 심화 (Advanced SHAP)
- **파일**: `analysis/shap_crash_severity.py` (Update)
- **역할**: Target Severity 지정 및 Group-wise Importance 집계 기능 추가.
- **문서**: `docs/xai_extension_ideas.md`

---

## 4. 결론 (Conclusion)
이 로드맵은 단순한 성능 개선을 넘어, **"데이터 불균형 해결", "안전 중심 설계", "일반화 능력", "설명 가능성"**이라는 4가지 학술적 기여점을 체계적으로 입증하기 위한 설계도이다. 이를 통해 T-ITS 등 Top-tier 저널의 요구 수준을 충족시킬 수 있다.
