# Journal Extension Plan: CrashSeverityNet

**목표**: IF 7~10급 국제 저널 (Q1) 투고를 위한 장기 연구 로드맵
**작성일**: 2025-11-30

---

## 1. 목표 저널 후보 (Target Journals)

| 저널명 | IF (Approx.) | 성격 및 적합성 |
| :--- | :--- | :--- |
| **IEEE Transactions on Intelligent Transportation Systems (T-ITS)** | 8.5 | 교통 AI 분야 최고 권위지. 딥러닝 아키텍처의 기술적 참신성과 실제 교통 문제 해결 능력을 모두 요구함. 가장 적합. |
| **Expert Systems with Applications (ESWA)** | 8.5 | 응용 AI 분야. 특정 도메인(교통)에 AI를 적용하여 성능을 개선한 사례를 선호. 방법론적 기여(Late Fusion, Fail-Safe) 강조 필요. |
| **Accident Analysis & Prevention (AAP)** | 6.3 | 교통 안전 분야 전통 강자. 딥러닝 모델 자체보다는 "사고 요인 분석"과 "안전 정책적 시사점"이 중요. SHAP 분석을 깊게 해야 함. |
| **Transportation Research Part C (TR-C)** | 8.3 | 교통 기술/시스템 분야. 자율주행/C-ITS 시스템 관점에서의 통합 설계가 중요. |
| **Reliability Engineering & System Safety (RESS)** | 8.1 | 시스템 신뢰성/안전 분야. "Fail-Safe", "Risk Assessment" 관점에서의 접근이 필수적. |

---

## 2. 현 수준(Mk.1/2) vs Q1 저널 요구사항 비교

| 구분 | 현재 수준 (Mk.1/2 - KSAE용) | Q1 저널 요구 수준 (Target) | Gap Analysis |
| :--- | :--- | :--- | :--- |
| **데이터셋** | US Accidents (Single) | **Multiple Datasets** (US + UK/Canada 등) | 일반화 성능(Generalization) 검증 필수. 다른 국가/환경 데이터에서도 작동함을 보여야 함. |
| **비교군** | RF, CatBoost, Simple MLP | **SOTA Deep Learning** (TabNet, FT-Transformer, Saint 등) | 최신 Tabular DL 모델들과의 정량적 비교 및 우위 입증 필요. |
| **분석 깊이** | Confusion Matrix, Recall | **In-depth XAI & Sensitivity Analysis** | SHAP뿐만 아니라, 입력 변화에 따른 민감도 분석, 반사실적 설명(Counterfactual) 등이 요구됨. |
| **실용성** | 개념적 제안 (V2V) | **Simulation / Case Study** | SUMO/Carla 시뮬레이터 연동 또는 실제 사고 사례에 대한 Case Study 필요. |

---

## 3. 단기 확장 계획 (6~12개월)

**목표**: 방법론의 견고함(Robustness) 강화 및 최신 SOTA 비교

- [ ] **Advanced Baselines 추가**:
    - `TabNet` (PyTorch 구현체 활용)
    - `FT-Transformer` (Feature Tokenizer + Transformer)
    - `ResNet` for Tabular Data
- [ ] **불균형 처리 고도화**:
    - `LDAM-DRW` (Learning with Defered Re-weighting) 손실 함수 적용.
    - `Mixup` 기반의 데이터 증강 기법 실험.
- [ ] **엄밀한 교차 검증**:
    - 5-Fold가 아닌 **10-Fold CV** 또는 **Nested CV** 적용.
    - 통계적 유의성 검정 (t-test, Wilcoxon signed-rank test) 수행.

---

## 4. 중기 확장 계획 (1~2년)

**목표**: 데이터 확장 및 멀티모달 융합

- [ ] **Cross-Dataset Validation**:
    - 영국 교통사고 데이터(Stats19) 또는 캐나다 NCDB 데이터 확보.
    - US 데이터로 학습한 모델을 타 국가 데이터에 적용(Transfer Learning)하여 성능 평가.
- [ ] **Multi-modal 확장 (NLP 융합)**:
    - US Accidents의 `Description` 컬럼(텍스트) 활용.
    - **BERT/RoBERTa**로 텍스트 임베딩 추출 후, 기존 Late Fusion 네트워크에 새로운 브랜치로 추가.
    - "정형 데이터 + 텍스트" 멀티모달 모델로 성능 극대화.

---

## 5. 장기 비전 (Long-term Vision)

**목표**: 실제 시스템 통합 및 안전성 보장 (System Integration)

- [ ] **V2X 시뮬레이션 연동**:
    - SUMO(Traffic Simulator)와 연동하여, 사고 발생 시 주변 차량 흐름 변화 시뮬레이션.
    - 심각도 예측값에 따른 차량 우회 경로 생성 알고리즘 제안.
- [ ] **Edge Device 최적화**:
    - Jetson Orin/Nano 등 엣지 디바이스에서의 추론 속도(Latency) 최적화.
    - 모델 경량화 (Quantization, Pruning) 실험.
- [ ] **Fail-Safe 메커니즘 구체화**:
    - 예측 불확실성(Uncertainty Estimation, Monte Carlo Dropout)을 측정.
    - 불확실성이 높을 경우 "운전자에게 제어권 이양" 등의 안전 프로토콜 설계.

---

## 6. 결론
현재의 `CrashSeverityNet`은 "안전 중심의 손실 함수 설계"라는 독창적인 아이디어를 가지고 있다. 이를 Q1 저널급으로 발전시키기 위해서는 **1) 최신 딥러닝 모델과의 비교**, **2) 다중 데이터셋 검증**, **3) 텍스트 등 추가 모달리티 활용**이 핵심이다. 이 로드맵을 따라 단계적으로 실험을 확장한다면 T-ITS나 ESWA 게재가 충분히 가능하다.
