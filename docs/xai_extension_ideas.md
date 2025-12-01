# XAI Extension Ideas: Interpretable Crash Prediction

**목표**: 블랙박스 모델(Deep Learning)의 의사결정 과정을 투명하게 설명하고, 이를 교통 안전 정책 수립에 활용할 수 있는 근거 자료로 제시.

---

## 1. Severity-wise SHAP Analysis (심각도별 분석)

기존 SHAP 분석은 주로 "모델의 전체 출력"에 집중하지만, 본 연구는 **"치명 사고(Severity 4)"와 "경미 사고(Severity 2)"의 결정 요인이 다름**을 밝히는 것이 중요하다.

- **Hypothesis**:
    - **Severity 2 (Minor)**: 교통량(Traffic), 날씨(Rain) 등 일상적인 요인이 지배적.
    - **Severity 4 (Fatal)**: 특정 시간대(Night), 고속도로 진출입로(Junction), 과속 구간 등 구조적/환경적 요인이 지배적.
- **Action**:
    - `analysis/shap_crash_severity.py`를 확장하여 `target_severity` 파라미터 추가.
    - Sev 2 vs Sev 4의 SHAP Summary Plot을 나란히 비교(Side-by-side).

## 2. Feature Group Importance (그룹 중요도)

개별 피처(예: Humidity)보다 **피처 그룹(Domain)** 단위의 기여도를 분석하는 것이 정책적으로 더 유의미하다.

- **Groups**:
    1.  **Temporal**: 시간, 요일, 월. (정책: 가변 속도 제한, 순찰 시간 조정)
    2.  **Weather**: 기상 조건. (정책: 악천후 경보 시스템)
    3.  **Road**: 도로 인프라. (정책: 도로 구조 개선, 신호등 설치)
    4.  **Spatial**: 위치. (정책: 사고 다발 구역 집중 관리)
- **Method**:
    - 각 그룹에 속한 피처들의 절대 SHAP 값 합계($\sum |SHAP|$)를 계산하여 그룹별 기여도 산출.

## 3. Advanced Plots (시각화 확장)

- **Partial Dependence Plot (PDP)**:
    - 특정 피처(예: `Distance(mi)`)가 변할 때 사고 심각도 확률이 어떻게 변하는지 선그래프로 표현.
    - 비선형 관계(예: 정체 길이가 특정 구간을 넘어가면 심각도 급증) 파악 가능.
- **Geospatial SHAP Map**:
    - `Start_Lat`, `Start_Lng`의 SHAP 값을 지도 위에 히트맵으로 시각화.
    - "어떤 지역이 모델에게 위험하게 인식되는가?"를 직관적으로 보여줌.

---

## 4. Implementation Plan

- [x] Basic SHAP Skeleton (`analysis/shap_crash_severity.py`)
- [ ] **Severity-wise Analysis**: 타겟 클래스 지정 기능 추가.
- [ ] **Group Importance**: 그룹별 집계 로직 구현.
- [ ] **PDP**: `sklearn.inspection.PartialDependenceDisplay` 활용.
