# 7. Preprocessing Summary

## 7.1 Data Cleaning
- **Missing Values**:
    - **Numerical Features**: Filled with the **Median** of the training set.
    - **Categorical Features**: Filled with the **Mode** (most frequent value) or marked as 'Unknown'.
    - **Target (Severity)**: Rows with missing severity labels were dropped.

## 7.2 Feature Engineering
We extracted meaningful features from the raw data to capture temporal, environmental, and spatial contexts.

| Group | Features | Engineering Logic |
| :--- | :--- | :--- |
| **Temporal** | Hour, DayOfWeek, Month | Extracted from `Start_Time`. |
| | Duration | `End_Time` - `Start_Time` (in minutes). |
| **Weather** | Temp, Humidity, Pressure, Visibility, Wind Speed | Used as continuous variables. |
| | Weather Condition | Categorical (e.g., Rain, Snow, Clear). |
| **Road** | Traffic Signal, Junction, Crossing, etc. | Binary flags (True/False $\to$ 1/0). |
| **Spatial** | Latitude, Longitude | Used as continuous coordinates. |
| | Distance(mi) | Length of the road segment affected. |

## 7.3 Data Splitting
To prevent data leakage and simulate real-world forecasting, we used a **Time-based Split**:
- **Train (60%)**: Earliest data.
- **Validation (20%)**: Middle segment.
- **Test (20%)**: Latest data.

## 7.4 Scaling and Encoding
- **Numerical Features**: Standardized using `StandardScaler` ($z = \frac{x - \mu}{\sigma}$).
- **Categorical Features**: Encoded using `OneHotEncoder`.
- **Leakage Prevention**: All scalers and encoders were **fitted ONLY on the Training set**. The Validation and Test sets were transformed using these pre-fitted parameters, ensuring that global statistics from the future (test set) did not leak into the training process.
