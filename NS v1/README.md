# NS Model - Nelson-Siegel for the Brazilian DI Yield Curve

Integrated system for modeling, analyzing, and backtesting DI yield curves using the Nelson-Siegel (NS) model, with risk factor decomposition (domestic vs. global via CDS) and inflation expectations analysis via PCA.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Methodology](#methodology)
  - [Nelson-Siegel](#1-nelson-siegel)
  - [CDS Decomposition](#2-cds-decomposition)
  - [Inflation PCA](#3-inflation-pca)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [How to Use](#how-to-use)
- [Examples](#examples)
- [Configuration](#configuration)
- [Outputs](#outputs)

---

## 🎯 Overview

This project provides a complete solution for:

1. **Yield Curve Modeling**: Daily fitting of the Nelson-Siegel model for DI curves, enabling interpolation and extrapolation of rates for different tenors
2. **Risk Decomposition**: Separation of sovereign risk (CDS) into **domestic** components (idiosyncratic to Brazil) and **global** components (external factors: DXY, VIX, commodities, UST10)
3. **Inflation Analysis**: Dimensionality reduction of inflation expectations (Focus survey) via PCA, generating interpretable factors (level, slope)
4. **Backtesting**: Framework for validating predictive models with an expanding-window walk-forward approach
5. **Sensitivity Analysis**: Quantification of how changes in CDS factors affect NS parameters (β0 = level, β1 = slope, β2 = curvature)

**Key features:**
- Cross-sectional outlier treatment via MAD (Median Absolute Deviation)
- Automatic conversion of Focus expectations reported by calendar year into fixed horizons (12m/24m/36m)
- Reusable PCA artifacts (scaler, loadings, explained variance)
- Standardized Excel export with intelligent merge behavior

---

## 📊 Methodology

### 1. Nelson-Siegel

The Nelson-Siegel model represents the yield curve as:

$$
y(\tau) = \beta_0 + \beta_1 \cdot L_1(\tau) + \beta_2 \cdot L_2(\tau)
$$

Where:
- $\tau$ = maturity in years
- $\lambda$ = decay parameter (fixed at 0.7308)
- $L_1(\tau) = \frac{1 - e^{-\lambda\tau}}{\lambda\tau}$ (slope loading)
- $L_2(\tau) = L_1(\tau) - e^{-\lambda\tau}$ (curvature loading)

**Factor interpretation:**
- **β0**: Level - long-term rate
- **β1**: Slope - term premium
- **β2**: Curvature - curve convexity

**Estimation:**
- Daily ordinary least squares (OLS)
- Input: observed DI tenors (1m, 3m, 5m, 6m, 12m, 14m, 24m, 36m, 48m, 60m)
- Output: smoothed curve for arbitrary target tenors (1m, 2m, 3m, 4m, 6m, 9m, 12m, 18m, 24m, 30m, 36m, 48m)

### 2. CDS Decomposition

Separates Brazil 10Y CDS into two components via OLS regression (Gabriel, S. 2026)¹:

$$
\Delta CDS_t = \alpha + \gamma_1 \Delta DXY_t + \gamma_2 \Delta CRB_t + \gamma_3 \Delta VIX_t + \gamma_4 \Delta UST10_t + \epsilon_t
$$

**Factors:**
- **CDS_glob** = $\hat{y}_t$ (fitted values): **global** risk explained by external factors
- **CDS_dom** = $\epsilon_t$ (residuals): **domestic** risk (fiscal, political, institutional)

**External variables:**
- DXY: U.S. Dollar Index
- CRB: Commodities
- VIX: S&P 500 volatility
- UST10: 10Y Treasury

### 3. Inflation PCA

Compresses Focus inflation expectations across multiple horizons into principal components:

**Input:**
- IPCA_12m, IPCA_24m, IPCA_36m (converted from annual expectations into fixed horizons)

**Output:**
- **InflPC1**: Inflation level (captures ~90% of variance)
- **InflPC2**: Inflation expectations curve slope

**Horizon conversion:**
- Focus expectations are reported as "IPCA year", "IPCA year_1", "IPCA year_2", "IPCA year_3"
- We convert them into fixed horizons using time-weighted interpolation based on the remaining time in the year:

$$
\\mathrm{IPCA}_{12m,t} = w_t \\cdot F_{year,t} + (1-w_t) \\cdot F_{year+1,t}
$$

where $w_t = \frac{\text{days until 12/31}}{\text{days in the year}}$

---

## 📁 Project Structure

```text
NS v1/
├── README.md                    # This file
├── configs/
│   ├── configs.yaml            # Model and backtest settings
│   └── logging.yaml            # Logging configuration
├── data/                       # Input data (Excel)
│   ├── di_swaps.xlsx          # Observed DI swaps
│   ├── Focus.xlsx             # Focus expectations
│   ├── bbg new.xlsx           # Bloomberg data (NTNB, IPCA, etc.)
│   ├── Database.xlsx          # Base for CDS decomposition
│   └── pca_fatores.xlsx       # Output: PCA factors + NS betas
├── reports/                    # Analysis outputs
│   ├── backtest/              # Backtest results
│   ├── ns/                    # Fitted NS curves
│   ├── pca_ipca/              # Inflation PCA analysis (loadings, correlations)
│   └── pca_real/              # Real-rate PCA analysis
└── src/
    ├── backtest_demo.py       # Backtest demo script
    ├── tests/                 # Unit tests
    │   ├── test_backtest.py
    │   ├── test_features.py
    │   └── test_splits.py
    └── yc/                    # Main package
        ├── __init__.py
        ├── data.py            # Data loading and preparation
        ├── modeling.py        # Nelson-Siegel and CDS decomposition
        ├── export.py          # Excel export
        ├── backtest.py        # Backtesting framework
        ├── advanced_backtest.py
        └── focus_scrap.py     # Focus web scraping (if needed)
```

---

## 🔧 Installation

### Prerequisites

- Python 3.9+

### Dependencies

```bash
pip install numpy pandas scikit-learn statsmodels openpyxl xlrd
```

### Setup

1. Clone or download the repository.
2. Update the absolute paths in `data.py` to match your folder structure:
   ```python
   focus_path = "[Your Saved Folder]/NS Model/data/Focus.xlsx"
   bbg_path = "[Your Saved Folder]/NS Model/data/bbg new.xlsx"
   db_path = "[Your Saved Folder]/Yield Curve - Decom/Database.xlsx"
   ```
3. Make sure the Excel files are present in `data/`.

---

## 🚀 How to Use

### 1. Simple Nelson-Siegel Fit

```python
from src.yc import data, modeling

# Load DI swaps
df_di = data.load_di_swaps_from_days(
    path="[Your Saved Folder]/NS Model/data/di_swaps.xlsx",
    maturities_months=[1, 3, 5, 6, 12, 14, 24, 36, 48, 60],
)

# Fit NS daily
df_betas, df_curve = modeling.fit_nelson_siegel_daily(
    df_di=df_di,
    maturities_fit_months=[1, 3, 5, 6, 12, 14, 24, 36, 48, 60],
    maturities_target_months=[1, 2, 3, 4, 6, 9, 12, 18, 24, 30, 36, 48],
    lam=0.7308,
    min_points_per_day=4,
    drop_outliers_mad=True,
)

print(df_betas.head())  # beta0, beta1, beta2, lambda, rmse_fit, n_points_fit
print(df_curve.head())  # DI_NS_1m, DI_NS_2m, ..., DI_NS_48m
```

### 2. CDS Decomposition

```python
from src.yc import data

df_cds = data.decompor_cds(
    db_path="[Your Saved Folder]/Yield Curve - Decom/Database.xlsx"
)

print(df_cds.head())  # CDS_glob, CDS_dom
```

### 3. NS + CDS Sensitivity

```python
from src.yc import modeling, data

# Load data
df_di = data.load_di_swaps_from_days(...)
df_cds = data.decompor_cds()

# Fit NS + CDS analysis
results = modeling.fit_ns_with_cds_decomposition(
    df_di=df_di,
    df_cds=df_cds,
    maturities_fit_months=[1, 3, 5, 6, 12, 14, 24, 36, 48, 60],
    maturities_target_months=[1, 2, 3, 4, 6, 9, 12, 18, 24, 30, 36, 48],
    rolling_window=60,
    standardize_cds=True,
)

# Outputs:
# results["df_betas"]         -> NS betas (beta0, beta1, beta2)
# results["df_curve"]         -> fitted NS curve
# results["df_sensitivities"] -> rolling sensitivities (beta0_sens_dom, beta0_sens_glob, ...)
# results["df_risk_contrib"]  -> CDS contributions to changes in betas
```

### 4. Inflation PCA

```python
from src.yc import data

df_merged, infl_artifacts, real_artifacts = data.PCA_IPCA(
    focus_path="[Your Saved Folder]/NS Model/data/Focus.xlsx",
    bbg_path="[Your Saved Folder]/NS Model/data/bbg new.xlsx",
    n_components=2,
    horizons_months=(12, 24, 36),
    make_real_pca=False,
)

# df_merged contains:
#   - IPCA_12m, IPCA_24m, IPCA_36m (fixed Focus horizons)
#   - InflPC1, InflPC2 (PCA factors)
#   - NTNB1Y, NTNB3Y (real rates)

# infl_artifacts contains:
#   - scaler, pca, input_cols, loadings, explained_variance_ratio
```

### 5. Run the Backtest

```python
from src.yc import Backtest

# Configure the backtest
bt = Backtest(
    df_features=df_features,  # DataFrame with features
    target_col="DI_12m_diff",
    feature_cols=["InflPC1", "InflPC2", "CDS_dom", "CDS_glob", "NTNB1Y"],
    train_min_days=252,
    test_days=21,
    gap_days=0,
)

# Run
results = bt.run()

# Analyze
print(results["metrics"])       # MAE, RMSE, R², etc.
print(results["predictions"])   # OOS predictions
```

---

## 📖 Examples

### Complete Example: NS → CDS → PCA → Backtest Pipeline

```python
from src.yc import data, modeling, Backtest
from src.yc.export import _fatores
import pandas as pd

# 1) Load DI swaps
df_di = data.load_di_swaps_from_days(
    path="[Your Saved Folder]/NS Model/data/di_swaps.xlsx",
    maturities_months=[1, 3, 5, 6, 12, 14, 24, 36, 48, 60],
)

# 2) CDS decomposition
df_cds = data.decompor_cds()

# 3) NS + CDS sensitivity
results_ns = modeling.fit_ns_with_cds_decomposition(
    df_di=df_di,
    df_cds=df_cds,
    maturities_fit_months=[1, 3, 5, 6, 12, 14, 24, 36, 48, 60],
    maturities_target_months=[1, 2, 3, 4, 6, 9, 12, 18, 24, 30, 36, 48],
    rolling_window=60,
)

# 4) Inflation PCA
df_infl, infl_art, _ = data.PCA_IPCA(n_components=2, make_real_pca=False)

# 5) Merge everything
df_full = results_ns["df_curve"].join(df_infl[["InflPC1", "InflPC2", "NTNB1Y", "NTNB3Y"]], how="inner")
df_full = df_full.join(df_cds, how="inner")

# 6) Create target (1-day-ahead difference in DI_12m)
df_full["DI_12m_diff"] = df_full["DI_NS_12m"].diff().shift(-1)

# 7) Backtest
bt = Backtest(
    df_features=df_full.dropna(),
    target_col="DI_12m_diff",
    feature_cols=["InflPC1", "InflPC2", "CDS_dom", "CDS_glob", "NTNB1Y"],
    train_min_days=252,
    test_days=21,
)

results = bt.run()
print(results["metrics"])

# 8) Export factors to Excel
_fatores.salvar_fator_em_excel(
    df_fator=df_full,
    colnames=["InflPC1", "InflPC2", "CDS_dom", "CDS_glob"],
    output_path="[Your Saved Folder]/NS Model/data/pca_fatores.xlsx",
)
```

---

## ⚙️ Configuration

### Nelson-Siegel Parameters (`modeling.fit_nelson_siegel_daily`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lam` | float | 0.7308 | NS λ (controls the decay of L1/L2) |
| `min_points_per_day` | int | 4 | Minimum observed tenors for a valid fit |
| `drop_outliers_mad` | bool | True | Removes cross-sectional outliers via MAD |
| `mad_z_thresh` | float | 8.0 | MAD z-score threshold |
| `y_min_ok` | float | None | Minimum acceptable rate (%) |
| `y_max_ok` | float | None | Maximum acceptable rate (%) |

### CDS Decomposition Parameters

- **Global variables**: DXY, CRB, VIX, UST10 (hardcoded in `data.decompor_cds`)
- **Transformations**: log returns for prices, simple differences for rates

### PCA Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_components` | int | 2 | Number of PCs to extract |
| `horizons_months` | tuple | (12, 24, 36) | Fixed IPCA horizons |
| `ffill_focus` | bool | True | Forward-fills Focus values (expectations change slowly) |
| `fit_start` | str | None | Start date for PCA fit (`YYYY-MM-DD`) |
| `fit_end` | str | None | End date for PCA fit |
| `make_real_pca` | bool | False | If True, creates RealPC1 from NTNB1Y/3Y |

---

## 📤 Outputs

### 1. Nelson-Siegel

**df_betas**: DataFrame with date index and columns:
- `beta0`, `beta1`, `beta2`: NS parameters
- `lambda`: λ value used
- `rmse_fit`: Root mean squared fitting error
- `n_points_fit`: Number of tenors used on the day
- `maturities_used_months`: String with tenors (for example, `"1,3,6,12,24,36"`)

**df_curve**: DataFrame with `DI_NS_{m}m` columns for each target maturity

### 2. CDS Decomposition

**df_cds**: DataFrame with:
- `CDS_glob`: Global component (fitted values)
- `CDS_dom`: Domestic component (residuals)

*Values are in log returns or % p.a. (depends on `standardize_cds`)*

### 3. CDS-NS Sensitivities

**df_sensitivities**: For each beta (β0, β1, β2):
- `beta{i}_sens_dom`: Sensitivity to CDS_dom (rolling regression coefficient)
- `beta{i}_sens_glob`: Sensitivity to CDS_glob
- `beta{i}_R2`: Rolling regression $R^2$

**df_risk_contrib**: Instantaneous contributions:
- `beta{i}_contrib_dom` = sens_dom × CDS_dom
- `beta{i}_contrib_glob` = sens_glob × CDS_glob
- `beta{i}_contrib_total` = sum of contributions

### 4. Inflation PCA

**df_merged**: DataFrame with:
- `IPCA_12m`, `IPCA_24m`, `IPCA_36m`: Fixed horizons
- `InflPC1`, `InflPC2`: Principal factors
- `NTNB1Y`, `NTNB3Y`: Real rates

**infl_artifacts** (`PCAArtifacts`):
- `scaler`: Fitted `StandardScaler`
- `pca`: Fitted `PCA`
- `input_cols`: List of inputs (`["IPCA_12m", "IPCA_24m", "IPCA_36m"]`)
- `loadings`: DataFrame with loadings for each PC
- `explained_variance_ratio`: Series with % of explained variance

### 5. Reports in `/reports`

- **pca_ipca/**:
  - `corr_matrix.csv`: Input correlation matrix
  - `corr_inputs_vs_pcs.csv`: Input × PC correlations
  - `pc_stats.csv`: Descriptive statistics for PCs
  - `pc_rolling_std_21d.csv`: 21-day rolling standard deviation
  - `top_25_InflPC2_shocks.csv`: Largest PC2 changes

- **backtest/**: Metrics and prediction outputs

---

## 🔬 Technical Details

### Outlier Treatment

The `_drop_cross_section_outliers_mad` method uses a robust z-score:

$$
z_i = 0.6745 \cdot \frac{y_i - \text{median}(y)}{\text{MAD}(y)}
$$

It removes observations with $|z_i| > 8.0$ (configurable).

### Focus Expectations Conversion

Because Focus reports "IPCA year" (expected value for the calendar year):
- On 03/15/2025, "IPCA year" refers to the cumulative value from Jan-Dec 2025
- To obtain the expected value 12 months ahead (03/15/2025 → 03/15/2026), we interpolate:

$$
\\mathrm{IPCA}_{12m} = w \\cdot \\text{IPCA}_{2025} + (1-w) \\cdot \\text{IPCA}_{2026}
$$

where $w = \frac{292}{365}$ (days remaining in 2025 / days in the year).

### NS Lambda

The default value $\lambda = 0.7308$ is empirically calibrated for Brazilian DI curves and maximizes fit quality at intermediate tenors (6m-24m).

---

## 📝 Notes

1. **Hardcoded paths**: The code assumes a fixed structure under `[Your Saved Folder]...`. For use in other environments, update the constants in [src/yc/data.py](src/yc/data.py).

2. **Data quality**: The system assumes that:
   - DI swaps are in % p.a.
   - Focus includes columns such as `"IPCA year"`, `"IPCA year_1"`, etc.
   - Bloomberg includes `"NTNB1Y"`, `"NTNB3Y"`, `"IPCA y/y"`

3. **Performance**: For approximately 2,000 days of data:
   - Nelson-Siegel: ~2s
   - CDS decomposition: ~1s
   - PCA: <1s
   - Backtest (252-day train): ~5-10s

4. **Future extensions**:
   - Migrate to relative paths or a config file
   - Add Svensson (4 factors) as an alternative to NS
   - Implement Dynamic Nelson-Siegel (DNS) with a Kalman filter
   - Build a web app for interactive visualization

---

## 👤 Authors

**Beatriz Monsanto**

**Gabriel de Macedo**

**Gustavo Machado**

**Giovana Katsuki**

---

## 📄 References

GABRIEL, S. Decomposition of Brazil's 5-year DI Futures in Basis Points. Available at: <https://arxiv.org/abs/2601.16995>. Accessed on: Feb. 12, 2026.

---

**Last updated**: February 2026
