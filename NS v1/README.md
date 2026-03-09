# NS Model - Nelson-Siegel para Curva de Juros DI Brasileira

Sistema integrado de modelagem, análise e backtesting de curvas de juros DI utilizando o modelo Nelson-Siegel (NS), com decomposição de fatores de risco (doméstico vs global via CDS) e análise de expectativas de inflação via PCA.

---

## 📋 Índice

- [Visão Geral](#visão-geral)
- [Metodologia](#metodologia)
  - [Nelson-Siegel](#1-nelson-siegel)
  - [Decomposição CDS](#2-decomposição-cds)
  - [PCA de Inflação](#3-pca-de-inflação)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Instalação](#instalação)
- [Como Usar](#como-usar)
- [Exemplos](#exemplos)
- [Configurações](#configurações)
- [Outputs](#outputs)

---

## 🎯 Visão Geral

Este projeto oferece uma solução completa para:

1. **Modelagem de Curva de Juros**: Ajuste diário do modelo Nelson-Siegel para curvas DI, permitindo interpolação e extrapolação de taxas para diferentes vértices
2. **Decomposição de Risco**: Separação do risco-país (CDS) em componentes **doméstico** (idiossincráticos ao Brasil) e **global** (fatores externos: DXY, VIX, commodities, UST10)
3. **Análise de Inflação**: Redução dimensional de expectativas de inflação (Focus) via PCA, gerando fatores interpretativos (nível, inclinação)
4. **Backtesting**: Framework para validação de modelos preditivos com walk-forward expanding window
5. **Sensibilidade**: Quantificação de como mudanças em fatores CDS impactam os parâmetros NS (β0=nível, β1=inclinação, β2=curvatura)

**Principais características:**
- Tratamento de outliers cross-sectionais via MAD (Median Absolute Deviation)
- Conversão automática de expectativas Focus "por ano calendário" para horizontes fixos (12m/24m/36m)
- Artefatos de PCA reutilizáveis (scaler, loadings, variância explicada)
- Exportação padronizada para Excel com merge inteligente

---

## 📊 Metodologia

### 1. Nelson-Siegel

O modelo Nelson-Siegel representa a curva de juros como:

$$
y(\tau) = \beta_0 + \beta_1 \cdot L_1(\tau) + \beta_2 \cdot L_2(\tau)
$$

Onde:
- $\tau$ = maturidade em anos
- $\lambda$ = parâmetro de decaimento (fixo em 0.7308)
- $L_1(\tau) = \frac{1 - e^{-\lambda\tau}}{\lambda\tau}$ (loading de inclinação)
- $L_2(\tau) = L_1(\tau) - e^{-\lambda\tau}$ (loading de curvatura)

**Interpretação dos fatores:**
- **β0**: Nível (level) - taxa de longo prazo
- **β1**: Inclinação (slope) - prêmio de prazo
- **β2**: Curvatura (curvature) - convexidade da curva

**Estimação:**
- Mínimos quadrados ordinários (OLS) diário
- Input: vértices observados DI (1m, 3m, 5m, 6m, 12m, 14m, 24m, 36m, 48m, 60m)
- Output: curva suavizada para vértices target arbitrários (1m, 2m, 3m, 4m, 6m, 9m, 12m, 18m, 24m, 30m, 36m, 48m)

### 2. Decomposição CDS

Separa o CDS Brasil 10Y em dois componentes via regressão OLS (Gabriel, S. 2026)¹:

$$
\Delta CDS_t = \alpha + \gamma_1 \Delta DXY_t + \gamma_2 \Delta CRB_t + \gamma_3 \Delta VIX_t + \gamma_4 \Delta UST10_t + \epsilon_t
$$

**Fatores:**
- **CDS_glob** = $\hat{y}_t$ (fitted values): risco **global** explicado por fatores externos
- **CDS_dom** = $\epsilon_t$ (resíduos): risco **doméstico** (fiscal, político, institucional)

**Variáveis externas:**
- DXY: Índice do dólar
- CRB: Commodities
- VIX: Volatilidade S&P 500
- UST10: Treasury 10Y

### 3. PCA de Inflação

Comprime expectativas de inflação Focus (horizontes múltiplos) em componentes principais:

**Input:**
- IPCA_12m, IPCA_24m, IPCA_36m (convertidos de expectativas anuais para horizontes fixos)

**Output:**
- **InflPC1**: Nível de inflação (captura ~90% da variância)
- **InflPC2**: Inclinação da curva de expectativas

**Conversão de horizontes:**
- Expectativas Focus vêm como "IPCA year", "IPCA year_1", "IPCA year_2", "IPCA year_3"
- Convertemos para horizontes fixos usando interpolação ponderada pelo tempo restante no ano:

$$
\text{IPCA}_{12m,t} = w_t \cdot F_{year,t} + (1-w_t) \cdot F_{year+1,t}
$$

onde $w_t = \frac{\text{dias até 31/12}}{\text{dias no ano}}$

---

## 📁 Estrutura do Projeto

```
NS v1/
├── README.md                    # Este arquivo
├── configs/
│   ├── configs.yaml            # Configurações do modelo e backtest
│   └── logging.yaml            # Configuração de logging
├── data/                       # Dados de entrada (Excel)
│   ├── di_swaps.xlsx          # DI swaps observados
│   ├── Focus.xlsx             # Expectativas Focus
│   ├── bbg new.xlsx           # Dados Bloomberg (NTNB, IPCA, etc.)
│   ├── Database.xlsx          # Base para decomposição CDS
│   └── pca_fatores.xlsx       # Output: fatores PCA + NS betas
├── reports/                    # Outputs de análises
│   ├── backtest/              # Resultados de backtesting
│   ├── ns/                    # Curvas NS ajustadas
│   ├── pca_ipca/              # Análises de PCA inflação (loadings, correlações)
│   └── pca_real/              # Análises de PCA juro real
└── src/
    ├── backtest_demo.py       # Script demonstrativo do backtest
    ├── tests/                 # Testes unitários
    │   ├── test_backtest.py
    │   ├── test_features.py
    │   └── test_splits.py
    └── yc/                    # Pacote principal
        ├── __init__.py
        ├── data.py            # Carregamento e preparação de dados
        ├── modeling.py        # Nelson-Siegel e decomposição CDS
        ├── export.py          # Exportação para Excel
        ├── backtest.py        # Framework de backtesting
        ├── advanced_backtest.py
        └── focus_scrap.py     # Web scraping Focus (se necessário)
```

---

## 🔧 Instalação

### Pré-requisitos

- Python 3.9+

### Dependências

```bash
pip install numpy pandas scikit-learn statsmodels openpyxl xlrd
```

### Setup

1. Clone ou baixe o repositório
2. Atualize os caminhos absolutos em `data.py` para refletir sua estrutura:
   ```python
   focus_path = "[Sua Pasta Salva]/NS Model/data/Focus.xlsx"
   bbg_path = "[Sua Pasta Salva]/NS Model/data/bbg new.xlsx"
   db_path = "[Sua Pasta Salva]/Yield Curve - Decom/Database.xlsx"
   ```

3. Certifique-se de que os arquivos Excel estão presentes em `data/`

---

## 🚀 Como Usar

### 1. Ajuste Simples de Nelson-Siegel

```python
from src.yc import data, modeling

# Carregar DI swaps
df_di = data.load_di_swaps_from_days(
    path="[Sua Pasta Salva]/NS Model/data/di_swaps.xlsx",
    maturities_months=[1, 3, 5, 6, 12, 14, 24, 36, 48, 60],
)

# Ajustar NS diariamente
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

### 2. Decomposição CDS

```python
from src.yc import data

df_cds = data.decompor_cds(
    db_path="[Sua Pasta Salva]/Yield Curve - Decom/Database.xlsx"
)

print(df_cds.head())  # CDS_glob, CDS_dom
```

### 3. NS + Sensibilidade a CDS

```python
from src.yc import modeling, data

# Carregar dados
df_di = data.load_di_swaps_from_days(...)
df_cds = data.decompor_cds()

# Ajustar NS + análise CDS
results = modeling.fit_ns_with_cds_decomposition(
    df_di=df_di,
    df_cds=df_cds,
    maturities_fit_months=[1, 3, 5, 6, 12, 14, 24, 36, 48, 60],
    maturities_target_months=[1, 2, 3, 4, 6, 9, 12, 18, 24, 30, 36, 48],
    rolling_window=60,
    standardize_cds=True,
)

# Outputs:
# results["df_betas"]         -> betas NS (beta0, beta1, beta2)
# results["df_curve"]         -> curva NS ajustada
# results["df_sensitivities"] -> sensibilidades rolling (beta0_sens_dom, beta0_sens_glob, ...)
# results["df_risk_contrib"]  -> contribuições de CDS para mudanças nos betas
```

### 4. PCA de Inflação

```python
from src.yc import data

df_merged, infl_artifacts, real_artifacts = data.PCA_IPCA(
    focus_path="[Sua Pasta Salva]/NS Model/data/Focus.xlsx",
    bbg_path="[Sua Pasta Salva]/NS Model/data/bbg new.xlsx",
    n_components=2,
    horizons_months=(12, 24, 36),
    make_real_pca=False,
)

# df_merged contém:
#   - IPCA_12m, IPCA_24m, IPCA_36m (Focus horizontes fixos)
#   - InflPC1, InflPC2 (fatores PCA)
#   - NTNB1Y, NTNB3Y (juro real)

# infl_artifacts contém:
#   - scaler, pca, input_cols, loadings, explained_variance_ratio
```

### 5. Executar Backtest

```python
from src.yc import Backtest

# Configurar backtest
bt = Backtest(
    df_features=df_features,  # DataFrame com features
    target_col="DI_12m_diff",
    feature_cols=["InflPC1", "InflPC2", "CDS_dom", "CDS_glob", "NTNB1Y"],
    train_min_days=252,
    test_days=21,
    gap_days=0,
)

# Executar
results = bt.run()

# Analisar
print(results["metrics"])       # MAE, RMSE, R², etc.
print(results["predictions"])   # Previsões OOS
```

---

## 📖 Exemplos

### Exemplo Completo: Pipeline NS → CDS → PCA → Backtest

```python
from src.yc import data, modeling, Backtest
from src.yc.export import _fatores
import pandas as pd

# 1) Carregar DI swaps
df_di = data.load_di_swaps_from_days(
    path="[Sua Pasta Salva]/NS Model/data/di_swaps.xlsx",
    maturities_months=[1, 3, 5, 6, 12, 14, 24, 36, 48, 60],
)

# 2) Decomposição CDS
df_cds = data.decompor_cds()

# 3) NS + sensibilidade CDS
results_ns = modeling.fit_ns_with_cds_decomposition(
    df_di=df_di,
    df_cds=df_cds,
    maturities_fit_months=[1, 3, 5, 6, 12, 14, 24, 36, 48, 60],
    maturities_target_months=[1, 2, 3, 4, 6, 9, 12, 18, 24, 30, 36, 48],
    rolling_window=60,
)

# 4) PCA de inflação
df_infl, infl_art, _ = data.PCA_IPCA(n_components=2, make_real_pca=False)

# 5) Merge tudo
df_full = results_ns["df_curve"].join(df_infl[["InflPC1", "InflPC2", "NTNB1Y", "NTNB3Y"]], how="inner")
df_full = df_full.join(df_cds, how="inner")

# 6) Criar target (diferença 1 dia à frente do DI_12m)
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

# 8) Exportar fatores para Excel
_fatores.salvar_fator_em_excel(
    df_fator=df_full,
    colnames=["InflPC1", "InflPC2", "CDS_dom", "CDS_glob"],
    output_path="[Sua Pasta Salva]/NS Model/data/pca_fatores.xlsx",
)
```

---

## ⚙️ Configurações

### Parâmetros Nelson-Siegel (`modeling.fit_nelson_siegel_daily`)

| Parâmetro | Tipo | Default | Descrição |
|-----------|------|---------|-----------|
| `lam` | float | 0.7308 | λ do NS (controla decaimento de L1/L2) |
| `min_points_per_day` | int | 4 | Mínimo de vértices observados para fit válido |
| `drop_outliers_mad` | bool | True | Remove outliers cross-sectionais via MAD |
| `mad_z_thresh` | float | 8.0 | Threshold z-score para MAD |
| `y_min_ok` | float | None | Taxa mínima aceitável (%) |
| `y_max_ok` | float | None | Taxa máxima aceitável (%) |

### Parâmetros Decomposição CDS

- **Variáveis globais**: DXY, CRB, VIX, UST10 (hardcoded em `data.decompor_cds`)
- **Transformações**: log-retornos para preços, diferenças simples para taxas

### Parâmetros PCA

| Parâmetro | Tipo | Default | Descrição |
|-----------|------|---------|-----------|
| `n_components` | int | 2 | Número de PCs a extrair |
| `horizons_months` | tuple | (12, 24, 36) | Horizontes fixos de IPCA |
| `ffill_focus` | bool | True | Forward-fill valores Focus (expectativas mudam lentamente) |
| `fit_start` | str | None | Data inicial para fit PCA (formato 'YYYY-MM-DD') |
| `fit_end` | str | None | Data final para fit PCA |
| `make_real_pca` | bool | False | Se True, cria RealPC1 de NTNB1Y/3Y |

---

## 📤 Outputs

### 1. Nelson-Siegel

**df_betas**: DataFrame com índice de data e colunas:
- `beta0`, `beta1`, `beta2`: Parâmetros NS
- `lambda`: Valor de λ usado
- `rmse_fit`: Erro quadrático médio do ajuste
- `n_points_fit`: Número de vértices usados no dia
- `maturities_used_months`: String com vértices (ex: "1,3,6,12,24,36")

**df_curve**: DataFrame com colunas `DI_NS_{m}m` para cada maturidade target

### 2. Decomposição CDS

**df_cds**: DataFrame com:
- `CDS_glob`: Componente global (fitted values)
- `CDS_dom`: Componente doméstico (resíduos)

*Valores em log-retornos ou % a.a. (depende de `standardize_cds`)*

### 3. Sensibilidades CDS-NS

**df_sensitivities**: Para cada beta (β0, β1, β2):
- `beta{i}_sens_dom`: Sensibilidade a CDS_dom (coef. da regressão rolling)
- `beta{i}_sens_glob`: Sensibilidade a CDS_glob
- `beta{i}_R2`: R² da regressão rolling

**df_risk_contrib**: Contribuições instantâneas:
- `beta{i}_contrib_dom` = sens_dom × CDS_dom
- `beta{i}_contrib_glob` = sens_glob × CDS_glob
- `beta{i}_contrib_total` = soma das contribuições

### 4. PCA Inflação

**df_merged**: DataFrame com:
- `IPCA_12m`, `IPCA_24m`, `IPCA_36m`: Horizontes fixos
- `InflPC1`, `InflPC2`: Fatores principais
- `NTNB1Y`, `NTNB3Y`: Juros reais

**infl_artifacts** (PCAArtifacts):
- `scaler`: StandardScaler fitted
- `pca`: PCA fitted
- `input_cols`: Lista de inputs (["IPCA_12m", "IPCA_24m", "IPCA_36m"])
- `loadings`: DataFrame com loadings de cada PC
- `explained_variance_ratio`: Série com % de variância explicada

### 5. Reportes em `/reports`

- **pca_ipca/**: 
  - `corr_matrix.csv`: Matriz de correlação dos inputs
  - `corr_inputs_vs_pcs.csv`: Correlação inputs × PCs
  - `pc_stats.csv`: Estatísticas descritivas dos PCs
  - `pc_rolling_std_21d.csv`: Desvio padrão rolling 21d
  - `top_25_InflPC2_shocks.csv`: Maiores variações do PC2

- **backtest/**: Outputs de métricas e previsões

---

## 🔬 Detalhes Técnicos

### Tratamento de Outliers

O método `_drop_cross_section_outliers_mad` usa z-score robusto:

$$
z_i = 0.6745 \cdot \frac{y_i - \text{median}(y)}{\text{MAD}(y)}
$$

Remove observações com $|z_i| > 8.0$ (configurável).

### Conversão de Expectativas Focus

Como Focus reporta "IPCA year" (expectativa para o ano calendário):
- No dia 15/03/2025, "IPCA year" refere-se ao acumulado de jan-dez/2025
- Para obter expectativa de 12 meses à frente (15/03/2025 → 15/03/2026), interpolamos:

$$
\text{IPCA}_{12m} = w \cdot \text{IPCA}_{2025} + (1-w) \cdot \text{IPCA}_{2026}
$$

onde $w = \frac{292}{365}$ (dias restantes em 2025 / dias no ano).

### Lambda do NS

O valor padrão $\lambda = 0.7308$ é calibrado empiricamente para curvas DI brasileiras e maximiza o ajuste em vértices intermediários (6m-24m).

---

## 📝 Notas

1. **Hardcoded paths**: O código assume estrutura fixa em `[Sua Pasta Salva]...`. Para uso em outros ambientes, atualize as constantes em [data.py](src/yc/data.py).

2. **Qualidade de dados**: O sistema assume que:
   - DI swaps estão em % a.a.
   - Focus tem colunas "IPCA year", "IPCA year_1", etc.
   - Bloomberg tem "NTNB1Y", "NTNB3Y", "IPCA y/y"

3. **Performance**: Para ~2000 dias de dados:
   - Nelson-Siegel: ~2s
   - Decomposição CDS: ~1s
   - PCA: <1s
   - Backtest (252d train): ~5-10s

4. **Extensões futuras**:
   - Migrar para paths relativos ou config file
   - Adicionar Svensson (4 fatores) como alternativa ao NS
   - Implementar Dynamic Nelson-Siegel (DNS) com filtro de Kalman
   - WebApp para visualização interativa

---

## 👤 Autores

**Beatriz Monsanto**

**Gabriel de Macedo**  

**Gustavo Machado**

**Giovana Katsuki**

---

## 📄 Referencias

GABRIEL, S. Decomposition of Brazil’s 5-year DI Futures in Basis Points. Disponível em: <https://arxiv.org/abs/2601.16995>. Acesso em: 12 fev. 2026.

---

**Última atualização**: Fevereiro 2026
