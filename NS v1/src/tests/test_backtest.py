"""
Testes unitários para o backtesting do Nelson-Siegel.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from yc import data, modeling, Backtest
from yc.advanced_backtest import AdvancedBacktestAnalysis

# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_di_data():
    """Cria dados de DI sintéticos para testes."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    np.random.seed(42)

    # Cria dados realistas de DI (% a.a.)
    data_dict = {
        "DI_1m": np.random.uniform(10.0, 12.0, 100),
        "DI_3m": np.random.uniform(10.1, 12.1, 100),
        "DI_6m": np.random.uniform(10.2, 12.2, 100),
        "DI_12m": np.random.uniform(10.3, 12.3, 100),
        "DI_24m": np.random.uniform(10.4, 12.4, 100),
    }

    df = pd.DataFrame(data_dict, index=dates)
    return df

@pytest.fixture
def sample_ns_fit(sample_di_data):
    """Ajusta NS em dados sintéticos."""
    maturities_fit = [1, 3, 6, 12, 24]
    maturities_target = [1, 3, 6, 12, 24]

    df_betas, df_curve = modeling.fit_nelson_siegel_daily(
        df_di=sample_di_data,
        maturities_fit_months=maturities_fit,
        maturities_target_months=maturities_target,
        lam=0.7308,
        min_points_per_day=3,
        drop_outliers_mad=False,
    )

    return df_betas, df_curve

@pytest.fixture
def backtest_instance(sample_di_data, sample_ns_fit):
    """Cria instância do Backtest."""
    df_betas, df_curve = sample_ns_fit

    bt = Backtest(
        df_di=sample_di_data,
        df_betas=df_betas,
        df_curve=df_curve,
        maturities_fit_months=[1, 3, 6, 12, 24],
        maturities_target_months=[1, 3, 6, 12, 24],
    )

    return bt

# ============================================================================
# Testes Básicos do Backtest
# ============================================================================

def test_backtest_initialization(backtest_instance):
    """Testa inicialização do Backtest."""
    assert len(backtest_instance.df_di) > 0
    assert len(backtest_instance.df_betas) > 0
    assert len(backtest_instance.df_curve) > 0

def test_compute_residuals(backtest_instance):
    """Testa cálculo de resíduos."""
    df_res = backtest_instance.compute_residuals()

    assert len(df_res) > 0
    assert all(c.startswith("residual_") for c in df_res.columns)

    # Resíduos não devem ser todos NaN
    assert not df_res.isna().all().all()

def test_compute_metrics(backtest_instance):
    """Testa cálculo de métricas."""
    metrics = backtest_instance.compute_metrics()

    # Verificações básicas
    assert metrics.rmse_overall >= 0
    assert metrics.mae_overall >= 0
    assert len(metrics.rmse_by_maturity) > 0
    assert len(metrics.mae_by_maturity) > 0

    # RMSE deve ser >= MAE (sempre verdadeiro)
    assert metrics.rmse_overall >= metrics.mae_overall

def test_metrics_by_maturity(backtest_instance):
    """Testa métricas por maturidade."""
    metrics = backtest_instance.compute_metrics()

    for mat, rmse in metrics.rmse_by_maturity.items():
        assert rmse > 0
        assert mat in metrics.mae_by_maturity
        assert rmse >= metrics.mae_by_maturity[mat]

def test_beta_stability(backtest_instance):
    """Testa estabilidade de betas."""
    metrics = backtest_instance.compute_metrics()

    assert "beta0" in metrics.beta_stability
    assert "beta1" in metrics.beta_stability
    assert "beta2" in metrics.beta_stability

    for beta, std in metrics.beta_stability.items():
        assert std >= 0

def test_coverage_calculation(backtest_instance):
    """Testa cálculo de cobertura."""
    metrics = backtest_instance.compute_metrics()

    for mat, cov in metrics.coverage_by_maturity.items():
        assert 0 <= cov <= 100

# ============================================================================
# Testes de Plots
# ============================================================================

def test_plot_betas_and_rmse(backtest_instance, tmp_path):
    """Testa geração de plot de betas e RMSE."""
    out_path = tmp_path / "test_betas_rmse.png"
    fig = backtest_instance.plot_betas_and_rmse(str(out_path))

    assert fig is not None
    assert out_path.exists()

def test_plot_residuals_distribution(backtest_instance, tmp_path):
    """Testa geração de histogramas."""
    out_path = tmp_path / "test_residuals_dist.png"
    fig = backtest_instance.plot_residuals_distribution(str(out_path))

    assert fig is not None
    assert out_path.exists()

def test_plot_residuals_timeseries(backtest_instance, tmp_path):
    """Testa geração de série temporal de resíduos."""
    out_path = tmp_path / "test_residuals_ts.png"
    fig = backtest_instance.plot_residuals_timeseries(str(out_path))

    assert fig is not None
    assert out_path.exists()

def test_plot_rmse_by_maturity(backtest_instance, tmp_path):
    """Testa geração de barplot de RMSE."""
    out_path = tmp_path / "test_rmse_by_mat.png"
    fig = backtest_instance.plot_rmse_by_maturity(str(out_path))

    assert fig is not None
    assert out_path.exists()

# ============================================================================
# Testes de Exportação
# ============================================================================

def test_export_report_to_excel(backtest_instance, tmp_path):
    """Testa exportação de relatório em Excel."""
    out_path = tmp_path / "test_backtest_report.xlsx"
    backtest_instance.export_report_to_excel(str(out_path))

    assert out_path.exists()

    # Verifica se tem as abas esperadas
    df_summary = pd.read_excel(out_path, sheet_name="Summary")
    assert len(df_summary) > 0

def test_backtest_run(backtest_instance, tmp_path):
    """Testa execução completa do backtest."""
    out_dir = tmp_path / "backtest_outputs"
    out_dir.mkdir()

    metrics = backtest_instance.run(str(out_dir), export_excel=True, verbose=False)

    assert metrics is not None
    assert (out_dir / "backtest_report.xlsx").exists()

    # Verifica se foram criados plots
    png_files = list(out_dir.glob("*.png"))
    assert len(png_files) > 0

# ============================================================================
# Testes de Análises Avançadas
# ============================================================================

def test_advanced_analysis_init(backtest_instance):
    """Testa inicialização de análises avançadas."""
    adv = AdvancedBacktestAnalysis(backtest_instance)

    assert adv.bt is not None
    assert len(adv.df_res) > 0

def test_rolling_metrics(backtest_instance):
    """Testa cálculo de métricas em rolling window."""
    adv = AdvancedBacktestAnalysis(backtest_instance)
    rolling = adv.rolling_metrics(window_days=20)

    assert len(rolling) > 0
    assert "RMSE" in rolling.columns
    assert "MAE" in rolling.columns
    assert not rolling["RMSE"].isna().all()

def test_residual_statistics(backtest_instance):
    """Testa estatísticas de resíduos."""
    adv = AdvancedBacktestAnalysis(backtest_instance)
    stats = adv.residual_statistics()

    assert len(stats) > 0
    assert "Mean" in stats.columns
    assert "Std" in stats.columns
    assert "Skewness" in stats.columns
    assert "Kurtosis" in stats.columns

def test_regime_analysis(backtest_instance):
    """Testa análise de regimes."""
    adv = AdvancedBacktestAnalysis(backtest_instance)
    regimes = adv.regime_analysis(volatility_percentile=50)

    assert isinstance(regimes, dict)
    assert len(regimes) > 0

def test_beta_stability_by_period(backtest_instance):
    """Testa estabilidade de betas por período."""
    adv = AdvancedBacktestAnalysis(backtest_instance)
    df_periods = adv.beta_stability_by_period(period_days=30)

    assert len(df_periods) > 0
    assert "Beta" in df_periods.columns
    assert "Mean" in df_periods.columns

def test_advanced_export(backtest_instance, tmp_path):
    """Testa exportação de análises avançadas."""
    adv = AdvancedBacktestAnalysis(backtest_instance)
    out_dir = tmp_path / "advanced"
    out_dir.mkdir()

    adv.export_advanced_analysis(str(out_dir))

    assert (out_dir / "advanced_backtest_analysis.xlsx").exists()

    # Verifica se foram criados plots
    png_files = list(out_dir.glob("*.png"))
    assert len(png_files) > 0

# ============================================================================
# Testes de Validação de Dados
# ============================================================================

def test_residuals_not_all_nan(backtest_instance):
    """Verifica que não há todos NaNs nos resíduos."""
    df_res = backtest_instance.compute_residuals()
    assert not df_res.isna().all().all()

def test_metrics_positive(backtest_instance):
    """Verifica que métricas são positivas."""
    metrics = backtest_instance.compute_metrics()

    assert metrics.rmse_overall > 0
    assert metrics.mae_overall > 0

def test_betas_reasonable_values(backtest_instance):
    """Verifica que betas têm valores razoáveis."""
    df_betas = backtest_instance.df_betas

    # beta0 (nível) deve estar próximo das taxas observadas
    assert df_betas["beta0"].min() > 5
    assert df_betas["beta0"].max() < 20

    # beta1 e beta2 têm amplitudes menores
    assert df_betas["beta1"].abs().max() < 5
    assert df_betas["beta2"].abs().max() < 5

# ============================================================================
# Executar testes
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
