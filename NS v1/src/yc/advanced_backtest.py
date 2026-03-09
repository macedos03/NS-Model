"""
Análises Avançadas de Backtesting do Nelson-Siegel.

Inclui:
1. Análise de rolling windows
2. Estabilidade de parâmetros por período
3. Comparação de performance em regimes diferentes
4. Análise de fat tails e distribuição de resíduos
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
import os


class AdvancedBacktestAnalysis:
    """Análises avançadas de backtesting."""

    def __init__(self, backtest_instance):
        """
        backtest_instance: instância da classe Backtest
        """
        self.bt = backtest_instance
        self.df_res = self.bt.compute_residuals()
        self.metrics = self.bt.compute_metrics()

    # =========================================================================
    # 1) Análise de Rolling Performance
    # =========================================================================

    def rolling_metrics(self, window_days: int = 60) -> pd.DataFrame:
        """
        Calcula RMSE e MAE em janelas móveis.

        window_days: tamanho da janela em dias
        """
        results = []

        for i in range(len(self.df_res) - window_days):
            window_start = self.df_res.index[i]
            window_end = self.df_res.index[i + window_days - 1]

            window_data = self.df_res.iloc[i : i + window_days]
            res_clean = window_data.values.flatten()
            res_clean = res_clean[~np.isnan(res_clean)]

            if len(res_clean) > 0:
                rmse = float(np.sqrt(np.mean(res_clean**2)))
                mae = float(np.mean(np.abs(res_clean)))
            else:
                rmse = np.nan
                mae = np.nan

            results.append({
                "Date_Start": window_start,
                "Date_End": window_end,
                "RMSE": rmse,
                "MAE": mae,
            })

        return pd.DataFrame(results)

    def plot_rolling_metrics(self, window_days: int = 60, out_path: str | None = None):
        """Plota métricas em rolling window."""
        rolling = self.rolling_metrics(window_days=window_days)

        fig, axes = plt.subplots(2, 1, figsize=(14, 8))

        # RMSE
        ax = axes[0]
        ax.plot(rolling["Date_End"], rolling["RMSE"], linewidth=1.5, color="#0B1F3A")
        ax.fill_between(rolling["Date_End"], rolling["RMSE"], alpha=0.3, color="#0B1F3A")
        ax.set_title(f"Rolling RMSE (window={window_days}d)", fontsize=12, fontweight="bold")
        ax.set_ylabel("RMSE (% a.a.)")
        ax.grid(True, alpha=0.3)

        # MAE
        ax = axes[1]
        ax.plot(rolling["Date_End"], rolling["MAE"], linewidth=1.5, color="#696969")
        ax.fill_between(rolling["Date_End"], rolling["MAE"], alpha=0.3, color="#696969")
        ax.set_title(f"Rolling MAE (window={window_days}d)", fontsize=12, fontweight="bold")
        ax.set_ylabel("MAE (% a.a.)")
        ax.set_xlabel("Date")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()

        if out_path:
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            fig.savefig(out_path, dpi=160, bbox_inches="tight")
            print(f"[OK] Plot saved: {out_path}")

        return fig

    # =========================================================================
    # 2) Análise de Distribuição de Resíduos
    # =========================================================================

    def residual_statistics(self) -> pd.DataFrame:
        """Calcula estatísticas por maturidade: mean, std, skew, kurtosis."""
        res_cols = [c for c in self.df_res.columns if c.startswith("residual_")]

        stats_list = []
        for col in sorted(res_cols):
            data = self.df_res[col].dropna()

            if len(data) > 0:
                stats_list.append({
                    "Maturity": col.replace("residual_", ""),
                    "Mean": float(data.mean()),
                    "Std": float(data.std()),
                    "Min": float(data.min()),
                    "Max": float(data.max()),
                    "Skewness": float(scipy_stats.skew(data)),
                    "Kurtosis": float(scipy_stats.kurtosis(data)),
                    "N": len(data),
                })

        return pd.DataFrame(stats_list)

    def plot_qq_plots(self, out_path: str | None = None):
        """Q-Q plots para testar normalidade dos resíduos."""
        res_cols = [c for c in self.df_res.columns if c.startswith("residual_")]
        n_cols = len(res_cols)

        if n_cols == 0:
            print("[WARN] Nenhum resíduo para Q-Q plots.")
            return None

        ncols = 3
        nrows = (n_cols + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
        axes_flat = axes.flatten() if n_cols > 1 else [axes]

        for idx, col in enumerate(sorted(res_cols)):
            ax = axes_flat[idx]
            data = self.df_res[col].dropna()

            scipy_stats.probplot(data, dist="norm", plot=ax)
            ax.set_title(f"Q-Q Plot: {col}", fontsize=11, fontweight="bold")
            ax.grid(True, alpha=0.3)

        for idx in range(len(res_cols), len(axes_flat)):
            axes_flat[idx].set_visible(False)

        fig.tight_layout()

        if out_path:
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            fig.savefig(out_path, dpi=160, bbox_inches="tight")
            print(f"[OK] Plot saved: {out_path}")

        return fig

    # =========================================================================
    # 3) Análise de Regimes (volatilidade alta vs baixa)
    # =========================================================================

    def regime_analysis(self, volatility_percentile: float = 75) -> dict[str, pd.DataFrame]:
        """
        Divide a série em regimes de volatilidade alta/baixa e compara performance.

        volatility_percentile: percentil de volatilidade para separação
        """
        # Calcula volatilidade móvel (30d)
        rolling_vol = (
            self.df_res.std(axis=1)
            .rolling(window=30, min_periods=1)
            .mean()
        )

        vol_threshold = rolling_vol.quantile(volatility_percentile / 100)

        # Separa períodos
        high_vol_mask = rolling_vol > vol_threshold
        low_vol_mask = ~high_vol_mask

        # Calcula métricas para cada regime
        regimes = {}

        for regime_name, mask in [("High_Volatility", high_vol_mask), ("Low_Volatility", low_vol_mask)]:
            subset = self.df_res.loc[mask]
            res_clean = subset.values.flatten()
            res_clean = res_clean[~np.isnan(res_clean)]

            if len(res_clean) > 0:
                regimes[regime_name] = pd.DataFrame({
                    "RMSE": [np.sqrt(np.mean(res_clean**2))],
                    "MAE": [np.mean(np.abs(res_clean))],
                    "N_Obs": [len(res_clean)],
                    "Mean_Residual": [np.mean(res_clean)],
                    "Std_Residual": [np.std(res_clean)],
                })

        return regimes

    def plot_regime_comparison(self, volatility_percentile: float = 75, out_path: str | None = None):
        """Compara performance entre regimes."""
        regimes = self.regime_analysis(volatility_percentile)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        metrics_list = ["RMSE", "MAE"]
        x_pos = np.arange(len(metrics_list))

        for metric_idx, metric in enumerate(metrics_list):
            ax = axes[metric_idx]

            values = [regimes[r][metric].iloc[0] if r in regimes else np.nan for r in ["Low_Volatility", "High_Volatility"]]
            colors = ["#0B1F3A", "#696969"]

            bars = ax.bar(["Low Vol Regime", "High Vol Regime"], values, color=colors, alpha=0.8, edgecolor="black", linewidth=1.2)

            ax.set_title(f"{metric} by Volatility Regime", fontsize=12, fontweight="bold")
            ax.set_ylabel(f"{metric} (% a.a.)")
            ax.grid(True, alpha=0.3, axis="y")

            # Adiciona valores
            for bar, val in zip(bars, values):
                if not np.isnan(val):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{val:.6f}", ha="center", va="bottom", fontsize=10)

        fig.tight_layout()

        if out_path:
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            fig.savefig(out_path, dpi=160, bbox_inches="tight")
            print(f"[OK] Plot saved: {out_path}")

        return fig

    # =========================================================================
    # 4) Análise de Estabilidade de Betas por Regime Temporal
    # =========================================================================

    def beta_stability_by_period(self, period_days: int = 90) -> pd.DataFrame:
        """
        Calcula a estabilidade de betas em períodos não-sobrepostos.

        Útil para detectar mudanças estruturais.
        """
        betas = self.bt.df_betas.copy()
        results = []

        for i in range(0, len(betas), period_days):
            period_betas = betas.iloc[i : i + period_days]

            if len(period_betas) > 1:
                period_start = period_betas.index.min()
                period_end = period_betas.index.max()

                for col in ["beta0", "beta1", "beta2"]:
                    if col in period_betas.columns:
                        vals = period_betas[col].dropna()
                        if len(vals) > 1:
                            results.append({
                                "Period_Start": period_start,
                                "Period_End": period_end,
                                "Beta": col,
                                "Mean": float(vals.mean()),
                                "Std": float(vals.std()),
                                "Min": float(vals.min()),
                                "Max": float(vals.max()),
                            })

        return pd.DataFrame(results)

    def plot_beta_stability_by_period(self, period_days: int = 90, out_path: str | None = None):
        """Plota estabilidade de betas por período."""
        df_periods = self.beta_stability_by_period(period_days)

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        for ax_idx, beta_name in enumerate(["beta0", "beta1", "beta2"]):
            ax = axes[ax_idx]

            data = df_periods[df_periods["Beta"] == beta_name]

            if len(data) > 0:
                x = np.arange(len(data))

                ax.bar(x, data["Mean"], yerr=data["Std"], alpha=0.7, capsize=5, color="#0B1F3A", edgecolor="black")

                ax.set_title(f"{beta_name} by Period ({period_days}d)", fontsize=12, fontweight="bold")
                ax.set_ylabel("Value (± Std Dev)")
                ax.set_xlabel("Period")
                ax.grid(True, alpha=0.3, axis="y")

        fig.tight_layout()

        if out_path:
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            fig.savefig(out_path, dpi=160, bbox_inches="tight")
            print(f"[OK] Plot saved: {out_path}")

        return fig

    # =========================================================================
    # 5) Exportar todas as análises avançadas
    # =========================================================================

    def export_advanced_analysis(self, out_dir: str, df_cds: pd.DataFrame | None = None):
        """Exporta todas as análises avançadas em Excel e plots."""
        os.makedirs(out_dir, exist_ok=True)

        # Excel com abas
        excel_path = os.path.join(out_dir, "advanced_backtest_analysis.xlsx")

        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            # Residual statistics
            self.residual_statistics().to_excel(writer, sheet_name="Residual_Stats", index=False)

            # Rolling metrics
            self.rolling_metrics(window_days=60).to_excel(writer, sheet_name="Rolling_Metrics_60d", index=False)
            self.rolling_metrics(window_days=30).to_excel(writer, sheet_name="Rolling_Metrics_30d", index=False)

            # Regime analysis
            regimes = self.regime_analysis()
            for regime_name, df_regime in regimes.items():
                df_regime.to_excel(writer, sheet_name=f"Regime_{regime_name}", index=False)

            # Beta stability by period
            self.beta_stability_by_period(period_days=90).to_excel(writer, sheet_name="Beta_Stability_90d", index=False)
            
            # CDS analysis (se disponível)
            if df_cds is not None:
                try:
                    cds_analysis = self.analyze_cds_impact(df_cds, rolling_window=60)
                    
                    if "df_rolling_corr" in cds_analysis:
                        cds_analysis["df_rolling_corr"].to_excel(writer, sheet_name="CDS_Rolling_Corr")
                    
                    if "regression_stats" in cds_analysis and cds_analysis["regression_stats"]:
                        pd.DataFrame([cds_analysis["regression_stats"]]).to_excel(
                            writer, sheet_name="CDS_Regression", index=False
                        )
                    
                    if "regime_stats" in cds_analysis and cds_analysis["regime_stats"]:
                        pd.DataFrame(cds_analysis["regime_stats"]).T.to_excel(
                            writer, sheet_name="CDS_Regime_Stats"
                        )
                except Exception as e:
                    print(f"[WARN] Erro ao exportar análise CDS: {e}")

        print(f"[OK] Advanced analysis exported to: {excel_path}")

        # Plots
        plots = {
            "advanced_rolling_metrics.png": lambda out_path: self.plot_rolling_metrics(window_days=60, out_path=out_path),
            "advanced_qq_plots.png": lambda out_path: self.plot_qq_plots(out_path=out_path),
            "advanced_regime_comparison.png": lambda out_path: self.plot_regime_comparison(volatility_percentile=75, out_path=out_path),
            "advanced_beta_stability_by_period.png": lambda out_path: self.plot_beta_stability_by_period(period_days=90, out_path=out_path),
        }
        
        # Adiciona plot CDS se disponível
        if df_cds is not None:
            plots["advanced_cds_impact.png"] = lambda out_path: self.plot_cds_impact(df_cds, rolling_window=60, out_path=out_path)

        for filename, plot_func in plots.items():
            try:
                out_path = os.path.join(out_dir, filename)
                plot_func(out_path)
                plt.close("all")
            except Exception as e:
                print(f"[WARN] Erro ao gerar plot {filename}: {e}")

        print(f"[OK] All advanced plots saved to: {out_dir}")

    def print_summary(self):
        """Imprime resumo das análises avançadas."""
        print("\n" + "="*80)
        print("ADVANCED BACKTEST ANALYSIS")
        print("="*80 + "\n")

        # Residual statistics
        print("1. RESIDUAL STATISTICS BY MATURITY:")
        print("-" * 80)
        res_stats = self.residual_statistics()
        print(res_stats.to_string(index=False))

        # Regime analysis
        print("\n2. PERFORMANCE BY VOLATILITY REGIME:")
        print("-" * 80)
        regimes = self.regime_analysis(volatility_percentile=75)
        for regime_name, df_regime in regimes.items():
            print(f"\n{regime_name}:")
            print(df_regime.to_string(index=False))

        # Rolling metrics
        print("\n3. ROLLING METRICS (60-day window):")
        print("-" * 80)
        rolling = self.rolling_metrics(window_days=60)
        print(f"RMSE: {rolling['RMSE'].min():.6f}% to {rolling['RMSE'].max():.6f}% (mean: {rolling['RMSE'].mean():.6f}%)")
        print(f"MAE:  {rolling['MAE'].min():.6f}% to {rolling['MAE'].max():.6f}% (mean: {rolling['MAE'].mean():.6f}%)")

        # Beta stability
        print("\n4. BETA STABILITY BY PERIOD (90-day):")
        print("-" * 80)
        beta_periods = self.beta_stability_by_period(period_days=90)
        for beta_name in ["beta0", "beta1", "beta2"]:
            subset = beta_periods[beta_periods["Beta"] == beta_name]
            if len(subset) > 0:
                print(f"\n{beta_name}:")
                print(f"  Mean (across periods): {subset['Mean'].mean():.6f}")
                print(f"  Std Dev range: {subset['Std'].min():.6f} to {subset['Std'].max():.6f}")

        print("\n" + "="*80 + "\n")

    # =========================================================================
    # 6) Análise de Impacto CDS nos Resíduos e Betas
    # =========================================================================

    def _standardize_cds_to_pct(self, df_cds: pd.DataFrame) -> pd.DataFrame:
        """
        Converte CDS_dom e CDS_glob de log-retornos para % a.a. acumulado.
        
        CDS vem como log-retornos (dif de log).
        Converte para série acumulada em basis points, depois em % a.a.
        """
        df_cds_std = df_cds[["CDS_dom", "CDS_glob"]].copy()
        
        # Exponencia e acumula para voltar ao nível
        for col in ["CDS_dom", "CDS_glob"]:
            # exp(log_ret) = valor relativo
            # cumsum de log_ret -> nível acumulado
            df_cds_std[f"{col}_level"] = (1.0 + df_cds_std[col]).cumprod() - 1.0
            # converte para basis points (* 10000) depois para % a.a.
            df_cds_std[f"{col}_pct"] = df_cds_std[f"{col}_level"] * 100.0
        
        return df_cds_std[["CDS_dom_pct", "CDS_glob_pct"]].rename(
            columns={"CDS_dom_pct": "CDS_dom", "CDS_glob_pct": "CDS_glob"}
        )

    def analyze_cds_impact(
        self,
        df_cds: pd.DataFrame,
        rolling_window: int = 60,
        standardize: bool = True,
    ) -> dict:
        """
        Analisa impacto de CDS_dom e CDS_glob nos resíduos do backtest.
        
        Args:
            df_cds: DataFrame com CDS_dom, CDS_glob (em log-retornos ou %)
            standardize: se True, converte para mesma escala (% a.a.)
        
        Retorna:
        - Correlação rolling entre resíduos e fatores CDS
        - Regressão de resíduos médios vs CDS
        - Análise de períodos de alto risco CDS
        """
        # Padroniza CDS para % a.a. se necessário
        if standardize:
            df_cds_use = self._standardize_cds_to_pct(df_cds)
        else:
            df_cds_use = df_cds[["CDS_dom", "CDS_glob"]].copy()
        
        # Merge resíduos com CDS
        df_merged = self.df_res.join(df_cds_use, how="inner")
        
        if len(df_merged) < rolling_window:
            print("[WARN] Dados insuficientes para análise CDS.")
            return {}
        
        # Resíduo médio por dia (média cross-section)
        res_cols = [c for c in df_merged.columns if c.startswith("residual_")]
        df_merged["residual_mean"] = df_merged[res_cols].mean(axis=1)
        df_merged["residual_std"] = df_merged[res_cols].std(axis=1)
        
        # 1) Correlação rolling
        rolling_corr = []
        for i in range(rolling_window, len(df_merged)):
            window = df_merged.iloc[i - rolling_window : i]
            
            corr_dom = window["residual_mean"].corr(window["CDS_dom"])
            corr_glob = window["residual_mean"].corr(window["CDS_glob"])
            
            rolling_corr.append({
                "Date": df_merged.index[i],
                "corr_CDS_dom": float(corr_dom) if not np.isnan(corr_dom) else 0.0,
                "corr_CDS_glob": float(corr_glob) if not np.isnan(corr_glob) else 0.0,
            })
        
        df_corr = pd.DataFrame(rolling_corr).set_index("Date")
        
        # 2) Regressão: residual_mean vs CDS (diferenças)
        data_clean = df_merged[["residual_mean", "CDS_dom", "CDS_glob"]].dropna()
        
        if len(data_clean) > 30:
            # Usa diferenças para remover tendências
            d_res = data_clean["residual_mean"].diff().dropna()
            d_cds_dom = data_clean["CDS_dom"].loc[d_res.index].diff().dropna()
            d_cds_glob = data_clean["CDS_glob"].loc[d_res.index].diff().dropna()
            
            # Alinha
            common_idx = d_res.index.intersection(d_cds_dom.index).intersection(d_cds_glob.index)
            
            if len(common_idx) > 30:
                X = np.column_stack([
                    np.ones(len(common_idx)),
                    d_cds_dom.loc[common_idx].values,
                    d_cds_glob.loc[common_idx].values,
                ])
                y = d_res.loc[common_idx].values
                
                coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
                yhat = X @ coefs
                
                ss_res = np.sum((y - yhat) ** 2)
                ss_tot = np.sum((y - y.mean()) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
                
                regression_stats = {
                    "intercept": float(coefs[0]),
                    "coef_CDS_dom": float(coefs[1]),
                    "coef_CDS_glob": float(coefs[2]),
                    "R2": float(r2),
                    "N": len(common_idx),
                    "interpretation": f"1% change in CDS_dom → {coefs[1]:.6f}% change in residual",
                }
            else:
                regression_stats = {}
        else:
            regression_stats = {}
        
        # 3) Análise por regime de risco CDS
        cds_total = df_merged["CDS_dom"].abs() + df_merged["CDS_glob"].abs()
        cds_high_threshold = cds_total.quantile(0.75)
        
        high_cds_mask = cds_total > cds_high_threshold
        low_cds_mask = ~high_cds_mask
        
        regime_stats = {}
        for regime_name, mask in [("High_CDS", high_cds_mask), ("Low_CDS", low_cds_mask)]:
            subset = df_merged.loc[mask, "residual_mean"].dropna()
            
            if len(subset) > 0:
                regime_stats[regime_name] = {
                    "Mean_Residual": float(subset.mean()),
                    "Std_Residual": float(subset.std()),
                    "RMSE": float(np.sqrt(np.mean(subset ** 2))),
                    "N": len(subset),
                }
        
        return {
            "df_rolling_corr": df_corr,
            "regression_stats": regression_stats,
            "regime_stats": regime_stats,
        }

    def plot_cds_impact(
        self,
        df_cds: pd.DataFrame,
        rolling_window: int = 60,
        out_path: str | None = None,
    ):
        """Plota análise de impacto CDS."""
        analysis = self.analyze_cds_impact(df_cds, rolling_window=rolling_window, standardize=True)
        
        if not analysis:
            print("[WARN] Análise CDS vazia.")
            return None
        
        # Padroniza CDS para plotar
        df_cds_use = self._standardize_cds_to_pct(df_cds)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # 1) Correlação rolling
        ax = axes[0, 0]
        df_corr = analysis["df_rolling_corr"]
        ax.plot(df_corr.index, df_corr["corr_CDS_dom"], label="CDS Doméstico", linewidth=1.5, color="#696969")
        ax.plot(df_corr.index, df_corr["corr_CDS_glob"], label="CDS Global", linewidth=1.5, color="#0B1F3A")
        ax.axhline(0, color="black", linestyle="--", alpha=0.3)
        ax.set_title(f"Correlação Rolling ({rolling_window}d): Resíduo vs CDS", fontsize=12, fontweight="bold")
        ax.set_ylabel("Correlação")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2) Regime comparison
        ax = axes[0, 1]
        regime_stats = analysis.get("regime_stats", {})
        if regime_stats:
            regimes = ["Low_CDS", "High_CDS"]
            rmse_vals = [regime_stats.get(r, {}).get("RMSE", 0) for r in regimes]
            colors = ["#0B1F3A", "#696969"]
            
            bars = ax.bar(regimes, rmse_vals, color=colors, alpha=0.8, edgecolor="black")
            ax.set_title("RMSE por Regime de Risco CDS", fontsize=12, fontweight="bold")
            ax.set_ylabel("RMSE (% a.a.)")
            ax.grid(True, alpha=0.3, axis="y")
            
            for bar, val in zip(bars, rmse_vals):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, val, f"{val:.4f}", 
                           ha="center", va="bottom", fontsize=10)
        
        # 3) Scatter: Δ Residual vs Δ CDS_dom
        ax = axes[1, 0]
        df_merged = self.df_res.join(df_cds_use, how="inner")
        res_cols = [c for c in df_merged.columns if c.startswith("residual_")]
        df_merged["residual_mean"] = df_merged[res_cols].mean(axis=1)
        
        # Usa diferenças para scatter (remover trends)
        d_res = df_merged["residual_mean"].diff().dropna()
        d_cds_dom = df_merged["CDS_dom"].loc[d_res.index].diff().dropna()
        
        common_idx = d_res.index.intersection(d_cds_dom.index)
        data_scatter = pd.DataFrame({
            "res": d_res.loc[common_idx],
            "cds": d_cds_dom.loc[common_idx]
        })
        
        ax.scatter(data_scatter["cds"], data_scatter["res"], 
                  alpha=0.3, s=10, color="#696969")
        ax.set_xlabel("Δ CDS Doméstico (% a.a.)")
        ax.set_ylabel("Δ Resíduo Médio (% a.a.)")
        ax.set_title("Variação de Resíduo vs Variação de CDS Doméstico", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        
        # Add regression line
        reg_stats = analysis.get("regression_stats", {})
        if reg_stats and "coef_CDS_dom" in reg_stats:
            x_range = np.linspace(data_scatter["cds"].min(), data_scatter["cds"].max(), 100)
            y_pred = reg_stats["intercept"] + reg_stats["coef_CDS_dom"] * x_range
            ax.plot(x_range, y_pred, color="red", linewidth=2, 
                   label=f"R² = {reg_stats['R2']:.3f}")
            ax.legend()
        
        # 4) Scatter: Δ Residual vs Δ CDS_glob
        ax = axes[1, 1]
        d_cds_glob = df_merged["CDS_glob"].loc[d_res.index].diff().dropna()
        
        common_idx = d_res.index.intersection(d_cds_glob.index)
        data_scatter = pd.DataFrame({
            "res": d_res.loc[common_idx],
            "cds": d_cds_glob.loc[common_idx]
        })
        
        ax.scatter(data_scatter["cds"], data_scatter["res"], 
                  alpha=0.3, s=10, color="#0B1F3A")
        ax.set_xlabel("Δ CDS Global (% a.a.)")
        ax.set_ylabel("Δ Resíduo Médio (% a.a.)")
        ax.set_title("Variação de Resíduo vs Variação de CDS Global", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        
        # Add regression line
        if reg_stats and "coef_CDS_glob" in reg_stats:
            x_range = np.linspace(data_scatter["cds"].min(), data_scatter["cds"].max(), 100)
            y_pred = reg_stats["intercept"] + reg_stats["coef_CDS_glob"] * x_range
            ax.plot(x_range, y_pred, color="#0B1F3A", linewidth=2, 
                   label=f"R² = {reg_stats['R2']:.3f}")
            ax.legend()
        
        fig.tight_layout()
        
        if out_path:
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            fig.savefig(out_path, dpi=160, bbox_inches="tight")
            print(f"[OK] Plot saved: {out_path}")
        
        return fig

# =====================================================================
# Main: Exemplo de execução
# =====================================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Permite importar do pacote yc
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from yc import data, modeling, Backtest

    print("="*80)
    print("ADVANCED NELSON-SIEGEL BACKTESTING ANALYSIS")
    print("="*80 + "\n")

    # =========================================================================
    # 1) Carrega dados de DI swaps
    # =========================================================================
    print("[1/4] Loading DI swaps data...")

    di_swaps_path = "Q:/Gabriel de Macedo/Política Monetária/NS Model/data/di_swaps.xlsx"

    maturities_fit = [1, 3, 5, 6, 12, 14, 24, 36, 48, 60]
    maturities_target = [1, 2, 3, 4, 6, 9, 12, 18, 24, 30, 36, 48]

    df_di = data.load_di_swaps_from_days(
        path=di_swaps_path,
        sheet_name=0,
        date_col="Dates",
        maturities_months=maturities_fit,
    )

    print(f"  ✓ Loaded {len(df_di)} days of DI swap data")
    print(f"  ✓ Maturities: {list(df_di.columns)}\n")

    # =========================================================================
    # 2) Ajusta o modelo Nelson-Siegel
    # =========================================================================
    print("[2/4] Fitting Nelson-Siegel model...")

    df_betas, df_curve = modeling.fit_nelson_siegel_daily(
        df_di=df_di,
        maturities_fit_months=maturities_fit,
        maturities_target_months=maturities_target,
        lam=0.7308,
        min_points_per_day=4,
        drop_outliers_mad=True,
        mad_z_thresh=8.0,
        y_min_ok=0.0,
        y_max_ok=40.0,
    )

    print(f"  ✓ Fitted {len(df_betas)} daily curves")
    print(f"  ✓ Betas shape: {df_betas.shape}")
    print(f"  ✓ Target maturities: {maturities_target}\n")

    # =========================================================================
    # 3) Executa backtesting
    # =========================================================================
    print("[3/4] Running backtest...")

    backtest_out_dir = "Q:/Gabriel de Macedo/Política Monetária/NS Model/reports/backtest"

    bt = Backtest(
        df_di=df_di,
        df_betas=df_betas,
        df_curve=df_curve,
        maturities_fit_months=maturities_fit,
        maturities_target_months=maturities_target,
    )

    metrics = bt.run(
        out_dir=backtest_out_dir,
        export_excel=True,
        verbose=True,
    )

    print(f"  ✓ Backtest completed\n")

    # =========================================================================
    # 4) Carrega decomposição CDS
    # =========================================================================
    print("[4/5] Loading CDS decomposition...")
    
    try:
        df_cds = data.decompor_cds()
        print(f"  ✓ Loaded {len(df_cds)} days of CDS data")
        print(f"  ✓ Components: CDS_dom, CDS_glob\n")
    except Exception as e:
        print(f"  [WARN] Could not load CDS data: {e}")
        df_cds = None

    # =========================================================================
    # 5) Executa análises avançadas
    # =========================================================================
    print("[5/5] Running advanced analysis...")

    advanced_out_dir = "Q:/Gabriel de Macedo/Política Monetária/NS Model/reports/backtest"

    advanced = AdvancedBacktestAnalysis(bt)

    # Imprime resumo
    advanced.print_summary()
    
    # Análise CDS (se disponível)
    if df_cds is not None:
        print("\n" + "="*80)
        print("CDS IMPACT ANALYSIS")
        print("="*80 + "\n")
        
        cds_analysis = advanced.analyze_cds_impact(df_cds, rolling_window=60)
        
        if "regression_stats" in cds_analysis and cds_analysis["regression_stats"]:
            reg = cds_analysis["regression_stats"]
            print("Regression: Residual = α + β₁·CDS_dom + β₂·CDS_glob")
            print(f"  Intercept:   {reg['intercept']:.6f}")
            print(f"  β CDS_dom:   {reg['coef_CDS_dom']:.6f}")
            print(f"  β CDS_glob:  {reg['coef_CDS_glob']:.6f}")
            print(f"  R²:          {reg['R2']:.4f}")
            print(f"  N:           {reg['N']}")
        
        if "regime_stats" in cds_analysis and cds_analysis["regime_stats"]:
            print("\nPerformance by CDS Regime:")
            for regime_name, stats in cds_analysis["regime_stats"].items():
                print(f"\n  {regime_name}:")
                print(f"    RMSE:          {stats['RMSE']:.6f}%")
                print(f"    Mean Residual: {stats['Mean_Residual']:.6f}%")
                print(f"    N:             {stats['N']}")
        
        print("\n" + "="*80 + "\n")

    # Exporta análises completas
    advanced.export_advanced_analysis(out_dir=advanced_out_dir, df_cds=df_cds)

    print("[OK] All analysis completed successfully!")
    print("\n" + "="*80)