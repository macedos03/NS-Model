from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import NamedTuple
import os


class BacktestMetrics(NamedTuple):
    """Métricas aggregadas de performance do backtest"""
    rmse_overall: float
    mae_overall: float
    rmse_by_maturity: dict[str, float]
    mae_by_maturity: dict[str, float]
    n_obs_by_maturity: dict[str, int]
    beta_stability: dict[str, float]  # std dev de cada beta
    coverage_by_maturity: dict[str, float]  # % de cobertura (não-NaN)


class Backtest:
    """
    Backtesting do modelo Nelson-Siegel.
    
    Calcula:
    - Fit diário do NS
    - Comparação curva teórica vs observada
    - Métricas de qualidade por maturidade
    - Estabilidade temporal dos parâmetros
    - Distribuição de resíduos
    """

    def __init__(
        self,
        df_di: pd.DataFrame,
        df_betas: pd.DataFrame,
        df_curve: pd.DataFrame,
        maturities_fit_months: list[int],
        maturities_target_months: list[int],
    ):
        """
        Inicializa backtest.

        df_di: DataFrame com colunas DI_{m}m observados (% a.a.)
        df_betas: DataFrame com beta0, beta1, beta2, rmse_fit, n_points_fit
        df_curve: DataFrame com colunas DI_NS_{m}m (curva ajustada)
        """
        self.df_di = df_di.sort_index()
        self.df_betas = df_betas.sort_index()
        self.df_curve = df_curve.sort_index()
        self.maturities_fit = sorted(maturities_fit_months)
        self.maturities_target = sorted(maturities_target_months)

    # =========================================================================
    # 1) Resíduos e erros diários
    # =========================================================================

    def compute_residuals(self) -> pd.DataFrame:
        """
        Calcula resíduos diários: residual[dt, m] = observed[dt, m] - fitted[dt, m]
        Apenas para maturidades comuns (fit + target).
        """
        residuals = []

        # maturidades comuns: as que existem tanto em fit quanto em target
        common_mats = sorted(set(self.maturities_fit) & set(self.maturities_target))

        for dt in self.df_curve.index:
            if dt not in self.df_di.index:
                continue

            row_res = {"Date": dt}
            for mat in common_mats:
                obs_col = f"DI_{mat}m"
                fit_col = f"DI_NS_{mat}m"

                if obs_col in self.df_di.columns and fit_col in self.df_curve.columns:
                    obs = self.df_di.loc[dt, obs_col]
                    fit = self.df_curve.loc[dt, fit_col]

                    if pd.notna(obs) and pd.notna(fit):
                        row_res[f"residual_{mat}m"] = float(obs - fit)

            residuals.append(row_res)

        df_res = pd.DataFrame(residuals).set_index("Date").sort_index()
        return df_res

    # =========================================================================
    # 2) Métricas globais
    # =========================================================================

    def compute_metrics(self) -> BacktestMetrics:
        """Calcula todas as métricas de performance."""

        df_res = self.compute_residuals()

        # --- Overall metrics ---
        all_residuals = df_res.values.flatten()
        all_residuals_clean = all_residuals[~np.isnan(all_residuals)]

        rmse_overall = float(np.sqrt(np.mean(all_residuals_clean**2)))
        mae_overall = float(np.mean(np.abs(all_residuals_clean)))

        # --- By maturity ---
        common_mats = sorted(set(self.maturities_fit) & set(self.maturities_target))
        rmse_by_mat = {}
        mae_by_mat = {}
        n_obs_by_mat = {}
        coverage_by_mat = {}

        for mat in common_mats:
            col = f"residual_{mat}m"
            if col in df_res.columns:
                res = df_res[col].dropna()
                n_nans = df_res[col].isna().sum()
                n_total = len(df_res)

                if len(res) > 0:
                    rmse_by_mat[f"{mat}m"] = float(np.sqrt(np.mean(res**2)))
                    mae_by_mat[f"{mat}m"] = float(np.mean(np.abs(res)))
                    n_obs_by_mat[f"{mat}m"] = int(len(res))
                    coverage_by_mat[f"{mat}m"] = float((n_total - n_nans) / n_total * 100)

        # --- Beta stability (std dev ao longo do tempo) ---
        beta_stability = {}
        for col in ["beta0", "beta1", "beta2"]:
            if col in self.df_betas.columns:
                beta_vals = self.df_betas[col].dropna()
                if len(beta_vals) > 1:
                    beta_stability[col] = float(beta_vals.std())

        metrics = BacktestMetrics(
            rmse_overall=rmse_overall,
            mae_overall=mae_overall,
            rmse_by_maturity=rmse_by_mat,
            mae_by_maturity=mae_by_mat,
            n_obs_by_maturity=n_obs_by_mat,
            beta_stability=beta_stability,
            coverage_by_maturity=coverage_by_mat,
        )

        return metrics

    def print_metrics(self, metrics: BacktestMetrics | None = None):
        """Imprime relatório das métricas."""
        if metrics is None:
            metrics = self.compute_metrics()

        print("\n" + "=" * 80)
        print("NELSON-SIEGEL BACKTEST REPORT")
        print("=" * 80)

        print(f"\nOverall Performance:")
        print(f"  RMSE: {metrics.rmse_overall:.6f}% a.a.")
        print(f"  MAE:  {metrics.mae_overall:.6f}% a.a.")

        print(f"\nBy Maturity:")
        print(f"  {'Maturity':<12} {'RMSE':<12} {'MAE':<12} {'N_obs':<10} {'Coverage':<10}")
        print(f"  {'-' * 54}")
        for mat in sorted(metrics.rmse_by_maturity.keys(), key=lambda x: int(x[:-1])):
            rmse_val = metrics.rmse_by_maturity.get(mat, np.nan)
            mae_val = metrics.mae_by_maturity.get(mat, np.nan)
            n_obs = metrics.n_obs_by_maturity.get(mat, 0)
            cov = metrics.coverage_by_maturity.get(mat, 0.0)

            print(f"  {mat:<12} {rmse_val:>11.6f}% {mae_val:>11.6f}% {n_obs:>9d} {cov:>8.1f}%")

        print(f"\nBeta Stability (std dev along time):")
        for beta, std_val in metrics.beta_stability.items():
            print(f"  {beta}: {std_val:.6f}")

        print("\n" + "=" * 80 + "\n")

        return metrics

    # =========================================================================
    # 3) Plots
    # =========================================================================

    def plot_betas_and_rmse(self, out_path: str | None = None) -> plt.Figure:
        """Plota betas e qualidade do fit (RMSE) ao longo do tempo."""
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))

        # --- Betas ---
        ax = axes[0]
        for col, color in [("beta0", "#0B1F3A"), ("beta1", "#696969"), ("beta2", "#7C94B1")]:
            if col in self.df_betas.columns:
                ax.plot(self.df_betas.index, self.df_betas[col], label=col, linewidth=1.5, color=color)

        ax.set_title("Nelson-Siegel Betas over Time", fontsize=12, fontweight="bold")
        ax.set_ylabel("Beta value (% a.a.)")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # --- RMSE ---
        ax = axes[1]
        if "rmse_fit" in self.df_betas.columns:
            ax.plot(self.df_betas.index, self.df_betas["rmse_fit"], linewidth=1.5, color="#0B1F3A")

        ax.set_title("Daily Fit Quality (RMSE)", fontsize=12, fontweight="bold")
        ax.set_ylabel("RMSE (% a.a.)")
        ax.set_xlabel("Date")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()

        if out_path:
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            fig.savefig(out_path, dpi=160, bbox_inches="tight")
            print(f"[OK] Plot saved: {out_path}")

        return fig

    def plot_residuals_distribution(self, out_path: str | None = None) -> plt.Figure:
        """Plota distribuição de resíduos por maturidade."""
        df_res = self.compute_residuals()

        # Filter colunas de resíduos
        res_cols = [c for c in df_res.columns if c.startswith("residual_")]
        n_cols = len(res_cols)

        if n_cols == 0:
            print("[WARN] Nenhum resídual para plotar.")
            return None

        ncols = 3
        nrows = (n_cols + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
        axes_flat = axes.flatten() if n_cols > 1 else [axes]

        for idx, col in enumerate(sorted(res_cols)):
            ax = axes_flat[idx]
            data = df_res[col].dropna()

            ax.hist(data, bins=50, alpha=0.7, color="#0B1F3A", edgecolor="black")
            ax.axvline(data.mean(), color="red", linestyle="--", linewidth=2, label=f"Mean={data.mean():.6f}")
            ax.axvline(0, color="black", linestyle="-", linewidth=1, alpha=0.5)

            ax.set_title(f"Residuals: {col}", fontsize=11, fontweight="bold")
            ax.set_xlabel("Residual (% a.a.)")
            ax.set_ylabel("Frequency")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        # Esconde axes não usados
        for idx in range(len(res_cols), len(axes_flat)):
            axes_flat[idx].set_visible(False)

        fig.tight_layout()

        if out_path:
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            fig.savefig(out_path, dpi=160, bbox_inches="tight")
            print(f"[OK] Plot saved: {out_path}")

        return fig

    def plot_residuals_timeseries(self, out_path: str | None = None) -> plt.Figure:
        """Plota série temporal de resíduos por maturidade."""
        df_res = self.compute_residuals()

        res_cols = [c for c in df_res.columns if c.startswith("residual_")]
        n_cols = len(res_cols)

        if n_cols == 0:
            print("[WARN] Nenhum resídual para plotar.")
            return None

        ncols = 3
        nrows = (n_cols + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
        axes_flat = axes.flatten() if n_cols > 1 else [axes]

        colors = ["#0B1F3A", "#244062", "#7C94B1", "#696969"]

        for idx, col in enumerate(sorted(res_cols)):
            ax = axes_flat[idx]
            data = df_res[col]
            color = colors[idx % len(colors)]

            ax.plot(data.index, data.values, linewidth=1, color=color, alpha=0.7)
            ax.axhline(0, color="black", linestyle="-", linewidth=1, alpha=0.5)
            ax.fill_between(data.index, data.values, 0, alpha=0.2, color=color)

            ax.set_title(f"Residuals: {col}", fontsize=11, fontweight="bold")
            ax.set_ylabel("Residual (% a.a.)")
            ax.grid(True, alpha=0.3)

        for idx in range(len(res_cols), len(axes_flat)):
            axes_flat[idx].set_visible(False)

        fig.tight_layout()

        if out_path:
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            fig.savefig(out_path, dpi=160, bbox_inches="tight")
            print(f"[OK] Plot saved: {out_path}")

        return fig

    def plot_curve_fit_key_dates(self, n_dates: int = 6, out_path: str | None = None) -> plt.Figure:
        """
        Plota curva teórica vs observada em datas selecionadas.
        Seleciona as datas com maior spread nas maturidades.
        """
        df_res = self.compute_residuals()

        # Seleciona datas com maior variação de resíduos
        res_std = (df_res.std(axis=1)).dropna()
        if len(res_std) == 0:
            print("[WARN] Sem datas para selecionar.")
            return None

        key_dates = res_std.nlargest(n_dates).index.tolist()

        common_mats = sorted(set(self.maturities_fit) & set(self.maturities_target))

        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        axes_flat = axes.flatten()

        colors_cycle = ["#0B1F3A", "#244062", "#7C94B1", "#696969", "#011242", "#6B6B6B1F"]

        for plot_idx, dt in enumerate(key_dates):
            ax = axes_flat[plot_idx]

            if dt not in self.df_curve.index or dt not in self.df_di.index:
                continue

            mats_arr = np.array(common_mats, dtype=float)

            # Curva teórica
            fit_vals = []
            for mat in common_mats:
                fit_col = f"DI_NS_{mat}m"
                if fit_col in self.df_curve.columns:
                    fit_vals.append(self.df_curve.loc[dt, fit_col])
                else:
                    fit_vals.append(np.nan)

            fit_vals = np.array(fit_vals, dtype=float)

            # Observados
            obs_vals = []
            for mat in common_mats:
                obs_col = f"DI_{mat}m"
                if obs_col in self.df_di.columns:
                    obs_vals.append(self.df_di.loc[dt, obs_col])
                else:
                    obs_vals.append(np.nan)

            obs_vals = np.array(obs_vals, dtype=float)

            # Plot
            color = colors_cycle[plot_idx % len(colors_cycle)]
            ax.plot(mats_arr, fit_vals, "o-", linewidth=2, markersize=6, label="NS Fitted", color=color, alpha=0.9)
            ax.scatter(mats_arr, obs_vals, s=50, marker="^", label="Observed", color=color, alpha=0.6, edgecolors="black")

            ax.set_title(f"{dt.date()}", fontsize=11, fontweight="bold")
            ax.set_xlabel("Maturity (months)")
            ax.set_ylabel("Yield (% a.a.)")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)

        # Esconde axes não usados
        for idx in range(len(key_dates), len(axes_flat)):
            axes_flat[idx].set_visible(False)

        fig.suptitle("Nelson-Siegel Fit: Key Dates (Highest Residual Spread)", fontsize=13, fontweight="bold")
        fig.tight_layout()

        if out_path:
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            fig.savefig(out_path, dpi=160, bbox_inches="tight")
            print(f"[OK] Plot saved: {out_path}")

        return fig

    def plot_rmse_by_maturity(self, out_path: str | None = None) -> plt.Figure:
        """Plota RMSE por maturidade (barplot)."""
        metrics = self.compute_metrics()

        mats = sorted(metrics.rmse_by_maturity.keys(), key=lambda x: int(x[:-1]))
        rmse_vals = [metrics.rmse_by_maturity[m] for m in mats]

        fig, ax = plt.subplots(figsize=(12, 6))

        bars = ax.bar(mats, rmse_vals, color="#0B1F3A", alpha=0.8, edgecolor="black", linewidth=1.2)

        # Destaca o máximo
        max_idx = rmse_vals.index(max(rmse_vals))
        bars[max_idx].set_color("#FF6B6B")

        ax.set_title("RMSE by Maturity", fontsize=13, fontweight="bold")
        ax.set_xlabel("Maturity")
        ax.set_ylabel("RMSE (% a.a.)")
        ax.grid(True, alpha=0.3, axis="y")

        # Adiciona valores nas barras
        for bar, val in zip(bars, rmse_vals):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{val:.4f}", ha="center", va="bottom", fontsize=9)

        fig.tight_layout()

        if out_path:
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            fig.savefig(out_path, dpi=160, bbox_inches="tight")
            print(f"[OK] Plot saved: {out_path}")

        return fig

    # =========================================================================
    # 4) Exportar relatório em Excel
    # =========================================================================

    def export_report_to_excel(self, out_path: str):
        """Exporta relatório completo em Excel (múltiplas abas)."""
        metrics = self.compute_metrics()
        df_res = self.compute_residuals()

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            # --- Aba 1: Summary ---
            summary = pd.DataFrame({
                "Metric": ["RMSE Overall", "MAE Overall"],
                "Value": [metrics.rmse_overall, metrics.mae_overall],
                "Unit": ["% a.a.", "% a.a."]
            })
            summary.to_excel(writer, sheet_name="Summary", index=False)

            # --- Aba 2: By Maturity ---
            mats = sorted(metrics.rmse_by_maturity.keys(), key=lambda x: int(x[:-1]))
            by_mat = pd.DataFrame({
                "Maturity": mats,
                "RMSE": [metrics.rmse_by_maturity[m] for m in mats],
                "MAE": [metrics.mae_by_maturity[m] for m in mats],
                "N_Obs": [metrics.n_obs_by_maturity[m] for m in mats],
                "Coverage_%": [metrics.coverage_by_maturity[m] for m in mats],
            })
            by_mat.to_excel(writer, sheet_name="By_Maturity", index=False)

            # --- Aba 3: Betas ---
            self.df_betas.reset_index().to_excel(writer, sheet_name="Betas", index=False)

            # --- Aba 4: Resíduos ---
            df_res.reset_index().to_excel(writer, sheet_name="Residuals", index=False)

        print(f"[OK] Report exported to: {out_path}")

    # =========================================================================
    # 5) Run completo
    # =========================================================================

    def run(self, out_dir: str, export_excel: bool = True, verbose: bool = True):
        """
        Executa backtesting completo: calcula métricas, gera plots e exporta relatório.

        out_dir: diretório para salvar outputs
        export_excel: se True, exporta relatório em Excel
        verbose: se True, imprime relatório
        """
        os.makedirs(out_dir, exist_ok=True)

        # Calcula e imprime métricas
        metrics = self.compute_metrics()
        if verbose:
            self.print_metrics(metrics)

        # Plots
        plots = {
            "backtest_betas_rmse.png": self.plot_betas_and_rmse,
            "backtest_residuals_dist.png": self.plot_residuals_distribution,
            "backtest_residuals_ts.png": self.plot_residuals_timeseries,
            "backtest_curve_fit_key_dates.png": lambda: self.plot_curve_fit_key_dates(n_dates=6),
            "backtest_rmse_by_maturity.png": self.plot_rmse_by_maturity,
        }

        for filename, plot_func in plots.items():
            try:
                out_path = os.path.join(out_dir, filename)
                plot_func(out_path)
                plt.close("all")
            except Exception as e:
                print(f"[WARN] Erro ao gerar plot {filename}: {e}")

        # Excel
        if export_excel:
            excel_path = os.path.join(out_dir, "backtest_report.xlsx")
            self.export_report_to_excel(excel_path)

        return metrics
