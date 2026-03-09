from __future__ import annotations

from pathlib import Path
import importlib.util
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _load_sibling(module_name: str, filename: str):
    here = Path(__file__).resolve().parent
    path = here / filename
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


# ----------------------------
# Imports compatíveis:
# - como pacote: from .data import ...
# - como script: carrega arquivos irmãos via importlib
# ----------------------------
if __package__ in (None, ""):
    _data_mod = _load_sibling("yc_data", "data.py")
    _model_mod = _load_sibling("yc_modeling", "modeling.py")
    _exp_mod = _load_sibling("yc_export", "export.py")
    _bt_mod = _load_sibling("yc_backtest", "backtest.py")

    data = _data_mod.data
    PCAArtifacts = _data_mod.PCAArtifacts
    modeling = _model_mod.modeling
    _fatores = _exp_mod._fatores
    Backtest = _bt_mod.Backtest
    BacktestMetrics = _bt_mod.BacktestMetrics
else:
    from .data import data, PCAArtifacts
    from .modeling import modeling
    from .export import _fatores
    from .backtest import Backtest, BacktestMetrics

# =========================
# BTG-like palette (se tiver os hex oficiais, substitua aqui)
# =========================
BTG_BLUE = "#0B1F3A"
BTG_MORE_BLUE = "#244062"
BTG_SLIGHT_BLUE = "#7C94B1"
BTG_GRAY = "#696969"

def _ensure_excel_readable_or_reset(path: str):
    """
    Se o arquivo existir mas estiver inválido/corrompido (ou sem engine detectável),
    remove para evitar erro no export.py ao ler.
    """
    if not path:
        return
    if os.path.exists(path):
        try:
            pd.read_excel(path, engine="openpyxl")
        except Exception:
            try:
                os.remove(path)
                print(f"[WARN] Arquivo Excel inválido removido e será recriado: {path}")
            except Exception as e:
                print(f"[WARN] Não consegui remover arquivo Excel inválido ({path}): {e}")


def exportar_cds_fatores(
    db_path: str | None = None,
    cols: tuple[str, str] = ("CDS_dom", "CDS_glob"),
    output_path: str | None = None,
    **kwargs,
):
    """
    Exporta CDS_dom / CDS_glob usando export._fatores.salvar_fator_em_excel.
    Faz hardening: se output_path existir e estiver inválido, remove e recria.
    """
    cds = data.decompor_cds(db_path=db_path) if db_path else data.decompor_cds()

    # define output_path (se não passar, mantém o default do export.py)
    if output_path is not None:
        _ensure_excel_readable_or_reset(output_path)
        _fatores.salvar_fator_em_excel(cds, list(cols), output_path=output_path, **kwargs)
    else:
        # tenta resetar o default também (se existir no export.py)
        try:
            default_out = getattr(_fatores, "__dict__", {}).get("fatores_path", None)
            if default_out:
                _ensure_excel_readable_or_reset(default_out)
        except Exception:
            pass
        _fatores.salvar_fator_em_excel(cds, list(cols), **kwargs)

    return cds


def export_ns_outputs_and_plots(
    df_di: pd.DataFrame,
    df_betas: pd.DataFrame,
    df_curve: pd.DataFrame,
    out_xlsx: str,
    out_dir: str,
    maturities_fit_months: list[int] | None = None,
):
    """
    Exporta outputs do Nelson-Siegel (betas + curva) para Excel e gera plots
    com heurísticas típicas de interpretação:
      - beta0 (level), beta1 (slope), beta2 (curvature)
      - datas-chave: última, maior choque de slope, maior choque de curvature, e ~1 ano atrás
      - qualidade do fit: rmse_fit e n_points_fit

    df_di: DI observados (colunas DI_{m}m) em % a.a.
    df_betas: saída do fit (beta0,beta1,beta2,rmse_fit,n_points_fit,...)
    df_curve: curva gerada (colunas DI_NS_{m}m)
    """
    os.makedirs(out_dir, exist_ok=True)

    # -------------------------
    # Excel (3 abas)
    # -------------------------
    summary = pd.DataFrame(index=[0])
    if "rmse_fit" in df_betas.columns:
        summary["rmse_mean"] = float(df_betas["rmse_fit"].mean())
        summary["rmse_p50"] = float(df_betas["rmse_fit"].median())
        summary["rmse_p90"] = float(df_betas["rmse_fit"].quantile(0.90))
        summary["rmse_p99"] = float(df_betas["rmse_fit"].quantile(0.99))

    out_xlsx_path = Path(out_xlsx)
    out_xlsx_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(str(out_xlsx_path), engine="openpyxl") as writer:
        df_betas.reset_index().to_excel(writer, sheet_name="NS_Betas", index=False)
        df_curve.reset_index().to_excel(writer, sheet_name="NS_Curve", index=False)
        summary.to_excel(writer, sheet_name="NS_Summary", index=False)

    # -------------------------
    # Helper: parse months grid
    # -------------------------
    curve_cols = [c for c in df_curve.columns if c.startswith("DI_NS_") and c.endswith("m")]
    if not curve_cols:
        raise ValueError("df_curve não tem colunas DI_NS_{m}m.")

    def _parse_months(cols: list[str]) -> np.ndarray:
        months = []
        for c in cols:
            mm = int(c.replace("DI_NS_", "").replace("m", ""))
            months.append(mm)
        return np.array(months, dtype=int)

    months_grid = _parse_months(curve_cols)
    order = np.argsort(months_grid)
    months_grid = months_grid[order]
    curve_cols = [curve_cols[i] for i in order]

    # -------------------------
    # Datas-chave (heurística)
    # -------------------------
    betas = df_betas.copy().sort_index()
    betas = betas.dropna(subset=["beta0", "beta1", "beta2"], how="any")
    if betas.empty:
        raise ValueError("df_betas está vazio após dropna(beta0,beta1,beta2).")

    last_dt = betas.index.max()
    d_beta1 = betas["beta1"].diff()
    d_beta2 = betas["beta2"].diff()

    dt_slope_shock = d_beta1.abs().idxmax() if d_beta1.dropna().size else last_dt
    dt_curv_shock = d_beta2.abs().idxmax() if d_beta2.dropna().size else last_dt

    target_1y = last_dt - pd.Timedelta(days=365)
    dt_1y = betas.index[betas.index.get_indexer([target_1y], method="nearest")[0]]

    key_dates = []
    for d in [last_dt, dt_slope_shock, dt_curv_shock, dt_1y]:
        if d not in key_dates:
            key_dates.append(d)

    # maturities fit (pontos observados)
    if maturities_fit_months is None:
        maturities_fit_months = sorted([
            int(c.replace("DI_", "").replace("m", ""))
            for c in df_di.columns
            if c.startswith("DI_") and c.endswith("m")
        ])

    # -------------------------
    # Plot 1 — Betas no tempo
    # -------------------------
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(betas.index, betas["beta0"], label="beta0 (Level)", linewidth=1.6, color=BTG_BLUE)
    ax.plot(betas.index, betas["beta1"], label="beta1 (Slope)", linewidth=1.6, color=BTG_GRAY)
    ax.plot(betas.index, betas["beta2"], label="beta2 (Curvature)", linewidth=1.6, color=BTG_SLIGHT_BLUE)

    ax.set_title("Nelson-Siegel — Betas (Level / Slope / Curvature)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Beta value (em % a.a.)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ns_betas_timeseries.png"), dpi=160)
    plt.close(fig)

    # -------------------------
    # Plot 2 — Curvas em datas-chave + pontos observados
    # -------------------------
    fig = plt.figure()
    ax = fig.add_subplot(111)

    date_styles = [
        (BTG_BLUE, 2.2, 1.0, "Última"),
        (BTG_MORE_BLUE, 1.8, 0.95, "Maior choque Slope"),
        (BTG_GRAY, 1.8, 0.95, "Maior choque Curvature"),
    ]

    for (dt, (col, lw, alpha, tag)) in zip(key_dates, date_styles):
        if dt not in df_curve.index:
            continue

        y_curve = df_curve.loc[dt, curve_cols].astype(float).values
        ax.plot(
            months_grid, y_curve,
            label=f"{tag}: {dt.date()}",
            linewidth=lw, alpha=alpha, color=col
        )

        if dt in df_di.index:
            obs_cols = [f"DI_{m}m" for m in maturities_fit_months if f"DI_{m}m" in df_di.columns]
            obs = df_di.loc[dt, obs_cols].dropna()
            if len(obs) > 0:
                obs_m = np.array([int(c.replace("DI_", "").replace("m", "")) for c in obs.index], dtype=int)
                ax.scatter(obs_m, obs.values.astype(float), s=18, color=col, alpha=0.85)

    ax.set_title("Nelson-Siegel — Curva DI (datas-chave) + vértices observados")
    ax.set_xlabel("Maturidade (meses)")
    ax.set_ylabel("Yield (% a.a.)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ns_curves_key_dates.png"), dpi=160)
    plt.close(fig)

    # -------------------------
    # Plot 3 — Qualidade do fit (RMSE) e robustez (n_points)
    # -------------------------
    fig = plt.figure()
    ax = fig.add_subplot(111)

    if "rmse_fit" in betas.columns:
        ax.plot(betas.index, betas["rmse_fit"], label="RMSE fit", linewidth=1.5, color=BTG_BLUE)

    ax.set_title("Nelson-Siegel — Qualidade do ajuste (RMSE)")
    ax.set_xlabel("Date")
    ax.set_ylabel("RMSE (% a.a.)")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ns_fit_quality.png"), dpi=160)
    plt.close(fig)

    print(f"[OK] Excel salvo em: {str(out_xlsx_path)}")
    print(f"[OK] Plots salvos em: {out_dir}")
    return str(out_xlsx_path)

def main():
    # 1) CDS
    cds_out = "Q:/Gabriel de Macedo/Política Monetária/NS Model/data/pca_fatores.xlsx"
    cds = exportar_cds_fatores(output_path=cds_out)
    print("[OK] CDS fatores exportados:", list(cds.columns))

    # 2) DI swaps
    df_di = data.read_di_swaps(
        path="Q:/Gabriel de Macedo/Política Monetária/NS Model/data/di_swaps.xlsx",
        sheet_name=0,
        date_col="Dates",
    )
    print(df_di.head())
    print("Início DI_48m:", df_di["DI_48m"].dropna().index.min() if "DI_48m" in df_di else None)
    print("Início DI_60m:", df_di["DI_60m"].dropna().index.min() if "DI_60m" in df_di else None)
    print(df_di.notna().sum())

    # 3) Nelson-Siegel (Caminho 1)
    maturities_fit = [1, 3, 5, 6, 12, 14, 24, 36, 48, 60]
    maturities_target = [1, 2, 3, 4, 6, 9, 12, 18, 24, 30, 36, 48]

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

    print("[OK] Betas:", df_betas.shape, "Curve:", df_curve.shape)
    print(df_betas.head())
    print(df_curve.head())

    # 4) Export Excel + plots (BTG-like)
    ns_out_xlsx = "Q:/Gabriel de Macedo/Política Monetária/NS Model/data/ns_outputs.xlsx"
    ns_out_dir = "Q:/Gabriel de Macedo/Política Monetária/NS Model/reports/ns"

    export_ns_outputs_and_plots(
        df_di=df_di,
        df_betas=df_betas,
        df_curve=df_curve,
        out_xlsx=ns_out_xlsx,
        out_dir=ns_out_dir,
        maturities_fit_months=maturities_fit,
    )

    # 5) BACKTESTING
    print("\n" + "="*80)
    print("INICIANDO BACKTESTING")
    print("="*80 + "\n")

    backtest_out_dir = "Q:/Gabriel de Macedo/Política Monetária/NS Model/reports/backtest"

    bt = Backtest(
        df_di=df_di,
        df_betas=df_betas,
        df_curve=df_curve,
        maturities_fit_months=maturities_fit,
        maturities_target_months=maturities_target,
    )

    metrics = bt.run(out_dir=backtest_out_dir, export_excel=True, verbose=True)

__all__ = [
    "data",
    "PCAArtifacts",
    "modeling",
    "exportar_cds_fatores",
    "export_ns_outputs_and_plots",
    "Backtest",
    "BacktestMetrics",
    "main",
]

if __name__ == "__main__":
    main()