"""
Demonstração de uso do Backtesting do modelo Nelson-Siegel.

Este script exemplifica:
1. Carregamento de dados de DI swaps
2. Ajuste do modelo Nelson-Siegel
3. Execução do backtesting completo
4. Visualização de resultados
"""

import sys
from pathlib import Path

# Permite importar do pacote yc
sys.path.insert(0, str(Path(__file__).parent))

from yc import data, modeling, Backtest


def main():
    print("="*80)
    print("NELSON-SIEGEL BACKTESTING DEMO")
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
        lam=0.7308,  # parâmetro clássico do NS
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
    # 4) Resumo de resultados
    # =========================================================================
    print("[4/4] Summary of results")
    print(f"  Overall RMSE: {metrics.rmse_overall:.6f}% a.a.")
    print(f"  Overall MAE:  {metrics.mae_overall:.6f}% a.a.")
    print(f"\n  Outputs saved to:")
    print(f"  - Excel report: {backtest_out_dir}/backtest_report.xlsx")
    print(f"  - Plots: {backtest_out_dir}/backtest_*.png")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
