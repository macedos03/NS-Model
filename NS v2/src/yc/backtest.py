from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _load_dataframe(path: Path) -> pd.DataFrame:
	"""Load parquet or CSV file."""
	if not path.exists():
		raise FileNotFoundError(f"Arquivo não encontrado: {path}")
	if path.suffix.lower() == ".parquet":
		return pd.read_parquet(path)
	return pd.read_csv(path)


def _consolidate_forecasts(forecast_paths: dict[str, Path]) -> pd.DataFrame:
	"""Consolidate all forecast csvs into a single dataframe."""
	frames = []
	for model_id, path in forecast_paths.items():
		df = _load_dataframe(path)
		if df.empty:
			continue
		# Ensure model_id matches path name
		if "model_id" in df.columns:
			df = df[df["model_id"] == model_id].copy() if model_id != "rw" else df.copy()
		else:
			df["model_id"] = model_id
		frames.append(df)

	if not frames:
		return pd.DataFrame()
	consolidated = pd.concat(frames, ignore_index=True)
	if "origin_week_ref" in consolidated.columns:
		consolidated["origin_week_ref"] = pd.to_datetime(consolidated["origin_week_ref"], errors="coerce")
	if "target_week_ref" in consolidated.columns:
		consolidated["target_week_ref"] = pd.to_datetime(consolidated["target_week_ref"], errors="coerce")
	sort_cols = [c for c in ["model_id", "origin_week_ref", "horizon_steps", "tau_years"] if c in consolidated.columns]
	return consolidated.sort_values(sort_cols).reset_index(drop=True)


def _filter_to_valid_obs(df: pd.DataFrame) -> pd.DataFrame:
	"""Keep only rows with valid observed yields."""
	before = len(df)
	df = df.dropna(subset=["yield_obs"]).copy()
	after = len(df)
	print(f"Obs com yield_obs válido: {before} → {after} ({100*after/before:.1f}%)")
	return df


def _build_coverage_table(df: pd.DataFrame) -> pd.DataFrame:
	"""Tabela de cobertura por (model, horizon, tau)."""
	required_cols = ["model_id", "horizon_steps", "tau_years"]
	if not all(col in df.columns for col in required_cols):
		return pd.DataFrame()

	rows = []
	for model_id in df["model_id"].unique():
		for h in sorted(df["horizon_steps"].unique()):
			for tau in sorted(df["tau_years"].unique()):
				subset = df[
					(df["model_id"] == model_id)
					& (df["horizon_steps"] == h)
					& (df["tau_years"] == tau)
				]
				rows.append(
					{
						"model_id": model_id,
						"horizon_steps": int(h),
						"tau_years": float(tau),
						"n_forecasts": int(len(subset)),
						"n_with_obs": int(subset["yield_obs"].notna().sum()),
						"coverage_pct": 100.0 * subset["yield_obs"].notna().mean() if len(subset) > 0 else 0.0,
					}
				)
	return pd.DataFrame(rows).sort_values(["model_id", "horizon_steps", "tau_years"]).reset_index(drop=True)


def _build_common_support_set(df: pd.DataFrame) -> pd.DataFrame:
	"""Filter to common support (same origin/target/h/tau across all models with valid obs)."""
	required_cols = ["model_id", "origin_week_ref", "target_week_ref", "horizon_steps", "tau_years", "yield_obs"]
	if not all(col in df.columns for col in required_cols):
		raise ValueError(f"Faltam colunas para common support. Cols presentes: {df.columns.tolist()}")

	df_valid = df.dropna(subset=["yield_obs"]).copy()
	models = df_valid["model_id"].unique()

	# For each (origin, target, h, tau), keep only if ALL models have it
	common_keys = None
	for model in models:
		model_keys = set(
			df_valid[df_valid["model_id"] == model][
				["origin_week_ref", "target_week_ref", "horizon_steps", "tau_years"]
			].apply(tuple, axis=1)
		)
		if common_keys is None:
			common_keys = model_keys
		else:
			common_keys = common_keys.intersection(model_keys)

	if not common_keys:
		print("⚠️ Aviso: nenhuma chave em comum entre todos os modelos no common support.")
		return pd.DataFrame()

	def check_key(row):
		key = (row["origin_week_ref"], row["target_week_ref"], row["horizon_steps"], row["tau_years"])
		return key in common_keys

	df_cs = df_valid[df_valid.apply(check_key, axis=1)].copy()
	print(f"Common support: {len(df_valid)} obs válidas → {len(df_cs)} em common support ({100*len(df_cs)/len(df_valid) if df_valid.shape[0] > 0 else 0:.1f}%)")
	return df_cs


def _calc_mae_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
	"""Calculate error metrics by (model, horizon, tau)."""
	tables = {}

	for metric_name in ["mae", "rmse", "bias"]:
		rows = []
		for model in sorted(df["model_id"].unique()):
			for h in sorted(df["horizon_steps"].unique()):
				row_data = {"model_id": model, "horizon_steps": int(h)}
				for tau in sorted(df["tau_years"].unique()):
					subset = df[
						(df["model_id"] == model)
						& (df["horizon_steps"] == h)
						& (df["tau_years"] == tau)
					]
					if subset.empty:
						row_data[f"tau_{int(tau)}y"] = np.nan
					else:
						if metric_name == "mae":
							val = float(subset["abs_error"].mean())
						elif metric_name == "rmse":
							val = float(np.sqrt((subset["error"] ** 2).mean()))
						else:  # bias
							val = float(subset["error"].mean())
						row_data[f"tau_{int(tau)}y"] = val
				rows.append(row_data)
		tables[metric_name] = pd.DataFrame(rows)

	return tables


def _calc_relmae_vs_rw(df: pd.DataFrame) -> pd.DataFrame:
	"""Calculate MAE relative to RW (MAE_model / MAE_RW)."""
	rw_mae = {}
	for h in sorted(df["horizon_steps"].unique()):
		for tau in sorted(df["tau_years"].unique()):
			rw_subset = df[
				(df["model_id"] == "rw_yield")
				& (df["horizon_steps"] == h)
				& (df["tau_years"] == tau)
			]
			if not rw_subset.empty:
				rw_mae[(h, tau)] = float(rw_subset["abs_error"].mean())

	rows = []
	for model in sorted(df["model_id"].unique()):
		if model == "rw_yield":
			continue
		row_data = {"model_id": model}
		for h in sorted(df["horizon_steps"].unique()):
			for tau in sorted(df["tau_years"].unique()):
				subset = df[
					(df["model_id"] == model)
					& (df["horizon_steps"] == h)
					& (df["tau_years"] == tau)
				]
				if subset.empty or (h, tau) not in rw_mae or rw_mae[(h, tau)] <= 1e-10:
					val = np.nan
				else:
					model_mae = float(subset["abs_error"].mean())
					val = model_mae / rw_mae[(h, tau)]
				row_data[f"tau_{int(tau)}y"] = val
		rows.append(row_data)

	return pd.DataFrame(rows) if rows else pd.DataFrame()


def _calc_winrate_vs_rw(df: pd.DataFrame) -> pd.DataFrame:
	"""Calculate % of forecasts where model beats RW."""
	# Build a pivot with model, origin, target, h, tau as index
	pivot_abs_error = df.pivot_table(
		index=["origin_week_ref", "target_week_ref", "horizon_steps", "tau_years"],
		columns="model_id",
		values="abs_error",
		aggfunc="first",
	)

	if "rw_yield" not in pivot_abs_error.columns:
		print("⚠️ RW não encontrado para cálculo de win rate.")
		return pd.DataFrame()

	rows = []
	rw_col = pivot_abs_error["rw_yield"]
	for model in pivot_abs_error.columns:
		if model == "rw_yield":
			continue
		model_col = pivot_abs_error[model]
		# Drop NaN pairs
		valid_mask = rw_col.notna() & model_col.notna()
		if valid_mask.sum() == 0:
			win_rate = np.nan
			n_obs = 0
		else:
			wins = (model_col[valid_mask] < rw_col[valid_mask]).sum()
			n_obs = int(valid_mask.sum())
			win_rate = 100.0 * wins / n_obs if n_obs > 0 else 0.0

		rows.append(
			{
				"model_id": model,
				"winrate_vs_rw_pct": win_rate,
				"n_obs": n_obs,
			}
		)
	return pd.DataFrame(rows)


def _format_table_for_display(df: pd.DataFrame, metric_name: str = "mae", decimals: int = 3) -> pd.DataFrame:
	"""Format MAE/RelMAE table for nice display."""
	tau_cols = [col for col in df.columns if col.startswith("tau_")]
	display_df = df[["model_id"] + tau_cols].copy()
	display_df.columns = ["Model"] + [col.replace("tau_", "").upper() for col in tau_cols]

	for col in display_df.columns[1:]:
		if metric_name == "relmae":
			display_df[col] = display_df[col].apply(lambda x: f"{x:.{decimals}f}" if pd.notna(x) else "-")
		else:
			display_df[col] = display_df[col].apply(lambda x: f"{x:.{decimals}f}" if pd.notna(x) else "-")
	return display_df


def _build_summary_stats(df: pd.DataFrame) -> dict:
	"""Build summary statistics for overall performance."""
	stats = {
		"n_total_obs": int(len(df)),
		"n_models": int(df["model_id"].nunique()),
		"models": sorted(df["model_id"].unique()),
		"horizons": sorted(df["horizon_steps"].unique()),
		"maturities": sorted(df["tau_years"].unique()),
		"date_range": {
			"start": str(df["origin_week_ref"].min().date()) if "origin_week_ref" in df.columns else None,
			"end": str(df["origin_week_ref"].max().date()) if "origin_week_ref" in df.columns else None,
		},
	}
	return stats


def run_evaluation(
	forecast_dir: Path,
	output_dir: Path,
	use_common_support: bool = True,
) -> dict:
	"""Run OOS evaluation pipeline."""
	output_dir.mkdir(parents=True, exist_ok=True)

	# Discover forecast files
	forecast_paths = {}
	for fpath in forecast_dir.glob("forecast_yields_*.csv"):
		model_id = fpath.stem.replace("forecast_yields_", "")
		if model_id and model_id != "rw":  # RW might be named differently
			forecast_paths[model_id] = fpath

	# Also check for RW
	rw_path = forecast_dir / "forecast_yields_rw.csv"
	if rw_path.exists():
		forecast_paths["rw_yield"] = rw_path

	if not forecast_paths:
		raise FileNotFoundError(f"Nenhum arquivo forecast_yields_* encontrado em {forecast_dir}")

	print(f"Carregando forecasts: {list(forecast_paths.keys())}")

	# Consolidate
	df = _consolidate_forecasts(forecast_paths)
	if df.empty:
		raise ValueError("Consolidação de forecasts resultou em dataframe vazio.")

	print(f"Total de linhas consolidadas: {len(df)}")

	# Filter to valid obs
	df_valid = _filter_to_valid_obs(df)

	# Coverage before common support
	coverage_before = _build_coverage_table(df_valid)
	coverage_before.to_csv(output_dir / "eval_oos_coverage_before_cs.csv", index=False)

	# Common support
	if use_common_support:
		df_eval = _build_common_support_set(df_valid)
		if df_eval.empty:
			raise ValueError("Common support resultou em dataset vazio.")
	else:
		df_eval = df_valid.copy()

	# Coverage after
	coverage_after = _build_coverage_table(df_eval)
	coverage_after.to_csv(output_dir / "eval_oos_coverage.csv", index=False)

	# MAE tables
	mae_tables = _calc_mae_tables(df_eval)
	mae_tables["mae"].to_csv(output_dir / "eval_oos_mae.csv", index=False)
	mae_tables["rmse"].to_csv(output_dir / "eval_oos_rmse.csv", index=False)
	mae_tables["bias"].to_csv(output_dir / "eval_oos_bias.csv", index=False)

	# RelMAE vs RW
	relmae = _calc_relmae_vs_rw(df_eval)
	if not relmae.empty:
		relmae.to_csv(output_dir / "eval_oos_relmae_vs_rw.csv", index=False)

	# Win rate vs RW
	winrate = _calc_winrate_vs_rw(df_eval)
	if not winrate.empty:
		winrate.to_csv(output_dir / "eval_oos_winrate_vs_rw.csv", index=False)

	# Summary stats
	summary = _build_summary_stats(df_eval)
	(output_dir / "eval_oos_summary.json").write_text(
		json.dumps(summary, indent=2, ensure_ascii=False, default=str),
		encoding="utf-8",
	)

	# Save consolidated panel
	df_eval.to_csv(output_dir / "eval_oos_panel_long.csv", index=False)

	# Display-friendly versions (formatted strings)
	mae_display = _format_table_for_display(mae_tables["mae"], metric_name="mae", decimals=4)
	mae_display.to_csv(output_dir / "eval_oos_mae_display.csv", index=False)
	rmse_display = _format_table_for_display(mae_tables["rmse"], metric_name="rmse", decimals=4)
	rmse_display.to_csv(output_dir / "eval_oos_rmse_display.csv", index=False)

	if not relmae.empty:
		relmae_display = _format_table_for_display(relmae, metric_name="relmae", decimals=3)
		relmae_display.to_csv(output_dir / "eval_oos_relmae_display.csv", index=False)

	return {
		"consolidated_df": df_eval,
		"mae_tables": mae_tables,
		"relmae": relmae,
		"winrate": winrate,
		"coverage": coverage_after,
		"summary": summary,
	}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Passo 8: Avaliação OOS de modelos de previsão de curva de juros")
	parser.add_argument(
		"--forecast-dir",
		type=Path,
		default=Path(r"Q:\Gabriel de Macedo\Política Monetária\NS Model\NS v2\data"),
		help="Diretório contendo forecast_yields_*.csv",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path(r"Q:\Gabriel de Macedo\Política Monetária\NS Model\NS v2\reports\evaluation"),
		help="Diretório para outputs da avaliação",
	)
	parser.add_argument(
		"--use-common-support",
		type=lambda x: str(x).strip().lower() in {"1", "true", "t", "yes", "y"},
		default=True,
		help="Se True, compara modelos em common support (mesmes obs para todos).",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	results = run_evaluation(
		forecast_dir=args.forecast_dir,
		output_dir=args.output_dir,
		use_common_support=args.use_common_support,
	)

	print("\n" + "=" * 70)
	print("Passo 8 (Avaliação OOS) concluído.")
	print("=" * 70)
	print(f"\nArquivos gerados em: {args.output_dir}")
	print("  - eval_oos_panel_long.csv (painel consolidado)")
	print("  - eval_oos_coverage.csv (cobertura por modelo/h/tau)")
	print("  - eval_oos_mae.csv (MAE absoluto)")
	print("  - eval_oos_rmse.csv (RMSE)")
	print("  - eval_oos_bias.csv (bias médio)")
	print("  - eval_oos_relmae_vs_rw.csv (MAE relativo ao RW)")
	print("  - eval_oos_winrate_vs_rw.csv (% ganho vs RW)")
	print("  - eval_oos_summary.json (resumo da avaliação)")

	summary = results["summary"]
	print(f"\nResumo da avaliação:")
	print(f"  - Total de obs: {summary['n_total_obs']}")
	print(f"  - Modelos: {', '.join(summary['models'])}")
	print(f"  - Horizontes: {sorted(summary['horizons'])}")
	print(f"  - Maturidades: {sorted(summary['maturities'])}")
	print(f"  - Período: {summary['date_range']['start']} → {summary['date_range']['end']}")

	if not results["mae_tables"]["mae"].empty:
		print("\nMAE por modelo (primeiras linhas):")
		print(results["mae_tables"]["mae"].head(6).to_string(index=False))

	if not results["mae_tables"]["rmse"].empty:
		print("\nRMSE por modelo (primeiras linhas):")
		print(results["mae_tables"]["rmse"].head(6).to_string(index=False))

	if not results["relmae"].empty:
		print("\nRelMAE vs RW (primeiras linhas):")
		print(results["relmae"].head(6).to_string(index=False))

	if not results["winrate"].empty:
		print("\nWin rate vs RW:")
		print(results["winrate"].to_string(index=False))


if __name__ == "__main__":
	main()
