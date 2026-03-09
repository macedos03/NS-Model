from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


LAMBDA_NS_DEFAULT = 0.266
CURVE_MAP_DEFAULT = {
	"bbg_bz_bond_1y": 1.0,
	"bbg_bz_bond_3y": 3.0,
	"bbg_bz_bond_5y": 5.0,
	"bbg_bz_bond_10y": 10.0,
}


def ns_loadings(tau: np.ndarray, lambda_: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	tau = np.asarray(tau, dtype=float)
	if np.any(tau <= 0):
		raise ValueError("Todas as maturidades tau devem ser > 0")
	if lambda_ <= 0:
		raise ValueError("lambda deve ser > 0")

	x = lambda_ * tau
	l2 = (1.0 - np.exp(-x)) / x
	l3 = l2 - np.exp(-x)
	X = np.column_stack([np.ones_like(tau), l2, l3])
	return l2, l3, X


def estimate_ns_betas_ols(y: np.ndarray, X: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
	y = np.asarray(y, dtype=float)
	beta, _residuals, rank, _svals = np.linalg.lstsq(X, y, rcond=None)

	y_hat = X @ beta
	resid = y - y_hat

	out = {
		"beta1_level": float(beta[0]),
		"beta2_slope": float(beta[1]),
		"beta3_curvature": float(beta[2]),
		"fit_mae": float(np.mean(np.abs(resid))),
		"fit_rmse": float(np.sqrt(np.mean(resid**2))),
		"fit_max_abs_err": float(np.max(np.abs(resid))),
		"rank_X": int(rank),
	}
	return out, y_hat, resid


def _parse_curve_map(text: str) -> dict[str, float]:
	parts = [p.strip() for p in text.split(",") if p.strip()]
	mapping: dict[str, float] = {}
	for item in parts:
		if ":" not in item:
			raise ValueError(f"Formato inválido em --curve-map: {item}")
		col, tau = item.split(":", 1)
		mapping[col.strip()] = float(tau)
	if len(mapping) < 3:
		raise ValueError("--curve-map precisa ter pelo menos 3 maturidades")
	return mapping


def _check_curve_columns(df: pd.DataFrame, curve_map: dict[str, float]) -> None:
	missing = [col for col in curve_map if col not in df.columns]
	if missing:
		raise ValueError(f"Colunas de curva ausentes na base semanal: {missing}")


def _unit_profile(df: pd.DataFrame, curve_cols: list[str]) -> dict[str, str]:
	profile = {}
	for col in curve_cols:
		s = df[col].dropna()
		if s.empty:
			profile[col] = "no_data"
			continue
		med = float(s.median())
		profile[col] = "decimal_like" if med < 1.0 else "percent_like"
	return profile


def _save_parquet_with_csv_fallback(df: pd.DataFrame, parquet_path: Path) -> Path:
	try:
		df.to_parquet(parquet_path, index=False)
		return parquet_path
	except Exception:
		csv_path = parquet_path.with_suffix(".csv")
		df.to_csv(csv_path, index=False)
		return csv_path


def _make_plots(ns_factors: pd.DataFrame, ns_fit: pd.DataFrame, reports_dir: Path) -> dict[str, Path]:
	plot_paths: dict[str, Path] = {}
	try:
		import matplotlib.pyplot as plt
	except Exception:
		return plot_paths

	reports_dir.mkdir(parents=True, exist_ok=True)

	# 1) Betas no tempo
	fig, ax = plt.subplots(figsize=(11, 5))
	series_style = [
		("beta1_level", "#0B1F3A"),
		("beta2_slope", "#696969"),
		("beta3_curvature", "#7C94B1"),
	]
	for col, color in series_style:
		ax.plot(ns_factors["week_ref"], ns_factors[col], label=col, color=color)
	ax.set_title("Nelson-Siegel betas (weekly)")
	ax.legend()
	ax.grid(alpha=0.25)
	fig.tight_layout()
	p1 = reports_dir / "ns_betas_timeseries.png"
	fig.savefig(p1, dpi=150)
	plt.close(fig)
	plot_paths["betas_timeseries"] = p1

	# 2) Observado vs fitted (1Y e 10Y)
	fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
	for ax, tenor in zip(axes, ["1y", "10y"]):
		obs_col = f"y_obs_{tenor}"
		fit_col = f"y_fit_{tenor}"
		if obs_col in ns_fit.columns and fit_col in ns_fit.columns:
			ax.plot(ns_fit["week_ref"], ns_fit[obs_col], label=f"observed {tenor}")
			ax.plot(ns_fit["week_ref"], ns_fit[fit_col], label=f"fitted {tenor}", alpha=0.85)
			ax.grid(alpha=0.25)
			ax.legend()
	axes[0].set_title("Observed vs fitted yields (NS)")
	fig.tight_layout()
	p2 = reports_dir / "ns_observed_vs_fitted_1y_10y.png"
	fig.savefig(p2, dpi=150)
	plt.close(fig)
	plot_paths["observed_vs_fitted"] = p2

	# 3) Distribuição do fit MAE
	fig, ax = plt.subplots(figsize=(8, 4.5))
	ax.hist(ns_fit["fit_mae"].dropna(), bins=30)
	ax.set_title("Distribution of weekly NS fit MAE")
	ax.set_xlabel("fit_mae")
	ax.grid(alpha=0.25)
	fig.tight_layout()
	p3 = reports_dir / "ns_fit_mae_hist.png"
	fig.savefig(p3, dpi=150)
	plt.close(fig)
	plot_paths["fit_mae_hist"] = p3

	return plot_paths


def run_ns_step(
	panel_weekly: pd.DataFrame,
	curve_map: dict[str, float],
	lambda_ns: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
	curve_cols = list(curve_map.keys())
	_check_curve_columns(panel_weekly, curve_map)

	df = panel_weekly[["week_ref"] + curve_cols].copy()
	df["week_ref"] = pd.to_datetime(df["week_ref"], errors="coerce")
	df = df.dropna(subset=["week_ref"]).sort_values("week_ref")

	unit_profile = _unit_profile(df, curve_cols)
	unit_labels = {v for v in unit_profile.values() if v != "no_data"}
	if len(unit_labels) > 1:
		raise ValueError(
			"Escala inconsistente entre maturidades (mistura de decimal e %). "
			f"Perfil detectado: {unit_profile}"
		)

	tau = np.array([curve_map[c] for c in curve_cols], dtype=float)
	_, _, X = ns_loadings(tau=tau, lambda_=lambda_ns)

	n_before = len(df)
	df = df.dropna(subset=curve_cols).reset_index(drop=True)
	n_after = len(df)

	factors_rows = []
	diag_rows = []
	fitted_rows = []

	for _, row in df.iterrows():
		y = row[curve_cols].to_numpy(dtype=float)
		est, y_hat, resid = estimate_ns_betas_ols(y, X)

		factors_rows.append(
			{
				"week_ref": row["week_ref"],
				"beta1_level": est["beta1_level"],
				"beta2_slope": est["beta2_slope"],
				"beta3_curvature": est["beta3_curvature"],
			}
		)

		diag_row = {
			"week_ref": row["week_ref"],
			"fit_mae": est["fit_mae"],
			"fit_rmse": est["fit_rmse"],
			"fit_max_abs_err": est["fit_max_abs_err"],
			"rank_X": est["rank_X"],
			"n_maturities_used": int(len(curve_cols)),
		}

		fitted_row = {"week_ref": row["week_ref"]}
		for i, col in enumerate(curve_cols):
			tenor_label = str(curve_map[col]).replace(".0", "")
			diag_row[f"y_obs_{tenor_label}y"] = float(y[i])
			diag_row[f"y_fit_{tenor_label}y"] = float(y_hat[i])
			diag_row[f"resid_{tenor_label}y"] = float(resid[i])

			fitted_row[f"y_obs_{tenor_label}y"] = float(y[i])
			fitted_row[f"y_fit_{tenor_label}y"] = float(y_hat[i])
			fitted_row[f"resid_{tenor_label}y"] = float(resid[i])

		diag_rows.append(diag_row)
		fitted_rows.append(fitted_row)

	ns_factors = pd.DataFrame(factors_rows).sort_values("week_ref")
	ns_diag = pd.DataFrame(diag_rows).sort_values("week_ref")
	ns_fitted = pd.DataFrame(fitted_rows).sort_values("week_ref")

	run_summary = {
		"lambda_ns": float(lambda_ns),
		"n_weeks_input": int(n_before),
		"n_weeks_estimated": int(n_after),
		"n_weeks_dropped_incomplete_curve": int(n_before - n_after),
		"curve_columns": curve_cols,
		"tau_years": [float(x) for x in tau],
		"unit_profile": unit_profile,
		"fit_mae_mean": float(ns_diag["fit_mae"].mean()) if not ns_diag.empty else None,
		"fit_rmse_mean": float(ns_diag["fit_rmse"].mean()) if not ns_diag.empty else None,
	}

	return ns_factors, ns_diag, ns_fitted, run_summary


def run_pipeline(
	input_file: Path,
	output_dir: Path,
	reports_dir: Path,
	lambda_ns: float,
	curve_map: dict[str, float],
) -> dict:
	if not input_file.exists():
		raise FileNotFoundError(f"Arquivo não encontrado: {input_file}")

	if input_file.suffix.lower() == ".parquet":
		panel_weekly = pd.read_parquet(input_file)
	else:
		panel_weekly = pd.read_csv(input_file)

	ns_factors, ns_diag, ns_fitted, run_summary = run_ns_step(
		panel_weekly=panel_weekly,
		curve_map=curve_map,
		lambda_ns=lambda_ns,
	)

	output_dir.mkdir(parents=True, exist_ok=True)
	reports_dir.mkdir(parents=True, exist_ok=True)

	factors_out = _save_parquet_with_csv_fallback(ns_factors, output_dir / "ns_factors_weekly.parquet")
	diag_out = output_dir / "ns_fit_diagnostics_weekly.csv"
	fitted_out = _save_parquet_with_csv_fallback(ns_fitted, output_dir / "ns_curve_fitted_weekly.parquet")
	summary_out = output_dir / "ns_run_summary.json"

	ns_diag.to_csv(diag_out, index=False)
	summary_out.write_text(json.dumps(run_summary, indent=2, ensure_ascii=False), encoding="utf-8")

	plots = _make_plots(ns_factors, ns_diag, reports_dir)

	return {
		"factors_out": factors_out,
		"diag_out": diag_out,
		"fitted_out": fitted_out,
		"summary_out": summary_out,
		"plots": plots,
		"run_summary": run_summary,
	}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Passo 3: extração de fatores NS semanais")
	parser.add_argument(
		"--input-file",
		type=Path,
		default=Path(r"Q:\Gabriel de Macedo\Política Monetária\NS Model\NS v2\data\panel_weekly_ns_curve_common.parquet"),
		help="Painel semanal com coluna week_ref e maturidades da curva",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path(r"Q:\Gabriel de Macedo\Política Monetária\NS Model\NS v2\data"),
		help="Diretório para arquivos de saída",
	)
	parser.add_argument(
		"--reports-dir",
		type=Path,
		default=Path(r"Q:\Gabriel de Macedo\Política Monetária\NS Model\NS v2\reports\ns"),
		help="Diretório para gráficos do QA NS",
	)
	parser.add_argument("--lambda-ns", type=float, default=LAMBDA_NS_DEFAULT)
	parser.add_argument(
		"--curve-map",
		type=str,
		default="bbg_bz_bond_1y:1,bbg_bz_bond_3y:3,bbg_bz_bond_5y:5,bbg_bz_bond_10y:10",
		help="Mapeamento coluna:tau_anos separado por vírgula",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	curve_map = _parse_curve_map(args.curve_map)
	outputs = run_pipeline(
		input_file=args.input_file,
		output_dir=args.output_dir,
		reports_dir=args.reports_dir,
		lambda_ns=args.lambda_ns,
		curve_map=curve_map,
	)

	print("Passo 3 (NS) concluído.")
	for key in ["factors_out", "diag_out", "fitted_out", "summary_out"]:
		print(f"{key}: {outputs[key]}")
	print(f"n_weeks_estimated: {outputs['run_summary']['n_weeks_estimated']}")
	print(f"lambda_ns: {outputs['run_summary']['lambda_ns']}")
	print(f"fit_mae_mean: {outputs['run_summary']['fit_mae_mean']}")
	if outputs["plots"]:
		for name, path in outputs["plots"].items():
			print(f"plot_{name}: {path}")


if __name__ == "__main__":
	main()
