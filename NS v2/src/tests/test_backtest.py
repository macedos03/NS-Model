from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

TESTS_DIR = Path(__file__).resolve().parent
YC_DIR = TESTS_DIR.parent / "yc"

BACKTEST_PATH = YC_DIR / "backtest.py"
spec = importlib.util.spec_from_file_location("backtest", BACKTEST_PATH)
if spec is None or spec.loader is None:
	raise ImportError(f"Could not load backtest module from {BACKTEST_PATH}")
backtest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backtest)


def _make_forecast_df(model_id: str, start: str = "2023-01-06") -> pd.DataFrame:
	weeks = pd.date_range(start, periods=4, freq="W-FRI")
	rows = []
	for i, origin in enumerate(weeks):
		target = origin + pd.Timedelta(days=7)
		for tau in [1.0, 3.0]:
			error = 0.01 * (i + 1) * (1 if model_id != "rw_yield" else 1.2)
			rows.append(
				{
					"model_id": model_id,
					"origin_week_ref": origin,
					"target_week_ref": target,
					"horizon_steps": 1,
					"tau_years": tau,
					"yield_obs": 0.10 + 0.005 * tau,
					"yield_fcst": 0.10 + 0.005 * tau + error,
					"error": error,
					"abs_error": abs(error),
				}
			)
	return pd.DataFrame(rows)


def test_consolidate_forecasts_parses_dates(tmp_path) -> None:
	path_model = tmp_path / "forecast_yields_var.csv"
	path_rw = tmp_path / "forecast_yields_rw.csv"

	_make_forecast_df("var").to_csv(path_model, index=False)
	_make_forecast_df("rw_yield").to_csv(path_rw, index=False)

	df = backtest._consolidate_forecasts({"var": path_model, "rw_yield": path_rw})

	assert not df.empty
	assert pd.api.types.is_datetime64_any_dtype(df["origin_week_ref"])
	assert pd.api.types.is_datetime64_any_dtype(df["target_week_ref"])
	assert set(df["model_id"].unique()) == {"var", "rw_yield"}


def test_build_common_support_set_keeps_intersection() -> None:
	df_var = _make_forecast_df("var")
	df_rw = _make_forecast_df("rw_yield")

	# Remove one key from VAR so intersection shrinks
	df_var = df_var.iloc[:-1].copy()
	df = pd.concat([df_var, df_rw], ignore_index=True)

	df_cs = backtest._build_common_support_set(df)

	assert not df_cs.empty
	assert len(df_cs) < len(df)
	counts_per_key = (
		df_cs.groupby(["origin_week_ref", "target_week_ref", "horizon_steps", "tau_years"])["model_id"]
		.nunique()
		.reset_index(name="n_models")
	)
	assert (counts_per_key["n_models"] == 2).all()


def test_run_evaluation_writes_expected_outputs(tmp_path) -> None:
	forecast_dir = tmp_path / "forecasts"
	output_dir = tmp_path / "evaluation"
	forecast_dir.mkdir()

	_make_forecast_df("var_dl").to_csv(forecast_dir / "forecast_yields_var_dl.csv", index=False)
	_make_forecast_df("var_dlfavar").to_csv(forecast_dir / "forecast_yields_var_dlfavar.csv", index=False)
	_make_forecast_df("rw_yield").to_csv(forecast_dir / "forecast_yields_rw.csv", index=False)

	results = backtest.run_evaluation(
		forecast_dir=forecast_dir,
		output_dir=output_dir,
		use_common_support=True,
	)

	assert not results["consolidated_df"].empty
	assert (output_dir / "eval_oos_mae.csv").exists()
	assert (output_dir / "eval_oos_bias.csv").exists()
	assert (output_dir / "eval_oos_summary.json").exists()
	assert (output_dir / "eval_oos_panel_long.csv").exists()
