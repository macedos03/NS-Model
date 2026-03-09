from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import NS_betas


def test_ns_loadings_valid_shape() -> None:
	tau = np.array([1.0, 3.0, 5.0, 10.0])
	l2, l3, X = NS_betas.ns_loadings(tau=tau, lambda_=0.266)

	assert l2.shape == (4,)
	assert l3.shape == (4,)
	assert X.shape == (4, 3)
	assert np.allclose(X[:, 0], 1.0)


def test_ns_loadings_invalid_inputs() -> None:
	with pytest.raises(ValueError):
		NS_betas.ns_loadings(tau=np.array([1.0, 0.0]), lambda_=0.266)

	with pytest.raises(ValueError):
		NS_betas.ns_loadings(tau=np.array([1.0, 3.0]), lambda_=-0.1)


def test_estimate_ns_betas_ols_recovers_parameters() -> None:
	tau = np.array([1.0, 3.0, 5.0, 10.0])
	_, _, X = NS_betas.ns_loadings(tau=tau, lambda_=0.266)
	true_beta = np.array([0.12, -0.03, 0.02])
	y = X @ true_beta

	est, y_hat, resid = NS_betas.estimate_ns_betas_ols(y=y, X=X)

	assert np.allclose(y_hat, y, atol=1e-10)
	assert np.allclose(resid, 0.0, atol=1e-10)
	assert est["beta1_level"] == pytest.approx(true_beta[0], abs=1e-10)
	assert est["beta2_slope"] == pytest.approx(true_beta[1], abs=1e-10)
	assert est["beta3_curvature"] == pytest.approx(true_beta[2], abs=1e-10)
	assert est["rank_X"] == 3


def test_run_ns_step_outputs_expected_columns() -> None:
	weeks = pd.date_range("2021-01-01", periods=6, freq="W-FRI")
	panel = pd.DataFrame(
		{
			"week_ref": weeks,
			"bbg_bz_bond_1y": np.linspace(0.08, 0.10, len(weeks)),
			"bbg_bz_bond_3y": np.linspace(0.09, 0.11, len(weeks)),
			"bbg_bz_bond_5y": np.linspace(0.10, 0.12, len(weeks)),
			"bbg_bz_bond_10y": np.linspace(0.11, 0.13, len(weeks)),
		}
	)

	ns_factors, ns_diag, ns_fitted, summary = NS_betas.run_ns_step(
		panel_weekly=panel,
		curve_map=NS_betas.CURVE_MAP_DEFAULT,
		lambda_ns=NS_betas.LAMBDA_NS_DEFAULT,
	)

	assert len(ns_factors) == len(weeks)
	assert len(ns_diag) == len(weeks)
	assert len(ns_fitted) == len(weeks)
	assert {"beta1_level", "beta2_slope", "beta3_curvature"}.issubset(ns_factors.columns)
	assert {"fit_mae", "fit_rmse", "fit_max_abs_err", "rank_X"}.issubset(ns_diag.columns)
	assert summary["n_weeks_estimated"] == len(weeks)
	assert summary["n_weeks_dropped_incomplete_curve"] == 0


def test_run_pipeline_writes_outputs(tmp_path) -> None:
	weeks = pd.date_range("2022-01-07", periods=5, freq="W-FRI")
	panel = pd.DataFrame(
		{
			"week_ref": weeks,
			"bbg_bz_bond_1y": [0.10, 0.101, 0.102, 0.103, 0.104],
			"bbg_bz_bond_3y": [0.11, 0.111, 0.112, 0.113, 0.114],
			"bbg_bz_bond_5y": [0.12, 0.121, 0.122, 0.123, 0.124],
			"bbg_bz_bond_10y": [0.13, 0.131, 0.132, 0.133, 0.134],
		}
	)

	input_file = tmp_path / "panel_weekly.csv"
	output_dir = tmp_path / "data_out"
	reports_dir = tmp_path / "reports_out"
	panel.to_csv(input_file, index=False)

	outputs = NS_betas.run_pipeline(
		input_file=input_file,
		output_dir=output_dir,
		reports_dir=reports_dir,
		lambda_ns=NS_betas.LAMBDA_NS_DEFAULT,
		curve_map=NS_betas.CURVE_MAP_DEFAULT,
	)

	assert outputs["diag_out"].exists()
	assert outputs["summary_out"].exists()
	assert outputs["factors_out"].exists()
	assert outputs["fitted_out"].exists()

	summary = json.loads(outputs["summary_out"].read_text(encoding="utf-8"))
	assert summary["n_weeks_estimated"] == 5
	assert summary["lambda_ns"] == pytest.approx(NS_betas.LAMBDA_NS_DEFAULT)
