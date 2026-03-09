from __future__ import annotations

import json

import numpy as np
import pandas as pd

import PCA


def test_apply_stationarity_transform_respects_manual_rules() -> None:
	panel = pd.DataFrame(
		{
			"week_ref": pd.date_range("2021-01-01", periods=5, freq="W-FRI"),
			"var_a": [1.0, 2.0, 3.0, 4.0, 5.0],
			"var_b": [10.0, 11.0, 12.0, 13.0, 14.0],
		}
	)
	metadata = pd.DataFrame(
		{
			"var_name_final": ["var_a", "var_b"],
			"transform_manual": ["diff1", "level"],
		}
	)
	fit_mask = pd.Series([True] * len(panel), index=panel.index)

	transformed, transform_meta = PCA._apply_stationarity_transform(
		panel_weekly=panel,
		feature_cols=["var_a", "var_b"],
		metadata=metadata,
		fit_mask=fit_mask,
		stationarity_mode="none",
	)

	row_a = transform_meta[transform_meta["var_name_final"] == "var_a"].iloc[0]
	row_b = transform_meta[transform_meta["var_name_final"] == "var_b"].iloc[0]
	assert row_a["transform_applied"] == "diff1"
	assert row_b["transform_applied"] == "level"
	assert pd.isna(transformed.loc[0, "var_a"])
	assert transformed.loc[1, "var_a"] == 1.0


def test_filter_by_coverage_drops_low_coverage_feature() -> None:
	df = pd.DataFrame(
		{
			"week_ref": pd.date_range("2022-01-01", periods=6, freq="W-FRI"),
			"good_feature": [1, 2, 3, 4, 5, 6],
			"bad_feature": [np.nan, np.nan, 1.0, np.nan, np.nan, np.nan],
		}
	)
	fit_mask = pd.Series([True] * len(df), index=df.index)

	filtered, coverage, dropped = PCA._filter_by_coverage(df, fit_mask=fit_mask, min_coverage=0.5)

	assert "good_feature" in filtered.columns
	assert "bad_feature" not in filtered.columns
	assert "bad_feature" in dropped
	assert set(coverage.columns) == {"var_name_final", "coverage_full", "coverage_fit"}


def test_run_pipeline_generates_outputs(tmp_path) -> None:
	n_rows = 26
	weeks = pd.date_range("2020-01-03", periods=n_rows, freq="W-FRI")

	panel = pd.DataFrame(
		{
			"week_ref": weeks,
			"infl_1": np.linspace(0.0, 1.0, n_rows),
			"infl_2": np.linspace(1.0, 2.0, n_rows),
			"infl_3": np.linspace(2.0, 3.0, n_rows),
			"ativ_1": np.linspace(5.0, 6.0, n_rows),
			"ativ_2": np.linspace(6.0, 7.0, n_rows),
			"ativ_3": np.linspace(7.0, 8.0, n_rows),
			"fisc_1": np.linspace(10.0, 11.0, n_rows),
			"fisc_2": np.linspace(11.0, 12.0, n_rows),
			"fisc_3": np.linspace(12.0, 13.0, n_rows),
			"bbg_selic": np.linspace(8.0, 9.0, n_rows),
		}
	)

	metadata = pd.DataFrame(
		{
			"var_name_final": [
				"infl_1",
				"infl_2",
				"infl_3",
				"ativ_1",
				"ativ_2",
				"ativ_3",
				"fisc_1",
				"fisc_2",
				"fisc_3",
			],
			"group_pca": [
				"inflacao",
				"inflacao",
				"inflacao",
				"atividade",
				"atividade",
				"atividade",
				"fiscal",
				"fiscal",
				"fiscal",
			],
			"group_pca_manual": [""] * 9,
			"transform_manual": ["level"] * 9,
		}
	)

	sample_config = {
		"train_start": str(weeks.min().date()),
		"train_end": str(weeks.max().date()),
	}

	panel_path = tmp_path / "panel.csv"
	metadata_path = tmp_path / "metadata.csv"
	sample_config_path = tmp_path / "sample_config.json"
	output_dir = tmp_path / "out"

	panel.to_csv(panel_path, index=False)
	metadata.to_csv(metadata_path, index=False)
	sample_config_path.write_text(json.dumps(sample_config), encoding="utf-8")

	outputs = PCA.run_pipeline(
		panel_path=panel_path,
		metadata_path=metadata_path,
		sample_config_path=sample_config_path,
		output_dir=output_dir,
		n_components_all=3,
		min_coverage_fit=0.8,
		stationarity_mode="none",
		top_n_corr=3,
		exclude_cols=[],
		groups=["inflacao", "atividade", "fiscal"],
		winsor_lower=None,
		winsor_upper=None,
		selic_col="bbg_selic",
	)

	assert outputs["out_scores_all"].exists()
	assert outputs["out_scores_group"].exists()
	assert outputs["out_fira"].exists()
	assert outputs["out_summary"].exists()

	summary = json.loads(outputs["out_summary"].read_text(encoding="utf-8"))
	assert summary["n_rows_total"] == n_rows
	assert summary["n_components_all"] == 3
	assert summary["n_features_initial"] == 9
