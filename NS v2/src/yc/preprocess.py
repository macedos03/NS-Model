from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd


NS_CURVE_DEFAULT = [
	"bbg_bz_bond_1y",
	"bbg_bz_bond_3y",
	"bbg_bz_bond_5y",
	"bbg_bz_bond_10y",
]


def _normalize_text(value: str) -> str:
	text = str(value).strip().lower()
	text = "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))
	text = re.sub(r"[^a-z0-9]+", "_", text)
	text = re.sub(r"_+", "_", text).strip("_")
	return text


def _build_focus_var_name(raw_name: str) -> str:
	mapped = {
		"brl": "fx_brl",
		"trade_b": "trade_balance",
		"ca": "ca",
		"fdi": "fdi",
		"dgdp": "dgdp",
		"indp": "indp",
		"imp": "imp",
		"exp": "exp",
		"gdp": "gdp",
		"ipca": "ipca",
		"selic": "selic",
		"primary": "primary",
		"net_debt": "net_debt",
		"nominal": "nominal",
	}
	name = _normalize_text(raw_name)
	tokens = [token for token in name.split("_") if token]
	if tokens:
		tokens[0] = mapped.get(tokens[0], tokens[0])
	return f"focus_{'_'.join(tokens)}"


def _build_bbg_var_name(raw_name: str) -> str:
	direct_map = {
		"selic": "bbg_selic",
		"bz_bond_1y": "bbg_bz_bond_1y",
		"bz_bond_3y": "bbg_bz_bond_3y",
		"bz_bond_5y": "bbg_bz_bond_5y",
		"bz_bond_10y": "bbg_bz_bond_10y",
		"citi_surprise_us": "bbg_citi_surprise_us",
		"citi_surpise_eur": "bbg_citi_surprise_eur",
		"bbg_surprise_inf_bz": "bbg_bbg_surprise_inf_bz",
		"bbg_surprise_inf_us": "bbg_bbg_surprise_inf_us",
		"vix": "bbg_vix",
		"bz_cds_5y": "bbg_bz_cds_5y",
		"bz_cds_10y": "bbg_bz_cds_10y",
		"ibov": "bbg_ibov",
		"wheat": "bbg_wheat",
		"energy_future": "bbg_energy_future",
		"fed_funds": "bbg_fed_funds",
		"us_5y_breakeven_inflation": "bbg_us_5y_breakeven_inflation",
		"bbg_comm": "bbg_comm",
	}
	key = _normalize_text(raw_name)
	if key in direct_map:
		return direct_map[key]
	return f"bbg_{key}"


def _load_wide_excel(file_path: Path) -> pd.DataFrame:
	df = pd.read_excel(file_path)
	if "Date" not in df.columns:
		raise ValueError(f"Arquivo sem coluna 'Date': {file_path}")
	return df


def _to_long_daily(df_wide: pd.DataFrame, source: str, var_builder) -> pd.DataFrame:
	value_cols = [col for col in df_wide.columns if col != "Date"]
	long_df = df_wide.melt(id_vars=["Date"], value_vars=value_cols, var_name="raw_name", value_name="value")
	long_df = long_df.rename(columns={"Date": "obs_date"})
	long_df["source"] = source
	long_df["var_name_final"] = long_df["raw_name"].map(var_builder)
	long_df["timestamp"] = pd.NaT
	long_df["available_date"] = pd.to_datetime(long_df["obs_date"], errors="coerce").dt.normalize()
	long_df["obs_date"] = pd.to_datetime(long_df["obs_date"], errors="coerce").dt.normalize()
	long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
	long_df["unit"] = pd.NA
	long_df["notes"] = pd.NA
	long_df["_ingest_order"] = range(len(long_df))
	long_df = long_df.dropna(subset=["obs_date"])
	return long_df[
		[
			"source",
			"raw_name",
			"var_name_final",
			"obs_date",
			"value",
			"timestamp",
			"available_date",
			"unit",
			"notes",
			"_ingest_order",
		]
	]


def _deduplicate_daily(long_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
	dup_mask = long_df.duplicated(subset=["var_name_final", "obs_date"], keep=False)
	dups = long_df.loc[dup_mask].copy()

	sorted_df = long_df.sort_values(
		["var_name_final", "obs_date", "timestamp", "_ingest_order"],
		na_position="first",
		kind="stable",
	)
	deduped = sorted_df.drop_duplicates(subset=["var_name_final", "obs_date"], keep="last")
	return deduped.drop(columns=["_ingest_order"]), dups


def _build_calendar_daily(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
	day_ref = pd.bdate_range(start=start_date, end=end_date)
	cal = pd.DataFrame({"day_ref": day_ref})
	cal["is_business_day"] = True
	cal["week_start"] = cal["day_ref"] - pd.to_timedelta(cal["day_ref"].dt.weekday, unit="D")
	cal["week_ref"] = cal.groupby("week_start")["day_ref"].transform("max")
	cal["month"] = cal["day_ref"].dt.to_period("M").astype(str)
	cal["quarter"] = cal["day_ref"].dt.to_period("Q").astype(str)
	cal["year"] = cal["day_ref"].dt.year
	cal["iso_week"] = cal["day_ref"].dt.isocalendar().week.astype(int)
	return cal


def _build_panel_daily_pit(calendar_daily: pd.DataFrame, long_df: pd.DataFrame) -> pd.DataFrame:
	panel = calendar_daily[["day_ref"]].copy().sort_values("day_ref")
	long_df = long_df.sort_values(["var_name_final", "available_date", "obs_date"])

	for var_name, grp in long_df.groupby("var_name_final", sort=True):
		series = grp[["available_date", "value"]].dropna(subset=["available_date"]).copy()
		series = series.sort_values("available_date", kind="stable")
		series = series.groupby("available_date", as_index=False).last()

		merged = pd.merge_asof(
			panel[["day_ref"]],
			series,
			left_on="day_ref",
			right_on="available_date",
			direction="backward",
			allow_exact_matches=True,
		)
		panel[var_name] = merged["value"].values

	return panel


def _build_panel_weekly_from_daily(panel_daily: pd.DataFrame, calendar_daily: pd.DataFrame) -> pd.DataFrame:
	map_week = calendar_daily[["day_ref", "week_ref"]].copy()
	merged = panel_daily.merge(map_week, on="day_ref", how="left")
	weekly = merged.loc[merged["day_ref"] == merged["week_ref"]].copy()
	weekly = weekly.drop(columns=["day_ref"]).rename(columns={"week_ref": "week_ref"})
	weekly = weekly.sort_values("week_ref").drop_duplicates(subset=["week_ref"], keep="last")
	return weekly


def _real_observation_flags(long_df: pd.DataFrame, calendar_daily: pd.DataFrame, vars_of_interest: list[str]) -> pd.DataFrame:
	base = calendar_daily[["day_ref", "week_ref"]].copy()
	for col in vars_of_interest:
		base[col] = False

	if not vars_of_interest:
		return base

	obs = long_df.loc[
		(long_df["var_name_final"].isin(vars_of_interest)) & long_df["value"].notna(),
		["var_name_final", "available_date"],
	].drop_duplicates()
	obs["flag"] = True
	pivot = obs.pivot_table(
		index="available_date",
		columns="var_name_final",
		values="flag",
		aggfunc="max",
		fill_value=False,
	)
	pivot = pivot.reset_index().rename(columns={"available_date": "day_ref"})

	out = base.merge(pivot, on="day_ref", how="left", suffixes=("", "_obs"))
	for col in vars_of_interest:
		if f"{col}_obs" in out.columns:
			out[col] = out[f"{col}_obs"].fillna(False).astype(bool)
			out = out.drop(columns=[f"{col}_obs"])
		else:
			out[col] = out[col].fillna(False).astype(bool)
	return out


def _build_panel_weekly_last_valid_curve(
	panel_daily: pd.DataFrame,
	calendar_daily: pd.DataFrame,
	long_df: pd.DataFrame,
	curve_vars: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
	available_curve_vars = [c for c in curve_vars if c in panel_daily.columns]
	if not available_curve_vars:
		return _build_panel_weekly_from_daily(panel_daily, calendar_daily), pd.DataFrame()

	real_flags = _real_observation_flags(long_df, calendar_daily, available_curve_vars)
	real_flags["real_curve_updates_count"] = real_flags[available_curve_vars].sum(axis=1)

	selected_rows = []
	for week_ref, grp in real_flags.groupby("week_ref", sort=True):
		grp = grp.sort_values("day_ref")
		calendar_week_ref = grp["day_ref"].max()

		with_real = grp.loc[grp["real_curve_updates_count"] > 0]
		if with_real.empty:
			snapshot_day = calendar_week_ref
			reason = "calendar_week_ref_fallback"
			best_count = 0
		else:
			best_count = int(with_real["real_curve_updates_count"].max())
			cand = with_real.loc[with_real["real_curve_updates_count"] == best_count]
			snapshot_day = cand["day_ref"].max()
			reason = "last_valid_curve_day"

		selected_rows.append(
			{
				"week_ref": week_ref,
				"calendar_week_ref": calendar_week_ref,
				"snapshot_day_ref": snapshot_day,
				"snapshot_rule": reason,
				"real_curve_updates_count": best_count,
				"friday_synthetic_flag": bool(snapshot_day != calendar_week_ref),
			}
		)

	snapshot_map = pd.DataFrame(selected_rows).sort_values("week_ref")
	weekly = snapshot_map.merge(panel_daily, left_on="snapshot_day_ref", right_on="day_ref", how="left")
	weekly = weekly.drop(columns=["day_ref"]) 
	value_cols = [c for c in panel_daily.columns if c != "day_ref"]
	weekly_panel = weekly[["week_ref"] + value_cols].copy()
	weekly_panel = weekly_panel.sort_values("week_ref").drop_duplicates(subset=["week_ref"], keep="last")
	return weekly_panel, snapshot_map


def _infer_category(var_name: str) -> str:
	if "bond" in var_name:
		return "yield_target"
	if var_name in {"bbg_selic", "focus_selic_year_0", "focus_selic_year_1"}:
		return "policy_rate"
	if var_name.startswith("focus_"):
		return "macro_feature"
	return "financial_feature"


def _infer_group_pca(var_name: str) -> str:
	if any(k in var_name for k in ["ipca", "inflation", "breakeven"]):
		return "inflacao"
	if any(k in var_name for k in ["gdp", "indp", "trade", "fdi", "exp", "imp"]):
		return "atividade"
	if any(k in var_name for k in ["primary", "debt", "nominal", "fiscal", "dgdp"]):
		return "fiscal"
	if any(k in var_name for k in ["cds", "vix", "risk"]):
		return "risco"
	if "surprise" in var_name:
		return "incerteza"
	return "financeiro"


def _infer_unit(var_name: str) -> str:
	if "cds" in var_name:
		return "bps"
	if "vix" in var_name:
		return "index_points"
	if "ibov" in var_name:
		return "points"
	if "bond" in var_name or "selic" in var_name or "fed_funds" in var_name or "breakeven" in var_name:
		return "percent_or_decimal_check"
	return "unknown"


def _build_metadata(long_df: pd.DataFrame) -> pd.DataFrame:
	rows = []
	for var_name, grp in long_df.groupby("var_name_final", sort=True):
		source = grp["source"].iloc[0]
		raw_name = grp["raw_name"].iloc[0]
		first_valid_date = grp.loc[grp["value"].notna(), "available_date"].min()
		last_valid_date = grp.loc[grp["value"].notna(), "available_date"].max()

		if source == "focus":
			raw_frequency_economic = "daily"
		elif any(k in var_name for k in ["surprise", "anfavea"]):
			raw_frequency_economic = "event"
		else:
			raw_frequency_economic = "daily"

		rows.append(
			{
				"var_name_final": var_name,
				"source": source,
				"raw_name": raw_name,
				"category": _infer_category(var_name),
				"group_pca": _infer_group_pca(var_name),
				"group_pca_manual": "",
				"manual_review_required": True,
				"raw_frequency": "daily",
				"raw_frequency_economic": raw_frequency_economic,
				"panel_base_frequency": "daily",
				"panel_frequency": "daily",
				"weekly_sampling_rule": "snapshot_last_valid_curve_day",
				"sampling_rule": "asof_snapshot",
				"availability_rule": "asof_le_day_ref",
				"first_valid_date": first_valid_date,
				"last_valid_date": last_valid_date,
				"unit": _infer_unit(var_name),
				"transform_candidate": "",
				"comments": "year_0...year_4 mantido como coletado para Focus",
			}
		)

	return pd.DataFrame(rows).sort_values(["source", "var_name_final"]).reset_index(drop=True)


def _qa_daily(panel_daily: pd.DataFrame, calendar_daily: pd.DataFrame) -> dict:
	qa = {}
	qa["day_ref_sorted"] = bool(panel_daily["day_ref"].is_monotonic_increasing)
	qa["day_ref_duplicated_count"] = int(panel_daily["day_ref"].duplicated().sum())

	expected = pd.bdate_range(panel_daily["day_ref"].min(), panel_daily["day_ref"].max())
	qa["business_day_continuity_ok"] = bool(len(expected) == len(panel_daily))
	qa["business_day_missing_count"] = int(len(expected.difference(panel_daily["day_ref"])))

	value_cols = [c for c in panel_daily.columns if c != "day_ref"]
	missing = panel_daily[value_cols].isna().mean().sort_values(ascending=False)
	qa["missing_ratio_top10"] = missing.head(10).to_dict()
	qa["column_name_unique"] = len(value_cols) == len(set(value_cols))
	qa["availability_rule"] = "asof_le_day_ref"
	qa["bfill_used"] = False

	curve_cols = [
		"bbg_bz_bond_1y",
		"bbg_bz_bond_3y",
		"bbg_bz_bond_5y",
		"bbg_bz_bond_10y",
	]
	curve_cols = [col for col in curve_cols if col in panel_daily.columns]
	curve_cov = {col: float(panel_daily[col].notna().mean()) for col in curve_cols}
	qa["curve_coverage_daily"] = curve_cov

	curve_scale = {}
	for col in curve_cols:
		med = panel_daily[col].median(skipna=True)
		curve_scale[col] = "decimal_like" if pd.notna(med) and med < 1.0 else "percent_like"
	qa["curve_scale_hint"] = curve_scale

	return qa


def _scale_flag_table(panel_daily: pd.DataFrame) -> pd.DataFrame:
	rows = []
	for col in [c for c in panel_daily.columns if c != "day_ref"]:
		s = panel_daily[col].dropna()
		if s.empty:
			rows.append({"var_name_final": col, "n_obs": 0, "out_of_range_count": 0, "flag": "no_data"})
			continue

		if any(k in col for k in ["bond", "selic", "fed_funds", "breakeven"]):
			scale_mode = "decimal_like" if s.median() < 1 else "percent_like"
			lb, ub = (-0.05, 0.60) if scale_mode == "decimal_like" else (-5.0, 60.0)
		elif "cds" in col:
			scale_mode = "bps"
			lb, ub = (0.0, 5000.0)
		elif "vix" in col:
			scale_mode = "index"
			lb, ub = (5.0, 120.0)
		elif "ibov" in col:
			scale_mode = "points"
			lb, ub = (1000.0, 300000.0)
		else:
			scale_mode = "not_checked"
			lb, ub = (float("-inf"), float("inf"))

		out_count = int(((s < lb) | (s > ub)).sum())
		flag = "ok" if out_count == 0 else "check_scale"
		rows.append(
			{
				"var_name_final": col,
				"n_obs": int(s.shape[0]),
				"scale_mode": scale_mode,
				"plausible_min": lb,
				"plausible_max": ub,
				"out_of_range_count": out_count,
				"out_of_range_ratio": float(out_count / s.shape[0]),
				"flag": flag,
			}
		)

	return pd.DataFrame(rows).sort_values(["flag", "out_of_range_ratio"], ascending=[True, False])


def _qa_weekly(panel_weekly: pd.DataFrame) -> dict:
	qa = {}
	qa["week_ref_sorted"] = bool(panel_weekly["week_ref"].is_monotonic_increasing)
	qa["week_ref_duplicated_count"] = int(panel_weekly["week_ref"].duplicated().sum())
	qa["n_weeks"] = int(len(panel_weekly))

	curve_cols = [
		"bbg_bz_bond_1y",
		"bbg_bz_bond_3y",
		"bbg_bz_bond_5y",
		"bbg_bz_bond_10y",
	]
	curve_cols = [col for col in curve_cols if col in panel_weekly.columns]
	qa["curve_coverage_weekly"] = {col: float(panel_weekly[col].notna().mean()) for col in curve_cols}
	return qa


def _longest_nan_run(series: pd.Series) -> int:
	mask = series.isna().tolist()
	best = 0
	cur = 0
	for is_na in mask:
		if is_na:
			cur += 1
			best = max(best, cur)
		else:
			cur = 0
	return int(best)


def _staleness_summary(panel_daily: pd.DataFrame, long_df: pd.DataFrame) -> pd.DataFrame:
	rows = []
	all_days = panel_daily[["day_ref"]].copy().sort_values("day_ref")

	for var_name in [c for c in panel_daily.columns if c != "day_ref"]:
		obs = long_df.loc[
			(long_df["var_name_final"] == var_name) & long_df["value"].notna(),
			["available_date"],
		].drop_duplicates().sort_values("available_date")

		if obs.empty:
			rows.append(
				{
					"var_name_final": var_name,
					"staleness_mean_days": pd.NA,
					"staleness_p95_days": pd.NA,
					"staleness_max_days": pd.NA,
				}
			)
			continue

		obs = obs.rename(columns={"available_date": "last_update_date"})
		asof_dates = pd.merge_asof(
			all_days,
			obs,
			left_on="day_ref",
			right_on="last_update_date",
			direction="backward",
		)
		age = (asof_dates["day_ref"] - asof_dates["last_update_date"]).dt.days
		rows.append(
			{
				"var_name_final": var_name,
				"staleness_mean_days": float(age.mean(skipna=True)),
				"staleness_p95_days": float(age.quantile(0.95)),
				"staleness_max_days": float(age.max(skipna=True)),
			}
		)

	return pd.DataFrame(rows).sort_values("var_name_final")


def _missing_report_full(
	panel_daily: pd.DataFrame,
	panel_weekly: pd.DataFrame,
	staleness_df: pd.DataFrame,
) -> pd.DataFrame:
	rows = []
	for col in [c for c in panel_daily.columns if c != "day_ref"]:
		d = panel_daily[col]
		w = panel_weekly[col] if col in panel_weekly.columns else pd.Series(dtype=float)
		first_valid = d.first_valid_index()
		last_valid = d.last_valid_index()
		rows.append(
			{
				"var_name_final": col,
				"missing_ratio_daily": float(d.isna().mean()),
				"coverage_daily": float(d.notna().mean()),
				"coverage_weekly": float(w.notna().mean()) if not w.empty else pd.NA,
				"first_valid_date": panel_daily.loc[first_valid, "day_ref"] if first_valid is not None else pd.NaT,
				"last_valid_date": panel_daily.loc[last_valid, "day_ref"] if last_valid is not None else pd.NaT,
				"longest_gap_daily": _longest_nan_run(d),
			}
		)

	report = pd.DataFrame(rows)
	report = report.merge(staleness_df, on="var_name_final", how="left")
	report = report.sort_values("missing_ratio_daily", ascending=False).reset_index(drop=True)
	return report


def _filter_columns_by_missing(panel: pd.DataFrame, missing_report: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, list[str]]:
	to_drop = missing_report.loc[missing_report["missing_ratio_daily"] > threshold, "var_name_final"].tolist()
	to_drop = [c for c in to_drop if c in panel.columns]
	filtered = panel.drop(columns=to_drop)
	return filtered, to_drop


def _build_ns_curve_common_sample(panel_weekly: pd.DataFrame, curve_cols: list[str]) -> tuple[pd.DataFrame, pd.Timestamp | None]:
	available = [c for c in curve_cols if c in panel_weekly.columns]
	if not available:
		return pd.DataFrame(columns=["week_ref"] + curve_cols), None

	mask = panel_weekly[available].notna().all(axis=1)
	if not mask.any():
		return pd.DataFrame(columns=["week_ref"] + available), None

	first_common_date = panel_weekly.loc[mask, "week_ref"].min()
	ns_panel = panel_weekly.loc[panel_weekly["week_ref"] >= first_common_date, ["week_ref"] + available].copy()
	return ns_panel, first_common_date


def _format_log(
	focus_rows: int,
	bbg_rows: int,
	dedup_removed: int,
	panel_daily: pd.DataFrame,
	panel_weekly: pd.DataFrame,
	qa_daily: dict,
	qa_weekly: dict,
) -> str:
	lines = []
	lines.append("PANEL BUILD LOG - STEP 1 (DAILY PIT -> WEEKLY SNAPSHOT)")
	lines.append("=" * 78)
	lines.append(f"focus_rows_ingested: {focus_rows}")
	lines.append(f"bbg_rows_ingested: {bbg_rows}")
	lines.append(f"duplicates_removed: {dedup_removed}")
	lines.append(f"panel_daily_shape: {panel_daily.shape}")
	lines.append(f"panel_weekly_shape: {panel_weekly.shape}")
	lines.append("")
	lines.append("QA DAILY")
	for k, v in qa_daily.items():
		lines.append(f"- {k}: {v}")
	lines.append("")
	lines.append("QA WEEKLY")
	for k, v in qa_weekly.items():
		lines.append(f"- {k}: {v}")
	return "\n".join(lines)


def _save_parquet_with_csv_fallback(df: pd.DataFrame, parquet_path: Path) -> Path:
	try:
		df.to_parquet(parquet_path, index=False)
		return parquet_path
	except Exception:
		csv_path = parquet_path.with_suffix(".csv")
		df.to_csv(csv_path, index=False)
		return csv_path


def run_pipeline(data_dir: Path, output_dir: Path) -> dict:
	missing_threshold = 0.70
	curve_cols = NS_CURVE_DEFAULT
	sample_config_overrides = {}

	return run_pipeline_with_config(
		data_dir=data_dir,
		output_dir=output_dir,
		missing_threshold=missing_threshold,
		curve_cols=curve_cols,
		sample_config_overrides=sample_config_overrides,
	)


def run_pipeline_with_config(
	data_dir: Path,
	output_dir: Path,
	missing_threshold: float,
	curve_cols: list[str],
	sample_config_overrides: dict,
) -> dict:
	focus_path = data_dir / "Focus.xlsx"
	bbg_path = data_dir / "bbg_data.xlsx"

	focus_wide = _load_wide_excel(focus_path)
	bbg_wide = _load_wide_excel(bbg_path)

	focus_long = _to_long_daily(focus_wide, source="focus", var_builder=_build_focus_var_name)
	bbg_long = _to_long_daily(bbg_wide, source="bbg", var_builder=_build_bbg_var_name)

	focus_rows = len(focus_long)
	bbg_rows = len(bbg_long)

	long_df = pd.concat([focus_long, bbg_long], ignore_index=True)
	deduped_df, dups = _deduplicate_daily(long_df)
	dedup_removed = len(long_df) - len(deduped_df)

	start_date = deduped_df["obs_date"].min()
	end_date = deduped_df["obs_date"].max()
	calendar_daily = _build_calendar_daily(start_date=start_date, end_date=end_date)

	panel_daily = _build_panel_daily_pit(calendar_daily, deduped_df)
	panel_weekly, weekly_snapshot_map = _build_panel_weekly_last_valid_curve(
		panel_daily=panel_daily,
		calendar_daily=calendar_daily,
		long_df=deduped_df,
		curve_vars=curve_cols,
	)
	metadata = _build_metadata(deduped_df)
	staleness_df = _staleness_summary(panel_daily, deduped_df)
	missing_report = _missing_report_full(panel_daily, panel_weekly, staleness_df)
	panel_weekly_filtered, removed_cols = _filter_columns_by_missing(panel_weekly, missing_report, threshold=missing_threshold)
	ns_curve_panel, ns_common_start = _build_ns_curve_common_sample(panel_weekly, curve_cols=curve_cols)
	scale_flags = _scale_flag_table(panel_daily)

	qa_daily = _qa_daily(panel_daily, calendar_daily)
	qa_weekly = _qa_weekly(panel_weekly)
	if not weekly_snapshot_map.empty:
		qa_weekly["synthetic_friday_weeks"] = int(weekly_snapshot_map["friday_synthetic_flag"].sum())
		qa_weekly["synthetic_friday_ratio"] = float(weekly_snapshot_map["friday_synthetic_flag"].mean())
	qa_weekly["missing_exclusion_threshold"] = missing_threshold
	qa_weekly["removed_columns_by_missing"] = removed_cols[:20]
	qa_weekly["removed_columns_count"] = len(removed_cols)

	scale_alerts = scale_flags.loc[scale_flags["flag"] == "check_scale"]
	qa_daily["scale_alert_count"] = int(len(scale_alerts))
	qa_daily["scale_alert_examples"] = scale_alerts.head(10)["var_name_final"].tolist()

	output_dir.mkdir(parents=True, exist_ok=True)
	daily_out = _save_parquet_with_csv_fallback(panel_daily, output_dir / "panel_daily_raw_pit.parquet")
	weekly_out = _save_parquet_with_csv_fallback(panel_weekly, output_dir / "panel_weekly_raw_pit.parquet")
	weekly_filtered_out = _save_parquet_with_csv_fallback(panel_weekly_filtered, output_dir / "panel_weekly_raw_pit_filtered.parquet")
	ns_curve_out = _save_parquet_with_csv_fallback(ns_curve_panel, output_dir / "panel_weekly_ns_curve_common.parquet")
	metadata_out = output_dir / "metadata_variables.csv"
	metadata_review_out = output_dir / "metadata_variables_review.csv"
	log_out = output_dir / "panel_build_log.txt"
	missing_out = output_dir / "missing_report_full.csv"
	staleness_out = output_dir / "qa_staleness_summary.csv"
	scale_flags_out = output_dir / "qa_scale_flags.csv"
	weekly_snapshot_out = output_dir / "qa_weekly_snapshot_days.csv"
	sample_config_out = output_dir / "sample_config.json"

	metadata.to_csv(metadata_out, index=False)
	metadata.to_csv(metadata_review_out, index=False)
	missing_report.to_csv(missing_out, index=False)
	staleness_df.to_csv(staleness_out, index=False)
	scale_flags.to_csv(scale_flags_out, index=False)
	if not weekly_snapshot_map.empty:
		weekly_snapshot_map.to_csv(weekly_snapshot_out, index=False)

	sample_config = {
		"train_start": sample_config_overrides.get("train_start"),
		"train_end": sample_config_overrides.get("train_end"),
		"oos_start": sample_config_overrides.get("oos_start"),
		"oos_end": sample_config_overrides.get("oos_end"),
		"replication_note": "Preencher janela comparável ao paper antes do backtest final.",
		"ns_curve_columns": curve_cols,
		"ns_common_start": str(ns_common_start.date()) if ns_common_start is not None else None,
		"panel_weekly_start": str(panel_weekly["week_ref"].min().date()) if not panel_weekly.empty else None,
		"panel_weekly_end": str(panel_weekly["week_ref"].max().date()) if not panel_weekly.empty else None,
		"panel_weekly_rows": int(len(panel_weekly)),
		"panel_weekly_ns_rows": int(len(ns_curve_panel)),
		"missing_exclusion_threshold": missing_threshold,
		"dropped_columns_by_missing": removed_cols,
	}
	sample_config_out.write_text(json.dumps(sample_config, indent=2, ensure_ascii=False), encoding="utf-8")

	log_text = _format_log(
		focus_rows=focus_rows,
		bbg_rows=bbg_rows,
		dedup_removed=dedup_removed,
		panel_daily=panel_daily,
		panel_weekly=panel_weekly,
		qa_daily=qa_daily,
		qa_weekly=qa_weekly,
	)
	log_out.write_text(log_text, encoding="utf-8")

	return {
		"daily_out": daily_out,
		"weekly_out": weekly_out,
		"weekly_filtered_out": weekly_filtered_out,
		"ns_curve_out": ns_curve_out,
		"metadata_out": metadata_out,
		"metadata_review_out": metadata_review_out,
		"log_out": log_out,
		"missing_out": missing_out,
		"staleness_out": staleness_out,
		"scale_flags_out": scale_flags_out,
		"weekly_snapshot_out": weekly_snapshot_out,
		"sample_config_out": sample_config_out,
		"duplicates_logged": len(dups),
	}

def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Build daily PIT and weekly snapshot panel from Focus + BBG")
	parser.add_argument(
		"--data-dir",
		type=Path,
		default=Path(r"Q:\Gabriel de Macedo\Política Monetária\NS Model\NS v2\data"),
		help="Directory containing Focus.xlsx and bbg_data.xlsx",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path(r"Q:\Gabriel de Macedo\Política Monetária\NS Model\NS v2\data"),
		help="Directory for outputs",
	)
	parser.add_argument(
		"--missing-threshold",
		type=float,
		default=0.70,
		help="Drop columns with daily missing ratio above this threshold in filtered panel",
	)
	parser.add_argument(
		"--curve-cols",
		type=str,
		default=",".join(NS_CURVE_DEFAULT),
		help="Comma-separated curve columns for robust weekly snapshot and NS common sample",
	)
	parser.add_argument("--train-start", type=str, default=None)
	parser.add_argument("--train-end", type=str, default=None)
	parser.add_argument("--oos-start", type=str, default=None)
	parser.add_argument("--oos-end", type=str, default=None)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	curve_cols = [c.strip() for c in args.curve_cols.split(",") if c.strip()]
	overrides = {
		"train_start": args.train_start,
		"train_end": args.train_end,
		"oos_start": args.oos_start,
		"oos_end": args.oos_end,
	}
	outputs = run_pipeline_with_config(
		data_dir=args.data_dir,
		output_dir=args.output_dir,
		missing_threshold=args.missing_threshold,
		curve_cols=curve_cols,
		sample_config_overrides=overrides,
	)
	print("Pipeline concluído.")
	for key, value in outputs.items():
		print(f"{key}: {value}")


if __name__ == "__main__":
	main()
