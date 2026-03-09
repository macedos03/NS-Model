from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


warnings.filterwarnings("ignore", message="invalid value encountered in divide", category=RuntimeWarning)


DEFAULT_GROUPS = ["inflacao", "atividade", "fiscal", "risco", "incerteza", "financeiro"]
DEFAULT_EXCLUDE = ["bbg_bz_bond_1y", "bbg_bz_bond_3y", "bbg_bz_bond_5y", "bbg_bz_bond_10y", "bbg_selic"]
VALID_TRANSFORMS = {"level", "diff1"}


@dataclass
class ScalerStats:
	mean_: pd.Series
	std_: pd.Series


def _save_parquet_with_csv_fallback(df: pd.DataFrame, parquet_path: Path) -> Path:
	try:
		df.to_parquet(parquet_path, index=False)
		return parquet_path
	except Exception:
		csv_path = parquet_path.with_suffix(".csv")
		df.to_csv(csv_path, index=False)
		return csv_path


def _parse_date(value: str | None) -> pd.Timestamp | None:
	if value is None or str(value).strip() == "":
		return None
	out = pd.to_datetime(value, errors="coerce")
	if pd.isna(out):
		return None
	return out


def _load_panel(panel_path: Path) -> pd.DataFrame:
	if not panel_path.exists():
		raise FileNotFoundError(f"Painel semanal não encontrado: {panel_path}")
	if panel_path.suffix.lower() == ".parquet":
		df = pd.read_parquet(panel_path)
	else:
		df = pd.read_csv(panel_path)

	if "week_ref" not in df.columns:
		raise ValueError("Painel semanal precisa conter coluna 'week_ref'")
	df["week_ref"] = pd.to_datetime(df["week_ref"], errors="coerce")
	df = df.dropna(subset=["week_ref"]).sort_values("week_ref").reset_index(drop=True)
	return df


def _load_metadata(metadata_path: Path) -> pd.DataFrame:
	if not metadata_path.exists():
		raise FileNotFoundError(f"Metadata não encontrado: {metadata_path}")
	md = pd.read_csv(metadata_path)
	if "var_name_final" not in md.columns:
		raise ValueError("metadata_variables.csv sem coluna 'var_name_final'")
	if "group_pca" not in md.columns:
		warnings.warn("metadata sem coluna 'group_pca'; grupos default podem ficar inconsistentes")
	if "group_pca_manual" not in md.columns:
		warnings.warn("metadata sem coluna 'group_pca_manual'; usando apenas 'group_pca'")
	if "transform_manual" not in md.columns:
		md["transform_manual"] = ""
		warnings.warn("metadata sem coluna 'transform_manual'; usando regra automática para todas as variáveis")
	return md


def _load_sample_config(sample_config_path: Path) -> dict:
	if not sample_config_path.exists():
		return {}
	return json.loads(sample_config_path.read_text(encoding="utf-8"))


def _build_feature_list(panel_weekly: pd.DataFrame, metadata: pd.DataFrame, exclude_cols: list[str]) -> list[str]:
	meta_vars = set(metadata["var_name_final"].tolist())
	exclude_set = set(exclude_cols)
	cols = [c for c in panel_weekly.columns if c != "week_ref" and c in meta_vars and c not in exclude_set]
	return cols


def _build_fit_mask(panel_weekly: pd.DataFrame, train_start: pd.Timestamp | None, train_end: pd.Timestamp | None) -> pd.Series:
	mask = pd.Series(True, index=panel_weekly.index)
	if train_start is not None:
		mask &= panel_weekly["week_ref"] >= train_start
	if train_end is not None:
		mask &= panel_weekly["week_ref"] <= train_end
	return mask


def _adf_nonstationary(series: pd.Series, alpha: float = 0.10) -> tuple[bool, float | None]:
	data = series.dropna()
	if len(data) < 80:
		return False, None
	try:
		from statsmodels.tsa.stattools import adfuller  # type: ignore
		pvalue = adfuller(data.values, autolag="AIC")[1]
		return bool(pvalue > alpha), float(pvalue)
	except Exception:
		x = data.to_numpy(dtype=float)
		if x.size < 3:
			return False, None
		x0 = x[:-1]
		x1 = x[1:]
		mask = np.isfinite(x0) & np.isfinite(x1)
		x0 = x0[mask]
		x1 = x1[mask]
		if x0.size < 3:
			return False, None

		x0c = x0 - x0.mean()
		x1c = x1 - x1.mean()
		den = np.sqrt((x0c * x0c).sum() * (x1c * x1c).sum())
		if not np.isfinite(den) or den <= 1e-12:
			return False, None
		auto1 = float((x0c * x1c).sum() / den)
		return bool(abs(auto1) > 0.97), None


def _apply_stationarity_transform(
	panel_weekly: pd.DataFrame,
	feature_cols: list[str],
	metadata: pd.DataFrame,
	fit_mask: pd.Series,
	stationarity_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
	transformed = panel_weekly[["week_ref"]].copy()
	meta_rows = []
	manual_map = dict(zip(metadata["var_name_final"], metadata.get("transform_manual", "")))

	for col in feature_cols:
		s = pd.to_numeric(panel_weekly[col], errors="coerce")
		train_s = s.loc[fit_mask]
		manual_raw = manual_map.get(col, "")
		manual = "" if pd.isna(manual_raw) else str(manual_raw).strip().lower()

		if manual in VALID_TRANSFORMS:
			diff_order = 1 if manual == "diff1" else 0
			is_nonstationary = False
			pvalue = None
			transform_source = "manual"
		else:
			if manual not in {"", "nan", "none"}:
				warnings.warn(f"transform_manual inválido para {col}: '{manual_raw}'. Usando regra automática.")

			if stationarity_mode == "none":
				diff_order = 0
				is_nonstationary = False
				pvalue = None
			elif stationarity_mode == "adf_or_acf":
				is_nonstationary, pvalue = _adf_nonstationary(train_s)
				diff_order = 1 if is_nonstationary else 0
			else:
				raise ValueError(f"stationarity_mode inválido: {stationarity_mode}")

			transform_source = "auto"

		if diff_order == 1:
			t = s.diff(1)
			transform = "diff1"
		else:
			t = s
			transform = "level"

		transformed[col] = t
		meta_rows.append(
			{
				"var_name_final": col,
				"transform_applied": transform,
				"transform_source": transform_source,
				"diff_order": diff_order,
				"adf_pvalue": pvalue,
				"fit_nonstationary_flag": is_nonstationary,
			}
		)

	transform_meta = pd.DataFrame(meta_rows)
	return transformed, transform_meta


def _filter_by_coverage(
	transformed: pd.DataFrame,
	fit_mask: pd.Series,
	min_coverage: float,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
	feature_cols = [c for c in transformed.columns if c != "week_ref"]
	coverage_rows = []
	keep_cols = []

	for col in feature_cols:
		cov_full = float(transformed[col].notna().mean())
		cov_fit = float(transformed.loc[fit_mask, col].notna().mean())
		coverage_rows.append(
			{
				"var_name_final": col,
				"coverage_full": cov_full,
				"coverage_fit": cov_fit,
			}
		)
		if cov_fit >= min_coverage:
			keep_cols.append(col)

	coverage_df = pd.DataFrame(coverage_rows).sort_values("coverage_fit")
	filtered = transformed[["week_ref"] + keep_cols].copy()
	dropped = [c for c in feature_cols if c not in keep_cols]
	return filtered, coverage_df, dropped


def _median_impute_fit(train_df: pd.DataFrame, full_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
	medians = train_df.median(axis=0, skipna=True)
	train_imp = train_df.fillna(medians)
	full_imp = full_df.fillna(medians)
	return train_imp, full_imp, medians


def _zscore_fit(train_df: pd.DataFrame, full_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, ScalerStats, list[str]]:
	mean_ = train_df.mean(axis=0)
	std_ = train_df.std(axis=0, ddof=0)
	zero_std_cols = std_[std_ <= 1e-12].index.tolist()

	if zero_std_cols:
		train_df = train_df.drop(columns=zero_std_cols)
		full_df = full_df.drop(columns=zero_std_cols)
		mean_ = mean_.drop(index=zero_std_cols)
		std_ = std_.drop(index=zero_std_cols)

	train_z = (train_df - mean_) / std_
	full_z = (full_df - mean_) / std_
	return train_z, full_z, ScalerStats(mean_=mean_, std_=std_), zero_std_cols


def _winsorize_fit(
	train_df: pd.DataFrame,
	full_df: pd.DataFrame,
	lower_q: float,
	upper_q: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	if not (0.0 <= lower_q < upper_q <= 1.0):
		raise ValueError("winsor quantiles inválidos: requer 0 <= lower < upper <= 1")

	lower = train_df.quantile(lower_q, axis=0)
	upper = train_df.quantile(upper_q, axis=0)
	train_w = train_df.clip(lower=lower, upper=upper, axis=1)
	full_w = full_df.clip(lower=lower, upper=upper, axis=1)
	bounds = pd.DataFrame({"var_name_final": lower.index, "winsor_lower": lower.values, "winsor_upper": upper.values})
	return train_w, full_w, bounds


def _pca_fit_transform(train_z: pd.DataFrame, full_z: pd.DataFrame, n_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	if n_components < 1:
		raise ValueError("n_components deve ser >= 1")
	if train_z.shape[0] < 2:
		raise ValueError("PCA requer ao menos 2 observações no treino")
	if train_z.shape[1] < 1:
		raise ValueError("PCA requer ao menos 1 variável após filtros")

	X_train = train_z.to_numpy(dtype=float)
	X_full = full_z.to_numpy(dtype=float)

	u, s, vt = np.linalg.svd(X_train, full_matrices=False)
	max_k = min(n_components, vt.shape[0])
	components = vt[:max_k, :]

	eigenvalues = (s[:max_k] ** 2) / max(1, (X_train.shape[0] - 1))
	total_var = (s**2).sum() / max(1, (X_train.shape[0] - 1))
	explained_ratio = eigenvalues / total_var if total_var > 0 else np.zeros_like(eigenvalues)

	scores_full = X_full @ components.T
	return scores_full, components, explained_ratio


def _align_component_signs(scores: np.ndarray, components: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	scores_adj = scores.copy()
	components_adj = components.copy()
	for i in range(components_adj.shape[0]):
		j = int(np.argmax(np.abs(components_adj[i, :])))
		if components_adj[i, j] < 0:
			components_adj[i, :] *= -1.0
			scores_adj[:, i] *= -1.0
	return scores_adj, components_adj


def _scores_df(week_ref: pd.Series, scores: np.ndarray, prefix: str) -> pd.DataFrame:
	data = {"week_ref": week_ref.values}
	for i in range(scores.shape[1]):
		data[f"{prefix}{i+1}"] = scores[:, i]
	return pd.DataFrame(data)


def _loadings_df(columns: list[str], components: np.ndarray, prefix: str) -> pd.DataFrame:
	rows = []
	for i in range(components.shape[0]):
		pc_name = f"{prefix}{i+1}"
		for j, col in enumerate(columns):
			rows.append({"component": pc_name, "var_name_final": col, "loading": float(components[i, j])})
	return pd.DataFrame(rows)


def _explained_df(explained_ratio: np.ndarray, prefix: str) -> pd.DataFrame:
	rows = []
	for i, ratio in enumerate(explained_ratio, start=1):
		rows.append({"component": f"{prefix}{i}", "explained_variance_ratio": float(ratio)})
	return pd.DataFrame(rows)


def _top_correlations(feature_df: pd.DataFrame, score_df: pd.DataFrame, score_cols: list[str], top_n: int) -> pd.DataFrame:
	rows = []
	for pc in score_cols:
		corrs = []
		for col in feature_df.columns:
			pair = pd.concat([feature_df[col], score_df[pc]], axis=1).dropna()
			if len(pair) < 20:
				continue
			x = pair.iloc[:, 0].to_numpy(dtype=float)
			y = pair.iloc[:, 1].to_numpy(dtype=float)
			finite_mask = np.isfinite(x) & np.isfinite(y)
			x = x[finite_mask]
			y = y[finite_mask]
			if x.size < 20:
				continue

			xc = x - x.mean()
			yc = y - y.mean()
			den = np.sqrt((xc * xc).sum() * (yc * yc).sum())
			if not np.isfinite(den) or den <= 1e-12:
				continue
			corr = float((xc * yc).sum() / den)
			if np.isfinite(corr):
				corrs.append((col, corr))

		corrs = sorted(corrs, key=lambda x: abs(x[1]), reverse=True)[:top_n]
		for rank, (col, corr) in enumerate(corrs, start=1):
			rows.append({"component": pc, "rank": rank, "var_name_final": col, "corr": corr})
	return pd.DataFrame(rows)


def _group_label_map(metadata: pd.DataFrame) -> dict[str, str]:
	md = metadata.copy()
	if "group_pca_manual" in md.columns:
		manual = md["group_pca_manual"].fillna("").astype(str).str.strip()
		md["group_final"] = np.where(manual != "", manual, md.get("group_pca", "financeiro"))
	else:
		md["group_final"] = md.get("group_pca", "financeiro")

	return dict(zip(md["var_name_final"], md["group_final"]))


def _run_group_pca(
	transformed_filtered: pd.DataFrame,
	fit_mask: pd.Series,
	group_map: dict[str, str],
	groups: list[str],
	top_n_corr: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	week_ref = transformed_filtered["week_ref"]
	all_cols = [c for c in transformed_filtered.columns if c != "week_ref"]

	scores_out = pd.DataFrame({"week_ref": week_ref.values})
	load_rows = []
	var_rows = []
	corr_full_frames = []
	corr_fit_frames = []

	for grp in groups:
		grp_cols = [c for c in all_cols if group_map.get(c, "financeiro") == grp]
		if len(grp_cols) < 3:
			continue

		grp_full = transformed_filtered[grp_cols]
		grp_train = grp_full.loc[fit_mask]
		grp_train_imp, grp_full_imp, _ = _median_impute_fit(grp_train, grp_full)
		grp_train_z, grp_full_z, scaler, zero_std = _zscore_fit(grp_train_imp, grp_full_imp)
		if grp_train_z.shape[1] < 3:
			continue

		scores, comps, ratio = _pca_fit_transform(grp_train_z, grp_full_z, n_components=1)
		scores, comps = _align_component_signs(scores, comps)

		pc_name = f"pc1_{grp}"
		scores_out[pc_name] = scores[:, 0]

		used_cols = grp_train_z.columns.tolist()
		for j, col in enumerate(used_cols):
			load_rows.append({"group": grp, "component": pc_name, "var_name_final": col, "loading": float(comps[0, j])})
		var_rows.append({"group": grp, "component": pc_name, "explained_variance_ratio": float(ratio[0]), "n_vars": len(used_cols), "zero_std_dropped": len(zero_std)})

		feature_std_full = pd.DataFrame(grp_full_z, columns=used_cols)
		score_full = pd.DataFrame({pc_name: scores[:, 0]}, index=feature_std_full.index)
		corr_df_full = _top_correlations(feature_std_full, score_full, [pc_name], top_n=top_n_corr)
		if not corr_df_full.empty:
			corr_df_full["group"] = grp
			corr_full_frames.append(corr_df_full)

		feature_std_fit = feature_std_full.loc[fit_mask]
		score_fit = score_full.loc[fit_mask]
		corr_df_fit = _top_correlations(feature_std_fit, score_fit, [pc_name], top_n=top_n_corr)
		if not corr_df_fit.empty:
			corr_df_fit["group"] = grp
			corr_fit_frames.append(corr_df_fit)

	load_df = pd.DataFrame(load_rows)
	var_df = pd.DataFrame(var_rows)
	corr_df_fit_all = pd.concat(corr_fit_frames, ignore_index=True) if corr_fit_frames else pd.DataFrame()
	corr_df_full_all = pd.concat(corr_full_frames, ignore_index=True) if corr_full_frames else pd.DataFrame()
	return scores_out, load_df, var_df, corr_df_fit_all, corr_df_full_all


def _build_fira(scores_group: pd.DataFrame, panel_weekly: pd.DataFrame, selic_col: str) -> pd.DataFrame:
	need = ["pc1_inflacao", "pc1_atividade", "pc1_fiscal"]
	available = [c for c in need if c in scores_group.columns]
	if len(available) < 3:
		return pd.DataFrame(columns=["week_ref"] + need + [selic_col])

	base = scores_group[["week_ref"] + need].copy()
	if selic_col in panel_weekly.columns:
		selic = panel_weekly[["week_ref", selic_col]].copy()
		base = base.merge(selic, on="week_ref", how="left")
	else:
		base[selic_col] = np.nan
	return base


def run_pipeline(
	panel_path: Path,
	metadata_path: Path,
	sample_config_path: Path,
	output_dir: Path,
	n_components_all: int,
	min_coverage_fit: float,
	stationarity_mode: str,
	top_n_corr: int,
	exclude_cols: list[str],
	groups: list[str],
	winsor_lower: float | None,
	winsor_upper: float | None,
	selic_col: str,
) -> dict:
	panel_weekly = _load_panel(panel_path)
	metadata = _load_metadata(metadata_path)
	sample_config = _load_sample_config(sample_config_path)
	if "filtered" in panel_path.name.lower():
		warnings.warn("Arquivo de painel parece pré-filtrado; valide se o critério está alinhado à cobertura semanal de treino.")

	train_start = _parse_date(sample_config.get("train_start"))
	train_end = _parse_date(sample_config.get("train_end"))
	fit_mask = _build_fit_mask(panel_weekly, train_start, train_end)
	if not fit_mask.any():
		raise ValueError("Janela de treino vazia após aplicar sample_config")

	feature_cols = _build_feature_list(panel_weekly, metadata, exclude_cols)
	transformed, transform_meta = _apply_stationarity_transform(panel_weekly, feature_cols, metadata, fit_mask, stationarity_mode)

	transformed_filtered, coverage_df, dropped_cov = _filter_by_coverage(transformed, fit_mask, min_coverage_fit)
	full_features = transformed_filtered.drop(columns=["week_ref"])
	train_features = full_features.loc[fit_mask]

	train_imp, full_imp, medians = _median_impute_fit(train_features, full_features)
	winsor_bounds_df = pd.DataFrame(columns=["var_name_final", "winsor_lower", "winsor_upper"])
	if winsor_lower is not None or winsor_upper is not None:
		if winsor_lower is None or winsor_upper is None:
			raise ValueError("Para winsorização, informe ambos: winsor_lower e winsor_upper")
		train_proc, full_proc, winsor_bounds_df = _winsorize_fit(train_imp, full_imp, winsor_lower, winsor_upper)
	else:
		train_proc, full_proc = train_imp, full_imp

	train_z, full_z, scaler, dropped_zero_std = _zscore_fit(train_proc, full_proc)

	scores_all, comps_all, expl_all = _pca_fit_transform(train_z, full_z, n_components=n_components_all)
	scores_all, comps_all = _align_component_signs(scores_all, comps_all)

	score_all_cols = [f"pc_all_{i+1}" for i in range(scores_all.shape[1])]
	scores_all_df = _scores_df(transformed_filtered["week_ref"], scores_all, prefix="pc_all_")
	scores_all_fit_df = _scores_df(transformed_filtered.loc[fit_mask, "week_ref"], scores_all[fit_mask.values, :], prefix="pc_all_")
	loadings_all_df = _loadings_df(train_z.columns.tolist(), comps_all, prefix="pc_all_")
	expl_all_df = _explained_df(expl_all, prefix="pc_all_")
	corr_all_fit_df = _top_correlations(train_z, scores_all_fit_df[score_all_cols], score_all_cols, top_n=top_n_corr)
	corr_all_full_df = _top_correlations(full_z, scores_all_df[score_all_cols], score_all_cols, top_n=top_n_corr)

	group_map = _group_label_map(metadata)
	scores_group_df, load_group_df, var_group_df, corr_group_fit_df, corr_group_full_df = _run_group_pca(
		transformed_filtered=transformed_filtered[["week_ref"] + train_z.columns.tolist()],
		fit_mask=fit_mask,
		group_map=group_map,
		groups=groups,
		top_n_corr=top_n_corr,
	)

	fira_df = _build_fira(scores_group_df, panel_weekly, selic_col=selic_col)

	output_dir.mkdir(parents=True, exist_ok=True)

	out_scores_all = _save_parquet_with_csv_fallback(scores_all_df, output_dir / "pca_all_scores_weekly.parquet")
	out_load_all = output_dir / "pca_all_loadings.csv"
	out_var_all = output_dir / "pca_all_explained_variance.csv"
	out_corr_all_fit = output_dir / "pca_all_top_correlations_fit.csv"
	out_corr_all_full = output_dir / "pca_all_top_correlations_full.csv"
	out_corr_all = output_dir / "pca_all_top_correlations.csv"

	out_scores_group = _save_parquet_with_csv_fallback(scores_group_df, output_dir / "pca_group_scores_weekly.parquet")
	out_load_group = output_dir / "pca_group_loadings.csv"
	out_var_group = output_dir / "pca_group_explained_variance.csv"
	out_corr_group_fit = output_dir / "pca_group_top_correlations_fit.csv"
	out_corr_group_full = output_dir / "pca_group_top_correlations_full.csv"
	out_corr_group = output_dir / "pca_group_top_correlations.csv"

	out_fira = _save_parquet_with_csv_fallback(fira_df, output_dir / "pca_fira_state_weekly.parquet")
	out_transform = output_dir / "pca_transform_rules.csv"
	out_coverage = output_dir / "pca_feature_coverage.csv"
	out_scaler = output_dir / "pca_scaler_stats.csv"
	out_medians = output_dir / "pca_imputer_medians.csv"
	out_winsor = output_dir / "pca_winsor_bounds.csv"
	out_feature_spec = output_dir / "pca_feature_spec.json"
	out_summary = output_dir / "pca_run_summary.json"

	loadings_all_df.to_csv(out_load_all, index=False)
	expl_all_df.to_csv(out_var_all, index=False)
	corr_all_fit_df.to_csv(out_corr_all_fit, index=False)
	corr_all_full_df.to_csv(out_corr_all_full, index=False)
	corr_all_full_df.to_csv(out_corr_all, index=False)
	load_group_df.to_csv(out_load_group, index=False)
	var_group_df.to_csv(out_var_group, index=False)
	corr_group_fit_df.to_csv(out_corr_group_fit, index=False)
	corr_group_full_df.to_csv(out_corr_group_full, index=False)
	corr_group_full_df.to_csv(out_corr_group, index=False)
	transform_meta.to_csv(out_transform, index=False)
	coverage_df.to_csv(out_coverage, index=False)

	pd.DataFrame({"var_name_final": scaler.mean_.index, "mean_fit": scaler.mean_.values, "std_fit": scaler.std_.values}).to_csv(out_scaler, index=False)
	pd.DataFrame({"var_name_final": medians.index, "median_fit": medians.values}).to_csv(out_medians, index=False)
	winsor_bounds_df.to_csv(out_winsor, index=False)

	features_by_group_final = {grp: [] for grp in groups}
	for col in train_z.columns.tolist():
		grp = group_map.get(col, "financeiro")
		features_by_group_final.setdefault(grp, []).append(col)

	feature_spec = {
		"features_initial": feature_cols,
		"dropped_coverage": dropped_cov,
		"dropped_zero_std": dropped_zero_std,
		"features_final_pca_all": train_z.columns.tolist(),
		"features_by_group_final": features_by_group_final,
	}
	out_feature_spec.write_text(json.dumps(feature_spec, indent=2, ensure_ascii=False), encoding="utf-8")

	summary = {
		"panel_path": str(panel_path),
		"metadata_path": str(metadata_path),
		"n_rows_total": int(len(panel_weekly)),
		"n_rows_fit": int(fit_mask.sum()),
		"fit_start": str(panel_weekly.loc[fit_mask, "week_ref"].min().date()),
		"fit_end": str(panel_weekly.loc[fit_mask, "week_ref"].max().date()),
		"stationarity_mode": stationarity_mode,
		"n_transform_manual": int((transform_meta["transform_source"] == "manual").sum()) if not transform_meta.empty else 0,
		"min_coverage_fit": min_coverage_fit,
		"winsor_lower": winsor_lower,
		"winsor_upper": winsor_upper,
		"n_features_initial": int(len(feature_cols)),
		"n_features_after_coverage": int(full_features.shape[1]),
		"n_features_dropped_coverage": int(len(dropped_cov)),
		"n_features_dropped_zero_std": int(len(dropped_zero_std)),
		"n_components_all": int(scores_all.shape[1]),
		"explained_variance_all": expl_all.tolist(),
		"groups_used": groups,
		"group_components": scores_group_df.columns.drop("week_ref").tolist(),
		"fira_columns": fira_df.columns.tolist(),
	}
	out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

	return {
		"out_scores_all": out_scores_all,
		"out_load_all": out_load_all,
		"out_var_all": out_var_all,
		"out_corr_all_fit": out_corr_all_fit,
		"out_corr_all_full": out_corr_all_full,
		"out_corr_all": out_corr_all,
		"out_scores_group": out_scores_group,
		"out_load_group": out_load_group,
		"out_var_group": out_var_group,
		"out_corr_group_fit": out_corr_group_fit,
		"out_corr_group_full": out_corr_group_full,
		"out_corr_group": out_corr_group,
		"out_fira": out_fira,
		"out_transform": out_transform,
		"out_coverage": out_coverage,
		"out_scaler": out_scaler,
		"out_medians": out_medians,
		"out_winsor": out_winsor,
		"out_feature_spec": out_feature_spec,
		"out_summary": out_summary,
		"feature_spec": feature_spec,
		"summary": summary,
	}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Passo 4: PCA(all), PCA(group) e FIRA no painel macro-financeiro")
	parser.add_argument(
		"--panel-path",
		type=Path,
		default=Path(r"Q:\Gabriel de Macedo\Política Monetária\NS Model\NS v2\data\panel_weekly_raw_pit_filtered.parquet"),
	)
	parser.add_argument(
		"--metadata-path",
		type=Path,
		default=Path(r"Q:\Gabriel de Macedo\Política Monetária\NS Model\NS v2\data\metadata_variables_review.csv"),
	)
	parser.add_argument(
		"--sample-config-path",
		type=Path,
		default=Path(r"Q:\Gabriel de Macedo\Política Monetária\NS Model\NS v2\data\sample_config.json"),
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path(r"Q:\Gabriel de Macedo\Política Monetária\NS Model\NS v2\data"),
	)
	parser.add_argument("--n-components-all", type=int, default=5)
	parser.add_argument("--min-coverage-fit", type=float, default=0.70)
	parser.add_argument("--stationarity-mode", type=str, choices=["adf_or_acf", "none"], default="adf_or_acf")
	parser.add_argument("--top-n-corr", type=int, default=5)
	parser.add_argument("--selic-col", type=str, default="bbg_selic")
	parser.add_argument(
		"--groups",
		type=str,
		default=",".join(DEFAULT_GROUPS),
		help="Grupos para PCA(group), separados por vírgula",
	)
	parser.add_argument(
		"--winsor-lower",
		type=float,
		default=None,
		help="Quantil inferior de winsorização (ex.: 0.01). Se informado, requer --winsor-upper.",
	)
	parser.add_argument(
		"--winsor-upper",
		type=float,
		default=None,
		help="Quantil superior de winsorização (ex.: 0.99). Se informado, requer --winsor-lower.",
	)
	parser.add_argument(
		"--exclude-cols",
		type=str,
		default=",".join(DEFAULT_EXCLUDE),
		help="Colunas excluídas do PCA (separadas por vírgula)",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	exclude_cols = [c.strip() for c in args.exclude_cols.split(",") if c.strip()]
	groups = [g.strip() for g in args.groups.split(",") if g.strip()]

	outputs = run_pipeline(
		panel_path=args.panel_path,
		metadata_path=args.metadata_path,
		sample_config_path=args.sample_config_path,
		output_dir=args.output_dir,
		n_components_all=args.n_components_all,
		min_coverage_fit=args.min_coverage_fit,
		stationarity_mode=args.stationarity_mode,
		top_n_corr=args.top_n_corr,
		exclude_cols=exclude_cols,
		groups=groups,
		winsor_lower=args.winsor_lower,
		winsor_upper=args.winsor_upper,
		selic_col=args.selic_col,
	)

	print("Passo 4 (PCA) concluído.")
	for key in [
		"out_scores_all",
		"out_scores_group",
		"out_fira",
		"out_load_all",
		"out_var_all",
		"out_summary",
	]:
		print(f"{key}: {outputs[key]}")
	print(f"n_features_after_coverage: {outputs['summary']['n_features_after_coverage']}")
	print(f"n_components_all: {outputs['summary']['n_components_all']}")


if __name__ == "__main__":
	main()