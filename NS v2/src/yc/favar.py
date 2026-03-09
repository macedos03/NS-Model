from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
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

MODEL_STATES = {
	"dl": ["beta1_level", "beta2_slope", "beta3_curvature"],
	"dlfavar_all": [
		"beta1_level",
		"beta2_slope",
		"beta3_curvature",
		"pc_all_1",
		"pc_all_2",
		"pc_all_3",
		"pc_all_4",
		"pc_all_5",
		"bbg_selic",
	],
	"dlfavar_group": [
		"beta1_level",
		"beta2_slope",
		"beta3_curvature",
		"pc1_inflacao",
		"pc1_atividade",
		"pc1_fiscal",
		"pc1_risco",
		"pc1_incerteza",
		"pc1_financeiro",
		"bbg_selic",
	],
	"dlfavar_fira": [
		"beta1_level",
		"beta2_slope",
		"beta3_curvature",
		"pc1_inflacao",
		"pc1_atividade",
		"pc1_fiscal",
		"bbg_selic",
	],
}


@dataclass
class VarFitResult:
	p: int
	state_columns: list[str]
	coef_B: np.ndarray
	intercept_c: np.ndarray
	A_matrices: list[np.ndarray]
	residuals: np.ndarray
	sigma_u: np.ndarray
	n_obs: int
	eigenvalues: np.ndarray
	is_stable: bool
	aic: float
	ljungbox_min_pvalue: float | None
	resid_autocorr_flag: bool | None


def _load_dataframe(path: Path) -> pd.DataFrame:
	if not path.exists():
		raise FileNotFoundError(f"Arquivo não encontrado: {path}")
	if path.suffix.lower() == ".parquet":
		return pd.read_parquet(path)
	return pd.read_csv(path)


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


def _parse_horizons(text: str) -> list[int]:
	out = sorted({int(x.strip()) for x in text.split(",") if x.strip()})
	if not out or any(x <= 0 for x in out):
		raise ValueError("--horizons precisa conter inteiros positivos (ex.: 4,13,26,52)")
	return out


def _ensure_week_ref(df: pd.DataFrame) -> pd.DataFrame:
	if "week_ref" not in df.columns:
		raise ValueError("DataFrame sem coluna 'week_ref'")
	out = df.copy()
	out["week_ref"] = pd.to_datetime(out["week_ref"], errors="coerce")
	out = out.dropna(subset=["week_ref"]).sort_values("week_ref")
	out = out.drop_duplicates(subset=["week_ref"], keep="last").reset_index(drop=True)
	return out


def _week_frequency_stats(week_ref: pd.Series) -> dict:
	if week_ref.size < 2:
		return {"n_gaps_non_7d": 0, "max_gap_days": None, "min_gap_days": None}
	delta = week_ref.diff().dt.days.dropna()
	return {
		"n_gaps_non_7d": int((delta != 7).sum()),
		"max_gap_days": int(delta.max()),
		"min_gap_days": int(delta.min()),
	}


def _build_state_tables(
	ns_factors: pd.DataFrame,
	pca_all: pd.DataFrame,
	pca_group: pd.DataFrame,
	pca_fira: pd.DataFrame,
	panel_weekly: pd.DataFrame,
	selic_col: str,
) -> dict[str, pd.DataFrame]:
	ns = _ensure_week_ref(ns_factors)
	all_df = _ensure_week_ref(pca_all)
	group_df = _ensure_week_ref(pca_group)
	fira_df = _ensure_week_ref(pca_fira)
	panel = _ensure_week_ref(panel_weekly)

	selic_df = None
	for candidate in [fira_df, panel, all_df, group_df]:
		if selic_col in candidate.columns:
			selic_df = candidate[["week_ref", selic_col]].copy()
			break
	if selic_df is None:
		raise ValueError(
			f"Coluna Selic '{selic_col}' não encontrada em nenhuma base candidata (fira/panel/all/group)"
		)

	state_dl = ns[["week_ref", "beta1_level", "beta2_slope", "beta3_curvature"]].copy()

	all_need = [f"pc_all_{i}" for i in range(1, 6)]
	for col in all_need:
		if col not in all_df.columns:
			raise ValueError(f"Coluna ausente em pca_all_scores_weekly: {col}")
	state_all = state_dl.merge(all_df[["week_ref"] + all_need], on="week_ref", how="inner")
	state_all = state_all.merge(selic_df, on="week_ref", how="inner")

	group_need = [
		"pc1_inflacao",
		"pc1_atividade",
		"pc1_fiscal",
		"pc1_risco",
		"pc1_incerteza",
		"pc1_financeiro",
	]
	for col in group_need:
		if col not in group_df.columns:
			raise ValueError(f"Coluna ausente em pca_group_scores_weekly: {col}")
	state_group = state_dl.merge(group_df[["week_ref"] + group_need], on="week_ref", how="inner")
	state_group = state_group.merge(selic_df, on="week_ref", how="inner")

	fira_need = ["pc1_inflacao", "pc1_atividade", "pc1_fiscal", selic_col]
	for col in fira_need:
		if col not in fira_df.columns:
			raise ValueError(f"Coluna ausente em pca_fira_state_weekly: {col}")
	state_fira = state_dl.merge(fira_df[["week_ref"] + fira_need], on="week_ref", how="inner")

	states = {
		"dl": state_dl,
		"dlfavar_all": state_all,
		"dlfavar_group": state_group,
		"dlfavar_fira": state_fira,
	}

	for model_id, df in states.items():
		cols = MODEL_STATES[model_id]
		df2 = _ensure_week_ref(df)
		df2 = df2[["week_ref"] + cols].copy()
		df2 = df2.dropna(subset=cols).reset_index(drop=True)
		states[model_id] = df2

	return states


def _build_var_matrices(z: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
	T, K = z.shape
	if T <= p:
		raise ValueError("Amostra insuficiente para o lag informado")

	X_rows = []
	Y_rows = []
	reg_names = ["intercept"]
	for lag in range(1, p + 1):
		for k in range(K):
			reg_names.append(f"lag{lag}_s{k}")

	for t in range(p, T):
		x = [1.0]
		for lag in range(1, p + 1):
			x.extend(z[t - lag, :].tolist())
		X_rows.append(x)
		Y_rows.append(z[t, :].tolist())

	X = np.asarray(X_rows, dtype=float)
	Y = np.asarray(Y_rows, dtype=float)
	return X, Y, reg_names


def _companion_eigenvalues(A_matrices: list[np.ndarray]) -> np.ndarray:
	if not A_matrices:
		return np.array([])
	K = A_matrices[0].shape[0]
	p = len(A_matrices)
	if p == 1:
		return np.linalg.eigvals(A_matrices[0])

	comp = np.zeros((K * p, K * p), dtype=float)
	comp[:K, : K * p] = np.concatenate(A_matrices, axis=1)
	comp[K:, :-K] = np.eye(K * (p - 1))
	return np.linalg.eigvals(comp)


def _ljungbox_min_pvalue(residuals: np.ndarray, max_lag: int = 12) -> float | None:
	try:
		from statsmodels.stats.diagnostic import acorr_ljungbox  # type: ignore
	except Exception:
		return None

	if residuals.size == 0:
		return None
	n_obs = residuals.shape[0]
	test_lag = int(min(max_lag, max(1, n_obs // 5)))
	if test_lag < 1:
		return None

	pvals = []
	for i in range(residuals.shape[1]):
		series = residuals[:, i]
		if np.all(~np.isfinite(series)):
			continue
		try:
			res = acorr_ljungbox(series, lags=[test_lag], return_df=True)
			pval = float(res["lb_pvalue"].iloc[-1])
			if np.isfinite(pval):
				pvals.append(pval)
		except Exception:
			continue

	if not pvals:
		return None
	return float(min(pvals))


def fit_var_ols(state_df: pd.DataFrame, state_columns: list[str], p: int) -> VarFitResult:
	z = state_df[state_columns].to_numpy(dtype=float)
	X, Y, _ = _build_var_matrices(z, p)

	coef_B, *_ = np.linalg.lstsq(X, Y, rcond=None)
	Y_hat = X @ coef_B
	U = Y - Y_hat

	T_eff, K = U.shape
	params_per_eq = X.shape[1]
	dof = T_eff - params_per_eq
	den = dof if dof > 0 else T_eff
	sigma_u = (U.T @ U) / max(1, den)

	c = coef_B[0, :]
	A_matrices = []
	idx = 1
	for _ in range(p):
		A = coef_B[idx : idx + K, :].T
		A_matrices.append(A)
		idx += K

	eigs = _companion_eigenvalues(A_matrices)
	is_stable = bool(np.all(np.abs(eigs) < 1.0)) if eigs.size else True

	sigma_ml = (U.T @ U) / max(1, T_eff)
	sign, logdet = np.linalg.slogdet(sigma_ml)
	if sign <= 0 or not np.isfinite(logdet):
		aic = float("inf")
	else:
		n_params_sys = K * (1 + K * p)
		aic = float(logdet + (2.0 * n_params_sys) / T_eff)

	lb_pmin = _ljungbox_min_pvalue(U)
	autocorr_flag = None if lb_pmin is None else bool(lb_pmin < 0.05)

	return VarFitResult(
		p=p,
		state_columns=state_columns,
		coef_B=coef_B,
		intercept_c=c,
		A_matrices=A_matrices,
		residuals=U,
		sigma_u=sigma_u,
		n_obs=T_eff,
		eigenvalues=eigs,
		is_stable=is_stable,
		aic=aic,
		ljungbox_min_pvalue=lb_pmin,
		resid_autocorr_flag=autocorr_flag,
	)


def select_lag_by_aic(
	state_df: pd.DataFrame,
	state_columns: list[str],
	p_grid: list[int],
) -> tuple[int, pd.DataFrame]:
	rows = []
	valid = []
	K = len(state_columns)
	T = len(state_df)

	for p in p_grid:
		params_per_eq = 1 + K * p
		T_eff = T - p
		if T_eff <= params_per_eq + 1:
			rows.append({"p": p, "aic": np.nan, "valid": False, "reason": "insufficient_dof"})
			continue
		try:
			fit = fit_var_ols(state_df, state_columns, p)
			rows.append({"p": p, "aic": fit.aic, "valid": np.isfinite(fit.aic), "reason": "ok"})
			if np.isfinite(fit.aic):
				valid.append((p, fit.aic))
		except Exception as exc:
			rows.append({"p": p, "aic": np.nan, "valid": False, "reason": f"fit_error:{type(exc).__name__}"})

	aic_df = pd.DataFrame(rows).sort_values("p").reset_index(drop=True)
	if not valid:
		raise ValueError("Nenhum lag válido para seleção por AIC")
	best_p = sorted(valid, key=lambda x: x[1])[0][0]
	return int(best_p), aic_df


def _iterative_state_forecast(last_history: np.ndarray, fit: VarFitResult, h_max: int) -> np.ndarray:
	if h_max < 1:
		return np.empty((0, len(fit.state_columns)))

	p = fit.p
	K = len(fit.state_columns)
	history = [last_history[i, :].copy() for i in range(last_history.shape[0])]
	preds = np.zeros((h_max, K), dtype=float)

	for h in range(h_max):
		zh = fit.intercept_c.copy()
		for lag in range(1, p + 1):
			zh += fit.A_matrices[lag - 1] @ history[-lag]
		preds[h, :] = zh
		history.append(zh)

	return preds


def _ns_design_matrix(curve_map: dict[str, float], lambda_ns: float) -> tuple[list[str], np.ndarray]:
	curve_cols = list(curve_map.keys())
	tau = np.asarray([curve_map[c] for c in curve_cols], dtype=float)
	x = lambda_ns * tau
	l2 = (1.0 - np.exp(-x)) / x
	l3 = l2 - np.exp(-x)
	X_ns = np.column_stack([np.ones_like(tau), l2, l3])
	return curve_cols, X_ns


def _choose_oos_window(
	state_df: pd.DataFrame,
	train_start: pd.Timestamp | None,
	oos_start: pd.Timestamp | None,
	oos_end: pd.Timestamp | None,
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
	weeks = state_df["week_ref"]
	min_w = weeks.min()
	max_w = weeks.max()

	if train_start is None:
		train_start = min_w

	if oos_start is None:
		split_idx = int(np.floor(0.70 * (len(state_df) - 1)))
		oos_start = pd.Timestamp(weeks.iloc[max(1, split_idx)])

	if oos_end is None:
		oos_end = max_w

	if not (train_start <= oos_start <= oos_end):
		raise ValueError("Janela inválida: requer train_start <= oos_start <= oos_end")

	return train_start, oos_start, oos_end


def _effective_p_grid(n_rows: int, n_states: int, p_max: int) -> list[int]:
	valid = []
	for p in range(1, p_max + 1):
		T_eff = n_rows - p
		params_per_eq = 1 + n_states * p
		if T_eff > params_per_eq + 1:
			valid.append(p)
	return valid


def _target_week_by_calendar(origin_week: pd.Timestamp, h: int) -> pd.Timestamp:
	return pd.Timestamp(origin_week + pd.Timedelta(days=7 * int(h)))


def _build_rw_yield_forecasts(
	model_id: str,
	origin_weeks: pd.Series,
	yields_obs: pd.DataFrame,
	curve_map: dict[str, float],
	horizons: list[int],
) -> pd.DataFrame:
	curve_cols = list(curve_map.keys())
	yields = _ensure_week_ref(yields_obs[["week_ref"] + curve_cols])
	yields_map = yields.set_index("week_ref")
	rows = []

	for origin_week in origin_weeks:
		origin_week = pd.Timestamp(origin_week)
		if origin_week not in yields_map.index:
			continue

		origin_vals = yields_map.loc[origin_week, curve_cols]
		for h in horizons:
			target_week = _target_week_by_calendar(origin_week, h)
			for col in curve_cols:
				y_pred = float(origin_vals[col])
				tau = float(curve_map[col])
				y_obs = np.nan
				if target_week in yields_map.index:
					y_obs = float(yields_map.loc[target_week, col])
				err = float(y_pred - y_obs) if np.isfinite(y_obs) else np.nan
				rows.append(
					{
						"model_id": model_id,
						"origin_week_ref": origin_week,
						"target_week_ref": target_week,
						"horizon_steps": int(h),
						"tau_years": tau,
						"yield_col": col,
						"yield_pred": y_pred,
						"yield_obs": y_obs,
						"error": err,
						"abs_error": float(abs(err)) if np.isfinite(err) else np.nan,
					}
				)

	return pd.DataFrame(rows)


def run_model_backtest(
	model_id: str,
	state_df: pd.DataFrame,
	yields_obs: pd.DataFrame,
	curve_map: dict[str, float],
	lambda_ns: float,
	horizons: list[int],
	p_max: int,
	train_start: pd.Timestamp | None,
	oos_start: pd.Timestamp | None,
	oos_end: pd.Timestamp | None,
	strict_weekly_horizon: bool,
) -> dict:
	state_columns = MODEL_STATES[model_id]
	df = _ensure_week_ref(state_df[["week_ref"] + state_columns])

	train_start_eff, oos_start_eff, oos_end_eff = _choose_oos_window(df, train_start, oos_start, oos_end)

	fit_for_lag = df.loc[(df["week_ref"] >= train_start_eff) & (df["week_ref"] < oos_start_eff)].copy()
	if fit_for_lag.empty:
		raise ValueError(f"Amostra vazia para seleção de lag antes do OOS ({model_id})")

	week_delta = fit_for_lag["week_ref"].diff().dt.days.dropna()
	n_non7 = int((week_delta != 7).sum()) if not week_delta.empty else 0
	if strict_weekly_horizon and n_non7 > 0:
		raise ValueError(
			f"Série de estado '{model_id}' não é semanal contínua antes do OOS (gaps != 7d: {n_non7})."
		)
	if (not strict_weekly_horizon) and n_non7 > 0:
		warnings.warn(
			f"Série de estado '{model_id}' tem {n_non7} gaps != 7d antes do OOS; horizonte será por calendário (7*h dias)."
		)

	p_grid = _effective_p_grid(len(fit_for_lag), len(state_columns), p_max)
	if not p_grid:
		raise ValueError(
			f"Amostra pré-OOS insuficiente para qualquer lag válido em '{model_id}'. "
			f"Reduza p_max (atual={p_max}) ou antecipe oos_start."
		)
	best_p, aic_df = select_lag_by_aic(fit_for_lag, state_columns, p_grid)

	curve_cols, X_ns = _ns_design_matrix(curve_map, lambda_ns)
	yields = _ensure_week_ref(yields_obs[["week_ref"] + curve_cols])
	yields_map = yields.set_index("week_ref")

	h_max = max(horizons)
	state_rows = []
	yield_rows = []
	origin_diag = []

	df = df.reset_index(drop=True)
	origin_mask = (df["week_ref"] >= oos_start_eff) & (df["week_ref"] <= oos_end_eff)
	origin_idx = df.index[origin_mask].tolist()

	beta_idx = [state_columns.index("beta1_level"), state_columns.index("beta2_slope"), state_columns.index("beta3_curvature")]

	for i in origin_idx:
		train_df = df.iloc[: i + 1].copy()
		train_df = train_df.loc[train_df["week_ref"] >= train_start_eff].copy()
		if len(train_df) <= best_p + 5:
			continue

		fit = fit_var_ols(train_df, state_columns, best_p)
		hist_arr = train_df[state_columns].to_numpy(dtype=float)
		preds = _iterative_state_forecast(hist_arr, fit, h_max=h_max)

		origin_week = pd.Timestamp(df.loc[i, "week_ref"])
		origin_diag.append(
			{
				"model_id": model_id,
				"origin_week_ref": origin_week,
				"lag_order_p": int(best_p),
				"n_obs_train": int(fit.n_obs),
				"aic_train": float(fit.aic),
				"is_stable": bool(fit.is_stable),
				"max_abs_eig": float(np.max(np.abs(fit.eigenvalues))) if fit.eigenvalues.size else 0.0,
				"ljungbox_min_pvalue": fit.ljungbox_min_pvalue,
				"resid_autocorr_flag": fit.resid_autocorr_flag,
			}
		)

		for h in horizons:
			pred_state = preds[h - 1, :]
			target_week = _target_week_by_calendar(origin_week, h)

			for j, name in enumerate(state_columns):
				state_rows.append(
					{
						"model_id": model_id,
						"origin_week_ref": origin_week,
						"target_week_ref": target_week,
						"horizon_steps": int(h),
						"state_name": name,
						"state_pred": float(pred_state[j]),
					}
				)

			beta_hat = pred_state[beta_idx]
			y_hat = X_ns @ beta_hat
			for m, col in enumerate(curve_cols):
				tau = float(curve_map[col])
				y_obs = np.nan
				if pd.notna(target_week) and target_week in yields_map.index:
					y_obs = float(yields_map.loc[target_week, col])
				err = float(y_hat[m] - y_obs) if np.isfinite(y_obs) else np.nan
				yield_rows.append(
					{
						"model_id": model_id,
						"origin_week_ref": origin_week,
						"target_week_ref": target_week,
						"horizon_steps": int(h),
						"tau_years": tau,
						"yield_col": col,
						"yield_pred": float(y_hat[m]),
						"yield_obs": y_obs,
						"error": err,
						"abs_error": float(abs(err)) if np.isfinite(err) else np.nan,
					}
				)

	base_fit = fit_var_ols(fit_for_lag, state_columns, best_p)

	return {
		"model_id": model_id,
		"state_columns": state_columns,
		"lag_order_p": int(best_p),
		"aic_by_p": aic_df,
		"base_fit": base_fit,
		"origin_var_diag": pd.DataFrame(origin_diag),
		"state_forecasts": pd.DataFrame(state_rows),
		"yield_forecasts": pd.DataFrame(yield_rows),
		"window": {
			"train_start": str(train_start_eff.date()),
			"oos_start": str(oos_start_eff.date()),
			"oos_end": str(oos_end_eff.date()),
		},
		"weekly_continuity_gaps_pre_oos": n_non7,
	}


def _coef_matrix_to_df(coef_B: np.ndarray, state_columns: list[str], p: int) -> pd.DataFrame:
	reg_names = ["intercept"]
	for lag in range(1, p + 1):
		for col in state_columns:
			reg_names.append(f"lag{lag}_{col}")

	df = pd.DataFrame(coef_B, columns=state_columns)
	df.insert(0, "regressor", reg_names)
	return df


def _a_matrices_long(A_matrices: list[np.ndarray], state_columns: list[str]) -> pd.DataFrame:
	rows = []
	for lag, A in enumerate(A_matrices, start=1):
		for i, row_name in enumerate(state_columns):
			for j, col_name in enumerate(state_columns):
				rows.append(
					{
						"lag": lag,
						"row_state": row_name,
						"col_state": col_name,
						"coef": float(A[i, j]),
					}
				)
	return pd.DataFrame(rows)


def _matrix_to_long(matrix: np.ndarray, names: list[str], value_col: str) -> pd.DataFrame:
	rows = []
	for i, rname in enumerate(names):
		for j, cname in enumerate(names):
			rows.append({"row_state": rname, "col_state": cname, value_col: float(matrix[i, j])})
	return pd.DataFrame(rows)


def run_pipeline(
	ns_factors_path: Path,
	pca_all_path: Path,
	pca_group_path: Path,
	pca_fira_path: Path,
	panel_weekly_path: Path,
	sample_config_path: Path,
	output_dir: Path,
	lambda_ns: float,
	curve_map: dict[str, float],
	p_max: int,
	horizons: list[int],
	selic_col: str,
	strict_weekly_horizon: bool,
) -> dict:
	ns_factors = _load_dataframe(ns_factors_path)
	pca_all = _load_dataframe(pca_all_path)
	pca_group = _load_dataframe(pca_group_path)
	pca_fira = _load_dataframe(pca_fira_path)
	panel_weekly = _load_dataframe(panel_weekly_path)

	sample_cfg = {}
	if sample_config_path.exists():
		sample_cfg = json.loads(sample_config_path.read_text(encoding="utf-8"))

	train_start = _parse_date(sample_cfg.get("train_start"))
	oos_start = _parse_date(sample_cfg.get("oos_start"))
	oos_end = _parse_date(sample_cfg.get("oos_end"))

	states = _build_state_tables(
		ns_factors=ns_factors,
		pca_all=pca_all,
		pca_group=pca_group,
		pca_fira=pca_fira,
		panel_weekly=panel_weekly,
		selic_col=selic_col,
	)

	output_dir.mkdir(parents=True, exist_ok=True)

	state_out_paths = {}
	for model_id, state_df in states.items():
		state_out = _save_parquet_with_csv_fallback(state_df, output_dir / f"state_{model_id}_weekly.parquet")
		state_out_paths[model_id] = state_out

	model_results = {}
	run_summary_models = {}
	yields_obs = _ensure_week_ref(panel_weekly)
	rw_frames = []

	for model_id, state_df in states.items():
		result = run_model_backtest(
			model_id=model_id,
			state_df=state_df,
			yields_obs=yields_obs,
			curve_map=curve_map,
			lambda_ns=lambda_ns,
			horizons=horizons,
			p_max=p_max,
			train_start=train_start,
			oos_start=oos_start,
			oos_end=oos_end,
			strict_weekly_horizon=strict_weekly_horizon,
		)
		model_results[model_id] = result

		rw_df = _build_rw_yield_forecasts(
			model_id="rw_yield",
			origin_weeks=result["origin_var_diag"]["origin_week_ref"] if not result["origin_var_diag"].empty else pd.Series(dtype="datetime64[ns]"),
			yields_obs=yields_obs,
			curve_map=curve_map,
			horizons=horizons,
		)
		rw_frames.append(rw_df)

		base_fit: VarFitResult = result["base_fit"]
		aic_df = result["aic_by_p"]
		coef_df = _coef_matrix_to_df(base_fit.coef_B, result["state_columns"], result["lag_order_p"])
		A_long_df = _a_matrices_long(base_fit.A_matrices, result["state_columns"])
		sigma_u_df = _matrix_to_long(base_fit.sigma_u, result["state_columns"], value_col="sigma_u")
		eig_df = pd.DataFrame(
			{
				"eig_real": np.real(base_fit.eigenvalues),
				"eig_imag": np.imag(base_fit.eigenvalues),
				"eig_abs": np.abs(base_fit.eigenvalues),
			}
		)

		aic_df.to_csv(output_dir / f"var_{model_id}_aic_by_p.csv", index=False)
		coef_df.to_csv(output_dir / f"var_{model_id}_coef_matrix_B.csv", index=False)
		A_long_df.to_csv(output_dir / f"var_{model_id}_A_matrices.csv", index=False)
		sigma_u_df.to_csv(output_dir / f"var_{model_id}_sigma_u.csv", index=False)
		eig_df.to_csv(output_dir / f"var_{model_id}_stability_eigenvalues.csv", index=False)
		result["origin_var_diag"].to_csv(output_dir / f"var_{model_id}_oos_origin_diag.csv", index=False)
		result["state_forecasts"].to_csv(output_dir / f"forecast_states_{model_id}.csv", index=False)
		result["yield_forecasts"].to_csv(output_dir / f"forecast_yields_{model_id}.csv", index=False)

		summary = {
			"model_id": model_id,
			"state_columns": result["state_columns"],
			"lag_order_p": int(result["lag_order_p"]),
			"window": result["window"],
			"n_rows_state": int(len(state_df)),
			"week_frequency": _week_frequency_stats(state_df["week_ref"]),
			"var_base_is_stable": bool(base_fit.is_stable),
			"var_base_max_abs_eig": float(np.max(np.abs(base_fit.eigenvalues))) if base_fit.eigenvalues.size else 0.0,
			"var_base_ljungbox_min_pvalue": base_fit.ljungbox_min_pvalue,
			"var_base_resid_autocorr_flag": base_fit.resid_autocorr_flag,
			"n_oos_origins": int(result["origin_var_diag"].shape[0]),
			"n_state_forecasts": int(result["state_forecasts"].shape[0]),
			"n_yield_forecasts": int(result["yield_forecasts"].shape[0]),
			"weekly_continuity_gaps_pre_oos": int(result["weekly_continuity_gaps_pre_oos"]),
		}
		run_summary_models[model_id] = summary
		(output_dir / f"var_{model_id}_summary.json").write_text(
			json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
		)

	run_summary = {
		"lambda_ns": float(lambda_ns),
		"curve_map": curve_map,
		"p_max": int(p_max),
		"horizons_steps": horizons,
		"selic_col": selic_col,
		"strict_weekly_horizon": bool(strict_weekly_horizon),
		"paths": {
			"ns_factors": str(ns_factors_path),
			"pca_all": str(pca_all_path),
			"pca_group": str(pca_group_path),
			"pca_fira": str(pca_fira_path),
			"panel_weekly": str(panel_weekly_path),
			"sample_config": str(sample_config_path),
		},
		"models": run_summary_models,
	}

	rw_all = pd.concat(rw_frames, ignore_index=True) if rw_frames else pd.DataFrame()
	if not rw_all.empty:
		rw_all = rw_all.drop_duplicates(
			subset=["model_id", "origin_week_ref", "target_week_ref", "horizon_steps", "tau_years", "yield_col"]
		).reset_index(drop=True)
	rw_all.to_csv(output_dir / "forecast_yields_rw.csv", index=False)

	out_summary = output_dir / "favar_run_summary.json"
	out_summary.write_text(json.dumps(run_summary, indent=2, ensure_ascii=False), encoding="utf-8")

	return {
		"state_outputs": state_out_paths,
		"model_results": model_results,
		"run_summary": run_summary,
		"out_summary": out_summary,
	}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Passos 5-7: montagem do estado, VAR(p) e forecast iterado NS/FAVAR")
	parser.add_argument(
		"--ns-factors-path",
		type=Path,
		default=Path(r"Q:\Gabriel de Macedo\Política Monetária\NS Model\NS v2\data\ns_factors_weekly.parquet"),
	)
	parser.add_argument(
		"--pca-all-path",
		type=Path,
		default=Path(r"Q:\Gabriel de Macedo\Política Monetária\NS Model\NS v2\data\pca_all_scores_weekly.parquet"),
	)
	parser.add_argument(
		"--pca-group-path",
		type=Path,
		default=Path(r"Q:\Gabriel de Macedo\Política Monetária\NS Model\NS v2\data\pca_group_scores_weekly.parquet"),
	)
	parser.add_argument(
		"--pca-fira-path",
		type=Path,
		default=Path(r"Q:\Gabriel de Macedo\Política Monetária\NS Model\NS v2\data\pca_fira_state_weekly.parquet"),
	)
	parser.add_argument(
		"--panel-weekly-path",
		type=Path,
		default=Path(r"Q:\Gabriel de Macedo\Política Monetária\NS Model\NS v2\data\panel_weekly_ns_curve_common.parquet"),
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
	parser.add_argument("--lambda-ns", type=float, default=LAMBDA_NS_DEFAULT)
	parser.add_argument(
		"--curve-map",
		type=str,
		default="bbg_bz_bond_1y:1,bbg_bz_bond_3y:3,bbg_bz_bond_5y:5,bbg_bz_bond_10y:10",
	)
	parser.add_argument("--p-max", type=int, default=12)
	parser.add_argument("--horizons", type=str, default="4,13,26,52")
	parser.add_argument("--selic-col", type=str, default="bbg_selic")
	parser.add_argument(
		"--strict-weekly-horizon",
		type=lambda x: str(x).strip().lower() in {"1", "true", "t", "yes", "y"},
		default=True,
		help="Se True, exige week_ref contínuo (7 dias) no pré-OOS; se False, usa alvo por calendário com warning.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	curve_map = _parse_curve_map(args.curve_map)
	horizons = _parse_horizons(args.horizons)

	outputs = run_pipeline(
		ns_factors_path=args.ns_factors_path,
		pca_all_path=args.pca_all_path,
		pca_group_path=args.pca_group_path,
		pca_fira_path=args.pca_fira_path,
		panel_weekly_path=args.panel_weekly_path,
		sample_config_path=args.sample_config_path,
		output_dir=args.output_dir,
		lambda_ns=args.lambda_ns,
		curve_map=curve_map,
		p_max=args.p_max,
		horizons=horizons,
		selic_col=args.selic_col,
		strict_weekly_horizon=args.strict_weekly_horizon,
	)

	print("Passos 5-7 (FAVAR/VAR/Forecast) concluídos.")
	print(f"summary: {outputs['out_summary']}")
	for model_id, out_path in outputs["state_outputs"].items():
		print(f"state_{model_id}: {out_path}")


if __name__ == "__main__":
	main()
