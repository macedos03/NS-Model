from __future__ import annotations

import numpy as np
import pandas as pd

class modeling:
    # --------------------------
    # Nelson–Siegel helpers
    # --------------------------
    @staticmethod
    def _ns_loadings(tau_years: np.ndarray, lam: float) -> np.ndarray:
        """
        Nelson-Siegel loadings matrix X(tau):
          y(tau) = β0*1 + β1*L1(tau) + β2*L2(tau)

        tau_years: maturidade em anos (ex.: 12m -> 1.0)
        lam: λ > 0
        """
        tau = np.asarray(tau_years, dtype=float)
        lam = float(lam)
        if lam <= 0:
            raise ValueError("lambda (lam) precisa ser > 0.")

        eps = 1e-8
        x = lam * np.maximum(tau, eps)

        L1 = (1.0 - np.exp(-x)) / x
        L2 = L1 - np.exp(-x)

        X = np.column_stack([np.ones_like(tau), L1, L2])
        return X

    @staticmethod
    def _drop_cross_section_outliers_mad(
        y: pd.Series,
        z_thresh: float = 8.0,
        min_keep: int = 4,
    ) -> pd.Series:
        """
        Remove outliers no corte transversal (mesmo dia) via z-score robusto (MAD).
        """
        y = y.dropna()
        if len(y) <= min_keep:
            return y

        med = float(y.median())
        mad = float((y - med).abs().median())
        if mad <= 1e-12:
            return y

        z = 0.6745 * (y - med) / mad
        y2 = y.loc[z.abs() <= z_thresh]

        if len(y2) < min_keep:
            return y
        return y2

    # --------------------------
    # Fit diário Nelson–Siegel (Caminho 1)
    # --------------------------
    @staticmethod
    def fit_nelson_siegel_daily(
        df_di: pd.DataFrame,
        maturities_fit_months: list[int],
        maturities_target_months: list[int],
        lam: float = 0.7308,
        min_points_per_day: int = 4,
        drop_outliers_mad: bool = True,
        mad_z_thresh: float = 8.0,
        y_min_ok: float | None = None,
        y_max_ok: float | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Ajusta Nelson-Siegel diariamente usando os vértices observados disponíveis no dia.
        """

        df = df_di.copy().sort_index()

        fit_cols = [f"DI_{m}m" for m in maturities_fit_months if f"DI_{m}m" in df.columns]
        if len(fit_cols) == 0:
            raise ValueError("Nenhuma coluna de fit encontrada em df_di para maturities_fit_months.")

        target_months = list(maturities_target_months)
        tau_target_years = np.array([m / 12.0 for m in target_months], dtype=float)
        X_target = modeling._ns_loadings(tau_target_years, lam=lam)

        betas_rows = []
        curve_rows = []

        for dt, row in df[fit_cols].iterrows():
            y = row.dropna()

            if y_min_ok is not None:
                y = y[y >= float(y_min_ok)]
            if y_max_ok is not None:
                y = y[y <= float(y_max_ok)]

            if drop_outliers_mad:
                y = modeling._drop_cross_section_outliers_mad(
                    y, z_thresh=mad_z_thresh, min_keep=min_points_per_day
                )

            if len(y) < min_points_per_day:
                continue

            used_months = [int(c.replace("DI_", "").replace("m", "")) for c in y.index]
            tau_years = np.array([m / 12.0 for m in used_months], dtype=float)

            X = modeling._ns_loadings(tau_years, lam=lam)
            yvec = y.values.astype(float)

            beta, *_ = np.linalg.lstsq(X, yvec, rcond=None)

            yhat = X @ beta
            rmse = float(np.sqrt(np.mean((yvec - yhat) ** 2)))

            y_target = X_target @ beta

            betas_rows.append({
                "Date": dt,
                "beta0": float(beta[0]),
                "beta1": float(beta[1]),
                "beta2": float(beta[2]),
                "lambda": float(lam),
                "rmse_fit": rmse,
                "n_points_fit": int(len(y)),
                "maturities_used_months": ",".join(map(str, used_months)),
            })

            curve_row = {"Date": dt}
            for m, val in zip(target_months, y_target):
                curve_row[f"DI_NS_{m}m"] = float(val)
            curve_rows.append(curve_row)

        df_betas = pd.DataFrame(betas_rows).set_index("Date").sort_index()
        df_curve = pd.DataFrame(curve_rows).set_index("Date").sort_index()

        return df_betas, df_curve

    @staticmethod
    def _standardize_cds_to_pct(df_cds: pd.DataFrame) -> pd.DataFrame:
        """
        Converte CDS_dom e CDS_glob de log-retornos para % a.a. acumulado.
        
        CDS vem como log-retornos (dif de log).
        Converte para série acumulada, depois em % a.a.
        """
        df_cds_std = df_cds[["CDS_dom", "CDS_glob"]].copy()
        
        # Exponencia e acumula para voltar ao nível
        for col in ["CDS_dom", "CDS_glob"]:
            # exp(log_ret) = valor relativo
            # cumsum de log_ret -> nível acumulado
            df_cds_std[f"{col}_level"] = (1.0 + df_cds_std[col]).cumprod() - 1.0
            # converte para % a.a.
            df_cds_std[f"{col}_pct"] = df_cds_std[f"{col}_level"] * 100.0
        
        return df_cds_std[["CDS_dom_pct", "CDS_glob_pct"]].rename( 
            columns={"CDS_dom_pct": "CDS_dom", "CDS_glob_pct": "CDS_glob"}
        )

    # --------------------------
    # Fit NS com decomposição CDS
    # --------------------------
    @staticmethod
    def fit_ns_with_cds_decomposition(
        df_di: pd.DataFrame,
        df_cds: pd.DataFrame,
        maturities_fit_months: list[int],
        maturities_target_months: list[int],
        lam: float = 0.7308,
        rolling_window: int = 60,
        standardize_cds: bool = True,
        **ns_kwargs,
    ) -> dict:
        """
        Ajusta NS + análise de sensibilidade aos fatores CDS.
        
        Quantifica quanto das mudanças nos betas (nível, slope, curvatura) 
        são explicadas por risco doméstico (CDS_dom) vs global (CDS_glob).
        
        Args:
            df_di: DataFrame com DI swaps observados
            df_cds: DataFrame com CDS_dom e CDS_glob (output de data.decompor_cds())
            rolling_window: janela para estimar sensibilidades
            standardize_cds: se True, converte CDS para % a.a. (mesma escala dos resíduos)
        
        Returns:
            dict com:
            - df_betas: betas NS diários
            - df_curve: curva NS ajustada
            - df_sensitivities: sensibilidades β vs CDS (rolling)
            - df_risk_contrib: contribuição de CDS para mudanças nos betas
        """
        # 1) Fit NS padrão
        df_betas, df_curve = modeling.fit_nelson_siegel_daily(
            df_di=df_di,
            maturities_fit_months=maturities_fit_months,
            maturities_target_months=maturities_target_months,
            lam=lam,
            **ns_kwargs,
        )
        
        # 2) Padroniza CDS (se necessário)
        if standardize_cds:
            df_cds = modeling._standardize_cds_to_pct(df_cds)
        
        # 3) Merge com CDS
        df_merged = df_betas.join(df_cds[["CDS_dom", "CDS_glob"]], how="inner")
        
        if len(df_merged) < rolling_window + 10:
            print("[WARN] Dados insuficientes para análise CDS rolling.")
            return {
                "df_betas": df_betas,
                "df_curve": df_curve,
                "df_sensitivities": pd.DataFrame(),
                "df_risk_contrib": pd.DataFrame(),
            }
        
        # 4) Calcula sensibilidades rolling
        sensitivities = []
        
        for i in range(rolling_window, len(df_merged)):
            window = df_merged.iloc[i - rolling_window : i]
            
            sens_row = {"Date": df_merged.index[i]}
            
            # Para cada beta, regredir mudanças vs mudanças em CDS
            for beta_name in ["beta0", "beta1", "beta2"]:
                d_beta = window[beta_name].diff().dropna()
                
                if len(d_beta) < 10:
                    sens_row[f"{beta_name}_sens_dom"] = np.nan
                    sens_row[f"{beta_name}_sens_glob"] = np.nan
                    sens_row[f"{beta_name}_R2"] = np.nan
                    continue
                
                # Alinha índices
                common_idx = d_beta.index
                d_cds_dom = window["CDS_dom"].loc[common_idx].diff().dropna()
                d_cds_glob = window["CDS_glob"].loc[common_idx].diff().dropna()
                
                # Alinha tudo novamente
                common_idx2 = d_beta.index.intersection(d_cds_dom.index).intersection(d_cds_glob.index)
                if len(common_idx2) < 10:
                    sens_row[f"{beta_name}_sens_dom"] = np.nan
                    sens_row[f"{beta_name}_sens_glob"] = np.nan
                    sens_row[f"{beta_name}_R2"] = np.nan
                    continue
                
                y = d_beta.loc[common_idx2].values
                x_dom = d_cds_dom.loc[common_idx2].values
                x_glob = d_cds_glob.loc[common_idx2].values
                
                # Regressão: Δβ = α + γ·ΔCDSdom + δ·ΔCDSglob
                X = np.column_stack([
                    np.ones(len(common_idx2)),
                    x_dom,
                    x_glob,
                ])
                
                try:
                    coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
                    
                    # R²
                    yhat = X @ coefs
                    ss_res = np.sum((y - yhat) ** 2)
                    ss_tot = np.sum((y - y.mean()) ** 2)
                    r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
                    
                    sens_row[f"{beta_name}_sens_dom"] = float(coefs[1])
                    sens_row[f"{beta_name}_sens_glob"] = float(coefs[2])
                    sens_row[f"{beta_name}_R2"] = float(r2)
                except:
                    sens_row[f"{beta_name}_sens_dom"] = np.nan
                    sens_row[f"{beta_name}_sens_glob"] = np.nan
                    sens_row[f"{beta_name}_R2"] = np.nan
            
            sensitivities.append(sens_row)
        
        df_sensitivities = pd.DataFrame(sensitivities).set_index("Date")
        
        # 5) Calcula contribuições para mudanças nos betas
        df_merged["d_beta0"] = df_merged["beta0"].diff()
        df_merged["d_beta1"] = df_merged["beta1"].diff()
        df_merged["d_beta2"] = df_merged["beta2"].diff()
        
        # Join sensibilidades
        df_risk_contrib = df_merged.join(df_sensitivities, how="inner")
        
        # Contribuição instantânea = sensibilidade × valor do fator
        for beta_name in ["beta0", "beta1", "beta2"]:
            df_risk_contrib[f"{beta_name}_contrib_dom"] = (
                df_risk_contrib[f"{beta_name}_sens_dom"] * df_risk_contrib["CDS_dom"]
            )
            df_risk_contrib[f"{beta_name}_contrib_glob"] = (
                df_risk_contrib[f"{beta_name}_sens_glob"] * df_risk_contrib["CDS_glob"]
            )
            df_risk_contrib[f"{beta_name}_contrib_total"] = (
                df_risk_contrib[f"{beta_name}_contrib_dom"] + 
                df_risk_contrib[f"{beta_name}_contrib_glob"]
            )
        
        return {
            "df_betas": df_betas,
            "df_curve": df_curve,
            "df_sensitivities": df_sensitivities,
            "df_risk_contrib": df_risk_contrib,
        }