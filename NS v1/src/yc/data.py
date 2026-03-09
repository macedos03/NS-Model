import numpy as np
import pandas as pd
import statsmodels.api as sm
import re
from dataclasses import dataclass
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ====== Caminhos dos arquivos ======
focus_path   = "Q:/Gabriel de Macedo/Política Monetária/NS Model/data/Focus.xlsx"
bbg_path     = "Q:/Gabriel de Macedo/Política Monetária/NS Model/data/bbg new.xlsx"
db_path      = "Q:/Gabriel de Macedo/Política Monetária/Yield Curve - Decom/Database.xlsx"

@dataclass
class PCAArtifacts:
    scaler: StandardScaler
    pca: PCA
    input_cols: list
    loadings: pd.DataFrame
    explained_variance_ratio: pd.Series

class data:
    # --------------------------
    # 1) CDS decomposition (seu código, só com pequenos hardenings)
    # --------------------------
    @staticmethod
    def decompor_cds(db_path=db_path):
        df_raw = pd.read_excel(db_path)
        df_raw.columns = [str(c).strip() for c in df_raw.columns]

        df = df_raw.rename(columns={
            "Date": "Date",
            "BRL": "BRL",
            "DXY Curncy": "DXY",
            "BRAZIL CDS USD SR 10Y D14 Corp": "CDS",
            "CRB CMDT Index": "CRB",
            "GT10 Govt": "UST10",
            "VIX": "VIX",
            "CRY": "CRY",
            "Cupom cambial": "CUPOM",
            "BZ10T": "BZ10T",
            "IBOV": "IBOV",
            "SELIC": "SELIC",
            "GT2 Govt": "BZ2T",
            "SPX": "SPX",
            "SWAP1A": "SWAP1A",
            "SWAP5A": "SWAP5A",
            "NASDAQ": "NASDAQ"
        })

        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date").set_index("Date")

        pct_vars  = ["BRL","DXY","CRB","VIX","CRY","CDS","IBOV","SPX","NASDAQ"]
        diff_vars = ["UST10","BZ10T","SELIC","BZ2T","SWAP5A","SWAP1A"]

        df_ret = df.copy()

        # log-retornos para pct_vars
        df_ret[pct_vars] = np.log(df_ret[pct_vars]) - np.log(df_ret[pct_vars].shift(1))

        # diferenças simples para diff_vars
        for col in diff_vars:
            df_ret[col] = df_ret[col] - df_ret[col].shift(1)

        df_ret = df_ret.dropna()

        extern_for_cds = ["DXY", "CRB", "VIX", "UST10"]

        X = sm.add_constant(df_ret[extern_for_cds])
        y = df_ret["CDS"]

        model = sm.OLS(y, X).fit()

        df_ret["CDS_glob"] = model.fittedvalues.astype(float)
        df_ret["CDS_dom"]  = model.resid.astype(float)

        return df_ret[["CDS_glob", "CDS_dom"]]

    # --------------------------
    # Helpers: leitura Focus e BBG
    # --------------------------
    @staticmethod
    def _read_focus(path=focus_path) -> pd.DataFrame:
        df = pd.read_excel(path)
        df.columns = [str(c).strip() for c in df.columns]

        # Padroniza coluna de data (no seu exemplo é "Data")
        if "Data" not in df.columns:
            raise ValueError("Focus.xlsx precisa ter coluna 'Data'.")

        df["Data"] = pd.to_datetime(df["Data"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["Data"]).sort_values("Data").set_index("Data")

        # Converte tudo que for número para float quando possível
        for c in df.columns:
            try:
                df[c] = pd.to_numeric(df[c])
            except:
                pass
        return df

    @staticmethod
    def _read_bbg(path=bbg_path) -> pd.DataFrame:
        na_vals = ["#N/A", "#N/A N/A", "N/A", "#VALUE!", "#REF!"]

        df = pd.read_excel(path, skiprows=[1], na_values=na_vals)
        df.columns = [str(c).strip() for c in df.columns]

        if "Date" not in df.columns:
            raise ValueError("bbg new.xlsx precisa ter coluna 'Date'.")

        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date").set_index("Date")

        rename_map = {
            "IPCA y/y": "IPCA_YoY",
            "IBOV Index": "IBOV",
            "Overnight Selic": "SELIC_ON",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        # Converte colunas relevantes para numérico (coerce!)
        for c in ["NTNB1Y", "NTNB3Y", "IPCA_YoY", "IBOV", "SELIC_ON"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        return df

    # --------------------------
    # 2) Focus anual -> horizontes fixos (12m/24m/36m)
    # --------------------------
    @staticmethod
    def _focus_year_to_fixed_horizon(
        df_focus: pd.DataFrame,
        base_col: str,
        horizons_months=(12, 24, 36),
        suffixes=("year", "year_1", "year_2", "year_3"),
    ) -> pd.DataFrame:
        """
        Converte séries Focus 'por ano calendário' em expectativas de maturidade fixa.
        Ex.: IPCA year, IPCA year_1, ... -> IPCA_12m, IPCA_24m, IPCA_36m.

        Fórmula (aprox):
        w(t) = fração do ano restante (dias até 31/12 / dias no ano)
        12m:  w*F_year + (1-w)*F_year_1
        24m:  w*F_year_1 + (1-w)*F_year_2
        36m:  w*F_year_2 + (1-w)*F_year_3
        """
        needed = [f"{base_col} {s}" for s in suffixes]
        missing = [c for c in needed if c not in df_focus.columns]
        if missing:
            raise ValueError(f"Faltam colunas no Focus para '{base_col}': {missing}")

        idx = df_focus.index
        years = idx.year

        # dias no ano (considera leap years)
        days_in_year = pd.to_datetime(years.astype(str) + "-12-31").dayofyear
        # dias restantes no ano a partir de t até 31/12 (inclusive o dia t -> aproximação simples)
        day_of_year = idx.dayofyear
        days_left = (days_in_year - day_of_year) / days_in_year
        w = pd.Series(days_left, index=idx).astype(float)

        out = pd.DataFrame(index=idx)

        F0 = pd.to_numeric(df_focus[f"{base_col} year"], errors="coerce")
        F1 = pd.to_numeric(df_focus[f"{base_col} year_1"], errors="coerce")
        F2 = pd.to_numeric(df_focus[f"{base_col} year_2"], errors="coerce")
        F3 = pd.to_numeric(df_focus[f"{base_col} year_3"], errors="coerce")

        for h in horizons_months:
            if h == 12:
                out[f"{base_col}_12m"] = w * F0 + (1 - w) * F1
            elif h == 24:
                out[f"{base_col}_24m"] = w * F1 + (1 - w) * F2
            elif h == 36:
                out[f"{base_col}_36m"] = w * F2 + (1 - w) * F3
            else:
                raise ValueError("Este helper suporta apenas horizontes 12, 24 e 36 meses.")

        return out

    # --------------------------
    # 3) PCA de inflação (Focus fixo + NTNB 1y/3y)
    # -------------------------
    @staticmethod
    def PCA_IPCA(
        focus_path=focus_path,
        bbg_path=bbg_path,
        n_components=2,
        horizons_months=(12, 24, 36),
        join_how="inner",
        ffill_focus=True,
        fit_start=None,
        fit_end=None,
        # opcional: criar também um fator de taxa real (compressão de NTNB1Y/3Y)
        make_real_pca=False,
    ):
        """
        PCA de inflação = SOMENTE Focus (IPCA_12m/24m/36m).
        NTN-B real (NTNB1Y/NTNB3Y) entra como variável separada (ou RealPC1, se make_real_pca=True).
        """

        # ---- Load ----
        df_focus = data._read_focus(focus_path)
        df_bbg = data._read_bbg(bbg_path)

        # ---- Focus IPCA -> horizontes fixos ----
        df_focus_ipca = data._focus_year_to_fixed_horizon(
            df_focus,
            base_col="IPCA",
            horizons_months=horizons_months
        )

        # ---- PCA INPUT: apenas inflação (Focus) ----
        infl_cols = list(df_focus_ipca.columns)  # IPCA_12m, IPCA_24m, IPCA_36m
        df_infl_in = df_focus_ipca[infl_cols].dropna().copy()

        # ---- Janela de fit (opcional) ----
        if fit_start is not None or fit_end is not None:
            fit_mask = pd.Series(True, index=df_infl_in.index)
            if fit_start is not None:
                fit_mask &= (df_infl_in.index >= pd.to_datetime(fit_start))
            if fit_end is not None:
                fit_mask &= (df_infl_in.index <= pd.to_datetime(fit_end))
            df_fit = df_infl_in.loc[fit_mask].copy()
            if len(df_fit) < 50:
                raise ValueError("Janela de fit do PCA (inflação) ficou pequena. Ajuste fit_start/fit_end.")
        else:
            df_fit = df_infl_in

        # ---- Standardize + PCA (inflação) ----
        infl_scaler = StandardScaler()
        X_fit = infl_scaler.fit_transform(df_fit[infl_cols].values)

        infl_pca = PCA(n_components=min(n_components, len(infl_cols)), random_state=42)
        infl_pca.fit(X_fit)

        X_all = infl_scaler.transform(df_infl_in[infl_cols].values)
        pcs = infl_pca.transform(X_all)

        # ---- Coloca PCs no dataframe Focus ----
        df_focus_factors = df_focus_ipca.copy()
        for i in range(infl_pca.n_components_):
            df_focus_factors.loc[df_infl_in.index, f"InflPC{i+1}"] = pcs[:, i]

        # ---- Artifacts de inflação ----
        infl_loadings = pd.DataFrame(
            infl_pca.components_.T,
            index=infl_cols,
            columns=[f"InflPC{i+1}" for i in range(infl_pca.n_components_)]
        )
        infl_evr = pd.Series(
            infl_pca.explained_variance_ratio_,
            index=[f"InflPC{i+1}" for i in range(infl_pca.n_components_)]
        )
        infl_artifacts = PCAArtifacts(
            scaler=infl_scaler,
            pca=infl_pca,
            input_cols=infl_cols,
            loadings=infl_loadings,
            explained_variance_ratio=infl_evr
        )

        # ---- Merge com BBG (para manter NTN-B real como variável) ----
        if join_how == "left":
            merged = df_bbg.join(df_focus_factors, how="left")
        else:
            merged = df_bbg.join(df_focus_factors, how=join_how)

        if ffill_focus:
            # forward fill dos horizontes e dos PCs (Focus é degrau)
            cols_to_ffill = infl_cols + [c for c in df_focus_factors.columns if c.startswith("InflPC")]
            merged[cols_to_ffill] = merged[cols_to_ffill].ffill()

        # ---- NTN-B real: garantir que existem e são numéricas ----
        for c in ["NTNB1Y", "NTNB3Y"]:
            if c in merged.columns:
                merged[c] = pd.to_numeric(merged[c], errors="coerce")

        # ---- (Opcional) PCA do juro real (NTNB1Y/3Y) para virar RealPC1 ----
        real_artifacts = None
        if make_real_pca:
            if "NTNB1Y" not in merged.columns or "NTNB3Y" not in merged.columns:
                raise ValueError("Para make_real_pca=True, BBG precisa ter NTNB1Y e NTNB3Y.")

            real_in = merged[["NTNB1Y", "NTNB3Y"]].dropna().copy()
            if len(real_in) < 50:
                raise ValueError("Poucas observações para PCA do juro real.")

            real_scaler = StandardScaler()
            Xr = real_scaler.fit_transform(real_in.values)

            real_pca = PCA(n_components=1, random_state=42)
            real_pca.fit(Xr)

            real_pc = real_pca.transform(Xr)[:, 0]
            merged.loc[real_in.index, "RealPC1"] = real_pc

            real_loadings = pd.DataFrame(
                real_pca.components_.T,
                index=["NTNB1Y", "NTNB3Y"],
                columns=["RealPC1"]
            )
            real_evr = pd.Series(real_pca.explained_variance_ratio_, index=["RealPC1"])

            real_artifacts = PCAArtifacts(
                scaler=real_scaler,
                pca=real_pca,
                input_cols=["NTNB1Y", "NTNB3Y"],
                loadings=real_loadings,
                explained_variance_ratio=real_evr
            )

        # Retorno:
        # - merged: já contém IPCA_12/24/36, InflPCs e NTNByields (e opcional RealPC1)
        # - infl_artifacts: artifacts do PCA de inflação (Focus)
        # - real_artifacts: artifacts do PCA do juro real (se make_real_pca=True), senão None
        return merged, infl_artifacts, real_artifacts
    
    @staticmethod
    def read_di_swaps(
        path: str = "Q:/Gabriel de Macedo/Política Monetária/NS Model/data/di_swaps.xlsx",
        sheet_name: str | int = 0,
        date_col: str = "Dates",
    ) -> pd.DataFrame:
        """
        Lê o Excel di_swaps.xlsx e devolve um DataFrame diário com colunas padronizadas:
          DI_1m, DI_3m, DI_5m, DI_6m, DI_12m, DI_14m, DI_24m, DI_36m, DI_48m, DI_60m
        em % a.a.

        Espera colunas do tipo:
          "DI Swap 21D", "DI SWAP 63D", "DI SWAP 105D", "DI SWAP 126D", "DI SWAP 252D",
          "DI SWAP 294D", "DI SWAP 504D", "DI SWAP 756D", "DI SWAP 1008D", "DI SWAP 1260D"
        e uma coluna de datas "Dates".
        """

        na_vals = ["#N/A", "#N/A N/A", "N/A", "#VALUE!", "#REF!"]

        df = pd.read_excel(path, sheet_name=sheet_name, na_values=na_vals)
        df.columns = [str(c).strip() for c in df.columns]

        if date_col not in df.columns:
            raise ValueError(f"di_swaps.xlsx precisa ter a coluna '{date_col}'.")

        df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
        df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)

        rename_map = {
            "DI Swap 21D": "DI_1m",
            "DI SWAP 63D": "DI_3m",
            "DI SWAP 105D": "DI_5m",
            "DI SWAP 126D": "DI_6m",
            "DI SWAP 252D": "DI_12m",
            "DI SWAP 294D": "DI_14m",
            "DI SWAP 504D": "DI_24m",
            "DI SWAP 756D": "DI_36m",
            "DI SWAP 1008D": "DI_48m",
            "DI SWAP 1260D": "DI_60m",
        }

        # Normaliza para pegar diferenças de maiúsculas/minúsculas e espaços
        cols_norm = {c: " ".join(c.split()).upper() for c in df.columns}
        inv_norm = {}
        for original, norm in cols_norm.items():
            inv_norm[norm] = original

        # aplica renomeação olhando versão "norm"
        to_rename = {}
        for k, v in rename_map.items():
            kk = " ".join(k.split()).upper()
            if kk in inv_norm:
                to_rename[inv_norm[kk]] = v

        df = df.rename(columns=to_rename)

        # Mantém só as DI padronizadas que existirem
        out_cols = [c for c in rename_map.values() if c in df.columns]
        if not out_cols:
            raise ValueError("Não encontrei colunas DI SWAP reconhecidas no Excel (ver rename_map).")

        df = df[out_cols].copy()

        # numérico
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        return df

    @staticmethod
    def load_di_swaps_from_days(
        path: str,
        sheet_name: str | int | None = 0,
        date_col: str = "Dates",
        maturities_months=(1, 3, 5, 6, 12, 14, 24, 36, 48, 60),
        day_to_month_map=None,
        scale_hint: str = "auto",   # "auto" | "pct" | "decimal"
        dedup: str = "last",        # "last" | "mean"
        drop_all_nan_rows: bool = True,
    ) -> pd.DataFrame:
        """
        Lê Excel com colunas do tipo '... 21D', '... 252D', etc.
        Retorna df wide:
          index=Date, colunas=DI_{m}m (m em maturities_months), valores em % a.a.
        """

        if day_to_month_map is None:
            # mapeamento base + seus novos vértices
            day_to_month_map = {
                21: 1,
                63: 3,
                105: 5,
                126: 6,
                252: 12,
                294: 14,
                504: 24,
                756: 36,
                1008: 48,
                1260: 60,
            }

        na_vals = ["#N/A", "#N/A N/A", "N/A", "#VALUE!", "#REF!"]

        df_raw = pd.read_excel(path, sheet_name=sheet_name, na_values=na_vals)
        df_raw.columns = [str(c).strip() for c in df_raw.columns]

        if date_col not in df_raw.columns:
            raise ValueError(
                f"Coluna de data '{date_col}' não encontrada. Colunas disponíveis: {list(df_raw.columns)}"
            )

        # parse dates
        df_raw[date_col] = pd.to_datetime(df_raw[date_col], dayfirst=True, errors="coerce")
        df_raw = df_raw.dropna(subset=[date_col]).sort_values(date_col)

        # dedup de data
        if df_raw[date_col].duplicated().any():
            if dedup == "last":
                df_raw = df_raw.drop_duplicates(subset=[date_col], keep="last")
            elif dedup == "mean":
                tmp = df_raw.copy()
                num_cols = [c for c in tmp.columns if c != date_col]
                tmp[num_cols] = tmp[num_cols].apply(pd.to_numeric, errors="coerce")
                df_raw = tmp.groupby(date_col, as_index=False)[num_cols].mean()
            else:
                raise ValueError("dedup precisa ser 'last' ou 'mean'.")

        df_raw = df_raw.set_index(date_col).sort_index()

        # identificar colunas com padrão "<n>D"
        col_rename = {}
        for c in df_raw.columns:
            s = str(c).strip().upper()
            m = re.search(r"(\d+)\s*D\b", s)
            if not m:
                continue
            days = int(m.group(1))
            if days in day_to_month_map:
                mm = day_to_month_map[days]
                col_rename[c] = f"DI_{mm}m"

        if not col_rename:
            raise ValueError(
                "Não consegui mapear nenhuma coluna com padrão '<n>D'. "
                "Verifique se os nomes têm algo como '... 21D', '... 252D', etc."
            )

        df = df_raw.rename(columns=col_rename)

        # manter apenas as colunas mapeadas
        mapped_cols = list(col_rename.values())
        df = df[mapped_cols].copy()

        # numeric coercion
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # garantir conjunto final de maturidades (as faltantes viram NaN)
        mats = list(maturities_months)
        for mm in mats:
            col = f"DI_{mm}m"
            if col not in df.columns:
                df[col] = np.nan

        df = df[[f"DI_{mm}m" for mm in mats]]

        # escala: valores típicos do DI vêm em % a.a.
        if scale_hint not in ("auto", "pct", "decimal"):
            raise ValueError("scale_hint precisa ser 'auto', 'pct' ou 'decimal'.")

        if scale_hint == "decimal":
            df = df * 100.0
        elif scale_hint == "auto":
            vals = df.stack().dropna()
            # se estiver em decimal (0.12), converte para %
            if len(vals) > 200 and (vals.median() < 1.0) and (vals.quantile(0.99) < 2.0):
                df = df * 100.0

        if drop_all_nan_rows:
            df = df.dropna(how="all")

        return df

    @staticmethod
    def di_fit_matrix(
        df_di: pd.DataFrame,
        maturities_fit_months=(1, 3, 5, 6, 12, 14, 24, 36, 48, 60),
        min_points_per_day: int = 4,
    ) -> pd.DataFrame:
        """
        Prepara matriz diária para o fit do Nelson–Siegel:
        - filtra apenas maturidades de fit
        - mantém dias com >= min_points_per_day observações válidas
        """
        cols = [f"DI_{m}m" for m in maturities_fit_months if f"DI_{m}m" in df_di.columns]
        if not cols:
            raise ValueError("Nenhuma coluna DI_{m}m encontrada para maturities_fit_months.")

        X = df_di[cols].copy()
        n_ok = X.notna().sum(axis=1)
        X = X.loc[n_ok >= min_points_per_day].copy()
        X["n_points_fit"] = n_ok.loc[X.index]
        return X
    
maturities_fit_months    = [1, 3, 5, 6, 12, 14, 24, 36, 48, 60]     # observado (fit)
maturities_target_months = [1, 2, 3, 4, 6, 9, 12, 18, 24, 30, 36, 48] # saída (gerar via NS)

df_di = data.load_di_swaps_from_days(
    path="Q:/Gabriel de Macedo/Política Monetária/NS Model/data/di_swaps.xlsx",
    sheet_name=0,
    date_col="Dates",
    maturities_months=maturities_fit_months,  # aqui você escolhe o observado
)

print(df_di.head())
print("Início DI_48m:", df_di["DI_48m"].first_valid_index() if "DI_48m" in df_di.columns else None)
print("Início DI_60m:", df_di["DI_60m"].first_valid_index() if "DI_60m" in df_di.columns else None)
print(df_di.notna().sum())

X_fit = data.di_fit_matrix(
    df_di,
    maturities_fit_months=maturities_fit_months,
    min_points_per_day=4,
)

print(X_fit.head())
print(X_fit["n_points_fit"].value_counts().sort_index())