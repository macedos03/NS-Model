import os
import numpy as np
import pandas as pd
from pathlib import Path

fatores_path = "Q:/Gabriel de Macedo/Política Monetária/NS Model/data/pca_fatores.xlsx"

class _fatores:
    @staticmethod
    def salvar_fator_em_excel(
        df_fator,
        colnames,
        output_path=fatores_path,
        sheet_name="Sheet1",
        engine="openpyxl",
    ):
        # ---------- prepara df_fator ----------
        fator = df_fator[colnames].copy().reset_index()

        # Normaliza nome da coluna de data para 'Data'
        if "Data" in fator.columns:
            pass
        elif "Date" in fator.columns:
            fator = fator.rename(columns={"Date": "Data"})
        else:
            fator = fator.rename(columns={fator.columns[0]: "Data"})

        fator["Data"] = pd.to_datetime(fator["Data"], errors="coerce")
        fator = fator.dropna(subset=["Data"]).set_index("Data").sort_index()

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        # ---------- tenta ler arquivo existente ----------
        fatores_df = None
        if out.exists() and out.is_file() and out.stat().st_size > 0:
            try:
                fatores_df = pd.read_excel(out, sheet_name=sheet_name, engine=engine)
                if "Data" not in fatores_df.columns:
                    # se vier sem Data, trata como inválido
                    fatores_df = None
                else:
                    fatores_df["Data"] = pd.to_datetime(fatores_df["Data"], errors="coerce")
                    fatores_df = fatores_df.dropna(subset=["Data"]).set_index("Data").sort_index()
            except Exception:
                # arquivo existe mas não é um xlsx válido (corrompido / conteúdo não-excel / etc.)
                fatores_df = None

        # ---------- se não conseguiu ler, cria do zero ----------
        if fatores_df is None:
            fatores_df = pd.DataFrame(index=fator.index).sort_index()

        # ---------- remove duplicatas antigas das colunas-base ----------
        cols_to_drop = []
        for base in colnames:
            for c in list(fatores_df.columns):
                if c != base and (c.startswith(base + "_") or c.startswith(base + ".") or c.startswith(base + " ")):
                    cols_to_drop.append(c)
        if cols_to_drop:
            fatores_df = fatores_df.drop(columns=cols_to_drop)

        # garante que bases existam
        for base in colnames:
            if base not in fatores_df.columns:
                fatores_df[base] = np.nan

        # une índices e sobrescreve colunas
        full_index = fatores_df.index.union(fator.index)
        fatores_df = fatores_df.reindex(full_index)

        for base in colnames:
            fatores_df.loc[fator.index, base] = fator[base].values

        # ---------- salva (sempre reescreve o arquivo) ----------
        fatores_df = fatores_df.sort_index().reset_index()  # Data vira coluna
        with pd.ExcelWriter(out, engine=engine, mode="w") as writer:
            fatores_df.to_excel(writer, sheet_name=sheet_name, index=False)

        return fatores_df