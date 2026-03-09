# -*- coding: utf-8 -*-
# ================== Imports ==================
import pandas as pd
from bcb import Expectativas
from pathlib import Path

# ================== Config ===================
DATA_MIN = "2007-01-01"

# Arquivo NOVO, só para os dados automáticos
FILE_PATH = Path(r"Q:\Gabriel de Macedo\Política Monetária\NS Model\data\Focus_Automatico.xlsx")

# ================== Endpoints =================
em = Expectativas()
ep_anuais = em.get_endpoint("ExpectativasMercadoAnuais")
# ================== Helpers ===================
def _to_dt(x):
    dt = pd.to_datetime(x, errors="coerce")
    if isinstance(dt, pd.Series):
        return dt.dt.normalize()
    try:
        return dt.normalize()
    except AttributeError:
        return dt

def consulta_anuais_mediana(endpoint, indicador, data_min=DATA_MIN, base_calculo=0):
    """
    Consulta anual 'geral' (baseCalculo=0).
    Retorna tabela pivot Data x Ano, com a Mediana.
    """
    df = (
        endpoint.query()
        .filter(endpoint.Indicador == indicador)
        .filter(endpoint.Data >= data_min)
        .filter(endpoint.baseCalculo == base_calculo)
        .select(endpoint.Data, endpoint.DataReferencia, endpoint.Mediana)
        .collect()
    )

    if df.empty:
        print(f"[AVISO] Sem dados anuais para {indicador}.")
        return pd.DataFrame()

    df["Data"] = _to_dt(df["Data"])
    df["DataReferencia"] = pd.to_numeric(df["DataReferencia"], errors="coerce").astype("Int64")
    df = df.drop_duplicates(subset=["Data", "DataReferencia"])

    pv = (
        df.pivot(index="Data", columns="DataReferencia", values="Mediana")
          .sort_index()
          .sort_index(axis=1)
    )

    print(f"[LOG] {indicador} anual (geral) shape: {pv.shape}")
    return pv

def consulta_top5_estatisticas(endpoint, indicador, data_min=DATA_MIN, preferencia=("M", "C", "L")):
    """
    Consulta anual Top5 Estatísticas.
    Pega apenas a Mediana, escolhendo tipoCalculo na ordem: M > C > L.
    Retorna tabela pivot Data x Ano.
    """
    df = (
        endpoint.query()
        .filter(endpoint.Indicador == indicador)
        .filter(endpoint.Data >= data_min)
        .select(endpoint.Data, endpoint.DataReferencia, endpoint.tipoCalculo, endpoint.Mediana)
        .collect()
    )

    if df.empty:
        print(f"[AVISO] Sem dados Top5 anuais para {indicador}.")
        return pd.DataFrame()

    df["Data"] = _to_dt(df["Data"])
    df["DataReferencia"] = pd.to_numeric(df["DataReferencia"], errors="coerce").astype("Int64")

    # Prioridade M > C > L
    cats = list(preferencia)
    df["tipoCalculo"] = pd.Categorical(df["tipoCalculo"], categories=cats, ordered=True)

    df = (
        df.sort_values(["Data", "DataReferencia", "tipoCalculo"])
          .drop_duplicates(subset=["Data", "DataReferencia"], keep="first")
    )

    df = df.drop(columns=["tipoCalculo"])

    pv = (
        df.pivot(index="Data", columns="DataReferencia", values="Mediana")
          .sort_index()
          .sort_index(axis=1)
    )

    print(f"[LOG] {indicador} Top5 shape: {pv.shape}")
    return pv

def encontrar_indicador_por_substring(endpoint, substring):
    """
    Procura no endpoint anual um indicador cujo nome contenha `substring`
    (case-insensitive). Retorna o primeiro nome encontrado.
    Levanta ValueError se não achar nada.
    """
    ind_df = (
        endpoint.query()
        .select(endpoint.Indicador)
        .collect()
    )

    ind_df = ind_df.drop_duplicates(subset=["Indicador"])

    mask = ind_df["Indicador"].str.contains(substring, case=False, na=False)

    if not mask.any():
        raise ValueError(f"Nenhum indicador contendo '{substring}' foi encontrado.")

    # Se quiser ver todos os matches, pode descomentar:
    print(f"[DEBUG] Indicadores contendo '{substring}':")
    print(ind_df.loc[mask, "Indicador"].sort_values().to_string(index=False))

    nome = ind_df.loc[mask, "Indicador"].iloc[0]
    print(f"[LOG] Indicador detectado para '{substring}': '{nome}'")
    return nome

def main_focus():
    # ================== Coletas principais ===================

    # Medianas anuais "geral" – já estavam ok
    dados_ipca   = consulta_anuais_mediana(ep_anuais, "IPCA")
    dados_selic  = consulta_anuais_mediana(ep_anuais, "Selic")
    dados_cambio = consulta_anuais_mediana(ep_anuais, "Câmbio")
    dados_pib    = consulta_anuais_mediana(ep_anuais, "PIB Total")

    # ===== FISCAL: Resultado primário, dívida líquida e resultado nominal =====
    dados_primario = consulta_anuais_mediana(ep_anuais, "Resultado primário")
    dados_div_liq = consulta_anuais_mediana(ep_anuais, "Dívida líquida do setor público")
    dados_res_nom = consulta_anuais_mediana(ep_anuais, "Resultado nominal")

    # ===== SETOR EXTERNO: Investimento direto, balança comercial e conta corrente =====
    dados_inv_dir = consulta_anuais_mediana(ep_anuais, "Investimento direto no país")
    dados_bal_com = consulta_anuais_mediana(ep_anuais, "Balança comercial")
    dados_cont_cor = consulta_anuais_mediana(ep_anuais, "Conta corrente")

    # Dicionário de abas -> DataFrames
    sheets = {
        "IPCA":   dados_ipca,
        "Selic":  dados_selic,
        "Câmbio": dados_cambio,
        "PIB":    dados_pib,
        "Resultado primário": dados_primario,
        "Dívida líquida do setor público": dados_div_liq,
        "Resultado nominal": dados_res_nom,
        "Investimento direto no país": dados_inv_dir,
        "Balança comercial": dados_bal_com,
        "Conta corrente": dados_cont_cor,
    }

    # ================== Escrita em Excel (arquivo novo) ===================
    with pd.ExcelWriter(FILE_PATH, engine="openpyxl", mode="w") as writer:
        for sheet_name, df in sheets.items():
            if not df.empty:
                df.to_excel(writer, sheet_name=sheet_name, index=True)
                print(f"[LOG] Aba '{sheet_name}' escrita com shape {df.shape}")

    print(f"Arquivo {FILE_PATH} salvo com sucesso.")

main_focus()