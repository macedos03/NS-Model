# -*- coding: utf-8 -*-
# ================== Imports ==================
import pandas as pd
from bcb import Expectativas
from pathlib import Path

# ================== Config ===================
DATA_MIN = "2007-01-01"

# Arquivo NOVO, só para os dados automáticos
FILE_PATH = Path(r"Q:\Gabriel de Macedo\Política Monetária\NS Model\NS v2\data\Focus_Automatico.xlsx")

# ================== Endpoints =================
em = Expectativas()
ep_anuais = em.get_endpoint("ExpectativasMercadoAnuais")
ep_mensais = em.get_endpoint("ExpectativaMercadoMensais")
ep_inflacao_12m = em.get_endpoint("ExpectativasMercadoInflacao12Meses")

# ================== Helpers ===================
def _to_dt(x):
    dt = pd.to_datetime(x, errors="coerce")
    if isinstance(dt, pd.Series):
        return dt.dt.normalize()
    try:
        return dt.normalize()
    except AttributeError:
        return dt

def consulta_anuais_metricas(
    endpoint,
    indicador,
    data_min=DATA_MIN,
    base_calculo=0,
    metricas=("Media", "Mediana", "DesvioPadrao"),
):
    """
    Consulta anual (baseCalculo=0) e retorna pivots Data x Ano
    para as métricas solicitadas (Media, Mediana, DesvioPadrao).
    """
    selects = [endpoint.Data, endpoint.DataReferencia]
    for metrica in metricas:
        if hasattr(endpoint, metrica):
            selects.append(getattr(endpoint, metrica))

    df = (
        endpoint.query()
        .filter(endpoint.Indicador == indicador)
        .filter(endpoint.Data >= data_min)
        .filter(endpoint.baseCalculo == base_calculo)
        .select(*selects)
        .collect()
    )

    if df.empty:
        print(f"[AVISO] Sem dados anuais para {indicador}.")
        return {}

    df["Data"] = _to_dt(df["Data"])
    df["DataReferencia"] = pd.to_numeric(df["DataReferencia"], errors="coerce").astype("Int64")
    df = df.drop_duplicates(subset=["Data", "DataReferencia"])

    out = {}
    for metrica in metricas:
        if metrica not in df.columns:
            continue
        pv = (
            df.pivot(index="Data", columns="DataReferencia", values=metrica)
            .sort_index()
            .sort_index(axis=1)
        )
        out[metrica] = pv
        print(f"[LOG] {indicador} anual ({metrica}) shape: {pv.shape}")

    return out


def consulta_anuais_mediana(endpoint, indicador, data_min=DATA_MIN, base_calculo=0):
    metricas = consulta_anuais_metricas(
        endpoint,
        indicador,
        data_min=data_min,
        base_calculo=base_calculo,
        metricas=("Mediana",),
    )
    return metricas.get("Mediana", pd.DataFrame())


def _year_offset_from_annual(pv, offset):
    if pv.empty:
        return pd.Series(dtype=float)

    vals = []
    for dt in pv.index:
        ano_alvo = dt.year + offset
        vals.append(pv.at[dt, ano_alvo] if ano_alvo in pv.columns else pd.NA)
    return pd.to_numeric(pd.Series(vals, index=pv.index), errors="coerce")


def _horizon_12m_from_annual(pv):
    if pv.empty:
        return pd.Series(dtype=float)

    idx = pv.index
    f0 = _year_offset_from_annual(pv, 0)
    f1 = _year_offset_from_annual(pv, 1)

    anos = idx.year.astype(str)
    dias_no_ano = pd.to_datetime(anos + "-12-31").dayofyear.to_numpy(dtype=float)
    frac_restante = (dias_no_ano - idx.dayofyear) / dias_no_ano

    return frac_restante * f0 + (1.0 - frac_restante) * f1


def _avg_year_range_from_annual(pv, start_offset, end_offset):
    if pv.empty:
        return pd.Series(dtype=float)

    series = [_year_offset_from_annual(pv, k) for k in range(start_offset, end_offset + 1)]
    return pd.concat(series, axis=1).mean(axis=1, skipna=True)


def _extrair_horizonte_annual(pv, horizonte):
    if horizonte == "12m":
        return _horizon_12m_from_annual(pv)
    if horizonte == "5y":
        return _year_offset_from_annual(pv, 5)
    if horizonte == "3to5y":
        return _avg_year_range_from_annual(pv, 3, 5)
    if isinstance(horizonte, str) and horizonte.startswith("y") and horizonte[1:].isdigit():
        return _year_offset_from_annual(pv, int(horizonte[1:]))
    raise ValueError(f"Horizonte inválido: {horizonte}")


def consulta_ipca_mes_corrente(endpoint, data_min=DATA_MIN):
    """
    IPCA esperado para o mês corrente (DataReferencia == mês/ano da Data).
    Retorna DataFrame com colunas Media e Mediana.
    """
    df = (
        endpoint.query()
        .filter(endpoint.Indicador == "IPCA")
        .filter(endpoint.Data >= data_min)
        .select(endpoint.Data, endpoint.DataReferencia, endpoint.Media, endpoint.Mediana)
        .collect()
    )

    if df.empty:
        print("[AVISO] Sem dados mensais de IPCA.")
        return pd.DataFrame()

    df["Data"] = _to_dt(df["Data"])
    df["DataReferencia"] = pd.to_datetime(df["DataReferencia"], format="%m/%Y", errors="coerce")

    mask = (
        (df["Data"].dt.year == df["DataReferencia"].dt.year)
        & (df["Data"].dt.month == df["DataReferencia"].dt.month)
    )
    df = df.loc[mask, ["Data", "Media", "Mediana"]].drop_duplicates(subset=["Data"], keep="last")
    out = df.set_index("Data").sort_index()

    print(f"[LOG] IPCA mês corrente shape: {out.shape}")
    return out


def consulta_ipca_12m(endpoint, data_min=DATA_MIN, suavizada_preferida=("S", "N")):
    """
    IPCA esperado em 12 meses (endpoint dedicado).
    Retorna DataFrame com Media e Mediana.
    """
    df = (
        endpoint.query()
        .filter(endpoint.Indicador == "IPCA")
        .filter(endpoint.Data >= data_min)
        .select(endpoint.Data, endpoint.Suavizada, endpoint.Media, endpoint.Mediana)
        .collect()
    )

    if df.empty:
        print("[AVISO] Sem dados de inflação 12 meses para IPCA.")
        return pd.DataFrame()

    df["Data"] = _to_dt(df["Data"])
    if "Suavizada" in df.columns:
        cats = list(suavizada_preferida)
        df["Suavizada"] = pd.Categorical(df["Suavizada"], categories=cats, ordered=True)
        df = df.sort_values(["Data", "Suavizada"]).drop_duplicates(subset=["Data"], keep="first")

    out = df[["Data", "Media", "Mediana"]].set_index("Data").sort_index()
    print(f"[LOG] IPCA 12m shape: {out.shape}")
    return out

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
    dados_div_pib = consulta_anuais_mediana(ep_anuais, "Dívida bruta do governo geral")

    # ===== FISCAL: Resultado primário, dívida líquida e resultado nominal =====
    dados_primario = consulta_anuais_mediana(ep_anuais, "Resultado primário")
    dados_div_liq = consulta_anuais_mediana(ep_anuais, "Dívida líquida do setor público")
    dados_res_nom = consulta_anuais_mediana(ep_anuais, "Resultado nominal")

    # ===== SETOR EXTERNO: Investimento direto, balança comercial e conta corrente =====
    dados_inv_dir = consulta_anuais_mediana(ep_anuais, "Investimento direto no país")
    dados_bal_com = consulta_anuais_mediana(ep_anuais, "Balança comercial")
    dados_cont_cor = consulta_anuais_mediana(ep_anuais, "Conta corrente")

    # ================== Novas séries de NÍVEL (Media/Mediana) ===================
    cache_anuais = {}

    def _get_anuais(indicador):
        if indicador not in cache_anuais:
            cache_anuais[indicador] = consulta_anuais_metricas(ep_anuais, indicador)
        return cache_anuais[indicador]

    def _nivel(indicador, horizonte):
        mm = _get_anuais(indicador)
        pv_media = mm.get("Media", pd.DataFrame())
        pv_mediana = mm.get("Mediana", pd.DataFrame())

        s_media = _extrair_horizonte_annual(pv_media, horizonte) if not pv_media.empty else pd.Series(dtype=float)
        s_mediana = _extrair_horizonte_annual(pv_mediana, horizonte) if not pv_mediana.empty else pd.Series(dtype=float)

        out = pd.DataFrame({"Media": s_media, "Mediana": s_mediana})
        out = out.dropna(how="all")
        return out

    def _incerteza(indicador, horizonte):
        mm = _get_anuais(indicador)
        pv_sd = mm.get("DesvioPadrao", pd.DataFrame())
        if pv_sd.empty:
            return pd.DataFrame()
        s_sd = _extrair_horizonte_annual(pv_sd, horizonte)
        out = pd.DataFrame({"DesvioPadrao": s_sd}).dropna(how="all")
        return out

    # nível
    dados_ipca_mes_corrente = consulta_ipca_mes_corrente(ep_mensais)
    dados_ipca_12m = consulta_ipca_12m(ep_inflacao_12m)
    dados_pib_12m = _nivel("PIB Total", "12m")
    dados_ind_12m = _nivel("Produção industrial", "12m")
    dados_imp_12m = _nivel("PIB Importação de bens e serviços", "12m")
    dados_exp_12m = _nivel("PIB Exportação de bens e serviços", "12m")
    dados_imp_3a5 = _nivel("PIB Importação de bens e serviços", "3to5y")
    dados_fiscal_3a5 = _nivel("Resultado nominal", "3to5y")
    dados_prim_12m = _nivel("Resultado primário", "12m")
    dados_div_12m = _nivel("Dívida bruta do governo geral", "12m")
    dados_div_3a5 = _nivel("Dívida bruta do governo geral", "3to5y")

    # incerteza (DesvioPadrao)
    sd_pib_serv_12m = _incerteza("PIB Serviços", "12m")
    sd_div_3a5 = _incerteza("Dívida bruta do governo geral", "3to5y")
    sd_exp_3a5 = _incerteza("PIB Exportação de bens e serviços", "3to5y")
    sd_ind_3a5 = _incerteza("Produção industrial", "3to5y")
    sd_cc_3a5 = _incerteza("Conta corrente", "3to5y")

    # Dicionário de abas -> DataFrames
    sheets = {
        "IPCA":   dados_ipca,
        "Selic":  dados_selic,
        "Câmbio": dados_cambio,
        "PIB":    dados_pib,
        "DebtGDP": dados_div_pib,
        "primário": dados_primario,
        "DLSP": dados_div_liq,
        "nominal": dados_res_nom,
        "idp": dados_inv_dir,
        "BC": dados_bal_com,
        "CA": dados_cont_cor,
        "CPI_CurrentMonth": dados_ipca_mes_corrente,
        "CPI_Next12M": dados_ipca_12m,
        "GDP_Next12M": dados_pib_12m,
        "IndProd_Next12M": dados_ind_12m,
        "Import_Next12M": dados_imp_12m,
        "Export_Next12M": dados_exp_12m,
        "Import_Avg_3to5Y": dados_imp_3a5,
        "FiscalBal_Avg_3to5Y": dados_fiscal_3a5,
        "PrimaryBal_Next12M": dados_prim_12m,
        "DebtGDP_Next12M": dados_div_12m,
        "DebtGDP_Avg_3to5Y": dados_div_3a5,
        "SD_GDPServices_12M": sd_pib_serv_12m,
        "SD_DebtGDP_3to5Y": sd_div_3a5,
        "SD_Export_3to5Y": sd_exp_3a5,
        "SD_IndProd_3to5Y": sd_ind_3a5,
        "SD_CurrAcc_3to5Y": sd_cc_3a5,
    }

    # ================== Escrita em Excel (arquivo novo) ===================
    with pd.ExcelWriter(FILE_PATH, engine="openpyxl", mode="w") as writer:
        for sheet_name, df in sheets.items():
            if df.empty:
                df = pd.DataFrame({"SemDados": [pd.NA]})
                print(f"[AVISO] Sem dados para '{sheet_name}'. Aba criada com placeholder.")

            df.to_excel(writer, sheet_name=sheet_name, index=True)
            print(f"[LOG] Aba '{sheet_name}' escrita com shape {df.shape}")

    print(f"Arquivo {FILE_PATH} salvo com sucesso.")

main_focus()