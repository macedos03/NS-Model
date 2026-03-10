"""
Visualizações de Avalição OOS — Passos 5–8

Gera gráficos de:
- Yields previstos vs observados
- Distribuição de erros
- Desempenho comparativo (MAE, RelMAE, Win Rate)
- Performance por horizonte e maturidade
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Config estético
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
COLORS = {
    'dl': '#002060',
    'dlfavar_all': '#0070C0',
    'dlfavar_group': '#A6CAEC',
    'dlfavar_fira': '#156082',
    'rw_yield': '#696969'
}

# Mapas de nomes amigáveis
MODEL_LABELS = {
    'dl': 'DL (Baseline)',
    'dlfavar_all': 'FAVAR(all)',
    'dlfavar_group': 'FAVAR(group) ⭐',
    'dlfavar_fira': 'FAVAR(fira)',
    'rw_yield': 'Random Walk'
}

TAU_LABELS = {
    1.0: '1Y',
    3.0: '3Y',
    5.0: '5Y',
    10.0: '10Y'
}

HORIZON_LABELS = {
    4: '4 weeks',
    13: '3 months',
    26: '6 months',
    52: '1 year'
}


def load_evaluation_data(output_dir: str = None):
    """Carrega todos os dados de avaliação OOS."""
    if output_dir is None:
        output_dir = Path('NS v2/reports/evaluation')
    else:
        output_dir = Path(output_dir)
    
    print(f"Carregando dados de {output_dir}")
    
    data = {
        'panel_long': pd.read_csv(output_dir / 'eval_oos_panel_long.csv'),
        'mae': pd.read_csv(output_dir / 'eval_oos_mae.csv'),
        'relmae': pd.read_csv(output_dir / 'eval_oos_relmae_vs_rw.csv'),
        'winrate': pd.read_csv(output_dir / 'eval_oos_winrate_vs_rw.csv'),
        'bias': pd.read_csv(output_dir / 'eval_oos_bias.csv'),
        'coverage': pd.read_csv(output_dir / 'eval_oos_coverage.csv')
    }
    
    # Parse datas
    data['panel_long']['target_week_ref'] = pd.to_datetime(
        data['panel_long']['target_week_ref']
    )
    
    return data


# ============================================================================
# 1. YIELDS PREVISTOS VS OBSERVADOS (séries temporais)
# ============================================================================

def plot_yields_timeseries(data: dict, output_dir: str = None, maturities: list = None):
    """
    Plota yields previstos vs observados ao longo do tempo por maturidade.
    Mostra trajetória para cada modelo em cores diferentes.
    """
    if maturities is None:
        maturities = [1.0, 5.0, 10.0]
    
    df = data['panel_long'].copy()
    
    # Filter para h=52 (1 ano ahead) para clareza
    df_h52 = df[df['horizon_steps'] == 52].sort_values('target_week_ref')
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Yields Previstos vs Observados (h=52 semanas)', fontsize=14, fontweight='bold')
    
    for idx, tau in enumerate(maturities):
        ax = axes[idx]
        df_tau = df_h52[df_h52['tau_years'] == tau].copy()
        
        if df_tau.empty:
            continue
        
        # Observed (único por data)
        observed = df_tau.groupby('target_week_ref')['yield_obs'].mean()
        ax.plot(observed.index, observed.values, 'k-', linewidth=3, label='Observed', zorder=10)
        
        # Previsões por modelo
        for model in df_tau['model_id'].unique():
            if model == 'rw_yield':
                continue
            df_m = df_tau[df_tau['model_id'] == model].sort_values('target_week_ref')
            ax.plot(df_m['target_week_ref'], df_m['yield_pred'],
                   color=COLORS.get(model, None),
                   label=MODEL_LABELS.get(model, model),
                   alpha=0.7, linewidth=1.5)
        
        ax.set_title(f'Maturity: {TAU_LABELS[tau]}', fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Yield (%)')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    if output_dir:
        fig.savefig(Path(output_dir) / '01_yields_timeseries.png', dpi=300, bbox_inches='tight')
    print("✓ Gráfico 1: Yields Timeseries")
    plt.close(fig)


# ============================================================================
# 2. DISTRIBUIÇÃO DE ERROS (box plots por horizonte)
# ============================================================================

def plot_error_distribution(data: dict, output_dir: str = None):
    """
    Box plots de erros de previsão por horizonte e modelo.
    Mostra a distribuição dos erros (não valor absoluto).
    """
    df = data['panel_long'].copy()
    
    # Exclude RW para foco em modelos
    df = df[df['model_id'] != 'rw_yield'].copy()
    df['horizon_label'] = df['horizon_steps'].map(HORIZON_LABELS)
    
    # Reorder
    model_order = ['dl', 'dlfavar_all', 'dlfavar_fira', 'dlfavar_group']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Distribution of Forecast Errors by Horizon', fontsize=14, fontweight='bold')
    
    for idx, h in enumerate([4, 13, 26, 52]):
        ax = axes[idx // 2, idx % 2]
        df_h = df[df['horizon_steps'] == h]
        
        sns.boxplot(data=df_h, x='tau_years', y='error', hue='model_id',
                   hue_order=model_order, ax=ax, palette=COLORS)
        
        ax.set_title(f'{HORIZON_LABELS[h]}', fontweight='bold', fontsize=12)
        ax.set_ylabel('Error (p.p.)')
        ax.set_xlabel('Maturity (years)')
        ax.axhline(0, color='red', linestyle='--', alpha=0.5)
        
        # Update legend labels
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, [MODEL_LABELS.get(m, m) for m in labels], 
                 title='Model', fontsize=9)
        
        # Format x-axis
        ax.set_xticklabels([TAU_LABELS.get(float(x.get_text()), x.get_text()) 
                            for x in ax.get_xticklabels()])
    
    plt.tight_layout()
    if output_dir:
        fig.savefig(Path(output_dir) / '02_error_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Gráfico 2: Error Distribution")
    plt.close(fig)


# ============================================================================
# 3. ABSOLUTE ERRORS (MAE) — Heatmap por modelo × maturidade
# ============================================================================

def plot_mae_heatmap(data: dict, output_dir: str = None):
    """
    Heatmap de MAE (erro absoluto médio) por modelo e maturidade.
    Mais escuro = melhor (menor erro).
    """
    df_mae = data['mae'].copy()
    
    # Reshape: (horizon, model) com valores para cada tau
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Mean Absolute Error (MAE) by Model and Maturity', 
                 fontsize=14, fontweight='bold')
    
    for idx, h in enumerate([4, 13, 26, 52]):
        ax = axes[idx // 2, idx % 2]
        df_h = df_mae[df_mae['horizon_steps'] == h].set_index('model_id')
        
        # Select только tau columns
        tau_cols = ['tau_1y', 'tau_3y', 'tau_5y', 'tau_10y']
        df_h = df_h[tau_cols].rename(columns={
            'tau_1y': '1Y', 'tau_3y': '3Y', 'tau_5y': '5Y', 'tau_10y': '10Y'
        })
        
        # Reorder models
        model_order = [m for m in ['dl', 'dlfavar_all', 'dlfavar_fira', 'dlfavar_group', 'rw_yield']
                      if m in df_h.index]
        df_h = df_h.loc[model_order]
        
        # Rename com labels amigáveis
        df_h.index = [MODEL_LABELS.get(m, m) for m in df_h.index]
        
        sns.heatmap(df_h, annot=True, fmt='.3f', cmap='RdYlGn_r', ax=ax,
                   cbar_kws={'label': 'MAE (p.p.)'})
        ax.set_title(f'h = {HORIZON_LABELS[h]}', fontweight='bold')
        ax.set_ylabel('Model')
        ax.set_xlabel('Maturity')
    
    plt.tight_layout()
    if output_dir:
        fig.savefig(Path(output_dir) / '03_mae_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ Gráfico 3: MAE Heatmap")
    plt.close(fig)


# ============================================================================
# 4. RELATIVE MAE vs RW — Ranking
# ============================================================================

def plot_relmae_ranking(data: dict, output_dir: str = None):
    """
    Ranking de RelMAE (MAE relativo ao benchmark RW).
    Barra por modelo, cor verde = melhor (< 1.0).
    """
    df_relmae = data['relmae'].copy()
    
    # Média across all horizons and maturities
    df_relmae['rel_mae_avg'] = df_relmae[[col for col in df_relmae.columns 
                                          if col.startswith('tau_')]].mean(axis=1)
    df_relmae = df_relmae.sort_values('rel_mae_avg')
    
    # Rename models
    df_relmae['model_label'] = df_relmae['model_id'].map(MODEL_LABELS)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Cores: verde se < 1, vermelho se > 1
    colors = ['#2ca02c' if x < 1 else '#d62728' 
              for x in df_relmae['rel_mae_avg']]
    
    bars = ax.barh(df_relmae['model_label'], df_relmae['rel_mae_avg'], color=colors, alpha=0.7)
    ax.axvline(1.0, color='black', linestyle='--', linewidth=2, label='RW Baseline')
    
    # Annotate values
    for i, (idx, row) in enumerate(df_relmae.iterrows()):
        ax.text(row['rel_mae_avg'] + 0.02, i, f"{row['rel_mae_avg']:.3f}", 
               va='center', fontweight='bold')
    
    ax.set_xlabel('Relative MAE (vs Random Walk)', fontweight='bold')
    ax.set_title('Model Ranking by Relative MAE\n(Lower = Better)', fontweight='bold', fontsize=12)
    ax.set_xlim(0.7, 1.1)
    ax.legend()
    ax.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    if output_dir:
        fig.savefig(Path(output_dir) / '04_relmae_ranking.png', dpi=300, bbox_inches='tight')
    print("✓ Gráfico 4: RelMAE Ranking")
    plt.close(fig)


# ============================================================================
# 5. WIN RATE vs RW — Comparative Bar Chart
# ============================================================================

def plot_winrate_vs_rw(data: dict, output_dir: str = None):
    """
    Win rate de cada modelo contra Random Walk.
    Mostra % de previsões que melhor RW (de forma absoluta).
    """
    df_wr = data['winrate'].copy()
    
    # Rename
    df_wr['model_label'] = df_wr['model_id'].map(MODEL_LABELS)
    df_wr = df_wr[df_wr['model_id'] != 'rw_yield'].sort_values('winrate_vs_rw_pct', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(df_wr['model_label'], df_wr['winrate_vs_rw_pct'], 
                  color=[COLORS.get(m, '#999999') for m in df_wr['model_id']],
                  alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Benchmark 50%
    ax.axhline(50, color='red', linestyle='--', linewidth=2, label='50% (neutral)', alpha=0.7)
    
    # Annotate values
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
               f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Win Rate (%)', fontweight='bold')
    ax.set_title('Win Rate vs Random Walk\n(% forecasts with lower absolute error)', 
                fontweight='bold', fontsize=12)
    ax.set_ylim(0, 65)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.xticks(rotation=12)
    plt.tight_layout()
    if output_dir:
        fig.savefig(Path(output_dir) / '05_winrate_vs_rw.png', dpi=300, bbox_inches='tight')
    print("✓ Gráfico 5: Win Rate vs RW")
    plt.close(fig)


# ============================================================================
# 6. PERFORMANCE POR HORIZONTE — Linha para cada modelo
# ============================================================================

def plot_mae_by_horizon(data: dict, output_dir: str = None):
    """
    MAE vs Horizonte (em semanas) para cada modelo.
    Mostra se acurácia piora com horizonte maior.
    """
    df = data['panel_long'].copy()
    
    # Exclude RW
    df = df[df['model_id'] != 'rw_yield']
    
    # Group por (model, horizonte) → média de abs_error
    df_mae_h = df.groupby(['model_id', 'horizon_steps'])['abs_error'].mean().reset_index()
    df_mae_h['model_label'] = df_mae_h['model_id'].map(MODEL_LABELS)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for model in df_mae_h['model_id'].unique():
        df_m = df_mae_h[df_mae_h['model_id'] == model].sort_values('horizon_steps')
        ax.plot(df_m['horizon_steps'], df_m['abs_error'],
               marker='o', linewidth=2.5, markersize=8,
               color=COLORS.get(model),
               label=MODEL_LABELS.get(model))
    
    ax.set_xlabel('Forecast Horizon (weeks)', fontweight='bold')
    ax.set_ylabel('Mean Absolute Error (p.p.)', fontweight='bold')
    ax.set_title('Error Growth by Forecast Horizon', fontweight='bold', fontsize=12)
    ax.set_xticks([4, 13, 26, 52])
    ax.set_xticklabels(['4\n(1mo)', '13\n(3mo)', '26\n(6mo)', '52\n(1yr)'])
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if output_dir:
        fig.savefig(Path(output_dir) / '06_mae_by_horizon.png', dpi=300, bbox_inches='tight')
    print("✓ Gráfico 6: MAE by Horizon")
    plt.close(fig)


# ============================================================================
# 7. PERFORMANCE POR MATURIDADE — Linha para cada modelo
# ============================================================================

def plot_mae_by_maturity(data: dict, output_dir: str = None):
    """
    MAE vs Maturidade (1Y, 3Y, 5Y, 10Y) para cada modelo.
    Mostra se modelo é melhor em certos segmentos.
    """
    df = data['panel_long'].copy()
    df = df[df['model_id'] != 'rw_yield']
    
    # Group por (model, tau)
    df_mae_tau = df.groupby(['model_id', 'tau_years'])['abs_error'].mean().reset_index()
    df_mae_tau['model_label'] = df_mae_tau['model_id'].map(MODEL_LABELS)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for model in df_mae_tau['model_id'].unique():
        df_m = df_mae_tau[df_mae_tau['model_id'] == model].sort_values('tau_years')
        ax.plot(df_m['tau_years'], df_m['abs_error'],
               marker='s', linewidth=2.5, markersize=8,
               color=COLORS.get(model),
               label=MODEL_LABELS.get(model))
    
    ax.set_xlabel('Yield Maturity (years)', fontweight='bold')
    ax.set_ylabel('Mean Absolute Error (p.p.)', fontweight='bold')
    ax.set_title('Error by Maturity Segment', fontweight='bold', fontsize=12)
    ax.set_xticks([1, 3, 5, 10])
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if output_dir:
        fig.savefig(Path(output_dir) / '07_mae_by_maturity.png', dpi=300, bbox_inches='tight')
    print("✓ Gráfico 7: MAE by Maturity")
    plt.close(fig)


# ============================================================================
# 8. HEATMAP: RelMAE por Horizonte × Maturidade (um por modelo)
# ============================================================================

def plot_relmae_heatmaps(data: dict, output_dir: str = None):
    """
    Heatmaps de RelMAE (vs RW) para cada modelo.
    Mostra padrão de desempenho por (horizonte, maturidade).
    """
    df_mae = data['mae'].copy()  # Use MAE data to extract horizons
    df_relmae = data['relmae'].copy()
    
    # Calcula RelMAE por horizonte: MAE_model / MAE_RW
    df_rw = df_mae[df_mae['model_id'] == 'rw_yield'].copy()
    
    models = [m for m in ['dl', 'dlfavar_all', 'dlfavar_fira', 'dlfavar_group']
             if m in df_mae['model_id'].values]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Relative MAE by Horizon and Maturity\n(vs Random Walk, Green=Better)', 
                fontweight='bold', fontsize=14)
    
    for idx, model in enumerate(models):
        ax = axes[idx // 2, idx % 2]
        
        df_m = df_mae[df_mae['model_id'] == model].copy()
        
        # Merge with RW baseline
        df_m = df_m.merge(df_rw[['horizon_steps', 'tau_1y', 'tau_3y', 'tau_5y', 'tau_10y']],
                         on='horizon_steps', suffixes=('_model', '_rw'))
        
        # Calcular RelMAE para cada tau
        for tau_col in ['tau_1y', 'tau_3y', 'tau_5y', 'tau_10y']:
            df_m[tau_col + '_rel'] = df_m[tau_col + '_model'] / df_m[tau_col + '_rw']
        
        # Pivot para heatmap
        df_heat = df_m[['horizon_steps', 'tau_1y_rel', 'tau_3y_rel', 'tau_5y_rel', 'tau_10y_rel']].set_index('horizon_steps')
        df_heat.columns = ['1Y', '3Y', '5Y', '10Y']
        
        # Reorder by horizonte
        df_heat = df_heat.reindex([4, 13, 26, 52])
        
        sns.heatmap(df_heat, annot=True, fmt='.3f', cmap='RdYlGn_r', ax=ax,
                   cbar_kws={'label': 'RelMAE vs RW'}, vmin=0.7, vmax=1.05)
        ax.set_title(MODEL_LABELS[model], fontweight='bold', fontsize=11)
        ax.set_ylabel('Horizon (weeks)')
        ax.set_xlabel('Maturity')
    
    plt.tight_layout()
    if output_dir:
        fig.savefig(Path(output_dir) / '08_relmae_heatmaps.png', dpi=300, bbox_inches='tight')
    print("✓ Gráfico 8: RelMAE Heatmaps")
    plt.close(fig)


# ============================================================================
# 9. FORECAST BIAS — Média de erros (deteta over/under prediction)
# ============================================================================

def plot_bias_comparison(data: dict, output_dir: str = None):
    """
    Bias (erro médio, não absoluto) para detectar se modelo
    sistematicamente over/under-predicts.
    """
    df_bias = data['bias'].copy()
    
    # Média across maturities por modelo
    df_bias['bias_avg'] = df_bias[[col for col in df_bias.columns 
                                    if col.startswith('tau_')]].mean(axis=1)
    
    df_bias['model_label'] = df_bias['model_id'].map(MODEL_LABELS)
    df_bias = df_bias[df_bias['model_id'] != 'rw_yield']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Cores: verde se próximo de 0, vermelho se negativo/positivo
    colors = ['#2ca02c' if abs(x) < 0.05 else ('#d62728' if x < -0.05 else '#ff7f0e')
              for x in df_bias['bias_avg']]
    
    bars = ax.bar(df_bias['model_label'], df_bias['bias_avg'], color=colors, alpha=0.7)
    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    
    # Annotate
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{height:.3f}', ha='center', va='bottom' if height > 0 else 'top',
               fontweight='bold', fontsize=10)
    
    ax.set_ylabel('Average Bias (p.p.)', fontweight='bold')
    ax.set_title('Forecast Bias by Model\n(Positive = Over-prediction, Negative = Under-prediction)',
                fontweight='bold', fontsize=12)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.xticks(rotation=12)
    plt.tight_layout()
    if output_dir:
        fig.savefig(Path(output_dir) / '09_bias_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Gráfico 9: Bias Comparison")
    plt.close(fig)


# ============================================================================
# 10. SCATTER: Previsão vs Observado (diagnóstico de viés)
# ============================================================================

def plot_pred_vs_obs_scatter(data: dict, output_dir: str = None):
    """
    Scatter plot: predicted vs observed yields.
    Pontos próximos da diagonal = bom ajuste.
    Spread vertical = underestimação de variância.
    """
    df = data['panel_long'].copy()
    df = df[df['model_id'] != 'rw_yield']
    
    models = sorted(df['model_id'].unique())
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Predicted vs Observed Yields\n(Perfect forecast on 45° line)', 
                fontweight='bold', fontsize=14)
    
    for idx, model in enumerate(models):
        ax = axes[idx // 2, idx % 2]
        df_m = df[df['model_id'] == model]
        
        # Sample para evitar overplot
        if len(df_m) > 5000:
            df_m = df_m.sample(5000, random_state=42)
        
        ax.scatter(df_m['yield_obs'], df_m['yield_pred'], alpha=0.3, s=10,
                  color=COLORS.get(model))
        
        # Diagonal perfeita
        ylim = (df_m['yield_obs'].min(), df_m['yield_obs'].max())
        ax.plot([ylim[0], ylim[1]], [ylim[0], ylim[1]], 'k--', linewidth=2, alpha=0.7)
        
        # R² approximado
        corr = df_m[['yield_pred', 'yield_obs']].corr().iloc[0, 1]
        
        ax.set_xlabel('Observed Yield (%)', fontweight='bold')
        ax.set_ylabel('Predicted Yield (%)', fontweight='bold')
        ax.set_title(f'{MODEL_LABELS[model]} (r={corr:.3f})', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if output_dir:
        fig.savefig(Path(output_dir) / '10_pred_vs_obs_scatter.png', dpi=300, bbox_inches='tight')
    print("✓ Gráfico 10: Pred vs Obs Scatter")
    plt.close(fig)


# ============================================================================
# 11. DAILY FIT QUALITY (RMSE) — estilo backtest_betas_rmse (parte 2)
# ============================================================================

def plot_dailyfit_quality_rmse(ns_diag_path: str = None, output_dir: str = None):
    """
    Série temporal de qualidade do ajuste diário (fit_rmse).
    Replica a parte 2 do gráfico backtest_betas_rmse.png em um painel dedicado.
    """
    if ns_diag_path is None:
        ns_diag_path = Path('NS v2/data/ns_fit_diagnostics_weekly.csv')
    else:
        ns_diag_path = Path(ns_diag_path)

    if output_dir is None:
        output_dir = Path('NS v2/reports/evaluation')
    else:
        output_dir = Path(output_dir)

    if not ns_diag_path.exists():
        raise FileNotFoundError(f"Arquivo de diagnóstico não encontrado: {ns_diag_path}")

    df_diag = pd.read_csv(ns_diag_path)
    required_cols = {'week_ref', 'fit_rmse'}
    if not required_cols.issubset(df_diag.columns):
        raise ValueError(
            f"Colunas obrigatórias ausentes em {ns_diag_path}. "
            f"Esperadas: {sorted(required_cols)} | Presentes: {sorted(df_diag.columns.tolist())}"
        )

    df_diag = df_diag[['week_ref', 'fit_rmse']].copy()
    df_diag['week_ref'] = pd.to_datetime(df_diag['week_ref'], errors='coerce')
    df_diag['fit_rmse'] = pd.to_numeric(df_diag['fit_rmse'], errors='coerce')
    df_diag = df_diag.dropna(subset=['week_ref', 'fit_rmse']).sort_values('week_ref')

    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.plot(df_diag['week_ref'], df_diag['fit_rmse'], linewidth=1.5, color='#0B1F3A')

    ax.set_title('Daily Fit Quality (RMSE)', fontsize=12, fontweight='bold')
    ax.set_ylabel('RMSE (% a.a.)')
    ax.set_xlabel('Date')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / '11_dailyfit_quality_rmse.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"✓ Gráfico 11: Daily Fit Quality (RMSE) | {out_path}")
    plt.close(fig)


# ============================================================================
# MAIN: Gera todos os gráficos
# ============================================================================

def generate_all_graphs(data_dir: str = None, output_dir: str = None):
    """
    Gera todos os 10 gráficos.
    
    Args:
        data_dir: Diretório com eval_oos_*.csv (default: NS v2/reports/evaluation)
        output_dir: Onde salvar PNGs (default: data_dir)
    """
    if data_dir is None:
        data_dir = 'NS v2/reports/evaluation'
    if output_dir is None:
        output_dir = data_dir
    
    print("\n" + "="*70)
    print("GERANDO GRÁFICOS DE AVALIAÇÃO OOS (Passos 5-8)")
    print("="*70 + "\n")
    
    # Load dados
    data = load_evaluation_data(data_dir)
    
    # Gera todos
    plot_yields_timeseries(data, output_dir)
    plot_error_distribution(data, output_dir)
    plot_mae_heatmap(data, output_dir)
    plot_relmae_ranking(data, output_dir)
    plot_winrate_vs_rw(data, output_dir)
    plot_mae_by_horizon(data, output_dir)
    plot_mae_by_maturity(data, output_dir)
    plot_relmae_heatmaps(data, output_dir)
    plot_bias_comparison(data, output_dir)
    plot_pred_vs_obs_scatter(data, output_dir)
    ns_diag_path = Path(data_dir).parents[1] / 'data' / 'ns_fit_diagnostics_weekly.csv'
    plot_dailyfit_quality_rmse(ns_diag_path=ns_diag_path, output_dir=output_dir)
    
    print("\n" + "="*70)
    print("✅ TODOS OS 11 GRÁFICOS GERADOS")
    print(f"📁 Salvos em: {Path(output_dir).absolute()}")
    print("="*70 + "\n")


if __name__ == '__main__':
    # Executar: python graphs.py
    generate_all_graphs()
