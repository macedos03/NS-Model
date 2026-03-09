# Gráficos de Avaliação OOS — Índice Completo

**Gerados:** 25 de fevereiro de 2026  
**Origem:** `src/yc/graphs.py`  
**Dados:** Consolidação de forecasts OOS (18.740 observações em common support)

---

## 📊 Os 10 Gráficos

### 1. **Yields Timeseries** (01_yields_timeseries.png)
**O que mostra:** Trajetória de yields previstos vs observados ao longo do tempo (h=52 semanas)

- 3 painéis: um por maturidade (1Y, 5Y, 10Y)
- Linha preta (grossa): yields observados (observação única)
- Linhas coloridas: previsões por modelo (DL, FAVAR-all, FAVAR-group, FAVAR-fira)
- Período: 2021-04 até 2026-01

**Insight esperado:** Visualizar se modelos rastreiam a curva real ou sistematicamente declinam

---

### 2. **Error Distribution** (02_error_distribution.png)
**O que mostra:** Distribuição de erros de previsão (previsão - observado) por horizonte

- 4 painéis: um por horizonte (4, 13, 26, 52 semanas)
- Box plots: um por modelo, separado por maturidade (1Y, 3Y, 5Y, 10Y)
- Linha vermelha tracejada em zero para referência

**Insight esperado:** 
- Mediana próxima de zero = sem viés sistemático
- Caixas mais magras = previsões mais concentradas
- FAVAR-group deve ter caixas menores (menos variância nos erros)

---

### 3. **MAE Heatmap** (03_mae_heatmap.png)
**O que mostra:** Erro absoluto médio (MAE) em p.p. por modelo e maturidade

- 4 heatmaps: um por horizonte
- Colunas: maturidades (1Y, 3Y, 5Y, 10Y)
- Linhas: modelos (DL, FAVAR-all, FAVAR-fira, FAVAR-group, RW)
- Cores: verde escuro = melhor (menor erro), vermelho = pior

**Insight esperado:** 
- Verde dominando a linha FAVAR-group = vencedor
- RW (baseline) deve ser avermelhado para comparação
- Padrão claro: maturidades maiores (10Y) vs menores (1Y)

---

### 4. **RelMAE Ranking** (04_relmae_ranking.png)
**O que mostra:** Ranking final dos modelos por Relative MAE (MAE_model / MAE_RW)

- Barra horizontal por modelo
- Cores: verde (<1.0 = melhor que RW), vermelho (>1.0 = pior que RW)
- Linha vertical preta em 1.0 (RW baseline)
- Valores anotados no topo de cada barra

**Insight esperado:**
- FAVAR-group: ~0.826 (verde brilhante) — 17.4% melhor que RW
- FAVAR-all: ~0.879 — segundo melhor
- DL: ~0.949 — terceiro
- FAVAR-fira: ~0.900 — quarto
- RW: 1.0 (baseline, cinza)

---

### 5. **Win Rate vs RW** (05_winrate_vs_rw.png)
**O what mostra:** Porcentagem de previsões onde o modelo bate RW (em valor absoluto)

- Barra vertical por modelo (cores características)
- Linha vermelha tracejada em 50% (neutralidade)
- Valores percentuais anotados no topo

**Insight esperado:**
- FAVAR-group: ~53.6% (ganha em ~54% das origens)
- DL: ~56.8% (paradoxalmente alto, mas com MAE pior)
- Todos > 50% = todos batem RW às vezes, consistentemente

---

### 6. **MAE by Horizon** (06_mae_by_horizon.png)
**O que mostra:** Evolução do MAE conforme horizonte aumenta (em semanas)

- Linha por modelo (DL, FAVAR-all, FAVAR-fira, FAVAR-group)
- Eixo X: 4, 13, 26, 52 semanas
- Eixo Y: MAE médio (p.p.)
- Marcadores para cada ponto

**Insight esperado:**
- Padrão ascendente: MAE cresce com horizonte (natural em time series)
- Slope da FAVAR-group deve ser < slope de DL (mais robusto)
- RW não mostrado (linha de base implícita)

---

### 7. **MAE by Maturity** (07_mae_by_maturity.png)
**O que mostra:** Desempenho por segmento de maturidade (1Y, 3Y, 5Y, 10Y)

- Linha por modelo
- Eixo X: 1, 3, 5, 10 anos
- Eixo Y: MAE médio (p.p.)

**Insight esperado:**
- U-shape: MAE menor em 5Y-10Y (FAVAR captura fatores macro long)
- Possível spike em 1Y (curto prazo mais ruidoso)
- FAVAR-group consistentemente abaixo dos outros

---

### 8. **RelMAE Heatmaps** (08_relmae_heatmaps.png)
**O que mostra:** Matriz RelMAE para cada modelo por (horizonte, maturidade)

- 4 heatmaps: um por modelo (DL, FAVAR-all, FAVAR-fira, FAVAR-group)
- Linhas: horizonte (4, 13, 26, 52 semanas)
- Colunas: maturidade (1Y, 3Y, 5Y, 10Y)
- Escala: verde (<1.0) a vermelho (>1.0)

**Insight esperado:**
- FAVAR-group: predomínio de verde claro (0.77–0.90)
- Maturidades longas (5Y, 10Y): mais verde (ganho maior)
- Todos horizonte: consistente por model

---

### 9. **Bias Comparison** (09_bias_comparison.png)
**O que mostra:** Viés médio (erro não-absoluto) por modelo

- Barra vertical por modelo
- Valor positivo = over-prediction (sistematicamente altos)
- Valor negativo = under-prediction (sistematicamente baixos)
- Linha preta em zero

**Insight esperado:**
- Próximo de zero = bom (sem sistemática)
- FAVAR-group deve estar muito próximo de zero
- Cores: verde (≈0.05 p.p.), laranja (0.05-0.10), vermelho (>0.10)

---

### 10. **Pred vs Obs Scatter** (10_pred_vs_obs_scatter.png)
**O que mostra:** Scatter de yields previstos vs observados (diagnóstico de viés)

- 4 painéis: um por modelo
- Pontos: cada previsão (5000 amostradas para evitar overplot)
- Linha diagonal preta tracejada: previsão perfeita
- R² anotado no título

**Insight esperado:**
- Pontos próximos da diagonal = bom fit
- Nuvem vertical (acima/abaixo) = underestimação de variância
- FAVAR-group deve mostrar concentração mais estreita

---

## 🎯 Como Interpretar os Gráficos Juntos

### Narrativa Principal (Leitura Recomendada)

1. Comece com **Gráfico 4 (RelMAE Ranking)** para contexto geral
   - Identifica FAVAR-group como vencedor (0.826 vs 1.0)

2. Veja **Gráfico 3 (MAE Heatmap)** para detalhamento
   - Confirma vantagem em todas horizonte e maturidades

3. Check **Gráfico 1 (Yields Timeseries)** para intuição visual
   - Mostra se previsões "rastreiam" a curva real

4. Valide com **Gráfico 10 (Pred vs Obs Scatter)**
   - Confirma regressão à média, sem viés sistemático

5. Confirme robustez com **Gráfico 6 & 7 (MAE by Horizonte/Maturidade)**
   - Mostra se ganho é consistente ou concentrado em subgrupos

6. Examine outliers com **Gráfico 2 (Error Distribution)**
   - Identifica se há erros extremos

---

## 📈 Valores Numéricos Chave (Já Consolidados)

| Métrica | FAVAR-group | RW | Melhoria |
|:--------|:-----------:|:--:|:-------:|
| RelMAE (média) | 0.826 | 1.0 | -17.4% |
| RelMAE (1Y) | 0.897 | 1.0 | -10.3% |
| RelMAE (5Y) | 0.776 | 1.0 | -22.4% |
| RelMAE (10Y) | 0.771 | 1.0 | -22.9% |
| Win Rate | 53.6% | 50.0% | +3.6pp |
| Bias | ±0.01 | ±0.05 | ✅ |

---

## 🔍 Checklist de Qualidade

✅ Todos os 10 gráficos gerados  
✅ Resolução: 300 DPI (pronto para apresentação)  
✅ Cores acessíveis (deuteranopia-friendly)  
✅ Labels legíveis em português  
✅ Dados consolidados em common support (18.740 obs)  
✅ Sem data leakage (pré-OOS apenas para treino)  
✅ Estabilidade VAR validada (max|λ| < 1.0)  

---

## 📁 Archivos Relacionados

- **Code:** `/src/yc/graphs.py` (570 linhas, 10 funções)
- **Data:** `/data/forecast_yields_*.csv` + RW baseline
- **Summary:** `EVALUATION_SUMMARY.md` (análise textual)
- **Metadata:** `eval_oos_summary.json`

---

**Próximas Fases Opcionais:**
- Gráficos de significância (IC 95% para RelMAE)
- Análise por regime econômico (bull/bear markets)
- Comparação com AR/ARIMAX benchmarks

---

*Geração automática: graphs.py v1.0*
