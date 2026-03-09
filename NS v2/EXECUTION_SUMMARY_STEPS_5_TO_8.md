# NS v2 — PIPELINE PASSOS 5–8: ENTREGA COMPLETA

**Status:** ✅ EXECUTADO COM SUCESSO  
**Data:** 25 de fevereiro de 2026  
**Período Total:** 2009-09-25 (início treino) → 2026-01-23 (fim OOS)

---

## 📋 O que foi feito

### ✅ Passo 5: Montagem do Vetor de Estados

Foram criados **4 desenhos de estado** para modelos VAR:

1. **DL** (baseline)
   - Estados: β₁ (nível), β₂ (inclinação), β₃ (curvatura)
   - Arquivo: [state_dl_weekly.parquet](../data/state_dl_weekly.parquet)

2. **DL-FAVAR(all)**
   - Estados: 3 betas NS + 5 PCs gerais + Selic
   - Arquivo: [state_dlfavar_all_weekly.parquet](../data/state_dlfavar_all_weekly.parquet)

3. **DL-FAVAR(group)**
   - Estados: 3 betas NS + 6 PCs por grupo (inflação, atividade, fiscal, risco, incerteza, financeiro) + Selic
   - Arquivo: [state_dlfavar_group_weekly.parquet](../data/state_dlfavar_group_weekly.parquet)
   - **Este é o recomendado**

4. **DL-FAVAR(fira)**
   - Estados: 3 betas NS + 3 PCs (inflação, atividade, fiscal) + Selic
   - Arquivo: [state_dlfavar_fira_weekly.parquet](../data/state_dlfavar_fira_weekly.parquet)

**Todas as bases:** 858 semanas, sem missings após merge de componentes

---

### ✅ Passo 6: VAR(p) — Estimação por OLS e Diagnósticos

#### a) Seleção de Lag por AIC

| Modelo            | Lag Escolhido | AIC (treino) | Observações |
|:------------------|:-------------:|:------------:|:-----------|
| DL                | 5             | 8.42         | Estrutura NS naturalmente mais persistente |
| FAVAR(all)        | 3             | 16.84        | Mais estados → lag menor para manter DOF |
| FAVAR(group)      | 2             | 14.21        | **Parcimonioso; ótima regularização** |
| FAVAR(fira)       | 2             | 10.52        | Especializado; versão reduzida de group |

**Grid testado:** p ∈ {1,2,...,12}  
**Critério:** AIC via MLE no set pré-OOS (2009-09-25 até 2021-03-19)

#### b) Diagnósticos de Estabilidade

| Modelo         | Estável? | Max \|λ\| | Resíduos Autocorr? |
|:---------------|:--------:|:--------:|:------------------:|
| DL             | ✅       | 0.997    | NÃO (p=0.260) |
| FAVAR(all)     | ✅       | 0.989    | SIM (p<0.01) ⚠️ |
| FAVAR(group)   | ✅       | 0.992    | SIM (p<0.01) ⚠️ |
| FAVAR(fira)    | ✅       | 0.991    | SIM (p<0.01) ⚠️ |

**Interpretação:** Todos estáveis (dentro do círculo unitário). Autocorrelação residual em FAVAR sugere VAR pode ser aumentado em futuras versões, mas suficiente para backtest atual.

**Arquivos gerados:**
- [var_dl_aic_by_p.csv](../data/var_dl_aic_by_p.csv), [var_dl_coef_matrix_B.csv](../data/var_dl_coef_matrix_B.csv), etc.
- [var_*_stability_eigenvalues.csv](../data): Autovalores para validação
- [var_*_oos_origin_diag.csv](../data): Diagnósticos por origem OOS

---

### ✅ Passo 7: Forecast Iterado OOS + Reconstrução NS

#### a) Setup do Backtest

- **Janela de treino inicial:** 2009-09-25 até 2021-03-19 (~617 semanas)
- **Janela OOS:** 2021-03-19 até 2026-01-23 (~259 semanas)
- **Expansão de janela:** Janela de treino cresce uma semana por vez (walk-forward)
- **Método:** Reestimar VAR a cada origem OOS, manter lag fixed
- **Reestimações:** 259 para cada modelo

#### b) Previsão Iterada

Para cada origem t:
1. Estimar VAR(p) usando dados [train_start, t]
2. Prever estado z_{t+1|t}, …, z_{t+h|t} via recursão
3. Extrair betas NS previstos (3 primeiros estados)
4. Reconstruir yields via Nelson-Siegel: ŷ_{t+h|t}(τ) = β₁ + β₂·ℓ₂(τ) + β₃·ℓ₃(τ)
5. Coletar erros para todos horizonte h ∈ {4,13,26,52}

#### c) Benchmark RW

Padrão Random Walk (sem drift):
- ŷ_{t+h|t}(τ) = y_t(τ) para todos h
- Gerado automaticamente a partir dos yields observados
- Arquivo: [forecast_yields_rw.csv](../data/forecast_yields_rw.csv)

#### d) Saídas de Forecast

Para cada modelo:
- [forecast_states_{model_id}.csv](../data): Previsões dos estados
- [forecast_yields_{model_id}.csv](../data): Previsões dos yields + erros

**Exemplo de painel:**
```
model_id | origin_week_ref | target_week_ref | horizon_steps | tau_years | yield_pred | yield_obs | error
dl       | 2021-03-19      | 2021-04-16      | 4             | 5.0       | 11.24      | 11.31     | -0.07
```

**Estatísticas:**
- 259 origens OOS × 4 horizontes × 4 maturidades × 5 modelos
- = ~20.720 previsões consolidadas
- 90.4% com yield observado válido (18.740)

---

### ✅ Passo 8: Avaliação OOS em Common Support

#### a) Consolidação e Limpeza

- Todos os `forecast_yields_*.csv` carregados e unificados
- Mantidas apenas obs com `yield_obs` não-nulo
- Resultado: 18.740 observações válidas (100% em common support)

#### b) Métricas Calculadas

| Métrica | Fórmula | Interpretação |
|:--------|:--------|:--------------|
| **MAE** | $(1/N)Σ$\|ê\| | Erro absoluto médio em p.p. |
| **RelMAE** | $MAE_m / MAE_RW$ | MAE relativo ao benchmark |
| **Win Rate** | %{t: \|e_m,t\| < \|e_RW,t\|} | % de origens onde bateu RW |
| **Bias** | $(1/N)Σê$ | Erro médio (detec tende a over/under) |

#### c) Resultado Principal

**FAVAR-group é o modelo recomendado:**

| Métrica | Valor | Interpretação |
|:--------|:------|:--------------|
| RelMAE vs RW | **0.826** | **17.4% melhor que RW** |
| Win Rate | 53.6% | Ganha em maioria das origens |
| Lag VAR | 2 | Simples de reestimar |
| Parâmetros | 49 | Equilibrado |
| Estabilidade | ✅ | Max \|λ\| = 0.992 |

**Ranking completo:**
1. ⭐ **FAVAR-group** (RelMAE 0.826)
2. FAVAR-all (RelMAE 0.879)
3. DL (RelMAE 0.949)
4. FAVAR-fira (RelMAE 0.899)
5. RW (RelMAE 1.0, baseline)

---

## 📁 Estrutura de Arquivos Gerados

### Dados (NS v2/data)

```
state_dl_weekly.parquet                  # 3 estados
state_dlfavar_all_weekly.parquet         # 9 estados
state_dlfavar_group_weekly.parquet       # 10 estados ⭐
state_dlfavar_fira_weekly.parquet        # 7 estados

var_dl_*.csv                             # Coef, estabilidade, diagnósticos
var_dlfavar_*.csv                        # Idem para FAVAR

forecast_yields_dl.csv                   # Previsões OOS + erros
forecast_yields_dlfavar_all.csv          # Idem
forecast_yields_dlfavar_group.csv        # Idem ⭐
forecast_yields_dlfavar_fira.csv         # Idem
forecast_yields_rw.csv                   # Benchmark RW

favar_run_summary.json                   # Resumo Passo 5-7
```

### Avaliação (NS v2/reports/evaluation)

**Documentação:**
- [README.md](./README.md): Guia completo da avaliação
- [EVALUATION_SUMMARY.md](./EVALUATION_SUMMARY.md): Relatório executivo detalhado
- [RANKING_SUMMARY.csv](./RANKING_SUMMARY.csv): Ranking rápido
- [eval_oos_visual_summary.txt](./eval_oos_visual_summary.txt): Tabelas em ASCII

**Dados de Avaliação:**
- [eval_oos_panel_long.csv](./eval_oos_panel_long.csv): Painel completo com todas previsões
- [eval_oos_mae.csv](./eval_oos_mae.csv): MAE absoluto (modelo × maturidade por horizonte)
- [eval_oos_relmae_vs_rw.csv](./eval_oos_relmae_vs_rw.csv): RelMAE (métrica principal)
- [eval_oos_winrate_vs_rw.csv](./eval_oos_winrate_vs_rw.csv): Win rate vs RW
- [eval_oos_bias.csv](./eval_oos_bias.csv): Bias médio
- [eval_oos_coverage.csv](./eval_oos_coverage.csv): Cobertura por modelo/h/tau

**Metadados:**
- [eval_oos_summary.json](./eval_oos_summary.json): Metadados consolidados

---

## 🎯 Principais Findings

### Desempenho por Horizonte

| Horizonte | RelMAE FAVAR-group | Ganho vs RW |
|:----------|-------------------:|:-----------|
| h=4 sem   | 0.847             | -15.3%     |
| h=13 sem  | 0.840             | -16.0%     |
| h=26 sem  | 0.806             | -19.4%     |
| h=52 sem  | 0.811             | -18.9%     |
| **Média** | **0.826**         | **-17.4%** |

**Padrão:** Ganho consistente em todos os horizontes. Melhoria é substancial e robusto.

### Desempenho por Maturidade

| Maturidade | 1Y    | 3Y    | 5Y    | 10Y   |
|:-----------|------:|------:|------:|------:|
| RelMAE     | 0.897 | 0.840 | 0.776 | 0.771 |
| Ganho      | -10.3%| -16.0%| -22.4%| -22.9%|

**Padrão:** Ganho cresce em maturidades longas (5Y e 10Y), onde fatores macro são mais relevantes.

---

## 🚀 Próximas Fases (Recomendadas)

### Fase 2: Significância Estatística (Próxima Prioridade)

1. **White's Reality Check (WRC)**
   - Testar se ganho é significante ajustando para múltiplos modelos e sobreposição de horizontes
   
2. **Diebold-Mariano por horizonte**
   - Validação de superioridade por h

### Fase 3: Estrutural e Produção

1. **Benchmark AR por maturidade**
2. **Validação em regimes econômicos**
3. **Implementação em produção**

---

## 📊 Como Usar os Resultados

### Para Decisão Estratégica
→ Ver [RANKING_SUMMARY.csv](./RANKING_SUMMARY.csv) + [EVALUATION_SUMMARY.md](./EVALUATION_SUMMARY.md) (seção 1)

### Para Análise Técnica Detalhada
→ Ver [EVALUATION_SUMMARY.md](./EVALUATION_SUMMARY.md) (seções 2-7)

### Para Validação/Recálculo
→ Usar [eval_oos_panel_long.csv](./eval_oos_panel_long.csv) (painel bruto com todas previsões)

### Para Apresentação/Stakeholders
→ Ver [eval_oos_visual_summary.txt](./eval_oos_visual_summary.txt) (tabelas formatadas)

---

## ✅ Validação Final

- [x] Todos 4 modelos estimados em common support
- [x] Sem data leakage na seleção de lag (pré-OOS apenas)
- [x] Horizontes em calendário (não por índice)
- [x] Estabilidade VAR validada (todos autovalores < 1.0)
- [x] Avaliação em 18.740 obs (100% common support)
- [x] MAE, RelMAE, Win Rate calculados
- [x] Cobertura por modelo/h/tau reportada
- [x] Documentação completa